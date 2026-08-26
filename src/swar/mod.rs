#![deny(clippy::inline_always)]

//! Searching eight bytes at a time in a general-purpose register.
//!
//! This is what runs where there are no vectors to reach for: a target `fearless_simd`
//! reports as its fallback level, where every lane operation is emulated one lane at a
//! time, and an explicitly requested [`crate::Backend::Scalar`].
//!
//! The win over a plain byte loop comes from the kernels an arithmetic trick can express:
//! one to three needles, or a range. A set that needs a table lookup has no such trick, because
//! SWAR has no gather, so it goes to [`crate::bytewise`] instead.

pub(crate) mod kernels;

use crate::{CHUNK_BYTES, FinderKind, IterState};

/// Bytes tested per general-purpose register.
pub(crate) const WORD_BYTES: usize = 8;

/// Bit 7 of every byte.
const HIGH: u64 = 0x8080_8080_8080_8080;

#[inline]
const fn splat(byte: u8) -> u64 {
    u64::from_ne_bytes([byte; WORD_BYTES])
}

/// Sets bit 7 of every non-zero byte. The bits below it are left as scratch.
///
/// `(b & 0x7f) + 0x7f` carries into bit 7 exactly when the low seven bits are non-zero
/// and cannot carry out of the byte, so OR-ing `b` back in covers `0x80` as well. The
/// classic `(w - LOW) & !w & HIGH` is two operations shorter but marks the wrong byte
/// when a zero borrows into its neighbour, and these marks have to be exact.
#[inline]
const fn nonzero_bytes(word: u64) -> u64 {
    ((word & !HIGH) + !HIGH) | word
}

/// Gathers bit 7 of each byte into the low eight bits.
///
/// Bit `8i + 7` times the multiplier's bit at `49 - 7i` lands on bit `56 + i`. No two of
/// the sixty-four partial products share a position, so nothing carries into the result,
/// and every off-diagonal one lands either below bit 56 or past bit 63.
#[inline]
pub(crate) const fn movemask(marks: u64) -> u64 {
    marks.wrapping_mul(0x0002_0408_1020_4081) >> 56
}

/// Tests [`WORD_BYTES`] bytes at a time, the word-at-a-time counterpart of [`crate::vector::Kernel`].
///
/// The result has bit 7 set in every matching byte and every other bit clear, which is
/// the form both [`movemask`] and [`u64::count_ones`] consume.
pub(crate) trait Kernel: Copy {
    fn from_kind(kind: &FinderKind) -> Option<Self>;

    fn matches(&self, word: u64) -> u64;
}

/// Marks every matching byte of a chunk, and the union of those marks.
///
/// The union is what lets [`find_next`] skip [`pack_marks`] for a chunk that did not
/// match at all: [`movemask`]'s multiply is the expensive half of a scan, and a scan
/// that finds nothing has no bit positions to report. This is the word-at-a-time
/// counterpart of the vector path's `any_lane_set` gate.
#[inline]
fn chunk_marks<K: Kernel>(
    kernel: &K,
    chunk: &[u8; CHUNK_BYTES],
) -> ([u64; CHUNK_BYTES / WORD_BYTES], u64) {
    let (words, []) = chunk.as_chunks::<WORD_BYTES>() else {
        unreachable!()
    };
    let mut marks = [0; CHUNK_BYTES / WORD_BYTES];
    let mut any = 0;
    for (i, word) in words.iter().enumerate() {
        marks[i] = kernel.matches(u64::from_le_bytes(*word));
        any |= marks[i];
    }
    (marks, any)
}

#[inline]
fn pack_marks(marks: [u64; CHUNK_BYTES / WORD_BYTES]) -> u64 {
    let mut bits = 0;
    for (i, marks) in marks.into_iter().enumerate() {
        bits |= movemask(marks) << (i * WORD_BYTES);
    }
    bits
}

#[inline]
fn chunk_bits<K: Kernel>(kernel: &K, chunk: &[u8; CHUNK_BYTES]) -> u64 {
    let (marks, _) = chunk_marks(kernel, chunk);
    pack_marks(marks)
}

/// Matches the final `len` bytes of `haystack`, returning their bits at positions
/// `0..len`.
///
/// Whole words are marked from the start of the tail, and whatever is left over is picked up
/// by re-reading the last whole word of the haystack, so a tail costs one word per eight bytes
/// rather than a whole chunk however short it is. Only a haystack too short to hold a whole
/// word has to be staged, which [`short_haystack_bits`] does.
#[inline]
fn tail_bits<K: Kernel>(kernel: &K, haystack: &[u8], len: usize) -> u64 {
    debug_assert!(0 < len && len < CHUNK_BYTES);
    let Some(last_word) = haystack.last_chunk::<WORD_BYTES>() else {
        return short_haystack_bits(kernel, &haystack[haystack.len() - len..]);
    };
    let (words, rest) = haystack[haystack.len() - len..].as_chunks::<WORD_BYTES>();
    debug_assert!(words.len() < CHUNK_BYTES / WORD_BYTES);

    let mut bits = 0;
    // The bound is a constant a tail can never reach, which is what lets the loop unroll
    // into shifts by constants instead of a variable-length chain.
    for (i, word) in words.iter().enumerate().take(CHUNK_BYTES / WORD_BYTES) {
        bits |= movemask(kernel.matches(u64::from_le_bytes(*word))) << (i * WORD_BYTES);
    }
    if !rest.is_empty() {
        // The re-read overlaps bytes that have already been scanned, either by the loop above
        // or by an earlier chunk; shifting their bits off the bottom discards them.
        let last = movemask(kernel.matches(u64::from_le_bytes(*last_word)));
        bits |= (last >> (WORD_BYTES - rest.len())) << (words.len() * WORD_BYTES);
    }
    bits
}

/// Matches a haystack shorter than one [`WORD_BYTES`], returning its bits at positions
/// `0..haystack.len()`.
///
/// Staged the way [`crate::vector`]'s short tail is, for the same reason. Write `n` for the
/// largest power of two that is at most the length: 4, 2 or 1. Copying the first and last `n`
/// bytes covers the whole of it, because the two ends overlap in the middle, and both copies
/// have a constant length and a constant destination. Sizing one copy to the length instead
/// lowers to a `memcpy` call that costs more than the scan it feeds.
#[inline]
fn short_haystack_bits<K: Kernel>(kernel: &K, haystack: &[u8]) -> u64 {
    /// Copies the first and last `N` bytes of `haystack` into the front of a word.
    #[inline]
    fn stage<const N: usize>(haystack: &[u8]) -> [u8; WORD_BYTES] {
        let mut buf = [0; WORD_BYTES];
        buf[..N].copy_from_slice(&haystack[..N]);
        buf[N..2 * N].copy_from_slice(&haystack[haystack.len() - N..]);
        buf
    }

    let len = haystack.len();
    debug_assert!(0 < len && len < WORD_BYTES);
    let (buf, staged) = match len {
        4.. => (stage::<4>(haystack), 4),
        2.. => (stage::<2>(haystack), 2),
        _ => (stage::<1>(haystack), 1),
    };
    // Neither end may keep more bits than it read bytes: above the front's sit the back's, and
    // above the back's sits the padding, which matches whenever the byte set contains zero.
    let bits = movemask(kernel.matches(u64::from_le_bytes(buf)));
    let kept = !(u64::MAX << staged);
    (bits & kept) | (((bits >> staged) & kept) << (len - staged))
}

/// Counts every matching byte of `haystack`.
#[inline]
pub(crate) fn count<K: Kernel>(haystack: &[u8], kernel: K) -> usize {
    let (words, tail) = haystack.as_chunks::<WORD_BYTES>();

    let mut total = 0;
    for word in words {
        total += kernel.matches(u64::from_le_bytes(*word)).count_ones() as usize;
    }
    // Only lane zero is inspected, so the zero padding above it cannot contribute a
    // spurious match even when the byte set contains zero.
    for &byte in tail {
        total += usize::from(kernel.matches(u64::from(byte)) & 0x80 != 0);
    }
    total
}

/// Scans from `from` for the first [`CHUNK_BYTES`] that contains a match.
#[inline]
pub(crate) fn find_next<K: Kernel>(state: &mut IterState<'_>, kernel: K) {
    let (haystack, from) = (state.haystack, state.pos);
    let (chunks, tail) = haystack[from..].as_chunks::<CHUNK_BYTES>();

    let mut chunks = chunks.iter().fuse().enumerate();
    // The first chunk packs its marks unconditionally. A dense haystack refills one
    // chunk at a time from `Iter::next`, so it nearly always matches right here, and
    // fusing the marking and packing passes keeps that path's instruction-level
    // parallelism. Only a sparse haystack reaches the gated loop below, where skipping
    // `movemask` across a long run of misses more than repays the pass wasted here.
    if let Some((_i, chunk)) = chunks.next() {
        let bits = chunk_bits(&kernel, chunk);
        if bits != 0 {
            state.bits = bits.into();
            state.bits_offset = from;
            state.pos = from + CHUNK_BYTES;
            return;
        }
    }

    for (i, chunk) in chunks {
        let (marks, any) = chunk_marks(&kernel, chunk);
        if any != 0 {
            let offset = from + i * CHUNK_BYTES;
            state.bits = pack_marks(marks).into();
            state.bits_offset = offset;
            state.pos = offset + CHUNK_BYTES;
            return;
        }
    }

    state.bits = if tail.is_empty() {
        0
    } else {
        tail_bits(&kernel, haystack, tail.len()).into()
    };
    state.bits_offset = haystack.len() - tail.len();
    state.pos = haystack.len();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn movemask_gathers_high_bits() {
        for expected in 0..256u64 {
            let mut marks = 0;
            for i in 0..WORD_BYTES {
                marks |= ((expected >> i) & 1) << (i * 8 + 7);
            }
            assert_eq!(movemask(marks), expected, "marks {marks:#018x}");
        }
    }
}
