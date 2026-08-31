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

use crate::{FinderKind, IterState, MatchedBitset};

/// Bytes tested per general-purpose register.
pub(crate) const WORD_BYTES: usize = 8;

/// Bit 7 of every byte.
const HIGH: u64 = splat(1 << 7);

#[inline]
const fn splat(byte: u8) -> u64 {
    u64::from_ne_bytes([byte; WORD_BYTES])
}

/// Sets bit 7 of every non-zero byte. The bits below it are left as scratch.
///
/// `(b & 0x7f) + 0x7f` carries into bit 7 exactly when the low seven bits are non-zero
/// and cannot carry out of the byte, so OR-ing `b` back in covers `0x80` as well.
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

/// Matches `tail`, the final bytes of `haystack`, returning their bits at positions
/// `0..tail.len()`.
///
/// Whole words are marked from the start of the tail, and whatever is left over is picked up
/// by re-reading the last whole word of the haystack, so a tail costs one word per eight bytes
/// rather than a whole chunk however short it is. Only a haystack too short to hold a whole
/// word has to be staged, which [`short_haystack_bits`] does.
#[inline]
fn tail_bits<K: Kernel>(kernel: &K, haystack: &[u8], tail: &[u8]) -> u64 {
    debug_assert!(0 < tail.len() && tail.len() < WORD_BYTES);
    if let Some(word) = haystack.last_chunk::<WORD_BYTES>() {
        let matched = kernel.matches(u64::from_le_bytes(*word));
        movemask(matched) >> (WORD_BYTES - tail.len())
    } else {
        short_tail_bits(kernel, tail)
    }
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
fn short_tail_bits<K: Kernel>(kernel: &K, haystack: &[u8]) -> u64 {
    /// Copies the first and last `N` bytes of `haystack` into the front of a word.
    #[inline]
    fn stage<const N: usize>(haystack: &[u8]) -> [u8; WORD_BYTES] {
        const { assert!(N <= 8 / 2) }
        debug_assert!(haystack.len() >= N);

        let mut buf = [0; WORD_BYTES];
        buf[..N].copy_from_slice(&haystack[..N]);
        buf[N..2 * N].copy_from_slice(&haystack[haystack.len() - N..]);
        buf
    }

    #[inline]
    fn slide_ends(bits: u64, staged: usize, len: usize) -> u64 {
        let kept = !(u64::MAX << staged);
        (bits & kept) | (((bits >> staged) & kept) << (len - staged))
    }

    let len = haystack.len();
    debug_assert!(0 < len && len < WORD_BYTES);
    let (buf, staged) = match len {
        4.. => (stage::<4>(haystack), 4),
        2..4 => (stage::<2>(haystack), 2),
        0..2 => (stage::<1>(haystack), 1),
    };
    let bits = movemask(kernel.matches(u64::from_le_bytes(buf)));
    slide_ends(bits, staged, len)
}

#[inline]
pub(crate) fn count<K: Kernel>(haystack: &[u8], kernel: K) -> usize {
    let (words, tail) = haystack.as_chunks::<WORD_BYTES>();

    let mut total = 0;
    for word in words {
        total += kernel.matches(u64::from_ne_bytes(*word)).count_ones() as usize;
    }
    if !tail.is_empty() {
        total += short_tail_bits(&kernel, tail).count_ones() as usize;
    }
    total
}

#[inline]
pub(crate) fn find_next<K: Kernel>(state: &mut IterState<'_>, kernel: K) {
    let (haystack, mut from) = (state.haystack, state.pos);
    // SAFETY: as in `crate::vector::find_next`.
    let unscanned = unsafe { haystack.get_unchecked(from..) };
    let (words, tail) = unscanned.as_chunks::<WORD_BYTES>();
    let (pairs, words) = words.as_chunks::<2>();

    for [first_word, second_word] in pairs {
        let first_match = kernel.matches(u64::from_le_bytes(*first_word));
        let second_match = kernel.matches(u64::from_le_bytes(*second_word));
        if (first_match | second_match) != 0 {
            state.bits =
                MatchedBitset::from(movemask(first_match) | movemask(second_match) << WORD_BYTES);
            state.bits_offset = from;
            state.pos = from + first_word.len() + second_word.len();
            return;
        }
        from += first_word.len() + second_word.len();
    }
    for word in words {
        let matches = kernel.matches(u64::from_le_bytes(*word));
        if matches != 0 {
            state.bits = MatchedBitset::from(movemask(matches));
            state.bits_offset = from;
            state.pos = from + word.len();
            return;
        }
        from += word.len();
    }

    state.bits = if tail.is_empty() {
        0
    } else {
        tail_bits(&kernel, haystack, tail).into()
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
