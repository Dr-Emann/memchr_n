#![deny(clippy::inline_always)]

//! Searches eight bytes at a time for [`crate::Backend::Swar`] and SIMD fallback levels.

pub(crate) mod kernels;

use crate::{IterState, KernelData, MatchedBitset, ScanOps};

pub(crate) const WORD_BYTES: usize = 8;

const HIGH: u64 = splat(1 << 7);

#[inline]
pub(crate) const fn splat(byte: u8) -> u64 {
    u64::from_ne_bytes([byte; WORD_BYTES])
}

/// Sets bit 7 of every nonzero byte.
///
/// `(b & 0x7f) + 0x7f` carries into bit 7 exactly when the low seven bits are non-zero
/// and cannot carry out of the byte, so OR-ing `b` back in covers `0x80` as well.
#[inline]
const fn nonzero_bytes(word: u64) -> u64 {
    ((word & !HIGH) + !HIGH) | word
}

/// Gathers bit 7 of each byte into the low byte.
///
/// The multiplier maps bit `8i + 7` to bit `56 + i`; other products fall outside the result.
#[inline]
pub(crate) const fn movemask(marks: u64) -> u64 {
    marks.wrapping_mul(0x0002_0408_1020_4081) >> 56
}

/// Tests [`WORD_BYTES`] bytes at a time and marks matches in each byte's high bit.
pub(crate) trait Kernel: Copy {
    /// Reads this kernel out of the field of `kernel_data` that holds it.
    ///
    /// # Safety
    ///
    /// `kernel_data` must have the field this kernel reads as its live field.
    unsafe fn from_data(kernel_data: &KernelData) -> Self;

    /// Marks each matching byte of `word` with `0x80` and each nonmatching byte with zero.
    fn matches(&self, word: u64) -> u64;

    /// Whether the byte set holds `byte`.
    fn matches_byte(&self, byte: u8) -> bool;
}

/// Returns match bits for the final partial word in `haystack`.
///
/// Re-reads the last full word when possible; shorter haystacks are staged.
#[inline]
fn tail_bits<K: Kernel>(kernel: &K, haystack: &[u8], tail: &[u8]) -> u64 {
    debug_assert!(!tail.is_empty() && tail.len() < WORD_BYTES);
    if let Some(word) = haystack.last_chunk::<WORD_BYTES>() {
        let matched = kernel.matches(u64::from_le_bytes(*word));
        movemask(matched) >> (WORD_BYTES - tail.len())
    } else {
        short_tail_bits(kernel, tail)
    }
}

/// Returns match bits for a haystack shorter than [`WORD_BYTES`].
///
/// Fixed-size copies of both ends avoid a `memcpy` call; overlapping bytes produce identical
/// bits and are merged.
#[inline]
fn short_tail_bits<K: Kernel>(kernel: &K, haystack: &[u8]) -> u64 {
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
    fn slide_ends(bits: u64, staged_len: usize, len: usize) -> u64 {
        let keep_mask = !(u64::MAX << staged_len);
        (bits & keep_mask) | (((bits >> staged_len) & keep_mask) << (len - staged_len))
    }

    let len = haystack.len();
    debug_assert!(!haystack.is_empty() && len < WORD_BYTES);
    let (buf, staged_len) = match len {
        4.. => (stage::<4>(haystack), 4),
        2..4 => (stage::<2>(haystack), 2),
        0..2 => (stage::<1>(haystack), 1),
    };
    let bits = movemask(kernel.matches(u64::from_le_bytes(buf)));
    slide_ends(bits, staged_len, len)
}

#[inline]
pub(crate) fn count_all<K: Kernel>(haystack: &[u8], kernel: K) -> usize {
    // Drain after 255 words to prevent byte-lane overflow.
    const CHUNKS_PER_ACCUMULATOR: usize = u8::MAX as usize;

    let (words, tail) = haystack.as_chunks::<WORD_BYTES>();

    let mut total = 0;
    for batch in words.chunks(CHUNKS_PER_ACCUMULATOR) {
        let mut counts = 0u64;

        for word in batch {
            let marks = kernel.matches(u64::from_le_bytes(*word)) >> 7;
            counts += marks;
        }

        for byte in counts.to_ne_bytes() {
            total += usize::from(byte);
        }
    }
    if !tail.is_empty() {
        total += short_tail_bits(&kernel, tail).count_ones() as usize;
    }
    total
}

/// Returns the offset of the first matching byte of `haystack`.
///
/// Uses the first marked byte directly instead of building a full [`movemask`].
#[inline]
pub(crate) fn find_first<K: Kernel>(haystack: &[u8], kernel: K) -> Option<usize> {
    #[inline]
    fn first_lane(marks: u64) -> usize {
        debug_assert!(marks != 0);
        marks.trailing_zeros() as usize / 8
    }

    // Byte-wise probing is faster than staging for fewer than eight bytes.
    if haystack.len() < WORD_BYTES {
        return haystack.iter().position(|&byte| kernel.matches_byte(byte));
    }

    let (words, tail) = haystack.as_chunks::<WORD_BYTES>();
    let (pairs, words) = words.as_chunks::<2>();

    let mut offset = 0;
    for [first_word, second_word] in pairs {
        let first_match = kernel.matches(u64::from_le_bytes(*first_word));
        let second_match = kernel.matches(u64::from_le_bytes(*second_word));
        if (first_match | second_match) != 0 {
            return Some(if first_match != 0 {
                offset + first_lane(first_match)
            } else {
                offset + WORD_BYTES + first_lane(second_match)
            });
        }
        offset += 2 * WORD_BYTES;
    }
    for word in words {
        let marks = kernel.matches(u64::from_le_bytes(*word));
        if marks != 0 {
            return Some(offset + first_lane(marks));
        }
        offset += WORD_BYTES;
    }

    if tail.is_empty() {
        return None;
    }
    let bits = tail_bits(&kernel, haystack, tail);
    (bits != 0).then(|| haystack.len() - tail.len() + bits.trailing_zeros() as usize)
}

#[inline]
pub(crate) fn find_next<K: Kernel>(state: &mut IterState<'_>, kernel: K) -> MatchedBitset {
    let (haystack, mut offset) = (state.haystack, state.scan_offset);
    // SAFETY: `state.scan_offset` never exceeds the haystack length.
    let unscanned = unsafe { haystack.get_unchecked(offset..) };
    let (words, tail) = unscanned.as_chunks::<WORD_BYTES>();
    let (pairs, words) = words.as_chunks::<2>();

    for [first_word, second_word] in pairs {
        let first_match = kernel.matches(u64::from_le_bytes(*first_word));
        let second_match = kernel.matches(u64::from_le_bytes(*second_word));
        if (first_match | second_match) != 0 {
            state.match_base = offset;
            state.scan_offset = offset + 2 * WORD_BYTES;
            return MatchedBitset::from(
                movemask(first_match) | movemask(second_match) << WORD_BYTES,
            );
        }
        offset += 2 * WORD_BYTES;
    }
    for word in words {
        let marks = kernel.matches(u64::from_le_bytes(*word));
        if marks != 0 {
            state.match_base = offset;
            state.scan_offset = offset + WORD_BYTES;
            return MatchedBitset::from(movemask(marks));
        }
        offset += WORD_BYTES;
    }

    state.match_base = haystack.len() - tail.len();
    state.scan_offset = haystack.len();
    if tail.is_empty() {
        0
    } else {
        tail_bits(&kernel, haystack, tail).into()
    }
}

/// The [`ScanOps`] whose entry points run `K`.
pub(crate) fn scan_ops<K: Kernel>() -> &'static ScanOps {
    unsafe fn find_next<K: Kernel>(
        kernel_data: &KernelData,
        state: &mut IterState<'_>,
    ) -> MatchedBitset {
        // SAFETY: `MemchrN` pairs `K` with its live `KernelData` field.
        let kernel = unsafe { K::from_data(kernel_data) };
        self::find_next(state, kernel)
    }

    unsafe fn count_all<K: Kernel>(kernel_data: &KernelData, haystack: &[u8]) -> usize {
        // SAFETY: `MemchrN` pairs `K` with its live `KernelData` field.
        let kernel = unsafe { K::from_data(kernel_data) };
        self::count_all(haystack, kernel)
    }

    unsafe fn find_first<K: Kernel>(kernel_data: &KernelData, haystack: &[u8]) -> Option<usize> {
        // SAFETY: `MemchrN` pairs `K` with its live `KernelData` field.
        let kernel = unsafe { K::from_data(kernel_data) };
        self::find_first(haystack, kernel)
    }

    &ScanOps {
        find_next: find_next::<K>,
        count_all: count_all::<K>,
        find_first: find_first::<K>,
    }
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
