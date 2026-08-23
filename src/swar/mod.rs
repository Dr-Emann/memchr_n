//! Searching eight bytes at a time in a general-purpose register.
//!
//! This is what runs on targets with no usable vectors: either no SIMD at all, where
//! `fearless_simd`'s fallback emulates every lane operation one lane at a time, or SSE2,
//! whose `swizzle_dyn` is a sixteen-way gather through memory. Against a plain byte loop
//! over the `sherlock` corpus it is 9.6x for one needle, 4.3x for three, and 5.3x for a
//! range. [`AnyByte`] is only 1.11x, because an arbitrary byte set still has to be
//! probed a byte at a time — SWAR has no gather.

use crate::bitset::Bitset;
use crate::{CHUNK_BYTES, IterState};
use core::range::RangeInclusive;

/// Bytes tested per general-purpose register.
const WORD: usize = 8;

/// Bit 7 of every byte.
const HIGH: u64 = 0x8080_8080_8080_8080;

#[inline]
const fn splat(byte: u8) -> u64 {
    u64::from_ne_bytes([byte; WORD])
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
const fn movemask(marks: u64) -> u64 {
    marks.wrapping_mul(0x0002_0408_1020_4081) >> 56
}

/// Tests [`WORD`] bytes at a time, the word-at-a-time counterpart of [`crate::Kernel`].
///
/// The result has bit 7 set in every matching byte and every other bit clear, which is
/// the form both [`movemask`] and [`u64::count_ones`] consume.
pub(crate) trait Kernel: Copy {
    fn matches(&self, word: u64) -> u64;
}

/// Marks the bytes a per-byte predicate accepts.
#[inline(always)]
fn probe_bytes(word: u64, accepts: impl Fn(u8) -> bool) -> u64 {
    let mut marks = 0;
    for i in 0..WORD {
        let byte = (word >> (i * 8)) as u8;
        marks |= u64::from(accepts(byte)) << (i * 8 + 7);
    }
    marks
}

#[derive(Copy, Clone)]
pub(crate) struct AnyOf<const N: usize> {
    pub(crate) needles: [u8; N],
}

impl<const N: usize> Kernel for AnyOf<N> {
    #[inline(always)]
    fn matches(&self, word: u64) -> u64 {
        let mut nonzero = !0;
        for &needle in &self.needles {
            nonzero &= nonzero_bytes(word ^ splat(needle));
        }
        !nonzero & HIGH
    }
}

/// Everything is precomputed so that `matches` stays branchless.
#[derive(Copy, Clone)]
pub(crate) struct OneRange {
    /// Splatted range start, ready to subtract from a `HIGH`-saturated word.
    start: u64,
    /// Complement of the splatted start, for the subtraction's high-bit fixup.
    not_start: u64,
    /// Splatted low seven bits of the span, saturated with `HIGH` to subtract from.
    span_low: u64,
    /// All ones when the span's high bit is set, all zeros otherwise.
    span_high: u64,
}

impl OneRange {
    pub(crate) fn new(range: RangeInclusive<u8>) -> Self {
        let RangeInclusive { start, last } = range;
        let span = last.wrapping_sub(start);
        let start = splat(start);
        Self {
            start: start & !HIGH,
            not_start: !start,
            span_low: splat(span & 0x7F) | HIGH,
            span_high: if span & 0x80 == 0 { 0 } else { u64::MAX },
        }
    }
}

impl Kernel for OneRange {
    #[inline(always)]
    fn matches(&self, word: u64) -> u64 {
        // A byte is in range exactly when `byte - start` wraps into `0..=span`.
        let shifted = ((word | HIGH) - self.start) ^ ((word ^ self.not_start) & HIGH);

        // `128 + span_low - shifted_low` stays inside the byte, so bit 7 answers
        // `span_low >= shifted_low` without borrowing into the next lane.
        let low_ge = (self.span_low - (shifted & !HIGH)) & HIGH;
        let clear_high = !shifted & HIGH;
        // A span with its high bit set admits every byte whose high bit is clear; one
        // without it admits none of them.
        (clear_high & low_ge) | (self.span_high & (clear_high | (shifted & low_ge)))
    }
}

/// The counterpart of the vector `AnyByte` kernel. Without a shuffle to run the table
/// lookup on there is nothing to do but probe the set one byte at a time.
#[derive(Copy, Clone)]
pub(crate) struct AnyByte {
    pub(crate) bytes: Bitset,
}

impl Kernel for AnyByte {
    #[inline(always)]
    fn matches(&self, word: u64) -> u64 {
        probe_bytes(word, |byte| self.bytes.contains(byte))
    }
}

/// Marks every matching byte of a chunk, and the union of those marks.
///
/// The union is what lets [`find_next`] skip [`pack_marks`] for a chunk that did not
/// match at all: [`movemask`]'s multiply is the expensive half of a scan, and a scan
/// that finds nothing has no bit positions to report. This is the word-at-a-time
/// counterpart of the vector path's `any_lane_set` gate.
#[inline(always)]
fn chunk_marks<K: Kernel>(
    kernel: &K,
    chunk: &[u8; CHUNK_BYTES],
) -> ([u64; CHUNK_BYTES / WORD], u64) {
    let (words, []) = chunk.as_chunks::<WORD>() else {
        unreachable!()
    };
    let mut marks = [0; CHUNK_BYTES / WORD];
    let mut any = 0;
    for (i, word) in words.iter().enumerate() {
        marks[i] = kernel.matches(u64::from_le_bytes(*word));
        any |= marks[i];
    }
    (marks, any)
}

#[inline(always)]
fn pack_marks(marks: [u64; CHUNK_BYTES / WORD]) -> u64 {
    let mut bits = 0;
    for (i, marks) in marks.into_iter().enumerate() {
        bits |= movemask(marks) << (i * WORD);
    }
    bits
}

#[inline(always)]
fn chunk_bits<K: Kernel>(kernel: &K, chunk: &[u8; CHUNK_BYTES]) -> u64 {
    let (marks, _) = chunk_marks(kernel, chunk);
    pack_marks(marks)
}

/// Matches the final `len` bytes of `haystack`, returning their bits at positions
/// `0..len`, the same way the vector path's tail handling does.
#[inline(always)]
fn tail_bits<K: Kernel>(kernel: &K, haystack: &[u8], len: usize) -> u64 {
    debug_assert!(0 < len && len < CHUNK_BYTES);
    if let Some(chunk) = haystack.last_chunk::<CHUNK_BYTES>() {
        chunk_bits(kernel, chunk) >> (CHUNK_BYTES - len)
    } else {
        let mut buf = [0; CHUNK_BYTES];
        buf[..len].copy_from_slice(&haystack[haystack.len() - len..]);
        chunk_bits(kernel, &buf) & !(u64::MAX << len)
    }
}

/// Counts every matching byte of `haystack`.
#[inline]
pub(crate) fn count<K: Kernel>(haystack: &[u8], kernel: K) -> usize {
    let (words, tail) = haystack.as_chunks::<WORD>();

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
            for i in 0..WORD {
                marks |= ((expected >> i) & 1) << (i * 8 + 7);
            }
            assert_eq!(movemask(marks), expected, "marks {marks:#018x}");
        }
    }

    fn assert_marks<K: Kernel>(kernel: &K, bytes: [u8; WORD], accepts: impl Fn(u8) -> bool) {
        let marks = kernel.matches(u64::from_le_bytes(bytes));
        assert_eq!(marks & !HIGH, 0, "scratch bits left set for {bytes:?}");
        for (i, &byte) in bytes.iter().enumerate() {
            let got = marks & (0x80 << (i * 8)) != 0;
            assert_eq!(got, accepts(byte), "lane {i} of {bytes:?}");
        }
    }

    /// Zero and one neighbours are what catch a borrow leaking between lanes, which is
    /// how the shorter `(w - LOW) & !w & HIGH` formulation goes wrong.
    fn hazardous_words(byte: u8) -> [[u8; WORD]; 3] {
        [
            [byte; WORD],
            core::array::from_fn(|i| if i % 2 == 0 { byte } else { 0 }),
            core::array::from_fn(|i| if i % 2 == 0 { byte } else { 1 }),
        ]
    }

    #[test]
    fn any_of_marks_exactly_the_matching_bytes() {
        for needle in 0..=u8::MAX {
            let kernel = AnyOf { needles: [needle] };
            for byte in 0..=u8::MAX {
                for bytes in hazardous_words(byte) {
                    assert_marks(&kernel, bytes, |b| b == needle);
                }
            }
        }
    }

    #[test]
    fn any_of_handles_several_needles() {
        let kernel = AnyOf {
            needles: [0x00, 0x41, 0xFF],
        };
        for byte in 0..=u8::MAX {
            for bytes in hazardous_words(byte) {
                assert_marks(&kernel, bytes, |b| b == 0x00 || b == 0x41 || b == 0xFF);
            }
        }
    }

    #[test]
    fn range_marks_boundaries_of_every_range() {
        for start in 0..=u8::MAX {
            for last in start..=u8::MAX {
                let kernel = OneRange::new(RangeInclusive { start, last });
                let probes = [
                    0,
                    start.wrapping_sub(1),
                    start,
                    start.wrapping_add(1),
                    last.wrapping_sub(1),
                    last,
                    last.wrapping_add(1),
                    u8::MAX,
                ];
                assert_marks(&kernel, probes, |b| start <= b && b <= last);
            }
        }
    }

    #[test]
    fn range_marks_every_byte_of_awkward_ranges() {
        // Spans either side of 128 are where the high-bit split in `matches` changes
        // which branch of the selection applies.
        let ranges = [(0, 9), (0x80, 0xFF), (0, 0xFF), (0x7F, 0x81), (0x41, 0x41)];
        for (start, last) in ranges {
            let kernel = OneRange::new(RangeInclusive { start, last });
            for byte in 0..=u8::MAX {
                for bytes in hazardous_words(byte) {
                    assert_marks(&kernel, bytes, |b| start <= b && b <= last);
                }
            }
        }
    }
}
