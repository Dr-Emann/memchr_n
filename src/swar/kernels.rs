use crate::KernelData;
use crate::bitset::Bitset;
use crate::swar::{HIGH, Kernel, nonzero_bytes, splat};
use core::range::RangeInclusive;

#[derive(Copy, Clone)]
pub(crate) struct AnyOf<const N: usize> {
    splatted_needles: [u64; N],
}

#[inline]
fn any_of_matches<const N: usize>(word: u64, splatted_needles: [u64; N]) -> u64 {
    let mut nonzero = !0;
    for &needle in &splatted_needles {
        nonzero &= nonzero_bytes(word ^ needle);
    }
    !nonzero & HIGH
}

impl<const N: usize> Kernel for AnyOf<N> {
    unsafe fn from_data(data: &KernelData) -> Self {
        const { assert!(N <= 3, "`splatted_needles` holds three") }
        // SAFETY: the caller guarantees `splatted_needles` is live; `N <= 3` bounds the reads.
        let splatted = unsafe { data.splatted_needles };
        Self {
            splatted_needles: core::array::from_fn(|i| {
                u64::from_ne_bytes(*splatted[i].first_chunk().unwrap())
            }),
        }
    }

    #[inline]
    fn matches(&self, word: u64) -> u64 {
        any_of_matches(word, self.splatted_needles)
    }

    #[inline]
    fn matches_byte(&self, byte: u8) -> bool {
        self.splatted_needles
            .iter()
            .any(|&needle| needle as u8 == byte)
    }
}

/// Stores precomputed masks for branchless range matching.
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
    unsafe fn from_data(data: &KernelData) -> Self {
        // SAFETY: the caller guarantees `range_masks` is live.
        unsafe { data.range_masks }
    }

    #[inline]
    fn matches(&self, word: u64) -> u64 {
        // A byte is in range when `byte - start` wraps into `0..=span`.
        let shifted = ((word | HIGH) - self.start) ^ ((word ^ self.not_start) & HIGH);

        // `128 + span_low - shifted_low` stays inside the byte, so bit 7 answers
        // `span_low >= shifted_low` without borrowing into the next lane.
        let low_ge = (self.span_low - (shifted & !HIGH)) & HIGH;
        let clear_high = !shifted & HIGH;
        // The span's high bit selects which half of the wrapping byte domain is valid.
        (clear_high & low_ge) | (self.span_high & (clear_high | (shifted & low_ge)))
    }

    #[inline]
    fn matches_byte(&self, byte: u8) -> bool {
        let start = !self.not_start as u8;
        let span = (self.span_low as u8 & !0x80) | (self.span_high as u8 & 0x80);
        byte.wrapping_sub(start) <= span
    }
}

/// Probes a 256-bit membership table one byte at a time.
#[derive(Copy, Clone)]
pub(crate) struct AnyByte {
    bitset: Bitset,
}

impl Kernel for AnyByte {
    unsafe fn from_data(data: &KernelData) -> Self {
        // SAFETY: the caller guarantees `bitset` is live.
        Self {
            bitset: unsafe { data.bitset },
        }
    }

    #[inline]
    fn matches(&self, word: u64) -> u64 {
        let mut marks = 0;
        for (i, &byte) in word.to_ne_bytes().iter().enumerate() {
            marks |= u64::from(self.matches_byte(byte)) << (i * 8 + 7);
        }
        marks
    }

    #[inline]
    fn matches_byte(&self, byte: u8) -> bool {
        self.bitset.contains(byte)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::swar::WORD_BYTES;

    fn assert_marks<K: Kernel>(kernel: &K, bytes: [u8; WORD_BYTES], accepts: impl Fn(u8) -> bool) {
        let marks = kernel.matches(u64::from_le_bytes(bytes));
        assert_eq!(marks & !HIGH, 0, "scratch bits left set for {bytes:?}");
        for (i, &byte) in bytes.iter().enumerate() {
            let got = marks & (0x80 << (i * 8)) != 0;
            assert_eq!(got, accepts(byte), "lane {i} of {bytes:?}");
        }
    }

    // Adjacent zeros and ones expose cross-lane borrows.
    fn hazardous_words(byte: u8) -> [[u8; WORD_BYTES]; 3] {
        [
            [byte; WORD_BYTES],
            core::array::from_fn(|i| if i % 2 == 0 { byte } else { 0 }),
            core::array::from_fn(|i| if i % 2 == 0 { byte } else { 1 }),
        ]
    }

    #[test]
    fn any_of_marks_exactly_the_matching_bytes() {
        for needle in 0..=u8::MAX {
            let kernel = AnyOf {
                splatted_needles: [splat(needle)],
            };
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
            splatted_needles: [splat(0x00), splat(0x41), splat(0xFF)],
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
        // Covers both sides of the high-bit split.
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

    #[test]
    fn any_byte_accepts_exactly_the_set() {
        let sets: [Vec<u8>; 4] = [
            Vec::new(),
            vec![0x41],
            (0..=u8::MAX).step_by(3).collect(),
            (0x80..=u8::MAX).collect(),
        ];
        for set in sets {
            let bitset = Bitset::from_bytes(&set);
            let kernel = AnyByte { bitset };
            for byte in 0..=u8::MAX {
                assert_eq!(
                    kernel.matches_byte(byte),
                    set.contains(&byte),
                    "{byte} in {set:?}"
                );
            }
        }
    }
}
