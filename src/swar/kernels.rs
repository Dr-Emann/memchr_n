use crate::KernelData;
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
        const { assert!(N <= 3, "`splatted_words` holds three") }
        // SAFETY: the caller guarantees `splatted_words` is live, and the assertion above
        // keeps the reads below inside it.
        let splatted = unsafe { data.splatted_words };
        Self {
            splatted_needles: core::array::from_fn(|i| splatted[i]),
        }
    }

    #[inline]
    fn matches(&self, word: u64) -> u64 {
        any_of_matches(word, self.splatted_needles)
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
    unsafe fn from_data(data: &KernelData) -> Self {
        // SAFETY: the caller guarantees `range_masks` is live.
        unsafe { data.range_masks }
    }

    #[inline]
    fn matches(&self, word: u64) -> u64 {
        // A byte is in range exactly when `byte - start` wraps into `0..=span`.
        let shifted = ((word | HIGH) - self.start) ^ ((word ^ self.not_start) & HIGH);

        // `128 + span_low - shifted_low` stays inside the byte, so bit 7 answers
        // `span_low >= shifted_low` without borrowing into the next lane.
        let low_ge = (self.span_low - (shifted & !HIGH)) & HIGH;
        let clear_high = !shifted & HIGH;
        // A byte past the halfway point is in range only when the span reaches that far:
        // with the span's high bit set every shifted byte below 128 fits outright, and
        // the ones above it fit when their low seven bits do; without it none of them do.
        (clear_high & low_ge) | (self.span_high & (clear_high | (shifted & low_ge)))
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

    /// Zero and one neighbours are what catch a borrow leaking between lanes, which is
    /// how the shorter `(w - LOW) & !w & HIGH` formulation goes wrong.
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
