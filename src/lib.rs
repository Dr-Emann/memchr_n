#![deny(unnameable_types, unreachable_pub)]

mod bitset;
mod bytewise;
mod kind;
mod swar;
mod vector;

use crate::bitset::Bitset;
use crate::kind::{ConstantNibble, Kind, KindTag, NibbleLookup};
use core::fmt;
use core::range::RangeInclusive;
use fearless_simd::Level;

/// Matches of one scan, the `i`th bit (numbered from lsb to msb) is 1 if the `i`th byte matched
///
/// Wide enough for the two `CHUNK_BYTES`s that [`vector::find_next`] scans per iteration.
type MatchedBitset = u128;

/// Which family of kernels a [`MemchrN`] is built from.
#[derive(Copy, Clone, Default, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Backend {
    /// The best kernels the running CPU supports.
    #[default]
    Auto,
    /// Word-at-a-time kernels, even on a target that has vectors.
    Scalar,
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    // Part of the public contract, and cheap to keep from regressing.
    assert_send_sync::<MemchrN>();
};

/// A searcher for a fixed set of bytes.
///
/// The set is fixed at construction, which is what lets everything about the search be
/// decided there too: the representation the bytes are collected into, the kernel that
/// matches them, and the data that kernel reads. Nothing about the set is recoverable
/// afterwards, and nothing can be added to it.
#[derive(Clone)]
pub struct MemchrN {
    /// Kept, with `family` and `kind`, only for [`Debug`]; `data` and `scan` are what
    /// searching goes through.
    level: Level,
    family: Family,
    kind: KindTag,
    data: KernelData,
    scan: Scan,
}

impl fmt::Debug for MemchrN {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MemchrN")
            .field("level", &self.level)
            .field("family", &self.family)
            .field("kind", &self.kind)
            .finish_non_exhaustive()
    }
}

/// Which kernels a [`MemchrN`] runs, resolved from [`Backend`] and the level.
#[derive(Copy, Clone, Debug)]
enum Family {
    Vector,
    Word,
}

impl MemchrN {
    /// Builds a searcher for the distinct bytes of `bytes`, on the best kernels the running
    /// CPU supports.
    ///
    /// Repeats are ignored, so the set is the bytes present, not how many times each
    /// appears.
    #[inline]
    pub fn new(bytes: &[u8]) -> Self {
        Self::new_with(bytes, Backend::Auto)
    }

    /// [`new`](Self::new), on a chosen [`Backend`].
    pub fn new_with(bytes: &[u8], backend: Backend) -> Self {
        Self::from_set(Bitset::from_bytes(bytes), backend)
    }

    /// Builds a searcher for every byte from the start of `range` through its end,
    /// inclusive.
    ///
    /// An empty range matches nothing.
    #[inline]
    pub fn from_range(range: core::ops::RangeInclusive<u8>) -> Self {
        Self::from_range_with(range, Backend::Auto)
    }

    /// [`from_range`](Self::from_range), on a chosen [`Backend`].
    ///
    /// Takes an [`ops::RangeInclusive`](core::ops::RangeInclusive) rather than the
    /// [`range::RangeInclusive`](RangeInclusive) used throughout, because it is worth
    /// supporting `..=` syntax at the boundary.
    pub fn from_range_with(range: core::ops::RangeInclusive<u8>, backend: Backend) -> Self {
        let mut set = Bitset::new();
        set.add_range(RangeInclusive {
            start: *range.start(),
            last: *range.end(),
        });
        Self::from_set(set, backend)
    }

    /// Resolves a collected set down to the one kernel that will scan for it.
    ///
    /// Every choice the search depends on is made here: the family, whether the target's
    /// byte shuffles are worth the kinds that need them, the kind itself, and from that
    /// pair the kernel's data and its entry points.
    fn from_set(set: Bitset, backend: Backend) -> Self {
        let level = Level::new();
        let vector = match backend {
            Backend::Scalar => None,
            Backend::Auto => vector::builder(level),
        };
        let family = if vector.is_some() {
            Family::Vector
        } else {
            Family::Word
        };
        // A word kernel has no shuffle to reach for, so it classifies as a vector target
        // without fast ones does.
        let fast_shuffles = vector.is_some() && vector::has_byte_shuffle(level);
        let kind = Kind::of(&set, fast_shuffles);
        Self {
            level,
            family,
            kind: KindTag::of(kind),
            data: KernelData::new(family, kind),
            scan: match vector {
                Some(build) => build(kind),
                None => word_build(kind),
            },
        }
    }

    /// Returns the offset of the first matching byte in `haystack`.
    #[inline]
    pub fn find(&self, haystack: &[u8]) -> Option<usize> {
        // SAFETY: as in `Iter::next`.
        unsafe { (self.scan.find_first)(&self.data, haystack) }
    }

    /// Returns an iterator over the offsets of every matching byte in `haystack`.
    #[inline]
    pub fn iter<'a>(&'a self, haystack: &'a [u8]) -> Iter<'a> {
        Iter {
            memchr_n: self,
            state: IterState {
                haystack,
                pos: 0,
                bits_offset: 0,
            },
            bits: 0,
        }
    }
}

/// Collects the bytes, then builds the searcher for them, on [`Backend::Auto`].
impl FromIterator<u8> for MemchrN {
    fn from_iter<T: IntoIterator<Item = u8>>(iter: T) -> Self {
        Self::from_set(Bitset::from_iter(iter), Backend::Auto)
    }
}

pub struct Iter<'a> {
    memchr_n: &'a MemchrN,
    state: IterState<'a>,
    /// Matches of the most recently scanned run that have not been yielded yet.
    ///
    /// Deliberately not in [`IterState`]: a scan neither reads nor writes it, so keeping it
    /// out of the struct the scan is handed keeps the caller from having to set it up before
    /// every call, and a scan returns its bits in registers instead.
    bits: MatchedBitset,
}

/// What a scan reads and writes, which is everything about the search but its matches.
struct IterState<'a> {
    haystack: &'a [u8],
    /// Offset of the first byte that has not been scanned yet.
    pos: usize,
    /// Offset of the first byte the most recent scan's bits describe.
    bits_offset: usize,
}

/// Everything a kernel needs, in the shape that kernel reads it, built once when the
/// [`MemchrN`] is.
///
/// Reading it back out of a [`Kind`] would cost an unaligned load at the enum's payload
/// offset. A union does not: the live field is decided once, by the same `build` that
/// installs the [`Scan`] beside it, and the alignment lets a whole block come back in one
/// load.
///
/// # Why the needles and range endpoints are splatted here
///
/// Storing them raw and splatting when the kernel loads was measured, and is the better
/// trade only for a single needle. Beyond that it costs two instructions per needle — a byte
/// into a general-purpose register, that register into a vector, then the broadcast — where a
/// pre-splatted block is one aligned load. On a short [`MemchrN::find`] that showed up as
/// 0.7-1.5ns of a ~4ns call for three needles and for a range.
///
/// It is not about build cost, which the original note here claimed: the two measured the
/// same to within noise, because building is dominated by collecting the set and choosing the
/// kernel, not by writing this out.
///
/// What splatting does cost is size — `[[u8; 16]; 3]` is what holds this union at 48 bytes,
/// and a [`MemchrN`] at 80 rather than 64. That is worth a second look, because a smaller one
/// measured 5-12% faster on dense iteration, where every refill touches both this and the
/// [`Scan`]. But it is a [`Scan`] problem rather than a splatting one: the win held even for a
/// single needle, where splatting on load is the *more* expensive of the two.
///
/// # Safety
///
/// Which field is live is fixed for a [`MemchrN`]'s lifetime by its [`Kind`], as
/// [`KernelData::new`] lays out. Nothing but the [`Scan`] built from that same kind may read
/// it.
#[derive(Copy, Clone)]
#[repr(align(16))]
union KernelData {
    /// [`vector::kernels::AnyOf`]: one to three needles, each splatted across a block.
    splatted_needles: [[u8; 16]; 3],
    /// [`vector::kernels::OneRange`]: the endpoints, each splatted across a block.
    splatted_range: [[u8; 16]; 2],
    /// [`vector::kernels::SmallSet`]: the low- and high-nibble tables.
    nibble_lookups: [NibbleLookup; 2],
    /// [`vector::kernels::SingleNibble`].
    nibble_table: NibbleTable,
    /// [`vector::kernels::AnyByte`] and [`bytewise::kernels::AnyByte`].
    bitset: Bitset,
    /// [`swar::kernels::AnyOf`]: one to three needles, each splatted across a word.
    splatted_words: [u64; 3],
    /// [`swar::kernels::OneRange`], whose masks are all derived up front.
    range_masks: swar::kernels::OneRange,
    /// never has no data
    never: (),
}

/// [`vector::kernels::SingleNibble`]'s table and the nibble it is indexed by.
#[derive(Copy, Clone)]
struct NibbleTable {
    which: ConstantNibble,
    table: [u8; 16],
}

impl KernelData {
    /// Builds the field the kernels for `family` and `kind` read. The arms line up
    /// one-for-one with [`word_build`] and the per-level `build` that `level_scans!` writes,
    /// which pick those kernels.
    fn new(family: Family, kind: Kind) -> Self {
        /// Splats one to three needles across whichever unit `family` scans in, leaving
        /// the slots past `N` zeroed for a kernel that will not read them.
        fn splat_needles<const N: usize>(family: Family, needles: [u8; N]) -> KernelData {
            const { assert!(N <= 3) }
            match family {
                Family::Vector => {
                    let mut splatted = [[0; 16]; 3];
                    for (slot, needle) in splatted.iter_mut().zip(needles) {
                        *slot = [needle; 16];
                    }
                    KernelData {
                        splatted_needles: splatted,
                    }
                }
                Family::Word => {
                    let mut splatted = [0; 3];
                    for (slot, needle) in splatted.iter_mut().zip(needles) {
                        *slot = swar::splat(needle);
                    }
                    KernelData {
                        splatted_words: splatted,
                    }
                }
            }
        }

        match kind {
            Kind::OneByte(needle) => splat_needles(family, [needle]),
            Kind::TwoBytes(needles) => splat_needles(family, needles),
            Kind::ThreeBytes(needles) => splat_needles(family, needles),
            // The other kind both families reach, and the only one where they want
            // different shapes: one splats the endpoints, the other derives masks from them.
            Kind::OneRange(range) => match family {
                Family::Vector => Self {
                    splatted_range: [[range.start; 16], [range.last; 16]],
                },
                Family::Word => Self {
                    range_masks: swar::kernels::OneRange::new(range),
                },
            },
            Kind::SmallSet {
                lo_lookup,
                hi_lookup,
            } => Self {
                nibble_lookups: [lo_lookup, hi_lookup],
            },
            Kind::ConstantNibble(which, table) => Self {
                nibble_table: NibbleTable { which, table },
            },
            Kind::AnyByte(bitset) => Self { bitset },
            Kind::Never => Self { never: () },
        }
    }
}

/// The search loops for one level-and-kernel pair, chosen when the [`MemchrN`] is built.
///
/// This is a `Box<dyn Scan>` written out by hand, to keep the allocation off callers that
/// build a [`MemchrN`] per search.
///
/// The level and the [`Kind`] are both fixed for a [`MemchrN`]'s lifetime, so resolving
/// them here also removes the two matches every refill used to run, and keeps `levels * kinds`
/// copies of the loop body out of [`Iter::next`].
#[derive(Copy, Clone)]
struct Scan {
    find_next: unsafe fn(&KernelData, &mut IterState<'_>) -> MatchedBitset,
    /// Counts every match in what is left of a haystack.
    ///
    /// Takes that remainder rather than the [`IterState`] it comes from: counting reads the
    /// haystack once and never resumes, so a scan has nothing to write back, and the state
    /// need not go to memory across the call the way [`find_next`](Scan::find_next)'s does.
    count_all: unsafe fn(&KernelData, &[u8]) -> usize,
    /// [`MemchrN::find`]'s whole search, rather than the first refill of an iteration.
    ///
    /// Both of the above are shaped for an iterator that will call them again: they take the
    /// [`IterState`] by pointer, which forces it to memory across the call, and they answer
    /// in a [`MatchedBitset`] the caller has to unpack. A search that stops at the first
    /// match wants neither, and pays for both — which on a short haystack is most of the
    /// call.
    find_first: unsafe fn(&KernelData, &[u8]) -> Option<usize>,
}

/// Builds the [`Scan`] for a byte set that nothing can match.
pub(crate) fn never_scan() -> Scan {
    fn find_next(_data: &KernelData, state: &mut IterState<'_>) -> MatchedBitset {
        state.pos = state.haystack.len();
        0
    }

    fn count_all(_data: &KernelData, _unscanned: &[u8]) -> usize {
        0
    }

    fn find_first(_data: &KernelData, _haystack: &[u8]) -> Option<usize> {
        None
    }

    Scan {
        find_next,
        count_all,
        find_first,
    }
}

/// The word-at-a-time counterpart of [`vector::builder`].
///
/// It stays here, where [`vector::builder`] does not, because it is the one build that spans
/// two modules: `swar`'s arithmetic covers the kinds it has a trick for, and the rest fall
/// through to `bytewise`'s table probe. Neither module chooses that; this is where they meet.
fn word_build(kind: Kind) -> Scan {
    match kind {
        Kind::OneByte(_) => swar::scan::<swar::kernels::AnyOf<1>>(),
        Kind::TwoBytes(_) => swar::scan::<swar::kernels::AnyOf<2>>(),
        Kind::ThreeBytes(_) => swar::scan::<swar::kernels::AnyOf<3>>(),
        Kind::OneRange(_) => swar::scan::<swar::kernels::OneRange>(),
        Kind::AnyByte(_) => bytewise::scan::<bytewise::kernels::AnyByte>(),
        Kind::Never => never_scan(),
        // Both scan by shuffling bytes within a vector, which is what picks them over
        // `AnyByte` in the first place; `Kind::of` only builds them for a family that
        // has shuffles to spend.
        Kind::SmallSet { .. } | Kind::ConstantNibble(..) => {
            unreachable!("shuffle kinds need vectors")
        }
    }
}

impl<'a> Iterator for Iter<'a> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.bits == 0 {
            if self.state.pos == self.state.haystack.len() {
                return None;
            }
            // SAFETY: `build_scan` installs each function for the kind passed back to it
            // here, and the `Level` that chose it proves the target has its features.
            self.bits = unsafe { (self.memchr_n.scan.find_next)(&self.memchr_n.data, &mut self.state) };
            if self.bits == 0 {
                return None;
            }
        }
        let bit = self.bits.trailing_zeros() as usize;
        self.bits &= self.bits - 1;
        Some(self.state.bits_offset + bit)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let min = self.bits.count_ones() as usize;
        let max = min.checked_add(self.state.haystack.len() - self.state.pos);
        (min, max)
    }

    fn count(self) -> usize {
        let mut total = self.bits.count_ones() as usize;
        // SAFETY: `pos` only ever moves to an offset a scan reached, so it is in bounds.
        let unscanned = unsafe { self.state.haystack.get_unchecked(self.state.pos..) };
        if !unscanned.is_empty() {
            let memchr_n = self.memchr_n;
            // SAFETY: as in `next`.
            total += unsafe { (memchr_n.scan.count_all)(&memchr_n.data, unscanned) };
        }
        total
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let mut remaining = n;
        while (self.bits.count_ones() as usize) <= remaining {
            remaining -= self.bits.count_ones() as usize;
            self.bits = 0;
            if self.state.pos == self.state.haystack.len() {
                return None;
            }
            // SAFETY: as in `next`.
            self.bits = unsafe { (self.memchr_n.scan.find_next)(&self.memchr_n.data, &mut self.state) };
            if self.bits == 0 {
                return None;
            }
        }
        for _ in 0..remaining {
            self.bits &= self.bits - 1;
        }
        let bit = self.bits.trailing_zeros() as usize;
        self.bits &= self.bits - 1;
        Some(self.state.bits_offset + bit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `MemchrN` is built per search often enough that its size is worth keeping honest.
    /// `KernelData` is the bulk of it, and everything else fits in the padding its
    /// alignment leaves behind.
    #[test]
    fn memchr_n_stays_small() {
        assert!(
            size_of::<MemchrN>() <= 80,
            "MemchrN is {} bytes",
            size_of::<MemchrN>()
        );
    }

    #[test]
    fn debug_names_the_chosen_kernel() {
        let debug = format!("{:?}", MemchrN::new(b"az"));
        assert!(debug.contains("TwoBytes"), "{debug}");
    }

    #[test]
    fn add_range_matches_adding_each_byte() {
        for start in 0..=u8::MAX {
            for last in start..=u8::MAX {
                let mut ranged = Bitset::new();
                ranged.add_range(RangeInclusive { start, last });

                let mut one_at_a_time = Bitset::new();
                for byte in start..=last {
                    one_at_a_time.add(byte);
                }

                assert_eq!(ranged, one_at_a_time, "{start}..={last}");
            }
        }
    }

    #[test]
    fn add_range_of_empty_range_adds_nothing() {
        let mut set = Bitset::from_bytes(b"abc");
        let before = set;
        set.add_range(RangeInclusive { start: 10, last: 9 });
        assert_eq!(set, before);
    }

    /// Every byte in `set`, recovered by searching a haystack of all 256 byte values.
    fn members(set: &Bitset) -> Vec<u8> {
        let all: Vec<u8> = (0..=u8::MAX).collect();
        MemchrN::from_set(*set, Backend::Auto)
            .iter(&all)
            .map(|offset| all[offset])
            .collect()
    }

    /// Both halves must agree on the kernel, not just the set: a bitset that covers a span
    /// exactly still has to reach [`Kind::OneRange`], and few enough distinct bytes
    /// still have to reach the kinds only the array representation can name.
    fn assert_same_set_and_kernel(bulk: &Bitset, one_at_a_time: &Bitset, case: &str) {
        assert_eq!(members(bulk), members(one_at_a_time), "{case}");
        for backend in [Backend::Auto, Backend::Scalar] {
            assert_eq!(
                format!("{:?}", MemchrN::from_set(*bulk, backend)),
                format!("{:?}", MemchrN::from_set(*one_at_a_time, backend)),
                "{case} on {backend:?}"
            );
        }
    }

    /// Past `ARRAY_MAX`, `from_bytes` dedups through a bitset rather than by rescanning the
    /// array, and has to land where the byte-at-a-time insert would.
    #[test]
    fn from_bytes_matches_adding_each_byte() {
        let alnum: Vec<u8> = (b'0'..=b'9')
            .chain(b'a'..=b'z')
            .chain(b'A'..=b'Z')
            .collect();
        let mut contiguous_out_of_order: Vec<u8> = (0..=23).collect();
        contiguous_out_of_order.push(100);
        contiguous_out_of_order.extend(24..=99);

        let cases: &[(&str, Vec<u8>)] = &[
            ("62 alnum, scattered", alnum),
            ("0..=200 contiguous", (0..=200).collect()),
            ("every byte", (0..=u8::MAX).collect()),
            ("60 bytes, 3 distinct", b"cba".repeat(20)),
            ("60 bytes, 1 distinct", b"z".repeat(60)),
            ("100 bytes, 24 distinct", (0..100).map(|i| i % 24).collect()),
            ("100 bytes, 25 distinct", (0..100).map(|i| i % 25).collect()),
            ("contiguous, out of order", contiguous_out_of_order),
            ("25 scattered", (0..25).map(|i: u8| i.wrapping_mul(7)).collect()),
            ("exactly ARRAY_MAX + 1", (0..25).collect()),
        ];

        for (case, bytes) in cases {
            let mut one_at_a_time = Bitset::new();
            for &byte in bytes {
                one_at_a_time.add(byte);
            }
            assert_same_set_and_kernel(&Bitset::from_bytes(bytes), &one_at_a_time, case);
        }
    }

    /// The same, for the range fast path taken when the array already holds something.
    #[test]
    fn add_wide_range_to_non_empty_matches_adding_each_byte() {
        let seeds: &[&[u8]] = &[b"", b"z", b"\x00", b"\x7f", b"az", b"\x00\xff", b"aeiouAEI"];
        for seed in seeds {
            for (start, last) in [(0u8, 255u8), (0x80, 0xFF), (10, 40), (100, 124), (60, 200)] {
                let mut ranged = Bitset::from_bytes(seed);
                ranged.add_range(RangeInclusive { start, last });

                let mut one_at_a_time = Bitset::from_bytes(seed);
                for byte in start..=last {
                    one_at_a_time.add(byte);
                }
                let case = format!("{seed:?} + {start}..={last}");
                assert_same_set_and_kernel(&ranged, &one_at_a_time, &case);
            }
        }
    }

    #[test]
    fn add_two_ranges_matches_adding_each_byte() {
        // Bounds that straddle every representation change: the array filling up, a bitset
        // word boundary, and the ends of the byte range.
        let bounds = [0u8, 1, 7, 23, 24, 25, 63, 64, 127, 128, 200, 254, 255];
        for &first_start in &bounds {
            for &first_last in bounds.iter().filter(|&&b| b >= first_start) {
                for &second_start in &bounds {
                    for &second_last in bounds.iter().filter(|&&b| b >= second_start) {
                        let mut ranged = Bitset::new();
                        ranged.add_range(RangeInclusive {
                            start: first_start,
                            last: first_last,
                        });
                        ranged.add_range(RangeInclusive {
                            start: second_start,
                            last: second_last,
                        });

                        let mut one_at_a_time = Bitset::new();
                        for byte in first_start..=first_last {
                            one_at_a_time.add(byte);
                        }
                        for byte in second_start..=second_last {
                            one_at_a_time.add(byte);
                        }

                        assert_eq!(
                            ranged, one_at_a_time,
                            "{first_start}..={first_last} then {second_start}..={second_last}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn add_keeps_a_byte_disjoint_from_an_existing_range() {
        let mut set = Bitset::new();
        set.add_range(RangeInclusive {
            start: 0,
            last: 100,
        });
        set.add(200);
        assert_eq!(members(&set), (0..=100).chain([200]).collect::<Vec<u8>>());
    }

    #[test]
    fn add_range_works_in_const_context() {
        const DIGITS: Bitset = {
            let mut set = Bitset::new();
            set.add_range(RangeInclusive {
                start: b'0',
                last: b'9',
            });
            set
        };
        assert_eq!(members(&DIGITS), b"0123456789");
    }

    fn build_word(bytes: &[u8]) -> MemchrN {
        MemchrN::new_with(bytes, Backend::Scalar)
    }

    fn build(bytes: &[u8]) -> MemchrN {
        MemchrN::new(bytes)
    }

    fn naive(set: &[u8], haystack: &[u8]) -> Vec<usize> {
        let mut offsets = Vec::new();
        for (offset, byte) in haystack.iter().enumerate() {
            if set.contains(byte) {
                offsets.push(offset);
            }
        }
        offsets
    }

    /// One byte set per [`Kind`], so every kernel is exercised.
    fn sets() -> Vec<Vec<u8>> {
        vec![
            vec![],
            b"z".to_vec(),
            b"az".to_vec(),
            b"azQ".to_vec(),
            b"aeiouAEI".to_vec(),
            b"abcdefghjl".to_vec(),
            b"0123456789abcdef".to_vec(),
            (b'0'..=b'9').collect(),
            (0x80..=0xFF).collect(),
            (0..=255u8).step_by(3).collect(),
        ]
    }

    fn haystack(len: usize) -> Vec<u8> {
        // A repeating byte pattern that hits every kernel's table entries, with the
        // period chosen so it does not line up with the 64-byte chunking.
        (0..len).map(|i| ((i * 37 + i / 7) % 251) as u8).collect()
    }

    #[test]
    fn matches_naive_across_lengths() {
        for set in sets() {
            for (name, searcher) in [("vector", build(&set)), ("word", build_word(&set))] {
                // Every length up to a chunk-and-change, so each way the tail can split
                // into whole words and a short remainder is covered, then the pair and
                // multi-chunk boundaries.
                let lens = (0..=80).chain([127, 128, 129, 255, 256, 1000]);
                for len in lens {
                    let haystack = haystack(len);
                    let expected = naive(&set, &haystack);
                    let got: Vec<usize> = searcher.iter(&haystack).collect();
                    assert_eq!(got, expected, "{name} set {set:?} len {len}");
                    assert_eq!(
                        searcher.iter(&haystack).count(),
                        expected.len(),
                        "{name} count for set {set:?} len {len}"
                    );
                }
            }
        }
    }

    #[test]
    fn find_matches_naive() {
        for set in sets() {
            for (name, searcher) in [("vector", build(&set)), ("word", build_word(&set))] {
                // Every length up to a chunk-and-change, so each way the tail can split
                // into whole words and a short remainder is covered, then the pair and
                // multi-chunk boundaries.
                let lens = (0..=80).chain([127, 128, 129, 255, 256, 1000]);
                for len in lens {
                    let haystack = haystack(len);
                    let expected = naive(&set, &haystack).first().copied();
                    assert_eq!(
                        searcher.find(&haystack),
                        expected,
                        "{name} set {set:?} len {len}"
                    );
                }
            }
        }
    }

    /// Both wide families scan the tail by re-reading the last whole unit they work in, so
    /// bits belonging to offsets the main loop already reported have to be shifted off, and
    /// the bytewise family instead has to advance past exactly the run it reported. A
    /// haystack that matches at every offset catches either going wrong.
    #[test]
    fn overlapping_tail_does_not_repeat_matches() {
        // The third set is large enough to reach `AnyByte`, and contains `x`.
        let dense: Vec<u8> = (0..=u8::MAX).step_by(3).collect();
        for searcher in [build(b"x"), build_word(b"x"), build_word(&dense)] {
            for len in 0..192 {
                let haystack = vec![b'x'; len];
                let expected: Vec<usize> = (0..len).collect();
                assert_eq!(
                    searcher.iter(&haystack).collect::<Vec<_>>(),
                    expected,
                    "{len}"
                );
                assert_eq!(searcher.iter(&haystack).count(), len, "{len}");
            }
        }
    }

    #[test]
    fn nth_matches_naive() {
        for set in sets() {
            for (name, searcher) in [("vector", build(&set)), ("word", build_word(&set))] {
                for len in (0..=80).chain([127, 128, 129, 255, 256, 1000]) {
                    let haystack = haystack(len);
                    let expected = naive(&set, &haystack);
                    for n in [0, 1, 2, 3, 7, 63, 64, 65, 100] {
                        assert_eq!(
                            searcher.iter(&haystack).nth(n),
                            expected.get(n).copied(),
                            "{name} set {set:?} len {len} nth {n}"
                        );
                    }
                }
            }
        }
    }

    /// `nth` has to consume the same matches `next` would, so mixing the two must walk the
    /// haystack exactly once.
    #[test]
    fn nth_advances_like_repeated_next() {
        let set = b"aeiouAEI";
        let searcher = build(set);
        let haystack = haystack(1000);
        let expected = naive(set, &haystack);
        for step in [0, 1, 5, 64, 200] {
            let mut iter = searcher.iter(&haystack);
            let mut index = 0;
            let mut got = Vec::new();
            while let Some(offset) = iter.nth(step) {
                index += step;
                got.push((index, offset));
                index += 1;
            }
            let mut want = Vec::new();
            let mut index = step;
            while index < expected.len() {
                want.push((index, expected[index]));
                index += step + 1;
            }
            assert_eq!(got, want, "step {step}");
        }
    }

    #[test]
    fn count_matches_iteration_after_partial_consumption() {
        let searcher = build(b"aeiouAEI");
        let haystack = haystack(1000);
        for skip in [0, 1, 2, 7, 40] {
            let mut iter = searcher.iter(&haystack);
            let mut taken = 0;
            for _ in 0..skip {
                if iter.next().is_some() {
                    taken += 1;
                }
            }
            assert_eq!(iter.count() + taken, naive(b"aeiouAEI", &haystack).len());
        }
    }

    /// Both halves of a scanned pair are reported in one set of bits, the upper half
    /// shifted up by [`CHUNK_BYTES`]. A lone match walked across the pair boundary catches a
    /// half packed at the wrong end.
    #[test]
    fn reports_a_match_from_either_half_of_a_pair() {
        for searcher in [build(b"x"), build_word(b"x")] {
            for len in [128, 192, 256, 300] {
                for offset in 0..len {
                    let mut haystack = vec![b'.'; len];
                    haystack[offset] = b'x';
                    assert_eq!(
                        searcher.iter(&haystack).collect::<Vec<_>>(),
                        vec![offset],
                        "len {len} offset {offset}"
                    );
                    // `find` walks the pair itself rather than unpacking the iterator's
                    // bits, so it has to pick the right half of one on its own.
                    assert_eq!(
                        searcher.find(&haystack),
                        Some(offset),
                        "find, len {len} offset {offset}"
                    );
                }
            }
        }
    }

    /// `find` duplicates the walk `Iter` does rather than driving it, so the two have to
    /// agree everywhere — including on the tails, where each family reads bytes it has
    /// already reported on.
    #[test]
    fn find_agrees_with_the_iterator() {
        for set in sets() {
            for (name, searcher) in [("vector", build(&set)), ("word", build_word(&set))] {
                for len in (0..=80).chain([127, 128, 129, 191, 192, 255, 256, 1000]) {
                    let haystack = haystack(len);
                    assert_eq!(
                        searcher.find(&haystack),
                        searcher.iter(&haystack).next(),
                        "{name} set {set:?} len {len}"
                    );
                }
            }
        }
    }

    /// The counting accumulator holds one byte per lane, so it must be drained before
    /// a lane can wrap.
    #[test]
    fn counts_do_not_overflow_the_accumulator() {
        let searcher = build(b"x");
        let len = 64 * (512 + 3);
        let haystack = vec![b'x'; len];
        assert_eq!(searcher.iter(&haystack).count(), len);
    }
}



