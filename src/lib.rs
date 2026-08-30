#![deny(unnameable_types, unreachable_pub)]

mod bitset;
mod bytes;
mod bytewise;
mod swar;
mod vector;

pub use bytes::Bytes;

use crate::bitset::Bitset;
use core::fmt;
use core::mem::transmute_copy;
use core::range::RangeInclusive;
use fearless_simd::{Level, Simd, dispatch};

/// Bytes matched per kernel invocation.
const CHUNK_BYTES: usize = 64;

/// Matches of one scan, the `i`th bit (numbered from lsb to msb) is 1 if the `i`th byte matched
///
/// Wide enough for the two [`CHUNK_BYTES`]s that [`vector::find_next`] scans per iteration.
type MatchedBitset = u128;

const _: () = assert!(MatchedBitset::BITS as usize >= CHUNK_BYTES * 2);

/// Which family of kernels a [`Finder`] is built from.
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
    assert_send_sync::<Finder>();
};

#[derive(Clone)]
pub struct Finder {
    /// Kept, with `family`, only for [`Debug`]; `kind` and `scan` are what searching goes
    /// through.
    level: Level,
    family: Family,
    kind: FinderKind,
    scan: Scan,
}

impl fmt::Debug for Finder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Finder")
            .field("level", &self.level)
            .field("family", &self.family)
            .field("kind", &self.kind)
            .finish_non_exhaustive()
    }
}

#[derive(Copy, Clone, Debug)]
enum FinderKind {
    AnyByte(Bitset),
    SmallSet {
        lo_lookup: NibbleLookup,
        hi_lookup: NibbleLookup,
    },
    ConstantNibble(ConstantNibble, [u8; 16]),
    OneByte(u8),
    TwoBytes([u8; 2]),
    ThreeBytes([u8; 3]),
    OneRange(RangeInclusive<u8>),
    Never,
}

/// Which kernels a [`Finder`] runs, resolved from [`Backend`] and the level.
#[derive(Copy, Clone, Debug)]
enum Family {
    Vector,
    Word,
}

impl Finder {
    fn new(level: Level, family: Family, kind: FinderKind) -> Self {
        Self {
            level,
            family,
            kind,
            scan: build_scan(level, family, kind),
        }
    }

    /// Returns the offset of the first matching byte in `haystack`.
    #[inline]
    pub fn find(&self, haystack: &[u8]) -> Option<usize> {
        self.iter(haystack).next()
    }

    /// Returns an iterator over the offsets of every matching byte in `haystack`.
    #[inline]
    pub fn iter<'a>(&'a self, haystack: &'a [u8]) -> Iter<'a> {
        Iter {
            finder: self,
            state: IterState {
                haystack,
                pos: 0,
                bits: 0,
                bits_offset: 0,
            },
        }
    }
}

pub struct Iter<'a> {
    finder: &'a Finder,
    state: IterState<'a>,
}

/// What a scan reads and writes: everything in [`Iter`] except which [`Finder`] it came from.
struct IterState<'a> {
    haystack: &'a [u8],
    /// Offset of the first byte that has not been scanned yet.
    pos: usize,
    /// Matches of the most recently scanned run that have not been yielded yet.
    bits: MatchedBitset,
    /// Offset of the first byte that `bits` describes.
    bits_offset: usize,
}

/// The search loops for one level-and-kernel pair, chosen when the [`Finder`] is built.
///
/// This is a `Box<dyn Scan>` written out by hand, to keep the allocation off callers that
/// build a [`Finder`] per search.
///
/// The level and the [`FinderKind`] are both fixed for a [`Finder`]'s lifetime, so resolving
/// them here also removes the two matches every refill used to run, and keeps `levels * kinds`
/// copies of the loop body out of [`Iter::next`].
#[derive(Copy, Clone)]
struct Scan {
    find_next: unsafe fn(&FinderKind, &mut IterState<'_>),
    count_all: unsafe fn(&FinderKind, &mut IterState<'_>) -> usize,
}

/// Builds the [`Scan`] for a vector kernel at one SIMD level.
///
/// A SIMD token is a zero-sized proof that the running target supports its level, so `S` alone
/// carries the level into the entry points below: each rebuilds its token and re-enters that
/// level's target-feature context through [`Simd::vectorize`].
fn vector_scan<S: Simd, K: vector::Kernel<S>>(simd: S) -> Scan {
    /// Rebuilds a SIMD token, which holds no data beyond the support it proves.
    ///
    /// # Safety
    ///
    /// The running target must support `S`'s level.
    #[inline(always)]
    unsafe fn token<S: Simd>() -> S {
        const {
            assert!(size_of::<S>() == 0);
            assert!(align_of::<S>() == 1);
        };
        // SAFETY: the assertion above makes this a zero-byte copy, and a zero-sized type has
        // exactly one value; the caller guarantees the support that value stands for.
        unsafe { transmute_copy(&()) }
    }

    unsafe fn find_next<S: Simd, K: vector::Kernel<S>>(
        kind: &FinderKind,
        state: &mut IterState<'_>,
    ) {
        // SAFETY: this function is only ever instantiated by `vector_scan`, which accepts an S,
        // so we know the caller proved the right target features are available
        let simd = unsafe { token::<S>() };
        simd.vectorize(
            #[inline(always)]
            move || {
                // SAFETY: the `Scan` below stores this function only for the variant `K` was
                // chosen for, so `from_kind` returns `Some`. Unwrapping it checked costs a
                // measurable amount on refill-heavy searches.
                let kernel = unsafe { K::from_kind(simd, kind).unwrap_unchecked() };
                vector::find_next(simd, state, kernel);
            },
        );
    }

    unsafe fn count_all<S: Simd, K: vector::Kernel<S>>(
        kind: &FinderKind,
        state: &mut IterState<'_>,
    ) -> usize {
        // SAFETY: as above.
        let simd = unsafe { token::<S>() };
        simd.vectorize(
            #[inline(always)]
            move || {
                // SAFETY: the `Scan` below stores this function only for the variant `K` was
                // chosen for, so `from_kind` returns `Some`. Unwrapping it checked costs a
                // measurable amount on refill-heavy searches.
                let kernel = unsafe { K::from_kind(simd, kind).unwrap_unchecked() };
                let total = vector::count(simd, &state.haystack[state.pos..], kernel);
                state.pos = state.haystack.len();
                total
            },
        )
    }

    // By taking a simd, we prove that it's safe for find_next/count_all to create one from thin
    // air when they are called.
    _ = simd;
    Scan {
        find_next: find_next::<S, K>,
        count_all: count_all::<S, K>,
    }
}

fn swar_scan<K: swar::Kernel>() -> Scan {
    unsafe fn find_next<K: swar::Kernel>(kind: &FinderKind, state: &mut IterState<'_>) {
        // SAFETY: as in `vector_scan`, the `Scan` below stores this function only for the
        // variant `K` was chosen for.
        let kernel = unsafe { K::from_kind(kind).unwrap_unchecked() };
        swar::find_next(state, kernel);
    }

    unsafe fn count_all<K: swar::Kernel>(kind: &FinderKind, state: &mut IterState<'_>) -> usize {
        // SAFETY: as above.
        let kernel = unsafe { K::from_kind(kind).unwrap_unchecked() };
        let total = swar::count(&state.haystack[state.pos..], kernel);
        state.pos = state.haystack.len();
        total
    }

    Scan {
        find_next: find_next::<K>,
        count_all: count_all::<K>,
    }
}

/// The byte-at-a-time counterpart of [`swar_scan`].
fn bytewise_scan<K: bytewise::Kernel>() -> Scan {
    unsafe fn find_next<K: bytewise::Kernel>(kind: &FinderKind, state: &mut IterState<'_>) {
        // SAFETY: as in `vector_scan`, the `Scan` below stores this function only for the
        // variant `K` was chosen for.
        let kernel = unsafe { K::from_kind(kind).unwrap_unchecked() };
        bytewise::find_next(state, kernel);
    }

    unsafe fn count_all<K: bytewise::Kernel>(
        kind: &FinderKind,
        state: &mut IterState<'_>,
    ) -> usize {
        // SAFETY: as above.
        let kernel = unsafe { K::from_kind(kind).unwrap_unchecked() };
        let total = bytewise::count(&state.haystack[state.pos..], kernel);
        state.pos = state.haystack.len();
        total
    }

    Scan {
        find_next: find_next::<K>,
        count_all: count_all::<K>,
    }
}

/// Builds the [`Scan`] for a byte set that nothing can match.
fn never_scan() -> Scan {
    fn find_next(_kind: &FinderKind, state: &mut IterState<'_>) {
        state.pos = state.haystack.len();
    }

    fn count_all(_kind: &FinderKind, state: &mut IterState<'_>) -> usize {
        state.pos = state.haystack.len();
        0
    }

    Scan {
        find_next,
        count_all,
    }
}

fn build_scan(level: Level, family: Family, kind: FinderKind) -> Scan {
    match family {
        Family::Vector => dispatch!(level, simd => vector_build(simd, kind)),
        Family::Word => word_build(kind),
    }
}

/// Picks the vector kernel for a kind. Each arm's kernel reads that same arm back in its
/// [`vector::Kernel`] impl.
fn vector_build<S: Simd>(simd: S, kind: FinderKind) -> Scan {
    match kind {
        FinderKind::OneByte(_) => vector_scan::<S, vector::kernels::AnyOf<S, 1>>(simd),
        FinderKind::TwoBytes(_) => vector_scan::<S, vector::kernels::AnyOf<S, 2>>(simd),
        FinderKind::ThreeBytes(_) => vector_scan::<S, vector::kernels::AnyOf<S, 3>>(simd),
        FinderKind::OneRange(_) => vector_scan::<S, vector::kernels::OneRange<S>>(simd),
        FinderKind::SmallSet { .. } => vector_scan::<S, vector::kernels::SmallSet>(simd),
        FinderKind::ConstantNibble(..) => vector_scan::<S, vector::kernels::SingleNibble>(simd),
        FinderKind::AnyByte(_) => vector_scan::<S, vector::kernels::AnyByte>(simd),
        FinderKind::Never => never_scan(),
    }
}

/// The word-at-a-time counterpart of [`vector_build`].
fn word_build(kind: FinderKind) -> Scan {
    match kind {
        FinderKind::OneByte(_) => swar_scan::<swar::kernels::AnyOf<1>>(),
        FinderKind::TwoBytes(_) => swar_scan::<swar::kernels::AnyOf<2>>(),
        FinderKind::ThreeBytes(_) => swar_scan::<swar::kernels::AnyOf<3>>(),
        FinderKind::OneRange(_) => swar_scan::<swar::kernels::OneRange>(),
        FinderKind::AnyByte(_) => bytewise_scan::<bytewise::kernels::AnyByte>(),
        FinderKind::Never => never_scan(),
        // Both scan by shuffling bytes within a vector, which is what picks them over
        // `AnyByte` in the first place; `Bytes::kind` only builds them for a family that has
        // shuffles to spend.
        FinderKind::SmallSet { .. } | FinderKind::ConstantNibble(..) => {
            unreachable!("shuffle kinds need vectors")
        }
    }
}

impl<'a> Iterator for Iter<'a> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.state.bits == 0 {
            if self.state.pos == self.state.haystack.len() {
                return None;
            }
            // SAFETY: `build_scan` installs each function for the kind passed back to it
            // here, and the `Level` that chose it proves the target has its features.
            unsafe { (self.finder.scan.find_next)(&self.finder.kind, &mut self.state) };
            if self.state.bits == 0 {
                return None;
            }
        }
        let bit = self.state.bits.trailing_zeros() as usize;
        self.state.bits &= self.state.bits - 1;
        Some(self.state.bits_offset + bit)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let min = self.state.bits.count_ones() as usize;
        let max = min.checked_add(self.state.haystack.len() - self.state.pos);
        (min, max)
    }

    fn count(mut self) -> usize {
        let mut total = self.state.bits.count_ones() as usize;
        if self.state.pos != self.state.haystack.len() {
            let finder = self.finder;
            // SAFETY: as in `next`.
            total += unsafe { (finder.scan.count_all)(&finder.kind, &mut self.state) };
        }
        total
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let mut remaining = n;
        while (self.state.bits.count_ones() as usize) <= remaining {
            remaining -= self.state.bits.count_ones() as usize;
            self.state.bits = 0;
            if self.state.pos == self.state.haystack.len() {
                return None;
            }
            // SAFETY: as in `next`.
            unsafe { (self.finder.scan.find_next)(&self.finder.kind, &mut self.state) };
            if self.state.bits == 0 {
                return None;
            }
        }
        for _ in 0..remaining {
            self.state.bits &= self.state.bits - 1;
        }
        let bit = self.state.bits.trailing_zeros() as usize;
        self.state.bits &= self.state.bits - 1;
        Some(self.state.bits_offset + bit)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum ConstantNibble {
    Lo,
    Hi,
}

#[derive(Debug, Default, Copy, Clone)]
struct NibbleLookup([u8; 16]);

impl NibbleLookup {
    #[inline]
    fn set(&mut self, nibble: u8, bit: u8) {
        debug_assert!(nibble < 16);
        debug_assert!(bit < 8);
        self.0[usize::from(nibble)] |= 1 << bit;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// Every byte in `bytes`, recovered by searching a haystack of all 256 byte values.
    fn members(bytes: &Bytes) -> Vec<u8> {
        let all: Vec<u8> = (0..=u8::MAX).collect();
        bytes
            .finder()
            .iter(&all)
            .map(|offset| all[offset])
            .collect()
    }

    #[test]
    fn bytes_add_range_matches_adding_each_byte() {
        for start in 0..=u8::MAX {
            for last in start..=u8::MAX {
                let mut ranged = Bytes::new();
                ranged.add_range(RangeInclusive { start, last });

                let mut one_at_a_time = Bytes::new();
                for byte in start..=last {
                    one_at_a_time.add(byte);
                }

                assert_eq!(
                    members(&ranged),
                    members(&one_at_a_time),
                    "{start}..={last}"
                );
            }
        }
    }

    #[test]
    fn bytes_add_two_ranges_matches_adding_each_byte() {
        // Bounds that straddle every representation change: the array filling up, a bitset
        // word boundary, and the ends of the byte range.
        let bounds = [0u8, 1, 7, 23, 24, 25, 63, 64, 127, 128, 200, 254, 255];
        for &first_start in &bounds {
            for &first_last in bounds.iter().filter(|&&b| b >= first_start) {
                for &second_start in &bounds {
                    for &second_last in bounds.iter().filter(|&&b| b >= second_start) {
                        let mut ranged = Bytes::new();
                        ranged.add_range(RangeInclusive {
                            start: first_start,
                            last: first_last,
                        });
                        ranged.add_range(RangeInclusive {
                            start: second_start,
                            last: second_last,
                        });

                        let mut one_at_a_time = Bytes::new();
                        for byte in first_start..=first_last {
                            one_at_a_time.add(byte);
                        }
                        for byte in second_start..=second_last {
                            one_at_a_time.add(byte);
                        }

                        assert_eq!(
                            members(&ranged),
                            members(&one_at_a_time),
                            "{first_start}..={first_last} then {second_start}..={second_last}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn bytes_add_range_of_empty_range_adds_nothing() {
        let mut bytes = Bytes::from_bytes(b"abc");
        bytes.add_range(RangeInclusive { start: 10, last: 9 });
        assert_eq!(members(&bytes), b"abc");
    }

    #[test]
    fn bytes_add_keeps_a_byte_disjoint_from_an_existing_range() {
        let mut bytes = Bytes::new();
        bytes.add_range(RangeInclusive {
            start: 0,
            last: 100,
        });
        bytes.add(200);
        assert_eq!(members(&bytes), (0..=100).chain([200]).collect::<Vec<u8>>());
    }

    #[test]
    fn bytes_add_range_works_in_const_context() {
        const DIGITS: Bytes = {
            let mut bytes = Bytes::new();
            bytes.add_range(RangeInclusive {
                start: b'0',
                last: b'9',
            });
            bytes
        };
        assert_eq!(members(&DIGITS), b"0123456789");
    }

    fn build_word(bytes: &[u8]) -> Finder {
        Bytes::from_bytes(bytes).finder_with(Backend::Scalar)
    }

    fn build(bytes: &[u8]) -> Finder {
        Bytes::from_bytes(bytes).finder()
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

    /// One byte set per [`FinderKind`], so every kernel is exercised.
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
            for (name, finder) in [("vector", build(&set)), ("word", build_word(&set))] {
                // Every length up to a chunk-and-change, so each way the tail can split
                // into whole words and a short remainder is covered, then the pair and
                // multi-chunk boundaries.
                let lens = (0..=80).chain([127, 128, 129, 255, 256, 1000]);
                for len in lens {
                    let haystack = haystack(len);
                    let expected = naive(&set, &haystack);
                    let got: Vec<usize> = finder.iter(&haystack).collect();
                    assert_eq!(got, expected, "{name} set {set:?} len {len}");
                    assert_eq!(
                        finder.iter(&haystack).count(),
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
            for (name, finder) in [("vector", build(&set)), ("word", build_word(&set))] {
                // Every length up to a chunk-and-change, so each way the tail can split
                // into whole words and a short remainder is covered, then the pair and
                // multi-chunk boundaries.
                let lens = (0..=80).chain([127, 128, 129, 255, 256, 1000]);
                for len in lens {
                    let haystack = haystack(len);
                    let expected = naive(&set, &haystack).first().copied();
                    assert_eq!(
                        finder.find(&haystack),
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
        for finder in [build(b"x"), build_word(b"x"), build_word(&dense)] {
            for len in 0..192 {
                let haystack = vec![b'x'; len];
                let expected: Vec<usize> = (0..len).collect();
                assert_eq!(
                    finder.iter(&haystack).collect::<Vec<_>>(),
                    expected,
                    "{len}"
                );
                assert_eq!(finder.iter(&haystack).count(), len, "{len}");
            }
        }
    }

    #[test]
    fn nth_matches_naive() {
        for set in sets() {
            for (name, finder) in [("vector", build(&set)), ("word", build_word(&set))] {
                for len in (0..=80).chain([127, 128, 129, 255, 256, 1000]) {
                    let haystack = haystack(len);
                    let expected = naive(&set, &haystack);
                    for n in [0, 1, 2, 3, 7, 63, 64, 65, 100] {
                        assert_eq!(
                            finder.iter(&haystack).nth(n),
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
        let finder = build(set);
        let haystack = haystack(1000);
        let expected = naive(set, &haystack);
        for step in [0, 1, 5, 64, 200] {
            let mut iter = finder.iter(&haystack);
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
        let finder = build(b"aeiouAEI");
        let haystack = haystack(1000);
        for skip in [0, 1, 2, 7, 40] {
            let mut iter = finder.iter(&haystack);
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
        for finder in [build(b"x"), build_word(b"x")] {
            for len in [128, 192, 256, 300] {
                for offset in 0..len {
                    let mut haystack = vec![b'.'; len];
                    haystack[offset] = b'x';
                    assert_eq!(
                        finder.iter(&haystack).collect::<Vec<_>>(),
                        vec![offset],
                        "len {len} offset {offset}"
                    );
                }
            }
        }
    }

    /// The counting accumulator holds one byte per lane, so it must be drained before
    /// a lane can wrap.
    #[test]
    fn counts_do_not_overflow_the_accumulator() {
        let finder = build(b"x");
        let len = CHUNK_BYTES * (512 + 3);
        let haystack = vec![b'x'; len];
        assert_eq!(finder.iter(&haystack).count(), len);
    }
}
