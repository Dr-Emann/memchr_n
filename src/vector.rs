//! Searching a [`CHUNK_BYTES`] of bytes at a time with SIMD.
//!
//! This is the path taken whenever `fearless_simd` reports anything better than its
//! scalar fallback, and — for the kinds built on a dynamic byte shuffle — whenever
//! [`has_byte_shuffle`] agrees the target has one. It mirrors [`crate::swar`] item for
//! item: a [`Kernel`] trait, one kernel per [`crate::FinderKind`], and [`count`] and
//! [`find_next`] drivers.

use crate::bitset::Bitset;
use crate::{CHUNK_BYTES, IterState, MatchedBitset};
use core::range::RangeInclusive;
use fearless_simd::prelude::*;
use fearless_simd::{Level, i8x64, kernel, mask8x64, u8x16, u8x32, u8x64};

/// Each lane of the counting accumulator gains at most one per chunk, so the
/// accumulator has to be drained before it can wrap.
pub(crate) const CHUNKS_PER_ACCUMULATOR: usize = u8::MAX as usize;

/// Whether the target has a single-instruction dynamic byte shuffle.
///
/// [`SmallSet`], [`SingleNibble`] and [`AnyByte`] are built on one. Where it is missing,
/// `swizzle_dyn` degrades into a per-lane gather through memory, which loses to probing
/// the byte set directly with [`crate::swar`].
pub(crate) fn has_byte_shuffle(level: Level) -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        // `pshufb` arrived with SSSE3, so plain SSE2 has nothing equivalent. `as_sse4_2`
        // also answers yes for AVX2 and AVX-512.
        level.as_sse4_2().is_some()
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        !level.is_fallback()
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) enum ConstantNibble {
    Lo,
    Hi,
}

#[derive(Debug, Default, Copy, Clone)]
pub(crate) struct NibbleLookup([u8; 16]);

impl NibbleLookup {
    #[inline]
    pub(crate) fn set(&mut self, nibble: u8, bit: u8) {
        debug_assert!(nibble < 16);
        debug_assert!(bit < 8);
        self.0[usize::from(nibble)] |= 1 << bit;
    }
}

/// Tests a chunk of [`CHUNK_BYTES`] bytes against a byte set.
pub(crate) trait Kernel<S: Simd>: Copy {
    fn matches(&self, chunk: u8x64<S>) -> mask8x64<S>;
}

/// Compares against each needle in turn, which beats a table lookup while the set is
/// small enough that the comparisons stay cheaper than the lookup they replace.
#[derive(Copy, Clone)]
pub(crate) struct AnyOf<S, const N: usize> {
    pub(crate) simd: S,
    pub(crate) needles: [u8; N],
}

impl<S: Simd, const N: usize> Kernel<S> for AnyOf<S, N> {
    #[inline(always)]
    fn matches(&self, chunk: u8x64<S>) -> mask8x64<S> {
        let mut matched = mask8x64::splat(self.simd, false);
        for &needle in &self.needles {
            matched |= chunk.simd_eq(needle);
        }
        matched
    }
}

#[derive(Copy, Clone)]
pub(crate) struct OneRange {
    pub(crate) range: RangeInclusive<u8>,
}

impl<S: Simd> Kernel<S> for OneRange {
    #[inline(always)]
    fn matches(&self, chunk: u8x64<S>) -> mask8x64<S> {
        let RangeInclusive { start, last } = self.range;
        chunk.simd_ge(start) & chunk.simd_le(last)
    }
}

#[derive(Copy, Clone)]
pub(crate) struct SmallSet<S> {
    pub(crate) simd: S,
    pub(crate) lo_lookup: NibbleLookup,
    pub(crate) hi_lookup: NibbleLookup,
}

impl<S: Simd> Kernel<S> for SmallSet<S> {
    #[inline(always)]
    fn matches(&self, chunk: u8x64<S>) -> mask8x64<S> {
        let lo_lookup = u8x64::block_splat(u8x16::simd_from(self.simd, self.lo_lookup.0));
        let hi_lookup = u8x64::block_splat(u8x16::simd_from(self.simd, self.hi_lookup.0));

        let lo = lo_lookup.swizzle_dyn_within_blocks(chunk & 0x0F);
        let hi = hi_lookup.swizzle_dyn_within_blocks(chunk >> 4);

        !(lo & hi).simd_eq(0)
    }
}

#[derive(Copy, Clone)]
pub(crate) struct SingleNibble<S> {
    pub(crate) simd: S,
    pub(crate) which: ConstantNibble,
    pub(crate) table: [u8; 16],
}

impl<S: Simd> Kernel<S> for SingleNibble<S> {
    #[inline(always)]
    fn matches(&self, chunk: u8x64<S>) -> mask8x64<S> {
        let table = u8x64::block_splat(u8x16::simd_from(self.simd, self.table));
        let non_const_nibbles = match self.which {
            ConstantNibble::Lo => chunk >> 4,
            ConstantNibble::Hi => chunk & 0x0F,
        };
        let should_match = table.swizzle_dyn_within_blocks(non_const_nibbles);
        chunk.simd_eq(should_match)
    }
}

#[derive(Copy, Clone)]
pub(crate) struct AnyByte<S> {
    pub(crate) simd: S,
    pub(crate) bytes: Bitset,
}

impl<S: Simd> Kernel<S> for AnyByte<S> {
    #[inline(always)]
    fn matches(&self, chunk: u8x64<S>) -> mask8x64<S> {
        let bits = u8x64::block_splat(u8x16::from_fn(self.simd, |i| 1 << (i % 8)));
        let bit = bits.swizzle_dyn_within_blocks(chunk & 0b0111);
        !(bit & self.membership_bits(chunk >> 3)).simd_eq(0)
    }
}

impl<S: Simd> AnyByte<S> {
    /// Looks each byte's high five bits up in the 256-bit table, giving the table byte
    /// that holds its membership bit.
    #[inline(always)]
    fn membership_bits(&self, indices: u8x64<S>) -> u8x64<S> {
        let table = u8x32::load_array_ref(self.simd, &self.bytes.as_array());
        // The table spans 32 lanes, so the lookup runs at whatever width the target
        // implements natively rather than at `u8x64` always.
        if S::u8s::N >= 64 {
            table.combine(table).swizzle_dyn(indices)
        } else {
            let (lo, hi) = indices.split();
            table.swizzle_dyn(lo).combine(table.swizzle_dyn(hi))
        }
    }
}

/// Counts every matching byte of `haystack`.
///
/// This is `#[inline(always)]` and must stay so, for two reasons: the kernel has to fuse
/// into the loop body, and on a target whose level is detected at runtime the whole loop has
/// to land inside the `#[target_feature]` entry point that `crate::levels!` generates.
#[inline(always)]
pub(crate) fn count<S: Simd, K: Kernel<S>>(simd: S, haystack: &[u8], kernel: K) -> usize {
    let (chunks, tail) = haystack.as_chunks::<CHUNK_BYTES>();

    let mut total = 0;
    for batch in chunks.chunks(CHUNKS_PER_ACCUMULATOR) {
        let mut counts = u8x64::splat(simd, 0);
        for chunk in batch {
            // A matching lane is all ones, so subtracting it adds one.
            counts -= mask_lanes(simd, kernel.matches(u8x64::from_slice(simd, chunk)));
        }
        total += sum_lanes(simd, counts);
    }
    if !tail.is_empty() {
        total += tail_bits(simd, &kernel, haystack, tail.len()).count_ones() as usize;
    }
    total
}

/// Scans from `from` for the first pair of [`CHUNK_BYTES`]s that contains a match.
///
/// Extracting the bitmask is the expensive half of a scan, so [`any_lane_set`] skips it
/// for the chunks that did not match at all. The check reduces to a general-purpose
/// register and feeds a branch, which is the loop's longest serial dependency, so two
/// chunks share one.
///
/// Both chunks of a matching pair are reported together. Reporting only the first and
/// resuming at the second would leave the work already done on the second to be redone
/// on the next call, which costs more than the shared check saves as soon as matches are
/// dense enough to land in most pairs.
#[inline(always)]
pub(crate) fn find_next<S: Simd, K: Kernel<S>>(simd: S, state: &mut IterState<'_>, kernel: K) {
    let (haystack, from) = (state.haystack, state.pos);
    let (chunks, tail) = haystack[from..].as_chunks::<CHUNK_BYTES>();
    let (pairs, rest) = chunks.as_chunks::<2>();

    for (i, [first, second]) in pairs.iter().enumerate() {
        let matched_first = kernel.matches(u8x64::load_array_ref(simd, first));
        let matched_second = kernel.matches(u8x64::load_array_ref(simd, second));
        if any_lane_set(simd, matched_first | matched_second) {
            let offset = from + i * (2 * CHUNK_BYTES);
            state.bits = MatchedBitset::from(bitmask(simd, matched_first))
                | MatchedBitset::from(bitmask(simd, matched_second)) << CHUNK_BYTES;
            state.bits_offset = offset;
            state.pos = offset + 2 * CHUNK_BYTES;
            return;
        }
    }

    if let [chunk] = rest {
        let matched = kernel.matches(u8x64::load_array_ref(simd, chunk));
        if any_lane_set(simd, matched) {
            let offset = from + pairs.len() * (2 * CHUNK_BYTES);
            state.bits = MatchedBitset::from(bitmask(simd, matched));
            state.bits_offset = offset;
            state.pos = offset + CHUNK_BYTES;
            return;
        }
    }

    state.bits = if tail.is_empty() {
        0
    } else {
        MatchedBitset::from(tail_bits(simd, &kernel, haystack, tail.len()))
    };
    state.bits_offset = haystack.len() - tail.len();
    state.pos = haystack.len();
}

/// Matches the final `len` bytes of `haystack`, returning their bits at positions
/// `0..len`.
///
/// A haystack of at least one full chunk is handled by re-reading the last chunk and
/// shifting the already-scanned bits off the bottom, which avoids staging the tail in
/// a padded buffer.
#[inline(always)]
fn tail_bits<S: Simd, K: Kernel<S>>(simd: S, kernel: &K, haystack: &[u8], len: usize) -> u64 {
    debug_assert!(0 < len && len < CHUNK_BYTES);
    if let Some(chunk) = haystack.last_chunk::<CHUNK_BYTES>() {
        let matched = kernel.matches(u8x64::from_slice(simd, chunk));
        bitmask(simd, matched) >> (CHUNK_BYTES - len)
    } else {
        let mut buf = [0; CHUNK_BYTES];
        buf[..len].copy_from_slice(&haystack[haystack.len() - len..]);
        let matched = kernel.matches(u8x64::from_slice(simd, &buf));
        bitmask(simd, matched) & !(u64::MAX << len)
    }
}

#[inline(always)]
fn mask_lanes<S: Simd>(simd: S, mask: mask8x64<S>) -> u8x64<S> {
    let lanes = <[i8; 64]>::from(mask);
    i8x64::simd_from(simd, lanes).bitcast()
}

/// Sums every lane of a counting accumulator.
///
/// Callers must drain the accumulator often enough that the total fits in `u16`, which
/// [`CHUNKS_PER_ACCUMULATOR`] guarantees.
#[inline(always)]
fn sum_lanes<S: Simd>(simd: S, counts: u8x64<S>) -> usize {
    #[cfg(target_arch = "aarch64")]
    if let Some(neon) = simd.level().as_neon() {
        let (lo, hi) = counts.split();
        let ((a, b), (c, d)) = (lo.split(), hi.split());
        return usize::from(aarch64_sum_lanes(
            neon,
            a.into(),
            b.into(),
            c.into(),
            d.into(),
        ));
    }
    let _ = simd;
    let mut total = 0;
    for &lane in counts.as_slice() {
        total += usize::from(lane);
    }
    total
}

/// Whether any lane matched, without extracting which ones did.
///
/// Matched on [`Level`] rather than its `as_*` accessors because those answer for
/// supersets too: AVX-512 keeps masks in `k` registers, where `any_true` is already a
/// single `kortestq` and the AVX2 arm would only get in the way.
#[inline(always)]
fn any_lane_set<S: Simd>(simd: S, matched: mask8x64<S>) -> bool {
    match simd.level() {
        #[cfg(target_arch = "aarch64")]
        Level::Neon(neon) => {
            // Four short-circuiting `vmaxvq`s is what `any_true` would cost here; one
            // or-reduce over the lanes is cheaper.
            let (lo, hi) = mask_lanes(simd, matched).split();
            let ((a, b), (c, d)) = (lo.split(), hi.split());
            aarch64_any_lane_set(neon, a.into(), b.into(), c.into(), d.into())
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Level::Avx2(avx2) => {
            // `any_true` lowers to the same `vpmovmskb`s that [`bitmask`] wants, so the
            // compiler folds the two together and re-splits the loop into one test and
            // branch per 32 lanes. `vptest` shares no work with the bitmask, which keeps
            // the extraction on the branch that found something.
            let (lo, hi) = mask_lanes(simd, matched).split();
            x86_any_lane_set(avx2, lo.into(), hi.into())
        }
        _ => matched.any_true(),
    }
}

#[inline(always)]
fn bitmask<S: Simd>(simd: S, matched: mask8x64<S>) -> u64 {
    #[cfg(target_arch = "aarch64")]
    if let Some(neon) = simd.level().as_neon() {
        let (lo, hi) = mask_lanes(simd, matched).split();
        let ((a, b), (c, d)) = (lo.split(), hi.split());

        return aarch64_bitmask(neon, a.into(), b.into(), c.into(), d.into());
    }
    let _ = simd;
    matched.to_bitmask()
}

kernel! {
    #[inline(always)]
    fn aarch64_any_lane_set(simd: Neon, a: [u8; 16], b: [u8; 16], c: [u8; 16], d: [u8; 16]) -> bool {
        use core::arch::aarch64::*;
        type V = u8x16<fearless_simd::Neon>;

        let a = V::simd_from(simd, a).into();
        let b = V::simd_from(simd, b).into();
        let c = V::simd_from(simd, c).into();
        let d = V::simd_from(simd, d).into();

        let any = vorrq_u8(vorrq_u8(a, b), vorrq_u8(c, d));
        vmaxvq_u32(vreinterpretq_u32_u8(any)) != 0
    }
}

kernel! {
    #[inline(always)]
    fn x86_any_lane_set(simd: Avx2, a: [u8; 32], b: [u8; 32]) -> bool {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;
        type V = u8x32<fearless_simd::Avx2>;

        let a = V::simd_from(simd, a).into();
        let b = V::simd_from(simd, b).into();

        let any = _mm256_or_si256(a, b);
        _mm256_testz_si256(any, any) == 0
    }
}

kernel! {
    #[inline(always)]
    fn aarch64_sum_lanes(simd: Neon, a: [u8; 16], b: [u8; 16], c: [u8; 16], d: [u8; 16]) -> u16 {
        use core::arch::aarch64::*;
        type V = u8x16<fearless_simd::Neon>;

        let a = vpaddlq_u8(V::simd_from(simd, a).into());
        let b = vpaddlq_u8(V::simd_from(simd, b).into());
        let c = vpaddlq_u8(V::simd_from(simd, c).into());
        let d = vpaddlq_u8(V::simd_from(simd, d).into());

        vaddvq_u16(vaddq_u16(vaddq_u16(a, b), vaddq_u16(c, d)))
    }
}

kernel! {
    #[inline(always)]
    fn aarch64_bitmask(simd: Neon, a: [u8; 16], b: [u8; 16], c: [u8; 16], d: [u8; 16]) -> u64 {
        use core::arch::aarch64::*;
        type V = u8x16<fearless_simd::Neon>;

        let a = V::simd_from(simd, a);
        let b = V::simd_from(simd, b);
        let c = V::simd_from(simd, c);
        let d = V::simd_from(simd, d);

        // The `sri` chain below packs lane `j` of `q0..q3` into bit `4j + k`, so it wants
        // the 64 lanes 4-way deinterleaved: `a` holds bytes 0..16, but `q0` has to hold
        // bytes 0, 4, 8, and so on. `ld4` would deinterleave on the way in, but it is far
        // slower than four plain loads on the cores this targets, so the masks get
        // deinterleaved after the compare instead.
        let (even_ab, odd_ab) = a.deinterleave(b);
        let (even_cd, odd_cd) = c.deinterleave(d);
        let (q0, q2) = even_ab.deinterleave(even_cd);
        let (q1, q3) = odd_ab.deinterleave(odd_cd);
        let (q0, q1, q2, q3) = (q0.into(), q1.into(), q2.into(), q3.into());

        // shift the second vector right by one, insert the top bit from the first vector
        // The top two bits each element of temp0 are from the first and second vector
        let temp0 = vsriq_n_u8::<1>(q1, q0);
        // shift the fourth vector right by one, insert the top bit from the third vector
        // The top two bits each element of temp1 are from the third and fourth vector
        let temp1 = vsriq_n_u8::<1>(q3, q2);
        // shift temp1 (the top two bits of which are from the third and fourth vector) right by 2,
        // insert the top two bits from temp0 (the top two bits of which are from the first and
        // second vector)
        // The top four bits of each element of temp2 are from the first, second, third, and fourth
        // vector
        let temp2 = vsriq_n_u8::<2>(temp1, temp0);
        // duplicate the top 4 bits into the bottom 4 bits of each element
        let temp3 = vsriq_n_u8::<4>(temp2, temp2);

        // Returns a value where the first 4 bits are the low 4 bits of the first element,
        // the next 4 bits are the high 4 bits of the second element, and so on.
        let result_vector = vshrn_n_u16::<4>(vreinterpretq_u16_u8(temp3));
        vget_lane_u64::<0>(vreinterpret_u64_u8(result_vector))
    }
}
