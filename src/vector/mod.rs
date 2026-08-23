pub(crate) mod kernels;

use crate::{CHUNK_BYTES, FinderKind, IterState, MatchedBitset};
use fearless_simd::prelude::*;
use fearless_simd::{Level, i8x64, kernel, mask8x64, u8x64};

/// Tests a chunk of [`CHUNK_BYTES`] bytes against a byte set.
pub(crate) trait Kernel: Copy {
    fn from_kind(kind: &FinderKind) -> Option<Self>;

    fn matches<S: Simd>(&self, chunk: u8x64<S>) -> mask8x64<S>;
}

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
pub(crate) fn find_next<S: Simd, K: Kernel>(simd: S, state: &mut IterState<'_>, kernel: K) {
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

/// Counts every matching byte of `haystack`.
#[inline(always)]
pub(crate) fn count<S: Simd, K: Kernel>(simd: S, haystack: &[u8], kernel: K) -> usize {
    // Each lane of the counting accumulator starts at zero, and gains at most one per chunk,
    // we can accumulate within a single vector until
    const CHUNKS_PER_ACCUMULATOR: usize = u8::MAX as usize;

    let (chunks, tail) = haystack.as_chunks::<CHUNK_BYTES>();

    // TODO: as_chunks
    let mut total = 0;
    for batch in chunks.chunks(CHUNKS_PER_ACCUMULATOR) {
        let mut counts = u8x64::splat(simd, 0);
        for chunk in batch {
            // A matching lane is all ones (-1), so subtracting it adds one.
            counts -= mask_to_u8s(simd, kernel.matches(u8x64::from_slice(simd, chunk)));
        }
        total += sum_lanes(simd, counts);
    }
    if !tail.is_empty() {
        total += tail_bits(simd, &kernel, haystack, tail.len()).count_ones() as usize;
    }
    total
}

/// Matches the final `len` bytes of `haystack`, returning their bits at positions
/// `0..len`.
///
/// A haystack of at least one full chunk is handled by re-reading the last chunk and
/// shifting the already-scanned bits off the bottom, which avoids staging the tail in
/// a padded buffer.
#[inline(always)]
fn tail_bits<S: Simd, K: Kernel>(simd: S, kernel: &K, haystack: &[u8], len: usize) -> u64 {
    debug_assert!(0 < len && len < CHUNK_BYTES);
    if let Some(chunk) = haystack.last_chunk::<CHUNK_BYTES>() {
        let matched = kernel.matches(u8x64::from_slice(simd, chunk));
        bitmask(simd, matched) >> (CHUNK_BYTES - len)
    } else {
        let mut buf = [0; CHUNK_BYTES];
        buf[..len].copy_from_slice(haystack.windows(len).last().unwrap());
        let matched = kernel.matches(u8x64::load_array(simd, buf));
        bitmask(simd, matched) & !(u64::MAX << len)
    }
}

#[inline(always)]
fn mask_to_u8s<S: Simd>(simd: S, mask: mask8x64<S>) -> u8x64<S> {
    let lanes = <[i8; 64]>::from(mask);
    i8x64::load_array(simd, lanes).bitcast()
}

/// Sums every lane of a vector.
#[inline(always)]
fn sum_lanes<S: Simd>(simd: S, counts: u8x64<S>) -> usize {
    // For some reason, llvm does a good job with x86 with a simple loop, but neon really benefits
    // from a custom kernel.
    #[cfg(target_arch = "aarch64")]
    if let Some(neon) = simd.level().as_neon() {
        return usize::from(aarch64_sum_lanes(neon, counts.into()));
    }
    let _ = simd;
    let mut total = 0;
    for &lane in counts.as_array_ref() {
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
    #[cfg(target_arch = "aarch64")]
    if let Some(neon) = simd.level().as_neon() {
        return aarch64_any_lane_set(neon, mask_to_u8s(simd, matched).into());
    }
    // Match directly, we don't want e.g. avx512, just avx2
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if let Level::Avx2(avx2) = simd.level() {
        // `any_true` lowers to the same `vpmovmskb`s that [`bitmask`] wants, so the
        // compiler folds the two together and re-splits the loop into one test and
        // branch per 32 lanes. `vptest` shares no work with the bitmask, which keeps
        // the extraction on the branch that found something.
        return avx2_any_lane_set(avx2, mask_to_u8s(simd, matched).into());
    }
    matched.any_true()
}

kernel! {
    #[inline(always)]
    fn aarch64_any_lane_set(simd: Neon, chunk: [u8; 64]) -> bool {
        use core::arch::aarch64::*;
        let chunk = u8x64::load_array(simd, chunk);
        let (l, r) = chunk.split();
        let ((a, b), (c, d)) = (l.split(), r.split());

        let any = vorrq_u8(vorrq_u8(a.into(), b.into()), vorrq_u8(c.into(), d.into()));
        vmaxvq_u32(vreinterpretq_u32_u8(any)) != 0
    }
}

kernel! {
    #[inline(always)]
    fn avx2_any_lane_set(simd: Avx2, bytes: [u8; 64]) -> bool {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;

        let bytes = u8x64::load_array(simd, bytes);
        let (a, b) = bytes.split();

        let any = _mm256_or_si256(a.into(), b.into());
        _mm256_testz_si256(any, any) == 0
    }
}

kernel! {
    #[inline(always)]
    fn aarch64_sum_lanes(simd: Neon, sums: [u8; 64]) -> u16 {
        use core::arch::aarch64::*;

        let sums = u8x64::load_array(simd, sums);
        let (l, r) = sums.split();
        let ((a, b), (c, d)) = (l.split(), r.split());

        // Sum pairs of elements into u16s
        let a = vpaddlq_u8(a.into());
        let b = vpaddlq_u8(b.into());
        let c = vpaddlq_u8(c.into());
        let d = vpaddlq_u8(d.into());

        vaddvq_u16(vaddq_u16(vaddq_u16(a, b), vaddq_u16(c, d)))
    }
}

#[inline(always)]
fn bitmask<S: Simd>(simd: S, matched: mask8x64<S>) -> u64 {
    #[cfg(target_arch = "aarch64")]
    if let Some(neon) = simd.level().as_neon() {
        return aarch64_to_bitmask(neon, matched.into());
    }
    let _ = simd;
    matched.to_bitmask()
}

kernel! {
    // This should probably be the `mask8x64::to_bitmask()` implementation in fearless_simd
    // See https://github.com/linebender/fearless_simd/issues/342
    #[inline(always)]
    fn aarch64_to_bitmask(simd: Neon, mask: [i8; 64]) -> u64 {
        use core::arch::aarch64::*;

        let mask_bytes: u8x64<_> = i8x64::load_array(simd, mask).to_bytes();
        let (l, r) = mask_bytes.split();
        let ((a, b), (c, d)) = (l.split(), r.split());

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
