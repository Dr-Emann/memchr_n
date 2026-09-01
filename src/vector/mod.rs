pub(crate) mod kernels;

use crate::{IterState, KernelData, MatchedBitset};
use fearless_simd::prelude::*;
use fearless_simd::{Level, i8x16, i8x64, kernel, u8x16, u8x32, u8x64};

const CHUNK_BYTES: usize = 64;
const _: () = assert!(MatchedBitset::BITS as usize >= CHUNK_BYTES * 2);

const BLOCK_BYTES: usize = 16;

/// Tests a chunk of [`CHUNK_BYTES`] bytes against a byte set.
pub(crate) trait Kernel<S: Simd>: Copy {
    /// Reads this kernel out of the field of `data` that holds it.
    ///
    /// # Safety
    ///
    /// `data`'s live field must be the one this kernel reads, as [`KernelData::new`] and
    /// [`crate::vector_build`] agree on for a [`crate::FinderKind`].
    unsafe fn from_data(simd: S, data: &KernelData) -> Self;

    fn matches<V: SimdInt<S, Element = u8, Block = u8x16<S>, ByteVector = V>>(
        &self,
        chunk: V,
    ) -> V::Mask;
}

/// Whether the target has a single-instruction dynamic byte shuffle.
///
/// [`kernels::SmallSet`], [`kernels::SingleNibble`] and [`kernels::AnyByte`] are built on one. Where it is missing,
/// `swizzle_dyn` degrades into a per-lane gather through memory, which loses to probing
/// the byte set directly with [`crate::bytewise`].
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
pub(crate) fn find_next<S: Simd, K: Kernel<S>>(
    simd: S,
    state: &mut IterState<'_>,
    kernel: K,
) -> MatchedBitset {
    let (haystack, mut from) = (state.haystack, state.pos);
    // SAFETY: `pos` only ever moves to an offset this function already scanned to, so it
    // never passes the end.
    let unscanned = unsafe { haystack.get_unchecked(from..) };
    let (chunks, tail) = unscanned.as_chunks::<CHUNK_BYTES>();
    let (pairs, rest) = chunks.as_chunks::<2>();

    for [first, second] in pairs {
        let matched_first = kernel.matches(u8x64::load_array_ref(simd, first));
        let matched_second = kernel.matches(u8x64::load_array_ref(simd, second));
        if (matched_first | matched_second).any_true() {
            state.bits_offset = from;
            state.pos = from + first.len() + second.len();
            return MatchedBitset::from(matched_first.to_bitmask())
                | MatchedBitset::from(matched_second.to_bitmask()) << CHUNK_BYTES;
        }
        from += first.len() + second.len();
    }

    if let [chunk] = rest {
        let matched = kernel.matches(u8x64::load_array_ref(simd, chunk));
        if matched.any_true() {
            state.bits_offset = from;
            state.pos = from + CHUNK_BYTES;
            return MatchedBitset::from(matched.to_bitmask());
        }
        from += chunk.len();
    }

    let (blocks, tail) = tail.as_chunks::<BLOCK_BYTES>();
    for block in blocks {
        let matched = kernel.matches(u8x16::load_array_ref(simd, block));
        if matched.any_true() {
            state.bits_offset = from;
            state.pos = from + block.len();
            return MatchedBitset::from(matched.to_bitmask());
        }
        from += block.len();
    }
    state.bits_offset = haystack.len() - tail.len();
    state.pos = haystack.len();
    if tail.is_empty() {
        0
    } else {
        MatchedBitset::from(tail_bits(simd, &kernel, haystack, tail))
    }
}

/// Counts every matching byte of `haystack`.
#[inline(always)]
pub(crate) fn count<S: Simd, K: Kernel<S>>(simd: S, haystack: &[u8], kernel: K) -> usize {
    // Each lane of the counting accumulator starts at zero, and gains at most one per chunk,
    // we can accumulate within a single vector until
    const CHUNKS_PER_ACCUMULATOR: usize = u8::MAX as usize;

    let (chunks, tail) = haystack.as_chunks::<CHUNK_BYTES>();
    let (blocks, tail) = tail.as_chunks::<BLOCK_BYTES>();

    let mut total = 0;
    for batch in chunks.chunks(CHUNKS_PER_ACCUMULATOR) {
        let mut counts = u8x64::splat(simd, 0);
        for chunk in batch {
            let matches = kernel.matches(u8x64::from_slice(simd, chunk));
            let matches: u8x64<_> = i8x64::load_array(simd, matches.into()).bitcast();
            // A matching lane is all ones (-1), so subtracting it adds one.
            counts -= matches;
        }
        total += sum_lanes_64(simd, counts);
    }
    if !blocks.is_empty() {
        let mut counts = u8x16::splat(simd, 0);
        for block in blocks {
            let matches = kernel.matches(u8x16::load_array_ref(simd, block));
            let matches: u8x16<_> = i8x16::load_array(simd, matches.into()).bitcast();
            counts -= matches;
        }
        let (count_l, count_r) = counts.widen();
        total += usize::from((count_l + count_r).reduce_sum());
    }
    if !tail.is_empty() {
        let bits = tail_bits(simd, &kernel, haystack, tail);
        total += bits.count_ones() as usize;
    }
    total
}

/// Matches `tail`, the final bytes of `haystack`, returning their bits at positions
/// `0..tail.len()`.
///
/// A haystack of at least one full block is handled by re-reading the last block and
/// shifting the already-scanned bits off the bottom, which avoids staging the tail in
/// a padded buffer.
#[inline(always)]
fn tail_bits<S: Simd, K: Kernel<S>>(simd: S, kernel: &K, haystack: &[u8], tail: &[u8]) -> u64 {
    debug_assert!(0 < tail.len() && tail.len() < BLOCK_BYTES);
    if let Some(chunk) = haystack.last_chunk::<BLOCK_BYTES>() {
        let matched = kernel.matches(u8x16::load_array_ref(simd, chunk));
        matched.to_bitmask() >> (BLOCK_BYTES - tail.len())
    } else {
        short_tail_bits(simd, kernel, tail)
    }
}

/// Matches a haystack shorter than one [`CHUNK_BYTES`], returning its bits at positions
/// `0..short_haystack.len()`.
///
/// llvm really likes to turn copies of dynamic size into actual calls to memcpy, even if we
/// can convince it the number of bytes is very small. We go to some lengths here to ensure
/// we keep all copies to constants here
#[inline(always)]
fn short_tail_bits<S: Simd, K: Kernel<S>>(simd: S, kernel: &K, short_haystack: &[u8]) -> u64 {
    // Copies the first and last `N` bytes of `haystack` into the front of a buffer, for a
    // haystack too short to load a `u8x16` from either end.
    #[inline]
    fn stage<const N: usize>(haystack: &[u8]) -> [u8; 16] {
        const { assert!(N <= 16 / 2) }
        debug_assert!(haystack.len() >= N);

        let mut buf = [0; 16];
        buf[..N].copy_from_slice(&haystack[..N]);
        buf[N..2 * N].copy_from_slice(&haystack[haystack.len() - N..]);
        buf
    }

    // Slides the bits of two `staged`-byte ends back to the positions they were read from,
    // discarding everything the ends did not cover.
    //
    // The two overlap in the middle, where they agree. Neither half may keep more bits than it
    // read bytes: above the front's sit the back's, and above the back's sit the lanes it was
    // duplicated into and the staging buffer's zero padding, which matches whenever the byte
    // set holds zero.
    #[inline]
    fn slide_ends(bits: u64, staged: usize, len: usize) -> u64 {
        let kept = !(u64::MAX << staged);
        (bits & kept) | (((bits >> staged) & kept) << (len - staged))
    }

    let len = short_haystack.len();
    debug_assert!(0 < len && len < BLOCK_BYTES);

    let (buf, staged) = match len {
        8.. => (stage::<8>(short_haystack), 8),
        4..8 => (stage::<4>(short_haystack), 4),
        2..4 => (stage::<2>(short_haystack), 2),
        0..2 => (stage::<1>(short_haystack), 1),
    };
    let ends = u8x16::load_array(simd, buf);
    slide_ends(kernel.matches(ends).to_bitmask(), staged, len)
}

/// Sums every lane of a vector.
#[inline(always)]
fn sum_lanes_64<S: Simd>(simd: S, counts: u8x64<S>) -> usize {
    // For some reason, llvm does a good job with x86 with a simple loop, but neon really benefits
    // from a custom kernel.
    #[cfg(target_arch = "aarch64")]
    if let Some(neon) = simd.level().as_neon() {
        use fearless_simd::u16x8;

        let (l, r) = counts.split();
        let ((a, b), (c, d)) = (l.split(), r.split());
        let a: u16x8<_> = aarch64_sum_widen(neon, a.into()).simd_into(simd);
        let b: u16x8<_> = aarch64_sum_widen(neon, b.into()).simd_into(simd);
        let c: u16x8<_> = aarch64_sum_widen(neon, c.into()).simd_into(simd);
        let d: u16x8<_> = aarch64_sum_widen(neon, d.into()).simd_into(simd);
        return usize::from(((a + b) + (c + d)).reduce_sum());
    }
    let _ = simd;
    let mut total = 0;
    for &lane in counts.as_array() {
        total += usize::from(lane);
    }
    total
}

kernel! {
    #[inline(always)]
    fn aarch64_sum_widen(simd: Neon, lanes: [u8; 16]) -> [u16; 8] {
        use core::arch::aarch64::*;
        use fearless_simd::u16x8;

        let lanes = u8x16::load_array(simd, lanes);
        let summed: u16x8<_> = vpaddlq_u8(lanes.into()).simd_into(simd);
        summed.into()
    }
}

kernel! {
    #[inline(always)]
    fn aarch64_swizzle_32_to_16(simd: Neon, table: [u8; 32], idx: [u8; 16]) -> [u8; 16] {
        use core::arch::aarch64::*;

        let table = u8x32::load_array(simd, table);
        let idx = u8x16::load_array(simd, idx);
        let res = vqtbl2q_u8(table.into(), idx.into());
        u8x16::simd_from(simd, res).into()
    }
}
