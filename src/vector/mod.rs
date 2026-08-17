pub(crate) mod kernels;

use crate::{IterState, KernelData, MatchedBitset, ScanOps};
use core::mem::transmute_copy;
use fearless_simd::prelude::*;
use fearless_simd::{Level, i8x16, i8x64, kernel, u8x16, u8x32, u8x64, u64x2};

const CHUNK_BYTES: usize = 64;
const _: () = assert!(MatchedBitset::BITS as usize >= CHUNK_BYTES * 2);

const BLOCK_BYTES: usize = 16;

/// Leading bytes probed before staging a sub-vector haystack.
///
/// One probe improves front-match latency; additional probes slow misses and later matches.
pub(crate) const PROBE_BYTES: usize = 1;

/// Tests a chunk of [`CHUNK_BYTES`] bytes against a byte set.
pub(crate) trait Kernel<S: Simd>: Copy {
    /// Reads this kernel out of the field of `kernel_data` that holds it.
    ///
    /// # Safety
    ///
    /// `kernel_data` must have the field this kernel reads as its live field.
    unsafe fn from_data(simd: S, kernel_data: &KernelData) -> Self;

    fn matches<V: SimdInt<S, Element = u8, Block = u8x16<S>, ByteVector = V>>(
        &self,
        chunk: V,
    ) -> V::Mask;

    /// Whether the byte set holds `byte`.
    ///
    /// This scalar path avoids staging for the leading bytes of short haystacks.
    fn matches_byte(&self, byte: u8) -> bool;
}

/// Whether the target has a single-instruction dynamic byte shuffle.
///
/// Shuffle-based kernels fall back to [`crate::BitsetLookup`] without one.
pub(crate) fn has_byte_shuffle(level: Level) -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        // SSE2 lacks `pshufb`; later levels accepted by `as_sse4_2` provide it.
        level.as_sse4_2().is_some()
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        !level.is_fallback()
    }
}

/// Scans from the current position until it finds matches or reaches the end.
///
/// Chunk pairs share an [`any_true`](fearless_simd::SimdMask::any_true) check and are returned
/// together so the second chunk is not rescanned.
#[inline(always)]
pub(crate) fn next_match_batch<S: Simd, K: Kernel<S>>(
    simd: S,
    state: &mut IterState<'_>,
    kernel: K,
) -> MatchedBitset {
    let (haystack, mut offset) = (state.haystack, state.scan_offset);
    // SAFETY: `state.scan_offset` never exceeds the haystack length.
    let unscanned = unsafe { haystack.get_unchecked(offset..) };
    let (chunks, tail) = unscanned.as_chunks::<CHUNK_BYTES>();
    let (pairs, rest) = chunks.as_chunks::<2>();

    for [first, second] in pairs {
        let matched_first = kernel.matches(u8x64::load_array_ref(simd, first));
        let matched_second = kernel.matches(u8x64::load_array_ref(simd, second));
        if (matched_first | matched_second).any_true() {
            state.match_base = offset;
            state.scan_offset = offset + 2 * CHUNK_BYTES;
            return MatchedBitset::from(matched_first.to_bitmask())
                | MatchedBitset::from(matched_second.to_bitmask()) << CHUNK_BYTES;
        }
        offset += 2 * CHUNK_BYTES;
    }

    if let [chunk] = rest {
        let matched = kernel.matches(u8x64::load_array_ref(simd, chunk));
        if matched.any_true() {
            state.match_base = offset;
            state.scan_offset = offset + CHUNK_BYTES;
            return MatchedBitset::from(matched.to_bitmask());
        }
        offset += CHUNK_BYTES;
    }

    let (blocks, tail) = tail.as_chunks::<BLOCK_BYTES>();
    for block in blocks {
        let matched = kernel.matches(u8x16::load_array_ref(simd, block));
        if matched.any_true() {
            state.match_base = offset;
            state.scan_offset = offset + BLOCK_BYTES;
            return MatchedBitset::from(matched.to_bitmask());
        }
        offset += BLOCK_BYTES;
    }
    state.match_base = haystack.len() - tail.len();
    state.scan_offset = haystack.len();
    if tail.is_empty() {
        0
    } else {
        MatchedBitset::from(tail_bits(simd, &kernel, haystack, tail))
    }
}

/// Returns the offset of the first matching byte of `haystack`.
///
/// Handles short haystacks and the first chunk before entering the paired scan loop.
#[inline(always)]
pub(crate) fn first_match<S: Simd, K: Kernel<S>>(
    simd: S,
    haystack: &[u8],
    kernel: K,
) -> Option<usize> {
    if haystack.len() < CHUNK_BYTES {
        return first_match_short(simd, haystack, kernel);
    }

    let (chunks, _) = haystack.as_chunks::<CHUNK_BYTES>();

    // Avoid loading the second chunk before checking the common first-chunk case.
    let mut offset = 0;
    let chunks = if let [first, rest @ ..] = chunks {
        let matched = kernel.matches(u8x64::load_array_ref(simd, first));
        if matched.any_true() {
            return Some(matched.to_bitmask().trailing_zeros() as usize);
        }
        offset = CHUNK_BYTES;
        rest
    } else {
        chunks
    };

    let (pairs, rest) = chunks.as_chunks::<2>();
    for [first, second] in pairs {
        let matched_first = kernel.matches(u8x64::load_array_ref(simd, first));
        let matched_second = kernel.matches(u8x64::load_array_ref(simd, second));
        if (matched_first | matched_second).any_true() {
            let bits = matched_first.to_bitmask();
            return Some(if bits != 0 {
                offset + bits.trailing_zeros() as usize
            } else {
                offset + CHUNK_BYTES + matched_second.to_bitmask().trailing_zeros() as usize
            });
        }
        offset += 2 * CHUNK_BYTES;
    }

    if let [chunk] = rest {
        let matched = kernel.matches(u8x64::load_array_ref(simd, chunk));
        if matched.any_true() {
            return Some(offset + matched.to_bitmask().trailing_zeros() as usize);
        }
        offset += CHUNK_BYTES;
    }

    // Re-read the last chunk; the overlapping prefix is known not to match.
    if offset == haystack.len() {
        return None;
    }
    let last = haystack
        .last_chunk::<CHUNK_BYTES>()
        .expect("the haystack is at least one chunk");
    let matched = kernel.matches(u8x64::load_array_ref(simd, last));
    matched
        .any_true()
        .then(|| haystack.len() - CHUNK_BYTES + matched.to_bitmask().trailing_zeros() as usize)
}

/// [`first_match`] for a haystack shorter than one [`CHUNK_BYTES`].
///
/// Overlapping front and back vectors avoid a loop. Sub-vector haystacks use a scalar probe
/// followed by staged ends.
#[inline(always)]
fn first_match_short<S: Simd, K: Kernel<S>>(simd: S, haystack: &[u8], kernel: K) -> Option<usize> {
    debug_assert!(haystack.len() < CHUNK_BYTES);
    let len = haystack.len();

    if let (Some(front), Some(back)) = (
        haystack.first_chunk::<{ 2 * BLOCK_BYTES }>(),
        haystack.last_chunk::<{ 2 * BLOCK_BYTES }>(),
    ) {
        let matched = kernel.matches(u8x32::load_array_ref(simd, front));
        if matched.any_true() {
            return Some(matched.to_bitmask().trailing_zeros() as usize);
        }
        let matched = kernel.matches(u8x32::load_array_ref(simd, back));
        return matched
            .any_true()
            .then(|| len - 2 * BLOCK_BYTES + matched.to_bitmask().trailing_zeros() as usize);
    }

    if let (Some(front), Some(back)) = (
        haystack.first_chunk::<BLOCK_BYTES>(),
        haystack.last_chunk::<BLOCK_BYTES>(),
    ) {
        let matched = kernel.matches(u8x16::load_array_ref(simd, front));
        if matched.any_true() {
            return Some(matched.to_bitmask().trailing_zeros() as usize);
        }
        let matched = kernel.matches(u8x16::load_array_ref(simd, back));
        return matched
            .any_true()
            .then(|| len - BLOCK_BYTES + matched.to_bitmask().trailing_zeros() as usize);
    }

    for (offset, &byte) in haystack.iter().take(PROBE_BYTES).enumerate() {
        if kernel.matches_byte(byte) {
            return Some(offset);
        }
    }
    if len <= PROBE_BYTES {
        return None;
    }

    // Keep staged ends separate so each half retains its original offset.
    let rest = &haystack[PROBE_BYTES..];
    let (bits, staged_len) = staged_ends_bits(simd, &kernel, rest);
    let keep_mask = !(u64::MAX << staged_len);
    let front = bits & keep_mask;
    if front != 0 {
        return Some(PROBE_BYTES + front.trailing_zeros() as usize);
    }
    let back = (bits >> staged_len) & keep_mask;
    (back != 0).then(|| len - staged_len + back.trailing_zeros() as usize)
}

/// Counts every matching byte of `haystack`.
#[inline(always)]
pub(crate) fn count_all<S: Simd, K: Kernel<S>>(simd: S, haystack: &[u8], kernel: K) -> usize {
    // Drain after 255 chunks to prevent byte-lane overflow.
    const CHUNKS_PER_ACCUMULATOR: usize = u8::MAX as usize;

    let (chunks, tail) = haystack.as_chunks::<CHUNK_BYTES>();
    let (blocks, tail) = tail.as_chunks::<BLOCK_BYTES>();

    let mut total = 0;
    for batch in chunks.chunks(CHUNKS_PER_ACCUMULATOR) {
        let mut counts = u8x64::splat(simd, 0);
        for chunk in batch {
            let match_mask = kernel.matches(u8x64::from_slice(simd, chunk));
            let match_mask: u8x64<_> = i8x64::load_array(simd, match_mask.into()).bitcast();
            // Subtracting an all-ones match mask adds one per lane.
            counts -= match_mask;
        }
        total += sum_lanes_64(simd, counts);
    }
    if !blocks.is_empty() {
        let mut counts = u8x16::splat(simd, 0);
        for block in blocks {
            let match_mask = kernel.matches(u8x16::load_array_ref(simd, block));
            let match_mask: u8x16<_> = i8x16::load_array(simd, match_mask.into()).bitcast();
            counts -= match_mask;
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

/// Returns match bits for the final partial block in `haystack`.
///
/// Re-reads the last full block when possible; shorter haystacks are staged.
#[inline(always)]
fn tail_bits<S: Simd, K: Kernel<S>>(simd: S, kernel: &K, haystack: &[u8], tail: &[u8]) -> u64 {
    debug_assert!(!tail.is_empty() && tail.len() < BLOCK_BYTES);
    if let Some(chunk) = haystack.last_chunk::<BLOCK_BYTES>() {
        let matched = kernel.matches(u8x16::load_array_ref(simd, chunk));
        matched.to_bitmask() >> (BLOCK_BYTES - tail.len())
    } else {
        short_tail_bits(simd, kernel, tail)
    }
}

/// Returns match bits for both ends of a haystack shorter than [`BLOCK_BYTES`].
///
/// General-purpose registers avoid store-forwarding stalls from a partially initialized vector
/// buffer. The returned halves may overlap and only their lowest `staged_len` bits are valid.
#[inline(always)]
fn staged_ends_bits<S: Simd, K: Kernel<S>>(
    simd: S,
    kernel: &K,
    short_haystack: &[u8],
) -> (u64, usize) {
    /// Packs the first and last `N` bytes into one word.
    #[inline]
    fn ends<const N: usize>(haystack: &[u8]) -> u64 {
        const { assert!(N <= 4) }
        debug_assert!(haystack.len() >= N);

        #[inline]
        fn end<const N: usize>(bytes: &[u8]) -> u64 {
            let mut buf = [0; 8];
            buf[..N].copy_from_slice(bytes);
            u64::from_le_bytes(buf)
        }

        let (first, last) = (&haystack[..N], &haystack[haystack.len() - N..]);
        end::<N>(first) | end::<N>(last) << (N * 8)
    }

    let len = short_haystack.len();
    debug_assert!(!short_haystack.is_empty() && len < BLOCK_BYTES);

    let (words, staged_len) = match len {
        8.. => {
            let first = u64::from_le_bytes(*short_haystack.first_chunk::<8>().unwrap());
            let last = u64::from_le_bytes(*short_haystack.last_chunk::<8>().unwrap());
            ([first, last], 8)
        }
        4..8 => ([ends::<4>(short_haystack), 0], 4),
        2..4 => ([ends::<2>(short_haystack), 0], 2),
        0..2 => ([ends::<1>(short_haystack), 0], 1),
    };
    let ends: u8x16<S> = u64x2::load_array(simd, words).bitcast();
    (kernel.matches(ends).to_bitmask(), staged_len)
}

/// [`staged_ends_bits`] with the two ends slid back to the positions they were read from, for
/// a caller that wants every match rather than the first.
#[inline(always)]
fn short_tail_bits<S: Simd, K: Kernel<S>>(simd: S, kernel: &K, short_haystack: &[u8]) -> u64 {
    let len = short_haystack.len();
    let (bits, staged_len) = staged_ends_bits(simd, kernel, short_haystack);
    let keep_mask = !(u64::MAX << staged_len);
    (bits & keep_mask) | (((bits >> staged_len) & keep_mask) << (len - staged_len))
}

/// Sums every lane of a vector.
#[inline(always)]
fn sum_lanes_64<S: Simd>(simd: S, counts: u8x64<S>) -> usize {
    // NEON benefits from explicit pairwise widening; x86 optimizes the scalar loop well.
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

pub(crate) fn scan_ops<S: Simd, K: Kernel<S>>(simd: S) -> &'static ScanOps {
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
        // SAFETY: `S` is zero-sized, and the caller guarantees target support.
        unsafe { transmute_copy(&()) }
    }
    /// Scans for the next matching chunk.
    ///
    /// # Safety
    ///
    /// The running target must support this module's level, and `kernel_data`'s live field
    /// must be the one `K` reads.
    unsafe fn next_match_batch_impl<S: Simd, K: Kernel<S>>(
        kernel_data: &KernelData,
        state: &mut IterState<'_>,
    ) -> MatchedBitset {
        // SAFETY: guaranteed by the caller.
        let simd = unsafe { token::<S>() };
        simd.vectorize(
            #[inline(always)]
            move || {
                // SAFETY: `kernel_data` has `K`'s live field.
                let kernel = unsafe { K::from_data(simd, kernel_data) };
                next_match_batch(simd, state, kernel)
            },
        )
    }

    /// Counts all matching bytes.
    ///
    /// # Safety
    ///
    /// The running target must support `S`, and `kernel_data` must have the field `K` reads as its
    /// live field.
    unsafe fn count_all_impl<S: Simd, K: Kernel<S>>(
        kernel_data: &KernelData,
        haystack: &[u8],
    ) -> usize {
        // SAFETY: guaranteed by the caller.
        let simd = unsafe { token::<S>() };
        simd.vectorize(
            #[inline(always)]
            move || {
                // SAFETY: `kernel_data` has `K`'s live field.
                let kernel = unsafe { K::from_data(simd, kernel_data) };
                count_all(simd, haystack, kernel)
            },
        )
    }

    /// Finds the first matching byte.
    ///
    /// # Safety
    ///
    /// The running target must support `S`, and `kernel_data` must have the field `K` reads as its
    /// live field.
    unsafe fn first_match_impl<S: Simd, K: Kernel<S>>(
        kernel_data: &KernelData,
        haystack: &[u8],
    ) -> Option<usize> {
        // SAFETY: guaranteed by the caller.
        let simd = unsafe { token::<S>() };
        simd.vectorize(
            #[inline(always)]
            move || {
                // SAFETY: `kernel_data` has `K`'s live field.
                let kernel = unsafe { K::from_data(simd, kernel_data) };
                first_match(simd, haystack, kernel)
            },
        )
    }
    _ = simd;

    &ScanOps {
        next_match_batch: next_match_batch_impl::<S, K>,
        count_all: count_all_impl::<S, K>,
        first_match: first_match_impl::<S, K>,
    }
}
