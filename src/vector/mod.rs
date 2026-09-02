pub(crate) mod kernels;

use crate::{IterState, KernelData, MatchedBitset};
use fearless_simd::prelude::*;
use fearless_simd::{Level, i8x16, i8x64, kernel, u8x16, u8x64, u64x2};

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
    /// the per-level `build` that `crate::level_scans!` writes agree on for a [`crate::Kind`].
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

/// Returns the offset of the first matching byte of `haystack`.
///
/// Broadly [`find_next`]'s walk, with everything an iterator would need afterwards left out:
/// no [`IterState`] to write back, and a bitmask extracted only for the one unit that
/// matched, whose lowest set bit is the answer.
///
/// Where it parts company is in what it is willing to spend to reach the end of a haystack it
/// will not reach. A refill is entered with whatever is left of one, so it is short only on
/// its last call and long on every other; a search is entered with the whole thing, and stops
/// at the first match. So this one leads with the cases that end early — a haystack below one
/// chunk, and a match in the first chunk — and only then falls into the paired loop that
/// carries a long scan.
#[inline(always)]
pub(crate) fn find_first<S: Simd, K: Kernel<S>>(
    simd: S,
    haystack: &[u8],
    kernel: K,
) -> Option<usize> {
    // A haystack shorter than one chunk reaches none of the chunk walk below, and taking its
    // length past three chunk sizes to discover that is most of what such a call costs.
    // `find_next` has no equivalent because an iterator arrives here with the length it has
    // left, which is short only on its last refill.
    if haystack.len() < CHUNK_BYTES {
        return find_first_short(simd, haystack, kernel);
    }

    let (chunks, tail) = haystack.as_chunks::<CHUNK_BYTES>();

    // The first chunk on its own, ahead of the pairing below. Pairing buys one `any_true`
    // per two chunks, which is what carries a long scan, but it also loads a second chunk
    // before it will look at the first one's answer. A search that returns early almost
    // always returns in the first chunk, and should not pay for a second to find that out.
    let mut from = 0;
    let chunks = if let [first, rest @ ..] = chunks {
        let matched = kernel.matches(u8x64::load_array_ref(simd, first));
        if matched.any_true() {
            return Some(matched.to_bitmask().trailing_zeros() as usize);
        }
        from = CHUNK_BYTES;
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
                from + bits.trailing_zeros() as usize
            } else {
                from + CHUNK_BYTES + matched_second.to_bitmask().trailing_zeros() as usize
            });
        }
        from += 2 * CHUNK_BYTES;
    }

    if let [chunk] = rest {
        let matched = kernel.matches(u8x64::load_array_ref(simd, chunk));
        if matched.any_true() {
            return Some(from + matched.to_bitmask().trailing_zeros() as usize);
        }
        from += CHUNK_BYTES;
    }

    let (blocks, tail) = tail.as_chunks::<BLOCK_BYTES>();
    for block in blocks {
        let matched = kernel.matches(u8x16::load_array_ref(simd, block));
        if matched.any_true() {
            return Some(from + matched.to_bitmask().trailing_zeros() as usize);
        }
        from += BLOCK_BYTES;
    }
    if tail.is_empty() {
        return None;
    }
    let bits = tail_bits(simd, &kernel, haystack, tail);
    (bits != 0).then(|| haystack.len() - tail.len() + bits.trailing_zeros() as usize)
}

/// [`find_first`] for a haystack shorter than one [`CHUNK_BYTES`].
///
/// Split out so the length ladder is one comparison deep for the two cases it can be in,
/// rather than the three the chunk walk would take it through on the way down.
#[inline(always)]
fn find_first_short<S: Simd, K: Kernel<S>>(
    simd: S,
    haystack: &[u8],
    kernel: K,
) -> Option<usize> {
    debug_assert!(haystack.len() < CHUNK_BYTES);
    let (blocks, tail) = haystack.as_chunks::<BLOCK_BYTES>();

    // No whole block means the haystack is shorter than one, which is the one case
    // [`tail_bits`] cannot serve by re-reading the last block. Going straight to the staged
    // path spares it that test, and the tail is the whole haystack.
    if blocks.is_empty() {
        if haystack.is_empty() {
            return None;
        }
        let bits = short_tail_bits(simd, &kernel, haystack);
        return (bits != 0).then(|| bits.trailing_zeros() as usize);
    }

    let mut from = 0;
    for block in blocks {
        let matched = kernel.matches(u8x16::load_array_ref(simd, block));
        if matched.any_true() {
            return Some(from + matched.to_bitmask().trailing_zeros() as usize);
        }
        from += BLOCK_BYTES;
    }
    if tail.is_empty() {
        return None;
    }
    let bits = tail_bits(simd, &kernel, haystack, tail);
    (bits != 0).then(|| haystack.len() - tail.len() + bits.trailing_zeros() as usize)
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
/// The two ends are staged through general-purpose registers rather than a `[u8; 16]` buffer.
/// A buffer written as two narrow stores and then read back as one 16-byte vector load is the
/// shape store forwarding cannot satisfy, so the load waits for both stores to reach the cache
/// — tens of cycles, on a path whose whole job is to be short. Assembling the same bytes in
/// two `u64`s and handing the pair over as a value keeps it off the stack entirely, and each
/// load is still of a constant width, which is what kept `copy_from_slice` from lowering to a
/// `memcpy` call.
#[inline(always)]
fn short_tail_bits<S: Simd, K: Kernel<S>>(simd: S, kernel: &K, short_haystack: &[u8]) -> u64 {
    /// The first and last `N` bytes of `haystack`, as two little-endian integers of `N` bytes
    /// each, packed into the bottom of a word.
    ///
    /// `N` is at most four, so both fit; the eight-byte case is the one below that needs a
    /// word each.
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

    // Slides the bits of two `staged`-byte ends back to the positions they were read from,
    // discarding everything the ends did not cover.
    //
    // The two overlap in the middle, where they agree. Neither half may keep more bits than it
    // read bytes: above the front's sit the back's, and above the back's sit the lanes it was
    // duplicated into and the zero padding, which matches whenever the byte set holds zero.
    #[inline]
    fn slide_ends(bits: u64, staged: usize, len: usize) -> u64 {
        let kept = !(u64::MAX << staged);
        (bits & kept) | (((bits >> staged) & kept) << (len - staged))
    }

    let len = short_haystack.len();
    debug_assert!(0 < len && len < BLOCK_BYTES);

    let (words, staged) = match len {
        // Eight bytes from each end fill both words, so this is the one case that reads them
        // separately rather than packing a pair into one.
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
        use fearless_simd::u8x32;

        let table = u8x32::load_array(simd, table);
        let idx = u8x16::load_array(simd, idx);
        let res = vqtbl2q_u8(table.into(), idx.into());
        u8x16::simd_from(simd, res).into()
    }
}
