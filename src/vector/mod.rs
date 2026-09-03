pub(crate) mod kernels;

use crate::{IterState, KernelData, Kind, MatchedBitset, Scan, Search, never_scan};
use core::mem::transmute_copy;
use fearless_simd::prelude::*;
use fearless_simd::{Level, i8x16, i8x64, kernel, u8x16, u8x32, u8x64, u64x2};

const CHUNK_BYTES: usize = 64;
const _: () = assert!(MatchedBitset::BITS as usize >= CHUNK_BYTES * 2);

const BLOCK_BYTES: usize = 16;

/// How many leading bytes [`find_first_short`] asks about one at a time before it stages a
/// haystack too short to fill a vector.
///
/// Staging costs the same whether the answer is the first byte or the last: the two ends go
/// through general-purpose registers into a vector, the vector produces a bitmask, and only
/// then is there an answer. A probe with [`Kernel::matches_byte`] is a compare and a branch,
/// so it answers its own byte in about a cycle — which is what `memchr` does below its own
/// vector width, and the whole of what it was beating us by there.
///
/// # Why one
///
/// Every byte probed is a byte the staging behind it pays for anyway, so a probe that does not
/// answer is wasted work. Swept over 0, 1, 2, 4, 8 and 16 on an AVX2 host, at 1, 4, 8 and 12
/// bytes, against a match at the front, a match halfway, and no match at all:
///
/// - the first byte carries the whole win. A match at the front went from 3.3-4.8ns to
///   2.7-3.3, and no longer probe beat that by more than a timer tick.
/// - the bytes after it only cost. A miss at twelve bytes ran 4.2ns at one probe, 4.5 at two,
///   5.0 at four, 5.9 at eight and 6.2 at sixteen — the last of which is `memchr`'s own walk,
///   against 3.6 for staging alone.
///
/// A one-byte haystack is the one length where probing the lot wins outright, and one probe
/// already covers it: it answers and returns without staging anything.
pub(crate) const PROBE_BYTES: usize = 1;

/// Tests a chunk of [`CHUNK_BYTES`] bytes against a byte set.
pub(crate) trait Kernel<S: Simd>: Copy {
    /// Reads this kernel out of the field of `data` that holds it.
    ///
    /// # Safety
    ///
    /// `data`'s live field must be the one this kernel reads, as [`KernelData::new`] and
    /// the per-level `build` that `level_scans!` writes agree on for a [`Kind`].
    unsafe fn from_data(simd: S, data: &KernelData) -> Self;

    fn matches<V: SimdInt<S, Element = u8, Block = u8x16<S>, ByteVector = V>>(
        &self,
        chunk: V,
    ) -> V::Mask;

    /// Whether the byte set holds `byte`.
    ///
    /// The scalar counterpart of [`matches`](Self::matches), and not a fallback for a target
    /// that cannot run it: every kernel here is cheaper per byte in a vector, and this exists
    /// for the haystack that cannot fill one. Below a vector's width the lanes have to be
    /// staged before they can be tested, and staging costs the same whether the answer is the
    /// first byte or the last, where a compare per byte answers the first byte in a compare.
    ///
    /// Each of these reads the same data the vector kernel does, so the two agree by
    /// construction rather than by keeping two descriptions of one set in step.
    fn matches_byte(&self, byte: u8) -> bool;
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
/// Extracting the bitmask is the expensive half of a scan, so [`any_true`](fearless_simd::SimdMask::any_true) skips it
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

    let (chunks, _) = haystack.as_chunks::<CHUNK_BYTES>();

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

    // Whatever is left is shorter than a chunk, and the haystack is not, so one read of its
    // last chunk covers it. That read reaches back over bytes already scanned, which costs
    // nothing and cannot mislead: this only runs because none of them matched.
    if from == haystack.len() {
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

/// [`find_first`] for a haystack shorter than one [`CHUNK_BYTES`].
///
/// Two reads that overlap in the middle cover any length between one vector and two, so this
/// is a ladder of widths rather than a walk: at most two vector operations and no loop,
/// where blocks and a tail would take up to four for the same bytes and branch between them.
/// The reads may see a byte twice, which costs nothing, because the front is tested first and
/// a match there is at the offset it reports.
///
/// Below the narrowest vector the ladder runs out, and the bottom rung is [`PROBE_BYTES`]
/// scalar probes ahead of the staged pair.
#[inline(always)]
fn find_first_short<S: Simd, K: Kernel<S>>(
    simd: S,
    haystack: &[u8],
    kernel: K,
) -> Option<usize> {
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

    // Below one vector there is nothing to overlap with, so the ladder runs out and the
    // leading bytes are asked for one at a time instead. See [`PROBE_BYTES`].
    for (offset, &byte) in haystack.iter().take(PROBE_BYTES).enumerate() {
        if kernel.matches_byte(byte) {
            return Some(offset);
        }
    }
    if len <= PROBE_BYTES {
        return None;
    }

    // What the probe did not reach is staged into one vector as two ends — and read back as
    // two halves rather than slid into place, since the front half answers whenever it can
    // and the back half is only reached when it cannot.
    let rest = &haystack[PROBE_BYTES..];
    let (bits, staged) = staged_ends_bits(simd, &kernel, rest);
    let kept = !(u64::MAX << staged);
    let front = bits & kept;
    if front != 0 {
        return Some(PROBE_BYTES + front.trailing_zeros() as usize);
    }
    // The back end was read from the end of the haystack either way, so its offsets are the
    // ones it would have had without the probe.
    let back = (bits >> staged) & kept;
    (back != 0).then(|| len - staged + back.trailing_zeros() as usize)
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

/// Matches a haystack shorter than one [`BLOCK_BYTES`], returning the bits of its two staged
/// ends and how many bytes each end covers.
///
/// The ends are staged through general-purpose registers rather than a `[u8; 16]` buffer. A
/// buffer written as two narrow stores and then read back as one 16-byte vector load is the
/// shape store forwarding cannot satisfy, so the load waits for both stores to reach the cache
/// — tens of cycles, on a path whose whole job is to be short. Assembling the same bytes in
/// two `u64`s and handing the pair over as a value keeps it off the stack entirely, and each
/// load is still of a constant width, which is what kept `copy_from_slice` from lowering to a
/// `memcpy` call.
///
/// Bit `i` below `staged` is byte `i`; bit `staged + i` is byte `len - staged + i`. The two
/// overlap in the middle, where they agree. Neither half means anything above its own
/// `staged` bits: past the front's sit the back's, and past the back's sit the lanes it was
/// duplicated into and the zero padding, which matches whenever the byte set holds zero.
#[inline(always)]
fn staged_ends_bits<S: Simd, K: Kernel<S>>(
    simd: S,
    kernel: &K,
    short_haystack: &[u8],
) -> (u64, usize) {
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
    (kernel.matches(ends).to_bitmask(), staged)
}

/// [`staged_ends_bits`] with the two ends slid back to the positions they were read from, for
/// a caller that wants every match rather than the first.
#[inline(always)]
fn short_tail_bits<S: Simd, K: Kernel<S>>(simd: S, kernel: &K, short_haystack: &[u8]) -> u64 {
    let len = short_haystack.len();
    let (bits, staged) = staged_ends_bits(simd, kernel, short_haystack);
    let kept = !(u64::MAX << staged);
    (bits & kept) | (((bits >> staged) & kept) << (len - staged))
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

/// Writes one level's [`Scan`] entry points, and the [`Kind`] match that picks between them.
///
/// Every entry point reaches its scan through [`Simd::vectorize`], which is what establishes
/// the target-feature context, and does it by calling into that context rather than by
/// putting the features on the entry point. So each of these is a trampoline, and what
/// matters is that it be a bare `jmp` and not a stack frame: a closure of more than two words
/// is passed indirectly, so it lands in the trampoline's own frame, which the call then has
/// to outlive.
///
/// Two words is why [`Search`] exists, and why `find_next` does not need it. Both leave these
/// carrying one or two pointers, all of them into the caller's memory, and all three come out
/// as `jmp`.
///
/// # Where there is no trampoline at all
///
/// `vectorize`'s target-feature function is `#[inline]`, and a caller may inline a callee
/// whose feature set it already covers. So wherever the level is what the crate is being
/// compiled for anyway, LLVM folds that function into these and the entry point *is* the
/// scan — the same code putting the attribute here by hand would give. That is every level
/// which is baseline for its architecture: all of `aarch64`, since NEON always is, where the
/// 42 boundaries this macro writes come out as none; `sse2` on `x86-64`; and `avx2` as well
/// in a build that enables it.
///
/// What still trampolines is a level above the compile-time baseline, reached by runtime
/// dispatch — `avx2` and `avx512` in an ordinary `x86-64` build. There the caller genuinely
/// lacks the features, and no amount of `#[inline]` may cross that.
///
/// # What the trampoline costs where it is left
///
/// Nothing that has survived measurement. Putting the level's `#[target_feature]` on the entry
/// point instead would remove it and let the arguments stay in registers rather than going
/// through [`Search`]; measured against that, `find` over the lengths where a `jmp` could
/// matter came out at a mean ratio of 1.015 over eleven points, alternating direction six
/// times. It was worth having when the closure was three words and the trampoline built a
/// frame to hold it — 1.13-1.30x — which is what [`Search`] is for.
///
/// It could not be taken anyway: there is no way to write that attribute from public API.
/// Spelling the feature list out here means keeping a copy of `fearless_simd`'s in step,
/// silently losing every lane operation to a call if it drifts; `fearless_simd::kernel!`
/// writes it correctly but takes only non-generic functions, so it would want one of these
/// per kind as well as per level. The remaining way in is the macro behind `kernel!`, which
/// is exported but `doc(hidden)`. `fearless_simd_macros::simd` is not a fourth option: it
/// expands to exactly the `vectorize` call below, and wants the token as a parameter, which
/// these cannot have and still share one fn-pointer type across levels.
macro_rules! level_scans {

    ($($(#[$cfg:meta])* $module:ident => $level:ident;)*) => {$(
        $(#[$cfg])*
        mod $module {
            use super::*;
            use fearless_simd::$level as Token;

            /// # Safety
            ///
            /// The running target must support this module's level, and `data`'s live field
            /// must be the one `K` reads.
            unsafe fn find_next<K: Kernel<Token>>(
                data: &KernelData,
                state: &mut IterState<'_>,
            ) -> MatchedBitset {
                // SAFETY: the caller's obligations, both of which `build` below discharges:
                // it is reached only through a `Level` that proves the support, and each arm
                // pairs a kernel with the kind whose `KernelData` field that kernel reads.
                let simd = unsafe { token::<Token>() };
                simd.vectorize(
                    #[inline(always)]
                    move || {
                        // SAFETY: as above.
                        let kernel = unsafe { K::from_data(simd, data) };
                        super::find_next(simd, state, kernel)
                    },
                )
            }

            /// # Safety
            ///
            /// As in [`find_next`].
            unsafe fn count_all<K: Kernel<Token>>(search: &Search<'_>) -> usize {
                // SAFETY: as in `find_next`.
                let simd = unsafe { token::<Token>() };
                simd.vectorize(
                    #[inline(always)]
                    move || {
                        // SAFETY: as above.
                        let kernel = unsafe { K::from_data(simd, search.data) };
                        super::count(simd, search.haystack, kernel)
                    },
                )
            }

            /// # Safety
            ///
            /// As in [`find_next`].
            unsafe fn find_first<K: Kernel<Token>>(search: &Search<'_>) -> Option<usize> {
                // SAFETY: as in `find_next`.
                let simd = unsafe { token::<Token>() };
                simd.vectorize(
                    #[inline(always)]
                    move || {
                        // SAFETY: as above.
                        let kernel = unsafe { K::from_data(simd, search.data) };
                        super::find_first(simd, search.haystack, kernel)
                    },
                )
            }

            fn scan<K: Kernel<Token>>() -> &'static Scan {
                &const {
                    Scan {
                        find_next: find_next::<K>,
                        count_all: count_all::<K>,
                        find_first: find_first::<K>,
                    }
                }
            }

            /// Picks the vector kernel for a kind. Each arm's kernel reads that same arm back
            /// in its [`Kernel`] impl.
            pub(super) fn build(kind: Kind) -> &'static Scan {
                match kind {
                    Kind::OneByte(_) => scan::<kernels::AnyOf<Token, 1>>(),
                    Kind::TwoBytes(_) => scan::<kernels::AnyOf<Token, 2>>(),
                    Kind::ThreeBytes(_) => scan::<kernels::AnyOf<Token, 3>>(),
                    Kind::OneRange(_) => scan::<kernels::OneRange<Token>>(),
                    Kind::SmallSet { .. } => scan::<kernels::SmallSet>(),
                    Kind::ConstantNibble(..) => scan::<kernels::SingleNibble>(),
                    Kind::AnyByte(_) => scan::<kernels::AnyByte>(),
                    Kind::Never => never_scan(),
                }
            }
        }
    )*};
}

level_scans! {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    sse2 => Sse2;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    sse4_2 => Sse4_2;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    avx2 => Avx2;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    avx512 => Avx512;
    #[cfg(target_arch = "aarch64")]
    neon => Neon;
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    wasm_simd128 => WasmSimd128;
}

/// The entry points for `level`'s vector kernels, or `None` for a level that has none: the
/// fallback level, and any level `level_scans!` has not been given.
///
/// Also what decides [`Family`](crate::Family), so a level without vector entry points
/// cannot be handed a [`Kind`] only a vector kernel can scan.
pub(crate) fn builder(level: Level) -> Option<fn(Kind) -> &'static Scan> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        // Strongest first: each accessor answers for its own level and every level above it.
        if level.as_avx512().is_some() {
            return Some(avx512::build);
        }
        if level.as_avx2().is_some() {
            return Some(avx2::build);
        }
        if level.as_sse4_2().is_some() {
            return Some(sse4_2::build);
        }
        if level.as_sse2().is_some() {
            return Some(sse2::build);
        }
    }
    #[cfg(target_arch = "aarch64")]
    if level.as_neon().is_some() {
        return Some(neon::build);
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if level.as_wasm_simd128().is_some() {
        return Some(wasm_simd128::build);
    }
    let _ = level;
    None
}
