#![deny(clippy::inline_always)]

//! Searching one byte at a time.
//!
//! This is where a byte set lands when testing it is a table probe and nothing else. Widening
//! the kernel cannot help: with no shuffle to run the lookup on, there is no way to answer for
//! more than one byte per probe, so [`crate::swar`]'s arithmetic has nothing to offer here and
//! its kernels' `u64` currency would only mean taking a word apart and putting it back together
//! around each probe.
//!
//! What the wide families do buy, and what this module keeps, is the shape of the scan around
//! the kernel: marks accumulated one byte apart so the probes stay branchless and vectorizable,
//! and packed down to one bit per byte only for a chunk that matched.

pub(crate) mod kernels;

use crate::swar::{WORD_BYTES, movemask};
use crate::{IterState, KernelData, MatchedBitset, Scan};

const CHUNK: usize = 32;

/// Accumulators one chunk's marks are spread across.
const WORDS: usize = CHUNK / WORD_BYTES;

const _: () = assert!(CHUNK <= u64::BITS as usize);
const _: () = assert!(CHUNK.is_multiple_of(WORD_BYTES));

/// Tests one byte at a time, the narrowest counterpart of [`crate::vector::Kernel`].
pub(crate) trait Kernel: Copy {
    /// Reads this kernel out of the field of `data` that holds it.
    ///
    /// # Safety
    ///
    /// As in [`crate::vector::Kernel::from_data`].
    unsafe fn from_data(data: &KernelData) -> Self;

    fn matches(&self, byte: u8) -> bool;
}

/// Marks each matching byte of a word at bit 7 of its own lane.
///
/// Spreading the marks a byte apart rather than packing them a bit apart is what makes this
/// vectorizable: every lane is independent, so the probes become vector work instead of a
/// serial chain of shifts into one register. Packing is deferred to [`movemask`], which only
/// the chunks that matched go on to pay for.
///
/// The bytes are shifted out of a `u64` rather than read from a `[u8; WORD_BYTES]`, which
/// measures faster: one load feeds all eight probes.
#[inline]
fn word_marks<K: Kernel>(kernel: &K, word: u64) -> u64 {
    let mut marks = 0;
    for i in 0..WORD_BYTES {
        marks |= u64::from(kernel.matches((word >> (i * 8)) as u8)) << (i * 8 + 7);
    }
    marks
}

/// Marks every matching byte of a chunk, and the union of those marks.
///
/// The union lets [`find_next`] skip [`pack_marks`] for a chunk that did not match at all, the
/// same gate `swar` puts in front of its own packing pass.
#[inline]
fn chunk_marks<K: Kernel>(kernel: &K, chunk: &[u8; CHUNK]) -> ([u64; WORDS], u64) {
    let (words, []) = chunk.as_chunks::<WORD_BYTES>() else {
        unreachable!()
    };
    let mut marks = [0; WORDS];
    let mut any = 0;
    for (i, word) in words.iter().enumerate() {
        marks[i] = word_marks(kernel, u64::from_le_bytes(*word));
        any |= marks[i];
    }
    (marks, any)
}

#[inline]
fn pack_marks(marks: [u64; WORDS]) -> u64 {
    let mut bits = 0;
    for (i, marks) in marks.into_iter().enumerate() {
        bits |= movemask(marks) << (i * WORD_BYTES);
    }
    bits
}

/// Matches `tail`, a run shorter than one [`CHUNK`], returning its bits at positions
/// `0..tail.len()`.
///
/// Whole words go through [`word_marks`] rather than straight into their final bit positions,
/// for the same reason a chunk does: the packed layout serialises the probes. A haystack
/// shorter than one chunk is *all* tail, so this is the whole cost of a short search.
///
/// What is not needed is either wider family's trick for the bytes past the last whole word.
/// There is no padding to mask off and no already-scanned byte to shift away, because a
/// per-byte scan reads exactly the bytes it reports on.
#[inline]
fn tail_bits<K: Kernel>(kernel: &K, tail: &[u8]) -> u64 {
    debug_assert!(tail.len() < CHUNK);
    let (words, rest) = tail.as_chunks::<WORD_BYTES>();

    let mut bits = 0;
    // The bound is a constant a tail can never reach, which is what lets the loop unroll into
    // shifts by constants instead of a variable-length chain.
    for (i, word) in words.iter().enumerate().take(WORDS) {
        bits |= movemask(word_marks(kernel, u64::from_le_bytes(*word))) << (i * WORD_BYTES);
    }
    let packed = words.len() * WORD_BYTES;
    for (i, &byte) in rest.iter().enumerate() {
        bits |= u64::from(kernel.matches(byte)) << (packed + i);
    }
    bits
}

/// Counts every matching byte of `haystack`.
#[inline]
pub(crate) fn count<K: Kernel>(haystack: &[u8], kernel: K) -> usize {
    let (words, tail) = haystack.as_chunks::<WORD_BYTES>();

    let mut total = 0;
    // One `count_ones` per eight probes, rather than one add per probe serialised on `total`.
    for word in words {
        total += word_marks(&kernel, u64::from_le_bytes(*word)).count_ones() as usize;
    }
    for &byte in tail {
        total += usize::from(kernel.matches(byte));
    }
    total
}

/// Returns the offset of the first matching byte of `haystack`.
///
/// The counterpart of [`crate::vector::find_first`]. A chunk's marks stay spread a byte
/// apart, so the first match is read straight off the accumulator that holds it: the union
/// already says which word to look in, and within a word a mark is at bit 7 of its own byte.
/// [`pack_marks`] never runs.
#[inline]
pub(crate) fn find_first<K: Kernel>(haystack: &[u8], kernel: K) -> Option<usize> {
    /// The offset within a chunk of its first marked byte.
    #[inline]
    fn first_mark(marks: [u64; WORDS]) -> usize {
        for (i, word) in marks.into_iter().enumerate() {
            if word != 0 {
                return i * WORD_BYTES + word.trailing_zeros() as usize / 8;
            }
        }
        unreachable!("only called for a chunk whose union is non-zero")
    }

    let (chunks, tail) = haystack.as_chunks::<CHUNK>();
    for (i, chunk) in chunks.iter().enumerate() {
        let (marks, any) = chunk_marks(&kernel, chunk);
        if any != 0 {
            return Some(i * CHUNK + first_mark(marks));
        }
    }
    let bits = tail_bits(&kernel, tail);
    (bits != 0).then(|| haystack.len() - tail.len() + bits.trailing_zeros() as usize)
}

/// Scans from `from` for the first [`CHUNK`] that contains a match.
#[inline]
pub(crate) fn find_next<K: Kernel>(state: &mut IterState<'_>, kernel: K) -> MatchedBitset {
    let (haystack, from) = (state.haystack, state.pos);
    // SAFETY: as in `crate::vector::find_next`.
    let unscanned = unsafe { haystack.get_unchecked(from..) };
    let (chunks, tail) = unscanned.as_chunks::<CHUNK>();

    let mut chunks = chunks.iter().fuse().enumerate();
    // As in `crate::swar::find_next`: the first chunk packs unconditionally, because a dense
    // haystack refills one chunk at a time from `Iter::next` and nearly always matches right
    // here. Only a sparse haystack reaches the gated loop below, where skipping `movemask`
    // across a long run of misses repays the pass wasted here.
    if let Some((_i, chunk)) = chunks.next() {
        let (marks, any) = chunk_marks(&kernel, chunk);
        if any != 0 {
            state.bits_offset = from;
            state.pos = from + CHUNK;
            return pack_marks(marks).into();
        }
    }

    // `from` stays at the start of the run: the enumeration was not restarted after the
    // chunk above, so `i` already counts it.
    for (i, chunk) in chunks {
        let (marks, any) = chunk_marks(&kernel, chunk);
        if any != 0 {
            let offset = from + i * CHUNK;
            state.bits_offset = offset;
            state.pos = offset + CHUNK;
            return pack_marks(marks).into();
        }
    }

    state.bits_offset = haystack.len() - tail.len();
    state.pos = haystack.len();
    tail_bits(&kernel, tail).into()
}

/// The [`Scan`] whose entry points run `K`.
pub(crate) fn scan<K: Kernel>() -> Scan {
    unsafe fn find_next<K: Kernel>(
        data: &KernelData,
        state: &mut IterState<'_>,
    ) -> MatchedBitset {
        // SAFETY: as in [`crate::swar::scan`]: the `Scan` below stores this function only
        // for the kind whose `KernelData` field `K` reads.
        let kernel = unsafe { K::from_data(data) };
        self::find_next(state, kernel)
    }

    unsafe fn count_all<K: Kernel>(data: &KernelData, unscanned: &[u8]) -> usize {
        // SAFETY: as above.
        let kernel = unsafe { K::from_data(data) };
        self::count(unscanned, kernel)
    }

    unsafe fn find_first<K: Kernel>(
        data: &KernelData,
        haystack: &[u8],
    ) -> Option<usize> {
        // SAFETY: as above.
        let kernel = unsafe { K::from_data(data) };
        self::find_first(haystack, kernel)
    }

    Scan {
        find_next: find_next::<K>,
        count_all: count_all::<K>,
        find_first: find_first::<K>,
    }
}
