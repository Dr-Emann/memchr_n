# memchr-n

Like [`memchr`], but for a set of bytes of any size.

`memchr-n` finds and iterates over the positions of every byte in a haystack that
belongs to any set of byte values, using heavily optimized SIMD where available.

## Overview

This section gives a brief high level overview of what this crate offers.

The [`MemchrN`] type is the entrypoint of the library. It offers a few means of construction:

- [`MemchrN::new`] creates a new `MemchrN` from a slice of bytes.
- [`MemchrN::from_range`] creates a new `MemchrN` which will match any byte in the given range.
- [`MemchrN::from_iter`] (also usable as `collection.collect()`) creates a new `MemchrN` from an
  arbitrary iterator of bytes.

Creating a [`MemchrN`] does the work of identifying the optimal way to search for that set of bytes:
constructing a [`MemchrN`] is somewhat expensive. Once a [`MemchrN`] is created, it should be reused,
it may be useful to store it in a `OnceLock` or the like globally if you have a known set of bytes to search for.

There are two main things you can do with a [`MemchrN`]:

- [`MemchrN::find`] finds the first position of a byte in a haystack that belongs to the set.
- [`MemchrN::iter`] returns an iterator which iterates over the positions of every byte in a haystack that belongs
  to the set.

## Performance

`memchr-n` manages to generally be faster than [`memchr`] for the same number of needles, despite supporting any
number of needles to search for. This is largely because [`memchr`] uses simd to find the next match, using simd
to check many bytes at once, but on finding a match, it returns the first match, and the next match is found by
starting the search over from the next byte after the first match. `memchr-n` instead keeps the result of the simd
search, and iterates directly over the positions of the matches already found, only beginning the next search once
all known matches found via simd have been iterated over.

This means `memchr-n` can be much faster when iterating over the matches when matches are dense. However, because
`memchr-n` has less of a trade off for sparse matches, it is able to go wider than [`memchr`], and check more bytes
simultaneously with simd, since it's not throwing away that information on the first match: [`memchr`] has to toe
the line between using wide simd to quickly skip ranges of non-matching bytes, without wasting too much work in the
case a match is found, but `memchr-n` does not have the same trade-off.

However, because `memchr-n` does use wider simd, there is a trade-off in latency to first match. This is somewhat
mitigated by the `MemchrN::find` method, but for very early matches, or very small haystacks, `memchr-n` may still
be slower than [`memchr`].

[`memchr`]: https://docs.rs/memchr