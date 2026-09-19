# memchr-n

Like [`memchr`], but for a set of bytes of any size.

`memchr-n` finds and iterates over the positions of every byte in a haystack that
belongs to any set of byte values, using heavily optimized SIMD where available.

## Overview

This section gives a brief high level overview of what this crate offers.

The [`MemchrN`] type is the entrypoint of the library. It offers a few means of construction:

- [`MemchrN::new`] creates a new `MemchrN` from a slice of bytes.
- [`MemchrN::from_not_byte`] creates a new `MemchrN` which matches every byte except the given byte.
- [`MemchrN::from_range`] creates a new `MemchrN` which will match any byte in the given range.
- [`MemchrN::from_iter`] (also usable as `collection.collect()`) creates a new `MemchrN` from an
  arbitrary iterator of bytes.
- [`MemchrN::from_byte_set`] creates a new `MemchrN` from a [`ByteSet`], which describes an arbitrary
  set of byte values.


A [`ByteSet`] is built up a byte, a range, or a slice at a time, and can be combined with the operators
(`|`, `&`, `^`, `-`, and `!`). This makes sets convenient to describe through unions, intersections, differences,
and complements.

All public inherent `ByteSet` methods are usable in a `const` context, so a set can be built once as a `const`.
In constant expressions, use the named methods such as `union`, `difference`, and `invert` instead of the overloaded
operators:

```rust
use memchr_n::{ByteSet, MemchrN};

const NOT_IDENTIFIER: ByteSet = {
    let mut set = ByteSet::from_bytes(b"_");
    set.add_range(b'0'..=b'9');
    set.add_range(b'a'..=b'z');
    set.add_range(b'A'..=b'Z');
    set.invert();
    set
};

let finder = MemchrN::from_byte_set(NOT_IDENTIFIER);
assert_eq!(finder.find(b"some_name(x)"), Some(9));

let mut hex = ByteSet::new();
hex.add_range(b'0'..=b'9');
hex.add_range(b'a'..=b'f');
let mut lower = ByteSet::new();
lower.add_range(b'a'..=b'z');

let finder = MemchrN::from_byte_set(lower - hex);
assert_eq!(finder.find(b"deadbeef zoo"), Some(9));
```

Creating a [`MemchrN`] selects and prepares a specialized search strategy based on the bytes to be matched and
available CPU features. Construction can be relatively expensive compared with an individual search, so reuse the
searcher when possible. For a fixed byte set used throughout a program, consider storing it globally in a
`LazyLock` or `OnceLock`.

There are two main things you can do with a [`MemchrN`]:

- [`MemchrN::find`] finds the first position of a byte in a haystack that belongs to the set.
- [`MemchrN::iter`] returns an iterator which iterates over the positions of every byte in a haystack that belongs
  to the set.

The [`Iter`] returned by [`MemchrN::iter`] can also be skipped forward with [`Iter::advance_to`], which discards
the matches before a byte offset. This is cheaper than calling `next` until the offset is reached: if the offset
falls within the batch of matches already found, it simply drops those matches, and otherwise it restarts the
search at that offset without examining the bytes in between.

## Performance

[`MemchrN`] is optimized for repeated searches with the same set of bytes, iterating over the matching positions,
and counting matches. The set of bytes is fixed at construction time, at which point a specialized search strategy
is selected, so reuse a constructed [`MemchrN`] instance to amortize that cost.

### Iterating Over Matches

When iterating over matches, the results of each SIMD batch are retained in a bitmask. Each step of the iteration
removes the first set bit. Only when the batch is exhausted does the next batch of SIMD operations run again.

[`memchr`]'s iterators, on the other hand, find a SIMD batch in a similar way, but once the first match is found,
the next iteration step must begin searching again starting from the next byte after the first match, even if the
first batch actually already had located that match. Avoiding these repeated searches can make `memchr-n` significantly
faster when matches are densely populated.

Both libraries compare multiple vectors per search-loop iteration. `memchr-n` processes up to 128 bytes per batch
and retains all matching positions from that batch. This lets it amortize the batch's work across multiple results.

### Counting Matches

Calling `finder.iter(haystack).count()` uses a dedicated counting implementation instead of iterating over the matches.
It accumulates into byte lane SIMD registers, only reducing to a single count when necessary.

[`memchr`] also has a specialized counting implementation, but only for a single needle, but its SIMD implementation
reduces the matching positions to a bitset and counts them each step. By keeping the counts in SIMD registers and
avoiding reduction for every step, `memchr-n` can be quite a bit faster than [`memchr`] for counting matches.
Because [`memchr`] does not have a specialized implementation for counting with two or three needles, `memchr-n` can
be much faster than [`memchr`] for counting matches of multiple needles.

### Finding the First Match

Use `MemchrN::find` when only the first match is needed. It has a separate implementation that checks an initial chunk
before entering the larger batch loop and uses smaller searches for short haystacks.

Batching favors throughput, but can do extra work before returning an early match. For very early matches or small
haystacks, `memchr-n` may be slower than [`memchr`]. Relative performance depends on the CPU, byte set, haystack
length, match density, and operation being performed.

## Minimum supported Rust version

This crate requires Rust 1.89 or later.

[`memchr`]: https://docs.rs/memchr
