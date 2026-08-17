# memchr-n

Like [`memchr`](https://docs.rs/memchr), but for a set of bytes of any size.

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