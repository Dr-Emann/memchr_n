# memchr-n

Like [`memchr`](https://docs.rs/memchr), but for a set of bytes of any size.

`memchr-n` finds and iterates over the positions of every byte in a haystack that
belongs to a caller-specified set of byte values, using SIMD where available.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
