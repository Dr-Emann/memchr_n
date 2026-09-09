//! Measures [`memchr_n::Backend::Scalar`] tail latency around the eight-byte word width.
//!
//! `memchr` uses its widest available backend, so its rows are reference values only.

mod timing;

use memchr_n::{Backend, MemchrN};
use std::hint::black_box;
use timing::best;

const HAYSTACK: &[u8] = include_bytes!("../benches/haystacks/sherlock/huge.txt");

const LENS: [usize; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 24, 63, 64, 65];

const ROUNDS: u32 = 100;

fn row(name: &str, mut f: impl FnMut(&[u8]) -> Option<usize>) {
    print!("{name:16}");
    for len in LENS {
        let hay = &HAYSTACK[..len];
        assert!(f(hay).is_none());
        let t = best(ROUNDS, 5000, || f(black_box(hay)));
        print!(" {:6.2}", t * 1e9);
    }
    println!();
}

fn main() {
    let onebyte = MemchrN::new_with(b"<", Backend::Scalar);
    let threebyte = MemchrN::new_with(b"<>@", Backend::Scalar);

    row("onebyte", |hay| onebyte.iter(hay).next());
    row("onebyte memchr", |hay| memchr::memchr(b'<', hay));
    row("threebyte", |hay| threebyte.iter(hay).next());
    row("threebyte memchr", |hay| {
        memchr::memchr3(b'<', b'>', b'@', hay)
    });

    print!("{:16}", "len");
    for len in LENS {
        print!(" {len:6}");
    }
    println!();
}
