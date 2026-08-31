//! Times the sub-word tail path of the word-at-a-time family in isolation.
//!
//! The counterpart of `vtail_probe` for [`memchr_n::Backend::Scalar`]. Lengths below eight
//! take the staged path, and eight and above re-read the last whole word.
//!
//! The `memchr` rows are a reference point only: that crate always uses the widest
//! backend the CPU has, so it is not scanning these tails the same way.

use memchr_n::{Backend, Bytes};
use std::hint::black_box;
use std::time::Instant;

const HAYSTACK: &[u8] = include_bytes!("../benches/haystacks/sherlock/huge.txt");

const LENS: [usize; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 24, 63, 64, 65];

const ROUNDS: u32 = 100;

fn best<T>(rounds: u32, iters: u32, mut f: impl FnMut() -> T) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..rounds {
        let start = Instant::now();
        for _ in 0..iters {
            black_box(f());
        }
        let elapsed = start.elapsed().as_secs_f64() / f64::from(iters);
        best = best.min(elapsed);
    }
    best
}

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
    let onebyte = Bytes::from_bytes(b"<").finder_with(Backend::Scalar);
    let threebyte = Bytes::from_bytes(b"<>@").finder_with(Backend::Scalar);

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
