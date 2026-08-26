//! Times the sub-word tail path of the word-at-a-time family in isolation.
//!
//! The counterpart of `vtail_probe` for [`memchr_n::Backend::Scalar`]. Lengths below eight
//! take the staged path, and eight and above re-read the last whole word.

use memchr_n::{Backend, Bytes, Finder};
use std::hint::black_box;
use std::time::Instant;

const HAYSTACK: &[u8] = include_bytes!("../benches/haystacks/sherlock/huge.txt");

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

fn main() {
    let lens = [1usize, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 24, 63, 64, 65];

    let onebyte = Bytes::from_bytes(b"<");
    let threebyte = Bytes::from_bytes(b"<>@");
    let cases: &[(&str, Finder)] = &[
        ("onebyte", onebyte.finder_with(Backend::Scalar)),
        ("threebyte", threebyte.finder_with(Backend::Scalar)),
    ];

    for (name, f) in cases {
        print!("{name:10}");
        for len in lens {
            let hay = &HAYSTACK[..len];
            assert!(f.iter(hay).next().is_none());
            let t = best(20, 5000, || f.iter(black_box(hay)).next());
            print!(" {:6.2}", t * 1e9);
        }
        println!();
    }
    print!("{:10}", "len");
    for len in lens {
        print!(" {len:6}");
    }
    println!();
}
