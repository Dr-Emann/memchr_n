//! Times the sub-chunk tail path of the vector family in isolation.

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
    let lens = [
        1usize, 2, 4, 8, 16, 24, 32, 40, 48, 56, 63, 64, 65, 96, 127, 128,
    ];

    let anybyte: Bytes = (0x80u8..=0xFF).step_by(6).collect();
    let onebyte = Bytes::from_bytes(b"<");
    let cases: &[(&str, Finder)] = &[
        ("anybyte", anybyte.finder_with(Backend::Auto)),
        ("onebyte", onebyte.finder_with(Backend::Auto)),
    ];

    for (name, f) in cases {
        print!("{name:8}");
        for len in lens {
            let hay = &HAYSTACK[..len];
            assert!(f.iter(hay).next().is_none());
            let t = best(20, 5000, || f.iter(black_box(hay)).next());
            print!(" {:6.2}", t * 1e9);
        }
        println!();
    }
    print!("{:8}", "len");
    for len in lens {
        print!(" {len:6}");
    }
    println!();
}
