//! Timing harness for the word-at-a-time backend: throughput per byte set, and the
//! latency of short haystacks and early matches.

use memchr_n::{Backend, Bytes, Finder};
use std::hint::black_box;
use std::time::Instant;

const HAYSTACK: &[u8] = include_bytes!("../benches/haystacks/sherlock/huge.txt");

fn finder(needles: &[u8]) -> Finder {
    Bytes::from_bytes(needles).finder_with(Backend::Scalar)
}

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

fn row(name: &str, secs: f64, bytes: usize) {
    let gbs = bytes as f64 / secs / 1e9;
    println!("{name:28} {:10.4} us {gbs:6.2} GB/s", secs * 1e6);
}

fn main() {
    let rounds = 20;
    let cases: &[(&str, &[u8])] = &[
        ("never1 <", b"<"),
        ("rare1 z", b"z"),
        ("rare3 zRJ", b"zRJ"),
        ("uncommon1 b", b"b"),
        ("common1 a", b"a"),
        ("common3 ato", b"ato"),
        ("verycommon1 sp", b" "),
    ];

    println!("== iterate (sum of offsets)");
    for &(name, needles) in cases {
        let f = finder(needles);
        let ours = best(rounds, 5, || {
            let mut sum = 0usize;
            for offset in f.iter(black_box(HAYSTACK)) {
                sum = sum.wrapping_add(offset);
            }
            sum
        });
        row(name, ours, HAYSTACK.len());
    }

    println!("== count");
    for &(name, needles) in cases {
        let f = finder(needles);
        let ours = best(rounds, 10, || f.iter(black_box(HAYSTACK)).count());
        row(name, ours, HAYSTACK.len());
    }

    println!("== find first (whole haystack)");
    for &(name, needles) in cases {
        let f = finder(needles);
        let ours = best(rounds, 100, || f.iter(black_box(HAYSTACK)).next());
        row(name, ours, HAYSTACK.len());
    }

    println!("== find first in a short prefix (latency, ns)");
    for len in [4usize, 8, 16, 32, 64, 128, 1024] {
        let f = finder(b"z");
        let hay = &HAYSTACK[..len];
        let t = best(rounds, 2000, || f.iter(black_box(hay)).next());
        println!("{:28} {:10.4} ns", format!("len {len}"), t * 1e9);
    }

    println!("== full scan of one chunk plus a tail, no match (ns)");
    for len in [64usize, 65, 72, 88, 96, 120, 127] {
        let hay = vec![b'.'; len];
        let f = finder(b"z");
        let t = best(rounds, 2000, || f.iter(black_box(&hay[..])).next());
        println!("{:28} {:10.4} ns", format!("len {len}"), t * 1e9);
    }

    println!("== find first at a known early offset (ns)");
    for off in [0usize, 3, 7, 9, 40, 70, 200] {
        let mut hay = vec![b'.'; 4096];
        hay[off] = b'z';
        let f = finder(b"z");
        let t = best(rounds, 2000, || f.iter(black_box(&hay[..])).next());
        println!("{:28} {:10.4} ns", format!("match at {off}"), t * 1e9);
    }
}
