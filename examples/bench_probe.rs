//! Low-noise timing harness for the public API, used to compare against `memchr`
//! without criterion's sampling in the way. Reports the best of several rounds, which
//! is far more stable run-to-run than criterion's `--quick` estimates.

use memchr_n::{Bytes, Finder};
use std::hint::black_box;
use std::time::Instant;

const HAYSTACK: &[u8] = include_bytes!("../benches/haystacks/sherlock/huge.txt");

fn finder(needles: &[u8]) -> Finder {
    Bytes::from_bytes(needles).finder()
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

fn row(name: &str, ours: f64, theirs: Option<f64>) {
    let gbs = HAYSTACK.len() as f64 / ours / 1e9;
    let cmp = match theirs {
        Some(t) => format!("{:9.3} us  {:5.2}x", t * 1e6, t / ours),
        None => String::from("        -         "),
    };
    println!("{name:28} {:9.3} us {gbs:6.1} GB/s   {cmp}", ours * 1e6);
}

fn offset_sum(finder: &Finder, haystack: &[u8]) -> usize {
    let mut sum = 0usize;
    for offset in finder.iter(haystack) {
        sum = sum.wrapping_add(offset);
    }
    sum
}

fn main() {
    let rounds = 20;
    let cases: &[(&str, &[u8])] = &[
        ("never1 <", b"<"),
        ("rare1 z", b"z"),
        ("rare2 zR", b"zR"),
        ("rare3 zRJ", b"zRJ"),
        ("uncommon1 b", b"b"),
        ("uncommon3 bp.", b"bp."),
        ("common1 a", b"a"),
        ("common3 ato", b"ato"),
        ("verycommon1 sp", b" "),
    ];

    println!("== count");
    for &(name, needles) in cases {
        let f = finder(needles);
        let ours = best(rounds, 30, || f.iter(black_box(HAYSTACK)).count());
        let theirs = match needles {
            [n1] => Some(best(rounds, 30, || {
                memchr::memchr_iter(*n1, black_box(HAYSTACK)).count()
            })),
            [n1, n2] => Some(best(rounds, 30, || {
                memchr::memchr2_iter(*n1, *n2, black_box(HAYSTACK)).count()
            })),
            [n1, n2, n3] => Some(best(rounds, 10, || {
                memchr::memchr3_iter(*n1, *n2, *n3, black_box(HAYSTACK)).count()
            })),
            _ => None,
        };
        row(name, ours, theirs);
    }

    println!("== iterate (sum of offsets)");
    for &(name, needles) in cases {
        let f = finder(needles);
        let ours = best(rounds, 10, || offset_sum(&f, black_box(HAYSTACK)));
        let theirs = match needles {
            [n1] => Some(best(rounds, 10, || {
                let mut sum = 0usize;
                for offset in memchr::memchr_iter(*n1, black_box(HAYSTACK)) {
                    sum = sum.wrapping_add(offset);
                }
                sum
            })),
            [n1, n2] => Some(best(rounds, 10, || {
                let mut sum = 0usize;
                for offset in memchr::memchr2_iter(*n1, *n2, black_box(HAYSTACK)) {
                    sum = sum.wrapping_add(offset);
                }
                sum
            })),
            [n1, n2, n3] => Some(best(rounds, 5, || {
                let mut sum = 0usize;
                for offset in memchr::memchr3_iter(*n1, *n2, *n3, black_box(HAYSTACK)) {
                    sum = sum.wrapping_add(offset);
                }
                sum
            })),
            _ => None,
        };
        row(name, ours, theirs);
    }

    println!("== find first");
    for &(name, needles) in cases {
        let f = finder(needles);
        let ours = best(rounds, 200, || f.iter(black_box(HAYSTACK)).next());
        let theirs = match needles {
            [n1] => Some(best(rounds, 200, || {
                memchr::memchr(*n1, black_box(HAYSTACK))
            })),
            [n1, n2] => Some(best(rounds, 200, || {
                memchr::memchr2(*n1, *n2, black_box(HAYSTACK))
            })),
            [n1, n2, n3] => Some(best(rounds, 200, || {
                memchr::memchr3(*n1, *n2, *n3, black_box(HAYSTACK))
            })),
            _ => None,
        };
        row(name, ours, theirs);
    }

    println!("== match counts");
    for &(name, needles) in cases {
        let f = finder(needles);
        println!("{name:28} {}", f.iter(HAYSTACK).count());
    }
}
