//! Compares scalar-backend throughput and latency with `memchr`'s word-at-a-time searchers.

mod timing;

use memchr::arch::all::memchr::{One, Three, Two};
use memchr_n::{Backend, MemchrN};
use std::hint::black_box;
use timing::best;

const HAYSTACK: &[u8] = include_bytes!("../benches/haystacks/sherlock/huge.txt");

const ROUNDS: u32 = 100;

fn finder(needles: &[u8]) -> MemchrN {
    MemchrN::new_with_backend(needles, Backend::Swar)
}

fn memchr_sum(needles: &[u8], iters: u32) -> Option<f64> {
    fn sum(it: impl Iterator<Item = usize>) -> usize {
        it.fold(0usize, |acc, offset| acc.wrapping_add(offset))
    }
    match needles {
        [a] => {
            let s = One::new(*a);
            Some(best(ROUNDS, iters, || sum(s.iter(black_box(HAYSTACK)))))
        }
        [a, b] => {
            let s = Two::new(*a, *b);
            Some(best(ROUNDS, iters, || sum(s.iter(black_box(HAYSTACK)))))
        }
        [a, b, c] => {
            let s = Three::new(*a, *b, *c);
            Some(best(ROUNDS, iters, || sum(s.iter(black_box(HAYSTACK)))))
        }
        _ => None,
    }
}

fn memchr_count(needles: &[u8], iters: u32) -> Option<f64> {
    match needles {
        [a] => {
            let s = One::new(*a);
            Some(best(ROUNDS, iters, || s.count(black_box(HAYSTACK))))
        }
        [a, b] => {
            let s = Two::new(*a, *b);
            Some(best(ROUNDS, iters, || s.iter(black_box(HAYSTACK)).count()))
        }
        [a, b, c] => {
            let s = Three::new(*a, *b, *c);
            Some(best(ROUNDS, iters, || s.iter(black_box(HAYSTACK)).count()))
        }
        _ => None,
    }
}

fn memchr_find(needles: &[u8], iters: u32) -> Option<f64> {
    match needles {
        [a] => {
            let s = One::new(*a);
            Some(best(ROUNDS, iters, || s.find(black_box(HAYSTACK))))
        }
        [a, b] => {
            let s = Two::new(*a, *b);
            Some(best(ROUNDS, iters, || s.find(black_box(HAYSTACK))))
        }
        [a, b, c] => {
            let s = Three::new(*a, *b, *c);
            Some(best(ROUNDS, iters, || s.find(black_box(HAYSTACK))))
        }
        _ => None,
    }
}

fn row(name: &str, ours: f64, theirs: Option<f64>, bytes: usize) {
    let gbs = |t: f64| bytes as f64 / t / 1e9;
    let cmp = match theirs {
        Some(t) => format!(
            "memchr {:10.4} us {:6.2} GB/s  {:5.2}x",
            t * 1e6,
            gbs(t),
            t / ours
        ),
        None => String::from("memchr          -"),
    };
    println!(
        "{name:28} {:10.4} us {:6.2} GB/s   {cmp}",
        ours * 1e6,
        gbs(ours)
    );
}

fn find_row(name: &str, ours: f64, theirs: Option<f64>, scanned: usize) {
    let cmp = match theirs {
        Some(t) => format!("memchr {:10.4} ns  {:5.2}x", t * 1e9, t / ours),
        None => String::from("memchr          -         "),
    };
    println!(
        "{name:28} {:10.4} ns   {cmp}   scanned {scanned:7}",
        ours * 1e9
    );
}

fn ns_row(name: &str, ours: f64, theirs: f64) {
    println!(
        "{name:28} {:10.4} ns   memchr {:10.4} ns  {:5.2}x",
        ours * 1e9,
        theirs * 1e9,
        theirs / ours,
    );
}

fn main() {
    let cases: &[(&str, &[u8])] = &[
        ("never1 <", b"<"),
        ("rare1 z", b"z"),
        ("rare3 zRJ", b"zRJ"),
        ("uncommon1 b", b"b"),
        ("common1 a", b"a"),
        ("common3 ato", b"ato"),
        ("verycommon1 sp", b" "),
        ("never4 <>{}", b"<>{}"),
        ("rare4 zQXJ", b"zQXJ"),
        ("uncommon4 qjkx", b"qjkx"),
        ("common4 atob", b"atob"),
        ("verycommon5 sp+aeto", b" aeto"),
    ];

    println!("== iterate (sum of offsets)");
    for &(name, needles) in cases {
        let f = finder(needles);
        let ours = best(ROUNDS, 5, || {
            let mut sum = 0usize;
            for offset in f.iter(black_box(HAYSTACK)) {
                sum = sum.wrapping_add(offset);
            }
            sum
        });
        row(name, ours, memchr_sum(needles, 5), HAYSTACK.len());
    }

    println!("== count");
    for &(name, needles) in cases {
        let f = finder(needles);
        let ours = best(ROUNDS, 10, || f.iter(black_box(HAYSTACK)).count());
        row(name, ours, memchr_count(needles, 10), HAYSTACK.len());
    }

    println!("== find first in the full haystack (latency, ns)");
    for &(name, needles) in cases {
        let f = finder(needles);
        let ours = best(ROUNDS, 100, || f.iter(black_box(HAYSTACK)).next());
        let scanned = f.find(HAYSTACK).map_or(HAYSTACK.len(), |offset| offset + 1);
        find_row(name, ours, memchr_find(needles, 100), scanned);
    }

    let theirs_z = One::new(b'z');

    println!("== find first in a short prefix (latency, ns)");
    for len in [4usize, 8, 16, 32, 64, 128, 1024] {
        let f = finder(b"z");
        let haystack = &HAYSTACK[..len];
        let ours = best(ROUNDS, 2000, || f.iter(black_box(haystack)).next());
        let theirs = best(ROUNDS, 2000, || theirs_z.find(black_box(haystack)));
        ns_row(&format!("len {len}"), ours, theirs);
    }

    println!("== full scan of one chunk plus a tail, no match (ns)");
    for len in [64usize, 65, 72, 88, 96, 120, 127] {
        let haystack = vec![b'.'; len];
        let f = finder(b"z");
        let ours = best(ROUNDS, 2000, || f.iter(black_box(&haystack[..])).next());
        let theirs = best(ROUNDS, 2000, || theirs_z.find(black_box(&haystack[..])));
        ns_row(&format!("len {len}"), ours, theirs);
    }

    println!("== anybyte full scan, no match, by length (ns)");
    for len in [4usize, 8, 16, 32, 64, 65, 72, 88, 96, 127, 128, 1024] {
        let f = finder(b"<>{}");
        let haystack = &HAYSTACK[..len];
        let ours = best(ROUNDS, 2000, || f.iter(black_box(haystack)).next());
        find_row(&format!("len {len}"), ours, None, len);
    }

    println!("== find first at a known early offset (ns)");
    for off in [0usize, 3, 7, 9, 40, 70, 200] {
        let mut haystack = vec![b'.'; 4096];
        haystack[off] = b'z';
        let f = finder(b"z");
        let ours = best(ROUNDS, 2000, || f.iter(black_box(&haystack[..])).next());
        let theirs = best(ROUNDS, 2000, || theirs_z.find(black_box(&haystack[..])));
        ns_row(&format!("match at {off}"), ours, theirs);
    }
}
