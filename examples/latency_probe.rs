//! First-match latency, split into the two things that make it up.
//!
//! The offset sweep varies how far the scan has to look; the length sweep pins the match at
//! offset 0 and varies the haystack length, so each row is the shortest path through
//! `find_next` that a haystack of that length can take. Together they separate the fixed
//! per-call cost from the work the scan does before it can answer.

mod timing;

use memchr_n::MemchrN;
use std::hint::black_box;
use timing::best;

const ROUNDS: u32 = 200;
const ITERS: u32 = 400;

fn main() {
    let one = MemchrN::new(b"x");
    let three = MemchrN::new(b"xyz");

    println!("== match offset swept, 1MiB haystack");
    println!(
        "{:>7} {:>10} {:>10} {:>10} {:>10}",
        "offset", "ours1", "memchr1", "ours3", "memchr3"
    );
    let offsets = [
        0usize, 1, 8, 15, 16, 17, 32, 63, 64, 65, 127, 128, 129, 192, 255, 256, 511, 1023,
    ];
    let mut hay = vec![b'.'; 1 << 20];
    for offset in offsets {
        hay[offset] = b'x';
        println!(
            "{offset:>7} {:>9.2}n {:>9.2}n {:>9.2}n {:>9.2}n",
            best(ROUNDS, ITERS, || one.find(black_box(&hay))) * 1e9,
            best(ROUNDS, ITERS, || memchr::memchr(b'x', black_box(&hay))) * 1e9,
            best(ROUNDS, ITERS, || three.find(black_box(&hay))) * 1e9,
            best(ROUNDS, ITERS, || memchr::memchr3(
                b'x',
                b'y',
                b'z',
                black_box(&hay)
            )) * 1e9,
        );
        hay[offset] = b'.';
    }

    println!("== match at offset 0, haystack length swept");
    println!(
        "{:>8} {:>10} {:>10} {:>10}",
        "len", "ours1", "ours3", "memchr1"
    );
    for len in [1usize, 8, 15, 16, 32, 63, 64, 65, 127, 128, 129, 1024] {
        let mut hay = vec![b'.'; len];
        hay[0] = b'x';
        let hay = hay.as_slice();
        println!(
            "{len:>8} {:>9.2}n {:>9.2}n {:>9.2}n",
            best(ROUNDS, ITERS, || one.find(black_box(hay))) * 1e9,
            best(ROUNDS, ITERS, || three.find(black_box(hay))) * 1e9,
            best(ROUNDS, ITERS, || memchr::memchr(b'x', black_box(hay))) * 1e9,
        );
    }
}
