//! Times the sub-chunk tail path of the vector family in isolation.
//!
//! `anybyte` has no `memchr` counterpart: that crate tops out at three needles.

mod timing;

use memchr_n::{Backend, MemchrN};
use std::hint::black_box;
use timing::best;

const HAYSTACK: &[u8] = include_bytes!("../benches/haystacks/sherlock/huge.txt");

const LENS: [usize; 16] = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 63, 64, 65, 96, 127, 128];

const ROUNDS: u32 = 100;

fn row(name: &str, mut f: impl FnMut(&[u8]) -> Option<usize>) {
    print!("{name:14}");
    for len in LENS {
        let hay = &HAYSTACK[..len];
        assert!(f(hay).is_none());
        let t = best(ROUNDS, 5000, || f(black_box(hay)));
        print!(" {:6.2}", t * 1e9);
    }
    println!();
}

fn main() {
    let anybyte: MemchrN = (0x80u8..=0xFF).step_by(6).collect();
    let onebyte = MemchrN::new_with(b"<", Backend::Auto);

    row("anybyte", |hay| anybyte.iter(hay).next());
    row("onebyte", |hay| onebyte.iter(hay).next());
    row("onebyte memchr", |hay| memchr::memchr(b'<', hay));

    print!("{:14}", "len");
    for len in LENS {
        print!(" {len:6}");
    }
    println!();
}
