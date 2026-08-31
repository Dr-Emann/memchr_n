#![allow(unsafe_op_in_unsafe_fn)]

//! Standalone timing probe for candidate NEON scanning kernels.
//!
//! Not a criterion benchmark: each variant is a hand-written loop over the same
//! haystack so the per-chunk instruction mix can be compared without the
//! iterator/dispatch machinery in between.
//!
//! Each variant is reported as the best of [`ROUNDS`] rounds, which is far steadier
//! run-to-run than a single averaged pass.

use std::arch::aarch64::*;
use std::hint::black_box;
use std::time::Instant;

const HAYSTACK: &[u8] = include_bytes!("../benches/haystacks/sherlock/huge.txt");

const ROUNDS: u32 = 100;

/// ld4-interleaved load, sri chain to an ordered 64-bit bitmask, popcount.
#[target_feature(enable = "neon")]
unsafe fn count_ld4_sri(needle: u8, haystack: &[u8]) -> usize {
    let n = vdupq_n_u8(needle);
    let mut count = 0usize;
    let mut p = haystack.as_ptr();
    let end = p.add(haystack.len() & !63);
    while p < end {
        let q = vld4q_u8(p);
        let e0 = vceqq_u8(q.0, n);
        let e1 = vceqq_u8(q.1, n);
        let e2 = vceqq_u8(q.2, n);
        let e3 = vceqq_u8(q.3, n);
        let t0 = vsriq_n_u8::<1>(e1, e0);
        let t1 = vsriq_n_u8::<1>(e3, e2);
        let t2 = vsriq_n_u8::<2>(t1, t0);
        let t3 = vsriq_n_u8::<4>(t2, t2);
        let bits = vget_lane_u64::<0>(vreinterpret_u64_u8(vshrn_n_u16::<4>(vreinterpretq_u16_u8(
            t3,
        ))));
        count += bits.count_ones() as usize;
        p = p.add(64);
    }
    count
}

/// Plain loads, per-lane accumulation via subtract, one horizontal add per block.
#[target_feature(enable = "neon")]
unsafe fn count_sub_acc(needle: u8, haystack: &[u8]) -> usize {
    let n = vdupq_n_u8(needle);
    let mut count = 0usize;
    let mut p = haystack.as_ptr();
    let end = p.add(haystack.len() & !63);
    // Each lane accumulates at most one per 16-byte vector, so flush before 255.
    while p < end {
        let block_end = end.min(p.add(64 * 63));
        let mut acc = vdupq_n_u8(0);
        while p < block_end {
            let a = vld1q_u8(p);
            let b = vld1q_u8(p.add(16));
            let c = vld1q_u8(p.add(32));
            let d = vld1q_u8(p.add(48));
            acc = vsubq_u8(acc, vceqq_u8(a, n));
            acc = vsubq_u8(acc, vceqq_u8(b, n));
            acc = vsubq_u8(acc, vceqq_u8(c, n));
            acc = vsubq_u8(acc, vceqq_u8(d, n));
            p = p.add(64);
        }
        count += vaddlvq_u8(acc) as usize;
    }
    count
}

/// Plain loads, four independent accumulators to break the dependency chain.
#[target_feature(enable = "neon")]
unsafe fn count_sub_acc4(needle: u8, haystack: &[u8]) -> usize {
    let n = vdupq_n_u8(needle);
    let mut count = 0usize;
    let mut p = haystack.as_ptr();
    let end = p.add(haystack.len() & !63);
    while p < end {
        let block_end = end.min(p.add(64 * 255));
        let mut a0 = vdupq_n_u8(0);
        let mut a1 = vdupq_n_u8(0);
        let mut a2 = vdupq_n_u8(0);
        let mut a3 = vdupq_n_u8(0);
        while p < block_end {
            let a = vld1q_u8(p);
            let b = vld1q_u8(p.add(16));
            let c = vld1q_u8(p.add(32));
            let d = vld1q_u8(p.add(48));
            a0 = vsubq_u8(a0, vceqq_u8(a, n));
            a1 = vsubq_u8(a1, vceqq_u8(b, n));
            a2 = vsubq_u8(a2, vceqq_u8(c, n));
            a3 = vsubq_u8(a3, vceqq_u8(d, n));
            p = p.add(64);
        }
        count += vaddlvq_u8(a0) as usize;
        count += vaddlvq_u8(a1) as usize;
        count += vaddlvq_u8(a2) as usize;
        count += vaddlvq_u8(a3) as usize;
    }
    count
}

/// memchr's approach: per-vector shrn movemask, mask, scalar popcount.
#[target_feature(enable = "neon")]
unsafe fn count_shrn(needle: u8, haystack: &[u8]) -> usize {
    #[inline(always)]
    unsafe fn movemask(v: uint8x16_t) -> u64 {
        let m = vshrn_n_u16::<4>(vreinterpretq_u16_u8(v));
        vget_lane_u64::<0>(vreinterpret_u64_u8(m)) & 0x8888_8888_8888_8888
    }
    let n = vdupq_n_u8(needle);
    let mut count = 0usize;
    let mut p = haystack.as_ptr();
    let end = p.add(haystack.len() & !63);
    while p < end {
        let a = vld1q_u8(p);
        let b = vld1q_u8(p.add(16));
        let c = vld1q_u8(p.add(32));
        let d = vld1q_u8(p.add(48));
        count += movemask(vceqq_u8(a, n)).count_ones() as usize;
        count += movemask(vceqq_u8(b, n)).count_ones() as usize;
        count += movemask(vceqq_u8(c, n)).count_ones() as usize;
        count += movemask(vceqq_u8(d, n)).count_ones() as usize;
        p = p.add(64);
    }
    count
}

/// ld4-interleaved load but sub-accumulate, isolating the cost of ld4 itself.
#[target_feature(enable = "neon")]
unsafe fn count_ld4_acc(needle: u8, haystack: &[u8]) -> usize {
    let n = vdupq_n_u8(needle);
    let mut count = 0usize;
    let mut p = haystack.as_ptr();
    let end = p.add(haystack.len() & !63);
    while p < end {
        let block_end = end.min(p.add(64 * 63));
        let mut acc = vdupq_n_u8(0);
        while p < block_end {
            let q = vld4q_u8(p);
            acc = vsubq_u8(acc, vceqq_u8(q.0, n));
            acc = vsubq_u8(acc, vceqq_u8(q.1, n));
            acc = vsubq_u8(acc, vceqq_u8(q.2, n));
            acc = vsubq_u8(acc, vceqq_u8(q.3, n));
            p = p.add(64);
        }
        count += vaddlvq_u8(acc) as usize;
    }
    count
}

/// Reorders four block-ordered mask vectors into the 4-way interleaved order that
/// the `sri` chain consumes, replacing `ld4`'s deinterleave with eight `uzp`s.
#[inline(always)]
unsafe fn interleave(
    a: uint8x16_t,
    b: uint8x16_t,
    c: uint8x16_t,
    d: uint8x16_t,
) -> [uint8x16_t; 4] {
    let e_ab = vuzp1q_u8(a, b);
    let o_ab = vuzp2q_u8(a, b);
    let e_cd = vuzp1q_u8(c, d);
    let o_cd = vuzp2q_u8(c, d);
    [
        vuzp1q_u8(e_ab, e_cd),
        vuzp1q_u8(o_ab, o_cd),
        vuzp2q_u8(e_ab, e_cd),
        vuzp2q_u8(o_ab, o_cd),
    ]
}

#[inline(always)]
unsafe fn sri_bitmask(m: [uint8x16_t; 4]) -> u64 {
    let t0 = vsriq_n_u8::<1>(m[1], m[0]);
    let t1 = vsriq_n_u8::<1>(m[3], m[2]);
    let t2 = vsriq_n_u8::<2>(t1, t0);
    let t3 = vsriq_n_u8::<4>(t2, t2);
    vget_lane_u64::<0>(vreinterpret_u64_u8(vshrn_n_u16::<4>(vreinterpretq_u16_u8(
        t3,
    ))))
}

/// Plain loads, `uzp` reorder, then the same `sri` bitmask chain.
#[target_feature(enable = "neon")]
unsafe fn count_uzp_sri(needle: u8, haystack: &[u8]) -> usize {
    let n = vdupq_n_u8(needle);
    let mut count = 0usize;
    let mut p = haystack.as_ptr();
    let end = p.add(haystack.len() & !63);
    while p < end {
        let a = vceqq_u8(vld1q_u8(p), n);
        let b = vceqq_u8(vld1q_u8(p.add(16)), n);
        let c = vceqq_u8(vld1q_u8(p.add(32)), n);
        let d = vceqq_u8(vld1q_u8(p.add(48)), n);
        count += sri_bitmask(interleave(a, b, c, d)).count_ones() as usize;
        p = p.add(64);
    }
    count
}

/// `uzp` reorder, but only on blocks that matched.
#[target_feature(enable = "neon")]
unsafe fn find_uzp_sri(needle: u8, haystack: &[u8]) -> Option<usize> {
    let n = vdupq_n_u8(needle);
    let mut p = haystack.as_ptr();
    let start = p;
    let end = p.add(haystack.len() & !63);
    while p < end {
        let a = vceqq_u8(vld1q_u8(p), n);
        let b = vceqq_u8(vld1q_u8(p.add(16)), n);
        let c = vceqq_u8(vld1q_u8(p.add(32)), n);
        let d = vceqq_u8(vld1q_u8(p.add(48)), n);
        let or = vorrq_u8(vorrq_u8(a, b), vorrq_u8(c, d));
        if vmaxvq_u32(vreinterpretq_u32_u8(or)) != 0 {
            let bits = sri_bitmask(interleave(a, b, c, d));
            return Some(p.offset_from(start) as usize + bits.trailing_zeros() as usize);
        }
        p = p.add(64);
    }
    None
}

/// Plain loads, or-reduce, and only extract a bitmask when the block matched.
#[target_feature(enable = "neon")]
unsafe fn find_or_reduce(needle: u8, haystack: &[u8]) -> Option<usize> {
    let n = vdupq_n_u8(needle);
    let mut p = haystack.as_ptr();
    let start = p;
    let end = p.add(haystack.len() & !63);
    while p < end {
        let a = vceqq_u8(vld1q_u8(p), n);
        let b = vceqq_u8(vld1q_u8(p.add(16)), n);
        let c = vceqq_u8(vld1q_u8(p.add(32)), n);
        let d = vceqq_u8(vld1q_u8(p.add(48)), n);
        let or = vorrq_u8(vorrq_u8(a, b), vorrq_u8(c, d));
        if vmaxvq_u32(vreinterpretq_u32_u8(or)) != 0 {
            for (i, v) in [a, b, c, d].into_iter().enumerate() {
                let m = vshrn_n_u16::<4>(vreinterpretq_u16_u8(v));
                let bits = vget_lane_u64::<0>(vreinterpret_u64_u8(m)) & 0x8888_8888_8888_8888;
                if bits != 0 {
                    return Some(
                        p.offset_from(start) as usize + i * 16 + bits.trailing_zeros() as usize / 4,
                    );
                }
            }
        }
        p = p.add(64);
    }
    None
}

/// ld4 load with a full bitmask on every block, as the current `Iter` does.
#[target_feature(enable = "neon")]
unsafe fn find_ld4(needle: u8, haystack: &[u8]) -> Option<usize> {
    let n = vdupq_n_u8(needle);
    let mut p = haystack.as_ptr();
    let start = p;
    let end = p.add(haystack.len() & !63);
    while p < end {
        let q = vld4q_u8(p);
        let e0 = vceqq_u8(q.0, n);
        let e1 = vceqq_u8(q.1, n);
        let e2 = vceqq_u8(q.2, n);
        let e3 = vceqq_u8(q.3, n);
        let t0 = vsriq_n_u8::<1>(e1, e0);
        let t1 = vsriq_n_u8::<1>(e3, e2);
        let t2 = vsriq_n_u8::<2>(t1, t0);
        let t3 = vsriq_n_u8::<4>(t2, t2);
        let bits = vget_lane_u64::<0>(vreinterpret_u64_u8(vshrn_n_u16::<4>(vreinterpretq_u16_u8(
            t3,
        ))));
        if bits != 0 {
            return Some(p.offset_from(start) as usize + bits.trailing_zeros() as usize);
        }
        p = p.add(64);
    }
    None
}

fn measure<T>(name: &str, iters: u32, expect: T, mut f: impl FnMut() -> T) -> f64
where
    T: PartialEq + std::fmt::Debug,
{
    let got = f();
    assert_eq!(got, expect, "{name} produced the wrong answer");
    let mut best = f64::MAX;
    for _ in 0..ROUNDS {
        let start = Instant::now();
        for _ in 0..iters {
            black_box(f());
        }
        best = best.min(start.elapsed().as_secs_f64() / f64::from(iters));
    }
    best
}

/// Reports a kernel that makes a full pass over `bytes`, where throughput is the
/// number that means something.
fn time<T>(name: &str, iters: u32, bytes: usize, expect: T, f: impl FnMut() -> T)
where
    T: PartialEq + std::fmt::Debug,
{
    let secs = measure(name, iters, expect, f);
    let gbs = bytes as f64 / secs / 1e9;
    println!("{name:24} {:8.3} us  {gbs:6.2} GB/s", secs * 1e6);
}

/// Reports a kernel that returns at the first match, where what is measured is
/// latency: it never reaches most of the haystack, so it has no throughput.
fn time_ns<T>(name: &str, iters: u32, expect: T, f: impl FnMut() -> T)
where
    T: PartialEq + std::fmt::Debug,
{
    let secs = measure(name, iters, expect, f);
    println!("{name:24} {:8.3} ns", secs * 1e9);
}

fn main() {
    let iters = 20;
    let short_iters = 2000;
    let needle = b'z';
    let expect_count = HAYSTACK.iter().filter(|&&b| b == needle).count();
    let truncated = &HAYSTACK[..HAYSTACK.len() & !63];
    let expect_count_truncated = truncated.iter().filter(|&&b| b == needle).count();

    println!(
        "== count (needle {:?}, {} matches)",
        needle as char, expect_count
    );
    unsafe {
        let e = count_ld4_sri(needle, truncated);
        time("ld4 + sri bitmask", iters, truncated.len(), e, || {
            count_ld4_sri(black_box(needle), black_box(truncated))
        });
        time("ld4 + sub acc", iters, truncated.len(), e, || {
            count_ld4_acc(black_box(needle), black_box(truncated))
        });
        time("ldr + sub acc", iters, truncated.len(), e, || {
            count_sub_acc(black_box(needle), black_box(truncated))
        });
        time("ldr + sub acc x4", iters, truncated.len(), e, || {
            count_sub_acc4(black_box(needle), black_box(truncated))
        });
        time("ldr + shrn popcount", iters, truncated.len(), e, || {
            count_shrn(black_box(needle), black_box(truncated))
        });
        time("ldr + uzp + sri", iters, truncated.len(), e, || {
            count_uzp_sri(black_box(needle), black_box(truncated))
        });
    }
    time(
        "memchr crate",
        iters,
        truncated.len(),
        expect_count_truncated,
        || memchr::memchr_iter(black_box(needle), black_box(truncated)).count(),
    );

    println!("== find first (no match)");
    unsafe {
        time("ld4 + sri bitmask", iters, truncated.len(), None, || {
            find_ld4(black_box(b'\x00'), black_box(truncated))
        });
        time("ldr + or reduce", iters, truncated.len(), None, || {
            find_or_reduce(black_box(b'\x00'), black_box(truncated))
        });
        time("ldr + or + uzp/sri", iters, truncated.len(), None, || {
            find_uzp_sri(black_box(b'\x00'), black_box(truncated))
        });
    }
    time("memchr crate", iters, truncated.len(), None, || {
        memchr::memchr(black_box(b'\x00'), black_box(truncated))
    });

    let expect = truncated.iter().position(|&b| b == needle);
    println!(
        "== find first (real needle at offset {}, checks bit ordering)",
        expect.unwrap()
    );
    unsafe {
        time_ns("ld4 + sri bitmask", short_iters, expect, || {
            find_ld4(black_box(needle), black_box(truncated))
        });
        time_ns("ldr + or reduce", short_iters, expect, || {
            find_or_reduce(black_box(needle), black_box(truncated))
        });
        time_ns("ldr + or + uzp/sri", short_iters, expect, || {
            find_uzp_sri(black_box(needle), black_box(truncated))
        });
    }
    time_ns("memchr crate", short_iters, expect, || {
        memchr::memchr(black_box(needle), black_box(truncated))
    });
}
