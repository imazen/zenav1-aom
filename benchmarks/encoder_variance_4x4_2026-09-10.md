# The ALLINTRA variance factor copied 16 pixels into a scratch to measure them — **−0.827 %**

**2026-09-10.** Byte-identical, **−0.827 % at 1024x1024 cq27 `--cpu-used 3`,
23 of 24 rounds faster, p < 0.0001**, against a same-binary null of −0.048 %
(13/24, p = 0.84). The largest landing of this session and the second largest of
the whole cycle.

## How it was found — the lever map, and a premise that did not survive

`encoder_lever_map_s3_2026-09-10.md` ranked the **variance family** second among
addressable rows: **89.7 ms port vs 18.3 ms C, 4.91x, +71.4 ms**. It had never
had a landing, and the one prior attempt (KB-PERF-47) rejected `#[autoversion]`
on `variance_raw` with the reason that variance's callers are *"inter/motion
paths largely dead on an ALLINTRA KEY frame"*.

**That premise is false at the shipping preset, and a frame-pointer attribution
says so in one run:**

```
dist::variance   80.9 %  aom_encode::intra_rd::intra_rd_variance_factor
                  9.1 %  aom_encode::partition_pick::log_sub_block_var
                  6.2 %  aom_encode::pack::pack_tile_from_trees_lr
                  3.8 %  aom_encode::pack::pack_tile_lr_stop
```

`intra_rd_variance_factor` is the ALLINTRA visual-quality RD scale — maximally
live on a KEY frame, not dead on one. KB-PERF-47's *measurement* stands (its
mechanism really did fail); its *explanation* pointed at the wrong callers, and
that explanation is what would have kept this row closed.

## The defect

`calc_normalized_variance_4x4` is called once per 4x4 unit of every block by
both of the top two callers. Its bd8 arm — the shipping path — did:

```rust
let mut w8 = [0u8; 16];
for r in 0..4 { for c in 0..4 {
    w8[r*4 + c] = buf[off + r*stride + c] as u8;      // 16 bounds-checked reads
}}
const ZEROS8: [u8; 4] = [0; 4];
aom_dsp::dist::variance(&w8, 4, &ZEROS8, 0, 4, 4).0 as i32
```

— a 16-pixel copy into a scratch, and then the **generic** variance kernel
walking those 16 values again against a stride-0 all-zero reference, with its own
bounds checks. libaom calls `aom_variance4x4_sse2` straight on the plane.

The replacement accumulates `sum` and `sum of squares` directly off the strided
window, one bounds check per row (KB-PERF-37's pattern), and applies the same
final expression.

## Bit-exact, by arithmetic rather than by assertion

* the reference is all-zero, so `diff == a` and the generic kernel reduces
  exactly to `tsum = sum(a)`, `tsse = sum(a*a)`;
* its `wrapping_add` cannot wrap: `sum(a*a) <= 16 * 255^2 = 1,040,400`;
* the final `tsse.wrapping_sub(((tsum as i64 * tsum as i64) / 16) as u32)` is
  copied verbatim — and by Cauchy-Schwarz `16*sum(a^2) >= sum(a)^2`, so it never
  wraps either;
* the `<= 255` bound is the same one the old `as u8` narrowing silently relied
  on, and it is **guaranteed, not assumed**: KB-51 made an out-of-range bd8
  sample a refusal at the public entry point (`KeyFrameError::SampleRange`), so
  the domain is unreachable. The `debug_assert` is kept.

The bd10/12 arm is deliberately untouched — it is not the hot path here
(`dist::highbd_variance`'s samples come 87 % from `dist_block_px_domain_into`),
and leaving it keeps the bite proof asymmetric.

## Measurement

Two sha256-distinct binaries from one tree, arms rotated each round,
same-binary null, 24 rounds x 4 reps.

| | effect | rounds | p |
|---|---:|---:|---:|
| **4x4 variance direct** | **−0.827 %** | **23/24** | **<0.0001** |
| null, same binary both sides | −0.048 % | 13/24 | 0.84 |

**Byte-identical on all four standard cells** — 40,237 / 39,694 / 10,912 /
11,961.

## Not covered

One box, one content class, one quantizer, `--cpu-used 3`, x86-64. The row is not
closed: the port's remaining variance cost is still generic where libaom is
size-specialised, and the bd10/12 arm plus `dist_block_px_domain_into`'s
`highbd_variance` (25.2 ms) are untouched.
