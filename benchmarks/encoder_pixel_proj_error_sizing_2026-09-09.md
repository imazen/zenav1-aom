# `pixel_proj_error`'s squaring tail: sized, NOT landed — the bound needs an audit, not an inequality

**2026-09-09. No code changed.** This records the measurement for the lever
KB-PERF-6 named and deliberately declined, so the next attempt starts from
evidence rather than from scratch — and records why it was declined a second
time.

## The lever, as KB-PERF-6 left it

> **`pixel_proj_error`: a bound DECLINED on purpose.** Squaring in `i32` lanes
> and reducing per chunk needs `8 * e^2 < 2^31`, i.e. `|e| < 16384` — and the
> arithmetic (`xq` reaches ~96 via `SGRPROJ_PRJ_MIN0/MAX0`, `flt - u` reaches
> ~2^16 at bd12) puts `|e|` AT that bound, not inside it. … To claim those
> milliseconds, DERIVE the bound and gate it at runtime the way
> `intra/dir_simd.rs` gates its tap bound.

The shipped kernel computes `e` in lanes and then squares scalar-side:

    for x in e.to_array() {
        err += x as i64 * x as i64;
    }

## What is now measured

`pixel_proj_error_impl_v3` is **2.24 % of the port at 1024x1024 cq27 speed 0**
(~224 ms) in the profile taken at HEAD after KB-PERF-16..19.

Disassembling the shipped `_v3` variant over its hot region:

| | count |
|---|---:|
| scalar `imul` | 3 |
| vector ops (`vpmulld`/`vpaddd`/`vpsubd`/`vpsrad`) | 11 |
| widening loads (`vpmovzxwd`/`vpmovsxwd`) | 2 |
| **stack traffic (`mov` to/from `%rsp`)** | **93** |

**So the cost is the round trip, not the multiplies.** `e.to_array()` forces a
store of the vector and eight scalar reloads per chunk; the widening loads are
already contracted (2, as expected for the two `widen` calls), so the `widen`
closure is NOT the problem — the same check that refuted the quantize `iscan`
hypothesis applies here and clears it.

That makes the lever real and worth roughly its 2.24 %, with the caveat that
removing a round trip does not recover all of it.

## Why it is still NOT landed

Vectorizing the squares needs TWO bounds, and neither is comfortable:

* `e * e` in `i32` lanes needs `|e| < 46341`;
* `reduce_add` over 8 lanes needs `8 * e^2 < 2^31`, i.e. **`|e| < 16384`**.

A back-of-envelope on the lowbd path — `e ~ (d - s) + (xq0*(flt0-u) +
xq1*(flt1-u)) / 2^11`, with `|d - s| < 2^bd`, `|xq| <= 96`, `|flt - u| ~ 2^16`
at bd12 — lands near **~7200**, i.e. inside 16384 but within a factor of ~2.3 of
it. **That is exactly the situation this codebase does not accept an inequality
for.** The standard here is an offline audit that enumerates or propagates
exact ranges — `xtask/audit_i16_fwd.py` (exact linear forms over `Fraction`
coefficients, which REJECTED `fadst4` and would have been missed by a triangle
inequality), `xtask/audit_nd16_lanes.py` (exhaustive enumeration over the whole
input space), and `intra/dir_simd.rs`'s runtime tap gate.

`pixel_proj_error` feeds loop-restoration RD decisions and therefore the byte
gates, so a bound that is wrong in a corner is a wrong bitstream, not a slow
one. KB-PERF-6 declined it on that reasoning with the same arithmetic in front
of it; overriding that on a back-of-envelope would be the weaker call.

## What the next attempt should do

1. Write `xtask/audit_sgr_proj_error.py` in the shape of the two existing
   audits: propagate exact bounds for `v`, then `e`, over `bd in {8, 10, 12}`,
   every `ep` in `SGR_PARAMS`, and the full `xq` range from
   `SGRPROJ_PRJ_MIN0/MAX0` / `MIN1/MAX1`, reporting `M*` — the largest `|e|`
   the kernel can produce.
2. If `M* < 16384` unconditionally, vectorize the squares outright.
3. If not, gate at RUNTIME on the data the way `dir_simd` does, and pin the gate
   in BOTH directions (it must fire on the real bd8 domain, and it must decline
   where the bound is exceeded) — `dir_simd`'s `reach` and
   `the_tap_bound_is_load_bearing` tests are the template.
4. Note that a per-chunk `max|e|` check is likely to cost more than it saves;
   prefer a per-CALL gate derived from `bd`, `ep` and `xq`, which are loop
   invariants.

## Also on the list, same class, not sized here

`acc_stat_line_impl_v3` 2.19 %, `wiener_impl_v3` 1.85 % (blocked on a magetypes
integer `interleave` that does not exist — `docs/MAGETYPES_VOCABULARY.md`),
`selfguided_restoration` 1.49 %, `calculate_intermediate::{closure#0}` 1.46 %.
Loop-restoration is ~9.2 % of the port at HEAD.
