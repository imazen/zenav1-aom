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

## RESOLVED THE SAME DAY: the lever is NULL, and the audit below should NOT be written

**Both halves were built and measured. Neither pays.** This section supersedes
the four-step plan that follows it, which is kept only to show what was
attempted.

### 1. Vectorized squares with an optimistic VERIFY (no audit needed)

Rather than prove `|e| < 16384` offline, the vector path ran optimistically —
`err += i64::from((e * e).reduce_add())` — while tracking the true range of `e`
with lane-wise `max`/`min`, discarding everything and recomputing scalar if the
range ever left the safe window. Correctness from the check, not an inequality;
wrapping en route is harmless because magetypes' integer `Mul`/`Add` wrap on
every backend and the verify throws the garbage away. (Range tracked as separate
max/min rather than `abs`, since `abs(i32::MIN)` wraps NEGATIVE and would slip
through.)

Byte-identical output, `pick_diff` 6/6 against the real exported C — and:

| | paired median | rounds faster | p |
|---|---:|---:|---:|
| verified variant | **+0.07 %** | 9/24 | 0.31 |
| same-binary null | −0.08 % | 16/24 | 0.15 |

**Null.**

### 2. The same thing with the verify STRIPPED — the theoretical best case

To separate "the verify costs too much" from "the vectorized square does not
pay", the tracking and the bail were deleted outright (unsafe, throwaway):

| | paired median | rounds faster | p |
|---|---:|---:|---:|
| no-verify probe | **−0.07 %** | 10/16 | 0.45 |
| same-binary null | −0.05 % | 10/16 | 0.45 |

**Also null, and indistinguishable from the null arm to two decimal places.**

### The conclusion, and it is stronger than "not worth it"

The verify is not the cost. **A per-chunk horizontal `reduce_add` costs about
what the eight scalar `i64` multiply-accumulates it replaces** — superscalar
hardware absorbs eight independent MACs well, while a horizontal reduce is a
serial dependency chain of shuffles and adds. So **no bound, however derived,
can make this lever pay**: an offline audit proving `M* < 16384` would license
exactly the code measured null in probe 2.

**Do not write `xtask/audit_sgr_proj_error.py`.** The recommendation below was
made before these two bands existed and is withdrawn. KB-PERF-6's named lever is
CLOSED by measurement rather than left open.

The 93-stack-op diagnosis above is still correct about the mechanism — the
round trip is real — it just does not follow that removing it is a win, because
what replaces it is not cheaper. That is the same shape as KB-PERF-6's own
box-sum lesson (a symbol's cost is not automatically a lever's size) arriving
from the arithmetic side instead of the profiler side.

## SUPERSEDED — what the next attempt was going to do

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
