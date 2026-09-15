# `acc_stat_line` v3 const-generic window specialization — 2026-09-15

Cell: 1024x1024 cq27 `--cpu-used 3` (the preset zenavif ships), eprof_x86 `port`,
reps=1 per round, `nice -n 19`, rotated base/new/baseB. Band TSV:
`benchmarks/encoder_acc_stat_constgeneric_2026-09-15.band1024s3.tsv`.
base = `e47d315`; new = working tree.

## Result

```
base     median   2396.85 ms  min   2386.49  n=24  bytes=['40237']
baseB    median   2399.25 ms  min   2389.65  n=24  bytes=['40237']
new      median   2397.22 ms  min   2389.09  n=24  bytes=['40237']
new vs base   : paired median -0.058 %   14/24 rounds faster   p=0.5413
null (baseB vs base): paired median +0.136 %   4/24 rounds faster   p=0.001544
```

**Null on wall** — does not clear the same-binary null and p is not
significant. Landed on the `txb_init_levels`/`wiener` precedent: the
kernel-level instruction win is real (−42% per call), output is
byte-identical (40,237 B every round, both arms), and the loop shape is
now statically provable rather than runtime-bounded. Shipping-cell wall
≈ 2397 ms vs C ≈ 1604 ms ≈ **1.49x** — still at the Gate-3 bar.

## Mechanism (kernel Ir, 196x196 cq27 s0, 5 reps, identical call counts)

| kernel | base | new | C |
|---|---:|---:|---:|
| restoration stats | 1,088.5M (92.6k/call, `acc_stat_line_impl_v3`) | **626.2M (53.2k/call)** | 243.6M (win5+win7 avx2, 156 unit calls) |

−42% Ir/call; the kernel's gap to libaom per restoration unit goes
**4.5x -> 2.57x**. Total cell Ir 34.21G -> 33.29G (−2.7%).

The generic magetypes v3 body ran every accumulator access through runtime
bounds (`wiener_win2`, `hstride`, slice lengths LLVM could not relate), so
each `ld!`/`st!` kept a `cmp+ja` + panic-thunk pair and each gather element
an index check. `wiener_halfwin` is only ever 2 (win5) or 3 (win7), so the
x86 tier is now `acc_stat_line_v3_x86<const WIN: usize>` — the same
four-pixel fold monomorphized on the window size:

* every `y`/`col` index is a literal-bounded value against a fixed-size
  array (`[i32; 56]`/`[i32; 2744]` views taken once per call after a
  single preflight — anything else delegates to the scalar recipe, whose
  panic behaviour is unchanged);
* the window gather takes one checked row slice per window row
  (WIN+3 columns) instead of a per-element index;
* M/H loops use raw `_mm256_loadu_si256`/`storeu`/`mullo`/`add` through
  `&[i32; 8]` referents — all checks statically elided.

## Correctness note (a real bug, found by the byte gates)

The first draft computed the window-row base as
`dgd_origin + (inner_i32) as usize`: a negative inner product (window rows
may address the extended border before the origin) wrapped to a huge
`usize` and panicked on the add in debug builds — four
`self_contained_key_frame` failures. `gather_window` always did the add in
`isize`; the specialization now does the same.

## What is NOT done

The remaining ~2.57x is structural and named in the function comment:
libaom accumulates **raw u16 products** into i64 `M_int`/`H_int` via
`_mm256_madd_epi16` two pixels at a time and folds the mean correction at
the end of the unit, while the port pre-subtracts the mean and
read-modify-writes i32 row accumulators. Mirroring C's shape changes the
caller's accumulator types — a bigger, riskier rewrite deferred until the
wall profile says it pays.

## Gates

* `pick_diff` 6/6 (both tiers vs real exported C)
* `self_contained_key_frame` 10/10 (incl. the four cells that caught the
  overflow bug)
* `encoder_gate_e2e_byte_match` 32/32
* shipping-cell output byte-identical: 40,237 B every round, both arms
