# KB-PERF-23 — the SIMD-preserving fused 8x8 forward transform: **−2.017 %**, 24 of 24 rounds

**2026-09-09.** Byte-identical; **−2.017 %** at 1024x1024 cq27 `--cpu-used 3`,
**24 of 24 rounds faster, p < 0.0001**, against a same-binary null of +0.128 %.

**The largest landing of this cycle by a factor of two** — the three that
preceded it compose to −1.054 %.

## It is the target the domain split identified, and the one the census warned about

Two records set this up, and both were needed:

* **`encoder_domain_split_2026-09-09.md`** measured that **52.8 % of the
  shipping-preset gap is COEFFICIENT-domain** against 24.4 % for the plane
  representation, with the transform class alone at **+789 ms = 40.5 %**. That
  is what pointed here rather than at the `u16`->`u8` refactor the ranking had
  been calling the structural root.
* **`encoder_txfm_size_census_2026-09-09.md`'s own correction** is why the
  earlier attempt failed and what shape a new one had to take: KB-PERF-16's 8x8
  fusion was **SCALAR**, measured **+7.07 %** and was reverted, because at 8x8
  both generic passes satisfy `n % 8 == 0` and run full-width — a scalar fusion
  trades the SIMD away. Its closing line: *"a future one would have to be
  VECTORISED itself to beat what is already there."*

8x8 is **25.74 % of forward transforms** at this preset, second only to the
already-fused 4x4 (40.51 %).

## What it does — keeps both passes, removes only the driver

Lane = COLUMN for the first pass: `v[r]` is input row `r`, so the column kernel
runs vertically across eight vectors, per lane — exactly `fwd_col_pass`, without
materialising `buf`. Then ONE in-register 8x8 `i32` transpose makes lane = ROW
and the row kernel runs the same way, followed by one contiguous vector store
per output row.

**The transpose is not an addition.** The generic row pass already pays one — it
loads `buf` through 8x8 tiles — so this moves it out of memory rather than
introducing it. What is actually removed is the DRIVER: `get_fwd_txfm_cfg`, the
two `try_*` gate functions, the scratch tiering, and the round trip through
`buf` between the passes.

## Exactness

Every step is the generic path's own, in its order, and `FWD_SHIFT[TX_8X8] =
[2, -1, 0]` was read from the table rather than assumed:

* `shl_clamp64v(_, 2)` is `round_shift_array(_, -shift[0])`;
* `rshiftv(_, 1)` is `round_shift_array(_, -shift[1])`;
* `shift[2] == 0` and `rect_type == 0`, so the row pass has no tail;
* **`lr_flip` is a lane REVERSE applied AFTER the column kernel** — the generic
  writes lane `j`'s result to column `col_n-1-j`, so reversing before the kernel
  would be wrong. That is the bite proof below;
* `ud_flip` is the source-row reversal at load; the transpose only moves lanes.

## Correctness — and the differentials earned their keep twice

* Encoder output **byte-identical** on four cells: 40,237 B (1024x1024 s3),
  39,694 (1024 s0), 10,912 (512 s0), 11,961 (512 s6).
* `-p zenav1-aom-dsp` **395/395 in both dispatch modes**; the
  `AOM_FORCE_SCALAR` pin and any non-8-point kernel decline to the generic
  driver.
* **The first version was WRONG and the gate caught it.** An index-computed
  `from_fn` transpose got stage 2's pairing wrong (`b[1]` computed
  `lo(t1,t3)` where it needs `hi(t0,t2)`), and `txfm2d_differential_fuzz` +
  `txfm2d_simd_equals_scalar_at_every_permutation` failed immediately. Rewritten
  as explicit named stages; a clever index expression is not worth the debugging
  it costs in a transpose.
* **Bite proof, on the documented trap**: dropping the `lr_flip` reverse fails
  `txfm2d_differential_fuzz` (against the **real exported C**) and
  `txfm2d_simd_perm_diff` while **393 stay green**.

## The band

Two sha256-distinct binaries from one tree, 24 rotated rounds, same-binary null:

| arm | median | paired median | rounds faster | p |
|---|---:|---:|---:|---:|
| base (KB-PERF-22 HEAD) | 3541.88 ms | — | — | — |
| baseB (same binary) | 3547.18 | +0.128 % | 8/24 | 0.1516 |
| **fused 8x8** | **3473.11** | **−2.017 %** | **24 of 24** | **<0.0001** |

~71 ms, inside the **60–125 ms** the sizing predicted before the work started
(from KB-PERF-16/17's 4x4 fusion at −2.30 % over ~48 % of calls, scaled to 8x8's
~26 % share and discounted for the fact that 8x8's passes stay SIMD so only the
driver is saved).

## Not covered

* **The INVERSE 8x8 is untouched.** It is **26.96 % of inverse transforms** and
  `INV_SHIFT[TX_8X8] = [-1, -4]` makes BOTH shifts live, so its recipe does not
  collapse the way the forward's does — a different proof, and the obvious next
  landing.
* **x86-64 only** (`cfg(target_arch = "x86_64")`); aarch64 and wasm keep the
  generic driver, which is unchanged.
* Non-8-point kernels (a tx_type whose 1-D type is not an 8-point DCT/ADST/IDTX)
  decline.
* One cell, one content class, bd8 4:2:0, cq27, `--cpu-used 3`.
