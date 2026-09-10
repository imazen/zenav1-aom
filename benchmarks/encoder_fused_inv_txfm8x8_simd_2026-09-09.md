# KB-PERF-24 — the SIMD-preserving fused 8x8 INVERSE transform: **−1.538 %**, 24 of 24 rounds

**2026-09-09.** Byte-identical; **−1.538 %** at 1024x1024 cq27 `--cpu-used 3`,
**24 of 24 rounds faster, p < 0.0001**, against a same-binary null of −0.035 %.

KB-PERF-23's twin, and the follow-up that record named. 8x8 is **26.96 % of
inverse transforms** at this preset.

## Why it needed its own proof rather than a copy

KB-PERF-23's closing note: *"`INV_SHIFT[TX_8X8] = [-1, -4]` makes BOTH shifts
live, so its recipe does not collapse the way the forward's does."* That held —
the forward's `[2, -1, 0]` has a no-op tail, and the inverse additionally
carries two `clamp_buf` calls and the final `highbd_clip_pixel_add`
reconstruction.

**One structural advantage in the other direction**, which is why this landed
faster than the forward did: the inverse input is **COLUMN-major**
(`mod_input[c * row_n + r]`), so the row pass needs **no** transpose — a
contiguous 8-lane load at `input[c*8..]` already has lane = r. The single
transpose moves to BETWEEN the passes, where the generic path pays it anyway by
writing and re-reading `buf`.

## Exactness — every step is the generic driver's, in its order

* row: `clampv(_, bd + 8)` is `clamp_buf(ti, (bd+8) as i8)`; `rshiftv(_, 1)` is
  `round_shift_array(_, -shift[0])`;
* `rect_type == 0` at 8x8, so the `NEW_INV_SQRT2` scaling does not apply;
* column: `clampv(_, col_clamp)` with `col_clamp = max(bd+6, 16)`;
  `rshiftv(_, 4)` is `round_shift_array(_, -shift[1])`;
* **`lr_flip` is a lane REVERSE on the TRANSPOSED vectors** — the generic reads
  buf column `col_n-1-c` for output column `c`;
* **`ud_flip` REORDERS the eight output vectors** (output row `r` takes
  `tout[row_n-1-r]`) — it is not a lane operation, and conflating the two is the
  easy mistake here;
* reconstruction is the same wrapping add then clamp to `[0, (1<<bd)-1]`, so the
  `as u16` narrowing is exact.

## The differentials caught the one thing that was wrong

**`opt_range(bd)` returns `(COL, ROW)`** — the driver destructures it as
`let (opt_range_col, opt_range_row) = opt_range(bd);`. The first version fed the
COL range to the ROW pass and vice versa, and
`inv_txfm2d_diff::inv_txfm2d_differential_fuzz` plus
`txfm2d_simd_equals_scalar_at_every_permutation` failed immediately. The call
site now names the two bindings `opt_col` / `opt_row` and carries a comment, so
the next reader cannot repeat it.

That is the second time in two landings that a **tuple-order or index-order**
slip was the only defect, and both times the differentials against the real
exported C found it in one run. (KB-PERF-23's was an index-computed transpose
stage.)

## Correctness

* Encoder output **byte-identical** on four cells: 40,237 B (1024x1024 s3),
  39,694 (1024 s0), 10,912 (512 s0), 11,961 (512 s6).
* `-p zenav1-aom-dsp` **395/395 in both dispatch modes**; the
  `AOM_FORCE_SCALAR` pin and any non-8-point kernel decline to the generic
  driver.
* **Bite proof, asymmetric**: swapping the row and column clamps fails
  `inv_txfm2d_differential_fuzz` (against the **real exported C**) and
  `txfm2d_simd_perm_diff` while **393 stay green**.

## The band

Two sha256-distinct binaries from one tree, 24 rotated rounds, same-binary null:

| arm | median | paired median | rounds faster | p |
|---|---:|---:|---:|---:|
| base (KB-PERF-23 HEAD) | 3426.93 ms | — | — | — |
| baseB (same binary) | 3423.48 | −0.035 % | 14/24 | 0.5413 |
| **fused inverse 8x8** | **3375.35** | **−1.538 %** | **24 of 24** | **<0.0001** |

Inside the **−1.3 to −1.6 %** predicted from KB-PERF-23's forward result scaled
by the inverse class's smaller share of the encode and 8x8's slightly larger
share of inverses.

## Not covered

* **x86-64 only** (`cfg(target_arch = "x86_64")`); aarch64 and wasm keep the
  generic driver, unchanged.
* Non-8-point kernels decline; so does any caller whose `INV_SHIFT` row is not
  `[-1, -4]` (checked at the hook, not assumed).
* The **lowbd `u8` inverse twin** is a separate entry point and is untouched —
  it is the decoder's path, and this is an encoder lever.
* One cell, one content class, bd8 4:2:0, cq27, `--cpu-used 3`.
