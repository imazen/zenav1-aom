# KB-PERF-32 — `generate_hog` did eight bounds-checked loads per pixel through a closure: **−0.20 %**, and the bounds checks were only a third of it

**2026-09-10.** Byte-identical; **−0.20 %** at 1024x1024 cq27 `--cpu-used 3`
(mean of two base copies: **−0.169 %** at 21/30, p=0.0428, and **−0.227 %** at
22/30, p=0.0161), over a 30-round rotated band. The null is **exactly flat**
(−0.00 %, 15/30, p=1.0000).

## How it was found — the ranking reordered first

`benchmarks/encoder_x86_reprofile_1024_s3_2026-09-10.md` is the first profile of
this encoder taken at the **shipping preset** rather than at `--cpu-used 0`, and
it moved **intra-pred/rd from 8.1 % to 22.6 % of the gap**, into second place.
Inside that class `generate_hog` had the worst ratio anywhere:

| | port | C |
|---|---:|---:|
| HOG | `generate_hog` **36.2 ms** | `prune_intra_mode_with_hog` **5.2 ms** |

**7.0x** — and C's symbol is the *whole* HOG path with its own generation
inlined, so the comparison is if anything generous to the port.

## The defect

The Sobel loop read every tap through a closure over the whole plane:

    let p = |r, c| -> i32 { i32::from(src[src_off + r * stride + c]) };

Eight distinct taps per pixel (three above, two beside, three below), each a
bounds check against a runtime slice length that the compiler cannot hoist out
of a doubly-nested loop with a computed base.

## The fix

Three row slices per row, walked as overlapping 3-wide windows:

    let above = &src[ra..ra + cols];   // and cur, below
    for ((wa, wc), wb) in above.windows(3).zip(cur.windows(3)).zip(below.windows(3)) {

Window `i` of a `cols`-long row is column `c = i + 1`, and `windows(3)` yields
`cols - 2` of them — which is exactly `1..cols - 1`. So the arithmetic, the walk
order and the f32 accumulation order are unchanged; what changes is **one bounds
check per row instead of eight per pixel**.

The `rows >= 3 && cols >= 3` guard selects the same set the old loop bounds did
(`1..rows - 1` and `1..cols - 1` are both empty below 3), except that `cols == 0`
now returns a normalized all-zero histogram where the old `cols - 1` would have
underflowed. No caller reaches that.

## §14, for the fifth time this project, and the number is worth having

| | |
|---|---:|
| said addressable by the class table | ~31 ms |
| `generate_hog` self cost, before | 36.2 ms |
| `generate_hog` self cost, after | **25.3 ms** |
| kernel delta | **−10.9 ms (1.43x)** |
| whole-encode delta | **−6.5 ms (−0.20 %)** |

**So the bounds checks were about a third of the kernel, not the kernel.** The
lever was 4.8x optimistic read against the class row and 1.7x optimistic read
against the kernel's own self cost — the second being the honest way to cost
this kind of change, and still not tight.

**What is left in it, named:** an integer division per interior pixel
(`get_hist_bin_idx`'s `(dy << 16) / dx`, ~20-26 cycles and not removable without
an exactness argument about truncation toward zero at negative `dy`), and u16
loads where libaom at bd8 reads u8 — the standing structural root, not this
kernel's to fix. At 25.3 ms against C's 5.2 it is still 4.9x and still the worst
ratio in its class.

## Correctness

* Encoder output **byte-identical** on four cells: 40,237 B (1024x1024 s3),
  39,694 (1024 s0), 10,912 (512 s0), 11,961 (512 s6) — two sha256-distinct
  binaries.
* `-p zenav1-aom-encode` **772/772**.
* The HOG differentials against the real exported C (`hog_prune_diff`, and the
  mask-level `prune_intra_mode_with_hog_matches_c` / `..._uv_...` /
  `generate_hog_matches_c`) are green. Those are what make "same arithmetic,
  same order" an assertion rather than a claim: the f32 accumulation order is
  observable there, and a reordered sum would fail them.

## The band

`.band1024s3.tsv`, 30 rotated rounds.

| arm | median ms | spread | paired median | rounds faster | p |
|---|---:|---:|---:|---:|---:|
| base | 3250.7 | 2.0 % | — | — | — |
| baseB | 3252.6 | 2.6 % | −0.00 % | 15/30 | 1.0000 |
| new vs base | 3246.5 | 2.3 % | −0.169 % | 21/30 | 0.0428 |
| new vs baseB | | | −0.227 % | 22/30 | 0.0161 |

## Not covered

One box, one content class, `--cpu-used 3`, x86-64. **`generate_hog` is
reachable only where the HOG prune runs** — `intra_pruning_with_hog` is 4 at
speed >= 6 and the chroma twin is force-disabled from speed 4
(KB-8), so this lever's size is speed-dependent and was measured at s3 only.
