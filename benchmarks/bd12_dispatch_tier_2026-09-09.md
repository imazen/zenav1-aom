# The bd12 dispatch-tier disagreement is CLOSED — and it closed unobserved

**2026-09-09.** One of the standing goal's two must-close classes is met. No
encoder code changed in this landing: the finding is a **measurement**, and the
deliverable is the **gate that should have been there**.

## What the goal required

> (b) anything where the port disagrees with ITSELF across dispatch tiers (the
> bd12 `1920x1080 cq24 cpu0` cell, +181 B default vs +55 B scalar), because a
> kernel whose tiers disagree is a differential hole (playbook §1).

That is stated as a must-CLOSE item, explicitly outside the "measured,
attributed, bounded and documented" cap that governs ordinary divergences.

## Measured at the exact cell

| arm | port_len | fnv-1a-64 | vs real aomenc |
|---|---:|---|---:|
| default dispatch | 158731 | `4b5f1efec0345f57` | **+59 B** |
| `AOM_FORCE_SCALAR=1` | 158731 | `4b5f1efec0345f57` | **+59 B** |

**Byte-identical across tiers**, against KB-38's 2026-08-04 reading of +181
default / +55 scalar. So the port no longer disagrees with itself, and what
remains at that cell is a plain DIVERGENCE (KB-38's own open root) which the
cap covers.

**No fix is attributable.** One of the ~74 landings since 2026-08-04 closed it.
That is not a satisfying answer and it is the honest one.

## Why nobody noticed — the transferable part

**The tree encoded bd12 nowhere.** `s4cov_hd_format_axis::
speed0_1080p_band_map_is_pinned`, the gate that owns this band, sweeps `bd8`
and `bd10` and stops. So the disagreement could neither be observed to persist
nor observed to close.

This is playbook §7 ("bugs live in the gap between two individually-green
rows") one level up, and it cuts both ways: **a must-close item can CLOSE
unobserved exactly as easily as it can open unobserved.** An item on a
must-close list with no gate under it is not being tracked; it is being
remembered.

## The gate, and how it tests tiers without running two of them

`crates/aom-bench/tests/all/bd12_dispatch_tier_agreement.rs`.

`AOM_FORCE_SCALAR` is read once per process (`dispatch::scalar_forced` is a
one-time pin), so no single test can compare the tiers. It does not need to:
**`just gate-landing` runs the whole suite twice**, as `test-next` and
`test-next-scalar`. A byte-identity assertion checked under both modes IS a
tier-agreement assertion — a reopened disagreement fails exactly one of the two
runs and names the cell.

* **Default tier, 23 s / 31 s (measured, both modes):** 18 bd12 cells,
  64x64..256x256 x cq {24, 32, 48}, spanning SB-exact (64/128/192/256) and
  partial-superblock (100/196) shapes because KB-23 and KB-34 were both
  reachable only through a partial SB. **All 18 byte-identical to real
  aomenc**, so the assertion is a hard equality rather than a pinned map — none
  of them can reach KB-38's `>= 1080p` arm.
* **`#[ignore]`d tier, 194.7 s measured:** the >=1080p map pinned, with the
  straddle count asserted so the grid cannot stop proving anything.

## A sharper and 1.8x cheaper reproducer for KB-38's remaining root

| cell | MP | term(s) satisfied | delta |
|---|---:|---|---:|
| `1072x1072 cq24` | 1.15 | qindex only | **0** |
| `1080x1080 cq32` | 1.17 | size only | **0** |
| **`1080x1080 cq24`** | **1.17** | **both** | **+138** |
| `1920x1080 cq24` | 2.07 | both | +59 |

`1080x1080 cq24` reproduces at **1.17 MP against 1920x1080's 2.07**, and its
razor is sharper than the one KB-38 had: BOTH neighbours are byte-exact, one
eight pixels under the `AOMMIN(w,h) >= 1080` term and one with `base_qindex`
128 above the `<= 108` term. **Both terms of the predicate are necessary,
measured rather than argued.** Use this cell for the decode-both probe.

## Also measured, and it bounds the search

**bd12 is byte-exact at every size below 1080p** — 18 cells x 3 quantizers,
both dispatch modes. So the remaining root is not a bd12 kernel defect
reachable from ordinary content; it needs the >=1080p arm to fire.

## Ruled out along the way (do not re-chase)

* **`highbd_variance64`'s SIMD tier.** Its i32 row accumulation is the natural
  bd12 overflow suspect, and it is sound: |diff| < 2^12 so each square < 2^24,
  and the widest AV1 row (128) totals `128 * 4095^2 = 2,146,435,200 < 2^31`.
  Tight — about 1M of headroom — but sound.
* **`block_error` / `highbd_block_error`** (the transform-domain distortion the
  KB-38 arm switches on via `tx_domain_dist_level`): **scalar only, no SIMD
  tier**, so it has no tiers to disagree.
* **The i16 transform path** (`lowbd16`, `lowbd16_fwd`): its `incant!` carries a
  `scalar` tier, so the decision to take the i16 path is tier-INDEPENDENT and
  both tiers run the same i16 arithmetic.

## Method

`crates/aom-bench/examples/bd12_tier_scan.rs` — sweeps bd12 cells and prints
`len` + payload hash; run it once with and once without `AOM_FORCE_SCALAR=1`
and diff. Kept because hunting a cheap reproducer is the expensive half of this
class of work, and 1920x1080 at ~90 s/encode is too slow to bisect against.
