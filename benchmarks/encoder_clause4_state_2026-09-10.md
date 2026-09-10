# Clause (4): state of play after the 2026-09-09/10 cycle

**2.287x -> 1.971x** at the shipping preset (1024x1024 cq27 `--cpu-used 3`,
port 3059 ms / C 1552 ms). **Seventeen landings, six documented rejections**,
every landing byte-identical on four cells from sha256-distinct binaries, with
`just gate-landing` green (1506/1506 twice) on each. **1.5x needs 731 ms more.**

This file is the index. Each claim links the record that measures it.

## Is the bar reachable? Yes — and the arithmetic is two lines

`encoder_clause4_reachability_2026-09-10.md`. The bar needs **50 % off every
class gap**, not 100 % of any one, and **halving all nine gives 2334 ms =
1.505x**. The transform class already fell **4.26x -> 2.78x in this cycle**
(a 45 % gap reduction) via eight fused kernels of 0.3-2.0 % each.

**Redundant work is structurally ruled out** — byte-identical output plus ported
speed features means the same decisions, the same candidates, the same kernel
invocation counts; and the drivers run at 1.29-1.37x while the kernels run
2.8-6.7x. The whole gap is per-operation cost.

## What this cycle established about METHOD

**1. Annotate, don't rank.** `perf annotate` on one symbol produced landings
averaging **1.4 pp**; class tables and symbol shares produced **0.3 pp**. A class
table cannot see a hot loop inlined into a large caller — `rd-driver` read as
"+261 ms of diffuse overhead" and 20 % of its top symbol was three bounds checks
per pixel (`encoder_subtract_block_2026-09-10.md`, **−3.17 %**).

**2. Check the symbol is actually slow.** The largest symbol in the profile
(12.8 %) is at PARITY with libaom, and the intra edge assembly is **0.85x —
faster**. Compare like-for-like before spending a landing.

**3. Re-annotate after landing, not just re-band.** The first `subtract_block`
fix turned a hot inlined loop into a standalone symbol and **lost its own
inlining**; the band said −3.17 % and looked finished, and a second annotate
found −0.23 % more.

**4. A new fast path can bypass an older one.** The eight fused transform
kernels each measured net-negative and byte-identical, and collectively they
route around the i16 path an earlier landing added — now ~97 % of transforms
(`encoder_clause4_narrowpath_2026-09-10.md`). No gate or band can see this.

**5. Reach and cost figures both carry their regime.** Three of this cycle's own
published numbers were corrected within hours: a class ranking taken at
`--cpu-used 0`, reach shares from another preset (wrong by ~2x in BOTH
directions), and a ceiling inflated ~2x by our own earlier measured null.

**6. sha256 every arm before trusting a band.** `cargo build | grep -E "^error"`
matches nothing (ANSI colour codes), a build failed invisibly, and a stale
binary entered an arm. Three different source states produced one identical
hash — the only reason it was caught.

## The six rejections, each naming a boundary

| rejected | measured | boundary it names |
|---|---:|---|
| doubled-run identity for `up == 1` (z1/z3) | +0.34 % | the kernel is **store-bound**; doubling arithmetic doubles the per-lane copy |
| exact `with_capacity` (3 sites) | +0.77 % | a capacity hint is **arithmetic**; it removed **0** allocations |
| thread-local scratch pool | +0.23 % | TLS + `RefCell` costs more than a glibc malloc |
| per-column `imul` removal (`txb_init_levels`) | null | wins only where **rows are short and numerous** |
| `#[autoversion]` on `variance_raw` | +0.25 % | adds a **dispatch per call**; needs large per-call work |
| in-loop config lookup (KB-PERF-14) | +0.20 % | check **which loop nest** a recomputation lands in |

**The allocation rule, from four attempts:** an allocation removal pays only if
its replacement is cheaper than the allocation was. Inline storage and stack
arrays clear that bar on glibc; TLS pools and extra arithmetic do not.

## Allocations

10,419,121 -> **7,378,761**; temporaries 1,931,251 -> **197,969 (−90 %)**
(`encoder_smallvec_winners_2026-09-10.md`, `encoder_alloc_pass_2026-09-10.md`).

**`SmallVec` beat `TinyVec` by 0.50 % of the whole encode at identical
allocation counts** — `TinyVec` is an enum and branches on `Inline | Heap` at
every access.

## What remains, with costs and risks

| item | size | risk |
|---|---:|---|
| **gather-based z2 left half** | ~38 ms addressable | raw-AVX2 gather + index-safety argument |
| **i16 inside the fused transforms** | 63 ms ceiling, **~1 %** realistic | 16x16 i16 in-register transpose (raw AVX2); forward half first, inverse needs its own band |
| **`aom_quantize_b_no_qmatrix`** (no SIMD at all, 7.5x) | ~23 ms | raw-order rewrite changes how `eob` is derived — the ONE order-sensitive output (KB-12) |
| **arena for `TxbEncode` coefficients** | 62 % of allocations | structural; **glibc masks its value — measure on Windows** |
| classes never annotated | intra-pred/rd +388, LR +113, distortion +108, postfilter +46 | low; the method is cheap |

**Leave `txb/trellis` alone at 1.37x** — `optimize_txb_core` is faster than
libaom's trellis, confirmed at both speeds.

## The platform caveat that governs two of these

**KB-PERF-2 measured allocation levers at 21 % of the win on Darwin and
86-99 % on Windows.** glibc is near a free-list pop on a hot reused size class,
so this box systematically **understates** allocation work. `winperf.yml`'s
`arms: prepost` mode should re-run the capacity and TLS-pool rejections and size
the arena before anyone concludes from Linux numbers.

## Honest position

Clause (4) is unmet and is the sole blocker for clause (1) (clauses 2, 3, 5, 6
are gated green; the zenavif seam is closed at 449 tests and needs two
`#[default]` flips). The cheap end is exhausted on this box: what is left is
gather kernels, an i16 transform rewrite, a structural arena, and a platform
this box is not. Whether to fund that is a decision, not a measurement — but it
should be made against a target the arithmetic says is reachable.
