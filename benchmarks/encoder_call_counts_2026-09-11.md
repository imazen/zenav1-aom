# Call-count profiling — the instrument sampling cannot be, and what it found on its first run (2026-09-11)

**Question asked:** does `perf record` capture call counts? **No.** Sampling attributes
time; a function called 9x too often at equal per-call cost is indistinguishable from one
9x too slow. The lever map's biggest row (memset+memmove, 7.29x) was unactionable for
exactly that reason — a count problem and a size problem look identical in samples.

**Instrument:** callgrind on the SHIPPING-path driver, both arms in one process —

```
cargo build --profile profiling -p zenav1-aom-bench --example eprof_x86
valgrind --tool=callgrind --cache-sim=yes --collect-jumps=yes \
    --callgrind-out-file=/tmp/cg_<cell>_<side>.out \
    ./target/profiling/examples/eprof_x86 <port|c> <W> <H> <CQ> <SPEED> <REPS>
python3 scripts/callgrind_counts.py port.out c.out [--top N]     # count diff
python3 scripts/callgrind_counts.py FILE.out --callers-of memcpy # who calls, how often
```

`eprof_x86`, not `gate3_profile` — the justfile's `just profile` recipe drives the
differential harness (1.9x slower, wrong regime; same trap as the discarded RD sweep).
The `profiling` cargo profile (release + `debug = true`) gives line attribution.
Cost: ~134x slowdown (196² port ≈ 16 s; 1024² port ≈ 7 min, C ≈ 3.5 min for
warm+1 rep — counts are exact and load-free, so one rep is a measurement, not a sample).

**What it sees / does not:** `calls=N` on every call arc — exact, and it counts through
C's RTCD dispatch (`av1_predict_intra_block` 6.25M calls, `aom_lpf_vertical_*_sse2` the
size-specialised family). It does NOT see calls LLVM inlined — Rust `copy_from_slice`
on a runtime size still lowers to a memcpy PLT call, but a callee LLVM absorbed leaves
no edge (per-line Ir still attributes its work; `aom_dsp::census` — the committed
feature-gated counter layer — is the source-site complement when a specific site needs
a count regardless of inlining). Asymmetric inlining is the comparability hazard:
`av1_predict_intra_block` shows 6.25M C calls vs ~0 port calls because the port inlines
its predictors — read count ratios together with Ir, never alone. SIMD kernels count
once per call regardless of lane width.

## The shipping cell, counted (1024x1024 cq27 s3, warm+1 = 2 encodes per arm)

| metric | port | C | ratio |
|---|---:|---:|---:|
| Ir (instructions) | 115.74 B | 50.07 B | 2.31x |
| Dw (data writes) | 13.69 B | 5.48 B | **2.50x** |
| Dr (data reads) | 23.14 B | 12.89 B | 1.80x |
| `__memcpy_avx` calls | **92,063,312** | 10,132,291 | **9.09x** |
| `__memset_avx2` calls | **47,808,559** | 15,035,358 | **3.18x** |
| memcpy+memset Ir | 4.40 B (3.8 %) | 0.41 B (0.8 %) | 10.7x |
| malloc family calls | ~3.81 M | ~1.06 M | ~3.6x (matches `eprof_alloc`) |

## Where the calls are — and the self-check that validates the method

**C's memcpy is edge assembly**: `cfl_load_dc_pred` 3.46M +
`build_directional_and_filter_intra_predictors` 2.59M + `build_non_directional` 2.11M
= **8.2M of its 10.1M calls (81 %)** — predictor setup copying reference edges into
owned arrays. The port's corresponding family — `assemble_dir_edges` 2.60M +
`assemble_nd_edges` 2.17M + `filter_intra_predict_high` 4.14M ≈ 8.9M — **matches**.
And the strongest validation: port `dist_block_px_domain_into` calls memcpy exactly
**3,712,400** times; C's `dist_block_px_domain` is called exactly **3,712,400** times —
one `convolve_copy` per call on both sides, 1:1. Where the port transcribes C's
structure, the counts agree. The 9x is all in the structure it doesn't share.

**The port-only layer — the publish round trip C doesn't have:**

| caller | memcpy calls | memset calls | what it is |
|---|---:|---:|---|
| `txfm_rd_in_plane_intra` | 19,401,352 | 8,768,632 | per-candidate `pred.clear+resize` + `predict→scratch` + per-row `copy_from_slice` publish (tx_search.rs:2237-2304) |
| `txfm_rd_in_plane_uv_p` | 8,387,872 | 4,192,672 | the chroma twin (intra_uv_rd.rs:~660) |
| `search_tx_type_intra_into` | 8,380,928 | — | `best_q/dqcoeff` winner saves, 2 per improvement (:1813-15) |
| `encode_intra_block_plane_y/uv` | 5.79M + 2.81M | 1.05M + 0.80M | same predict→scratch→publish on the encode path |
| `predict_uv_txb` | 3,537,940 | — | chroma predict publish |
| `fwd_rect48_fused` | 8,649,056 | — | transform kernel output staging |
| `two_tap_run` / `smooth` / `hadamard*` | ~14M | — | kernel-internal staging (needs its own look) |
| `optimize_txb_core` | — | 5,978,132 | scratch memsets incl. the `[0u8; TX_PAD_2D]` entry |

In C the prediction goes straight into `pd->dst` — the recon plane (`reconintra.c`
ref==dst form, KB-34). A losing candidate's prediction overwrites the plane harmlessly
because intra predictors read only above/left edges; `recon_intra` then writes the
winner's reconstruction over it. The port instead pays **memset + predict-to-scratch +
per-row publish** per candidate mode per txb — KB-PERF-56 converted
`intra_model_rd_y` (−0.50 %) but `txfm_rd_in_plane_intra`, the UV walk, and the encode
path kept the round trip. **This is the named mechanism behind the map's largest row**
(memset+memmove, +96.5 ms of gap): predominantly a count problem, and a pure
work-REMOVAL lever — the kind every band this cycle paid on.

`residual.clear()+resize` is dead work too: `highbd_subtract_block` writes every
element of `[0, txw*txh)` before any read — the re-zero is never observed. Same for
`coeff`/`dqcoeff` buffers where a full-overwrite callee follows.

## Second finding: the write traffic is also a LAYOUT problem

Port Dw = 2.50x C's — wider than Ir (2.31x). The u16-at-bd8 plane root already on the
map (24 % of gap) shows up here as *twice the bytes per copy* on every publish, edge
assembly and convolve_copy. Counts separate the two halves of the row: the call-count
excess (~9x memcpy) is removable by structure; the bytes-per-call excess (2x) needs the
u8-plane conversion.

## Caveats

* Counts at `profiling` opt level — release codegen, but the binary differs from the
  band binary (debug info). Call structure is the same; absolute Ir differs slightly.
* `calls=` counts edges that exist; summed `calls` for a callee = total invocations.
  Inlined Rust callees are absent (port intra predictors), C's dispatched callees are
  present — asymmetric rows are annotated, not missing.
* Arc Ir under a `cfn=` is inclusive cost of that edge; used here for weight, not proof.
* One box, one cell, `--cpu-used 3`. The 196² spot-check reproduced the same structure
  (memcpy 8.28x, memset 2.82x, Ir 2.31x) — the mechanism is not size-regime-dependent.

## Third finding (the important one): restoration is a per-call KERNEL gap, not a count gap

The biggest Ir divergence the count diff surfaced is the restoration family —
**never on the lever map**. Search-kernel call counts match C exactly
(`pixel_proj_error` 2,002 = C's 2,002; `calc_proj_params` 254 = C's 254), so the
excess is per-call implementation, not excess work:

| kernel (1024², per call) | port | C AVX2 | ratio |
|---|---:|---:|---:|
| `pixel_proj_error` (generated magetypes v3) | 721,342 Ir | 114,791 | 6.3x |
| `calc_proj_params` (pure scalar i64 loop) | 1,930,589 | 256,736 | 7.5x |
| wiener apply (`wiener_pass_madd_v3`, H and V separate) | 30,048 × 51,744 calls | 11,942 × 26,716 | ~2x calls + 2.5x/call — C fuses both passes in `av1_wiener_convolve_add_src` |

Two lessons landed:

1. **A first-pass "better AVX2" kernel (i32 lanes, i64-lane accumulate, safe-load
   windows) only recovered ~8 %** — LLVM had already auto-vectorised the
   magetypes body's scalar widenings/squares inside the target-feature region.
   The real gap is ALGORITHMIC: C's `av1_lowbd_pixel_proj_error_avx2` processes
   16 px/iter in **i16 lanes** with `vpmaddwd` pair tricks (`[xq0,xq1]` against
   interleaved `f1,f2` computes `xq0·f1 + xq1·f2` in one instruction; the
   single-filter cases fold `u = d<<4` into the coefficient `−xq<<4`), packs
   `flt` i32→i16 (justified by C's own `|flt| < 2^15` assert), and widens the
   i32 row accumulator to i64 once per ROW.
2. Transcribing that structure (plus `as_chunks` row iteration to remove the
   per-window bounds-check ceremony under `forbid(unsafe_code)`) measured at
   196²: `pixel_proj_error` 458k → **162k Ir/call** (2.8x), `calc_proj_params`
   1.14M → **572k Ir/call** (2x, C's `mul_epi32` even/odd i64-product scheme —
   no input bound needed, exact for all i32 lanes).

## Data

`/tmp/cg_1024_port.out`, `/tmp/cg_1024_c.out`, `/tmp/cg_test_{port,c}.out` (196²);
`scripts/callgrind_counts.py` (new). The copy-count lever was run to ground:
the in-place conversion + winner-swap + grow-only buffers landed (byte-identical
on all gates) and the 24-round band measured it a **null** (+0.082 %, p=0.54)
— the removed calls were temporally small (~2 % of Ir) and ~850k of them moved
into `dist_block_px_domain_into`'s now-strided (C-faithful) row reads rather
than disappearing. The lever map's memset+memmove row is therefore NOT the
+96.5 ms it suggested: counts were structurally excess but the work was cheap.

## Fourth finding: `quantize_b` — i32-domain SIMD, not C's saturating kernel

Port `aom_quantize_b_no_qmatrix` was the single largest per-call gap the count
diff surfaced: **2,390 Ir/call vs C's `aom_quantize_b_avx2` at 149** (~16x,
1.29B Ir of the 115.7B encode). C's kernel runs saturating i16 lanes — which
diverges from C's own scalar on `|coeff| > i16::MAX` — while this port's
contract is the C scalar on the full i32 domain (`quantize_b_diff` fuzzes
|coeff| < 2^19). The landed kernel keeps i32 lanes and widens only where the
scalar's i64 intermediates need it:

- `clamped·quant ≤ 2^30` stays in `mullo` after reassociating
  `((clamped*32)*quant)>>16` → `(clamped*quant)>>11`;
- `t·quant_shift ≤ 2^36` is the one `mul_epi32` even/odd widen — odd lanes are
  always AC class (the DC lane is even), so the odd product uses the all-AC
  splat — the one bug the fuzz caught before landing;
- `srl_epi64` suffices for the arithmetic shift because the scalar keeps only
  the low 32 bits (`as i32`);
- `eob` = max of `iscan+1` under `tmp32 != 0`, order-free.

**2,469 → 467 Ir/call; banded −1.522 % (24/24, p=1.19e-7, null +0.074 %)**
on the shipping cell; byte-identical on 9 cells and through the 240k-case
real-C fuzz.

## Fifth: `highbd_variance64` — C's i16-lane madd shape, w=4 absorbed

The port's planes are u16 even at bd8, so it must run C's *highbd* variance
shape (`highbd_variance_sse4.c`) rather than the u8 `aom_variance4x4_sse2` —
but the magetypes body ran i32 lanes with a per-lane scalar `widen` and
per-row `reduce_add` pairs, and routed w=4 (the hottest size — 1.5M calls at
1024² via `dist_block_px_domain`) to scalar. The v3 body now runs i16 lanes
with `madd_epi16(d, ones)`/`madd_epi16(d, d)` pair-sums, packs FOUR rows per
ymm at w=4 (zero padding diffs to 0 — contributes nothing), takes an 8-lane
xmm tail at w%16==8, and does one `hadd` pair-reduce per row-group.

`highbd_variance` inclusive Ir at 196²: **88.5M → 53.1M (−40 %)**; the SIMD
call count rose 80k→146k as w=4 calls moved off scalar. Bit-exactness: i16
diffs are exact on the pixel domain (a,b < 1<<bd ≤ 4096 — the bound the
module doc already claims and the SIMD differential tests at every tier).
Band: **NULL — +0.241 %, p=0.064 (7/24 faster), null +0.078 %** at the
shipping cell. Third instance of count-vs-cost this cycle: the removed work
was ALU-cheap and pipelined for free. Kept as the structurally right shape
(bit-exact, lower Ir), counted as a measured null, not a lever.

## Sixth: trellis inlining + dead-init memset batch

Two landings off the post-variance re-profile (callgrind at 196² and 1024²).

**Trellis/txb inlining — −1.324 % (24/24, p=1.19e-7, null +0.016 %).** C's
per-coefficient cost helpers are `static INLINE` in `txb_rdopt_utils.h`; the
port had them as 8–13-arg non-inlined calls plus a `&dyn Fn` dequant callback
that blocked `get_dqv` inlining. `#[inline]`/`#[inline(always)]` on the
mid-level cost fns and passing `dequant`/`iqmatrix` directly into
`update_coeff_general` cut the txb family's Ir 839M → 627M at 196² (−25 %)
and measured −1.324 % on the shipping cell. The port still calls
`optimize_txb` ~2.5x more often than C (547k vs 217k at 196²) — that is
search-breadth, a different question, not touched here.

**Dead-init memset batch — −0.851 % (21/24, p=2.8e-4, null −0.076 %).**
memset was the #2 self-Ir symbol at 1024² (3.7B) and the port calls it 5.7x
more often than C. Three call sites were provably dead — buffers Rust
zero-initialises that C leaves uninitialized because the first pass writes
every byte anything later reads:

- `optimize_txb`'s `levels: [u8; TX_PAD_2D]` (1,312 B/call): moved to
  `XformQuantScratch.levels` via new `optimize_txb{,_qm}_scratch` entries;
  `txb_init_levels` covers every read position.
- `fwd_txfm2d_core`'s `buf.clear()`: the col pass writes all `col_n*row_n`
  before the row pass reads (batch gates accept only full-coverage shapes).
- `av1_inv_txfm2d{,_add}_into`'s `buf.clear()` ×2: same argument through the
  row pass. (`remap_input` KEPT its clear — its uncopied tail IS read.)

memset calls 3.50M → 2.94M at 196².

**Two traps worth recording.** (a) Passing `levels` as `&mut [u8]` instead of
`&mut [u8; TX_PAD_2D]` erased the fixed-extent knowledge and reintroduced
per-access bounds checks — net WORSE on Ir (+65M). Slice-to-array params
matter for hot indexed buffers. (b) A `thread_local!` `RefCell` edge-scratch
pool for the directional predictor's `[u16; 160]`×2 arrays cost ~+70M Ir —
`LocalKey::with` + `borrow_mut` + the now non-inlined 17-arg call boundary
exceed the 640-byte memset it removed. Reverted; the edge arrays need
explicit scratch threading through the six call sites' env structs if they
are ever worth it.

The `quantize_fp` iscan-widen attempt (`i16x16::from_slice` + `widen_low`)
was an Ir wash — LLVM already recognises the `from_fn` per-lane extension
pattern. Reverted; `quantize_fp` remains ~250M self-Ir at 196² / 4.3B at
1024² — its gap is now structural (per-call setup on small `n`, and 2.5x C's
call count), not per-chunk.

## Fused 4x4 forward kernel (`av1_lowbd_fwd_txfm2d_4x4_sse2`) — LANDED −1.324 %

The 4x4 forward transform paid EIGHT scalar kernel calls per block: the fused
scalar path (`fwd_txfm2d_4x4_fused`) loops 4 columns + 4 rows through
`av1_fdct4`/`av1_fadst4`/`fidtx4`, ~2.3M scalar kernel calls at 196². C's
`av1_lowbd_fwd_txfm2d_4x4_sse2` does the whole block in ~120 SSE2 instructions.

`try_fwd_txfm2d_4x4_fused` (`transform/simd/mod.rs`) transcribes that kernel
verbatim — i16 `madd` pairs built by `pair_set_epi16`, the three 4-point
kernels (DCT/ADST/IDTX — the port's `Fwd1dI16` has no `fadst4`, but THIS kernel
is not bound by that table: it is a whole-block transcription whose safety is
its own runtime gate), `transpose_16bit_4x4`, lr/ud flips, sign-extend store
into column-major `output[c*4+r]`. Gate: `max_abs_i16_strided(input,stride,4,4)
<= 512`; over-bound input declines to the existing scalar fused path. At 196²
307,955 calls, zero declines; `av1_fdct4`/`fadst4` vanish from the encode
profile; whole-cell Ir 9.74B → 9.52B (−2.2 %).

Band (24 rounds, rotated arms, same-binary null): **−1.324 % (24/24,
p=1.19e-7, null +0.094 %)**, 40,237 B on the shipping cell; the two pinned
divergent cells (256² cq20 s0, 1024² cq35 s9) produce byte-identical output to
the pre-change base — divergences are pre-existing near-ties, not this change.
`txfm2d_simd_perm_diff` gained a ±512 gate-edge arm — the bd8 arm never
reached the gate band and full-i16 arms always declined, so the accept-side
edge was previously undriven.

## i16-in-fused-transforms via dispatcher decline — REJECTED

The scoped restoration of i16 passes for `fwd_16x16_fused`,
`fwd_rect816_fused`, `inv_16x16_fused`, `inv_rect816_fused` was implemented as
"decline to the generic driver when its i16 gates fire". Callgrind showed the
i16 paths did fire (`fwd_row_pass_i16` 51M, `inv_row_pass_i16` 57.8M Ir) — but
the band never needed to finish: every round read ~+85 ms WORSE (~+3 %).
Mechanism: the generic driver pays an extra i32 intermediate buffer round-trip
and per-pass setup the fused kernels avoid, and the forward side ran
`fwd_col_i16_applies`'s input scan twice (dispatcher + `try_fwd_col_pass`).
Reverted. The remaining form of this lever is fusing i16 lanes INTO the fused
kernels themselves — real work per shape, unpriced.

## Fused 4x4 inverse kernel (counterpart of `lowbd_inv_txfm2d_add_4x4_ssse3`) — LANDED −0.594 %

Symmetric to the forward landing: the 4x4 inverse ran EIGHT scalar kernel
calls per block (4 row + 4 col through `av1_idct4`/`av1_iadst4`/
`av1_iidentity4`), ~1.6M scalar calls at 196². `try_inv_txfm2d_4x4_fused`
transcribes C's w4 kernel shape (`idct4_w4`/`iadst4_w4`/`iidentity4` madd
structure) adapted to the port's column-major input (`input[c*4+r]` — the
contiguous loads are columns, not rows as in C's row-major lowbd input) and
u16 destination (`highbd_clip_pixel_add`, not C's u8 packus store).

**The gate is the durable lesson.** The first version declined only on stage
bounds and FAILED the permutation differential (SIMD 255 vs scalar 203 on a
±2^20 fuzz cell): the port's scalar inverse keeps `step` intermediates in i32
(and `av1_iadst4` in i64, no stage clamps at all), while the i16 transcription
saturates in `packs_epi32`/`adds_epi16` wherever an intermediate exceeds
±32767 — legal for inputs as small as |in| ~ 12k. C's own lowbd kernel has
the same saturation, and gets away with it because the decode-path contract
bounds coefficients; the port's u16 path promises C-scalar behaviour on
arbitrary i32 input. Fix: a runtime input bound `max_abs(input[..16]) <= 4096`
that keeps EVERY intermediate inside i16 through both passes (row outputs ≤
~11.2k, col outputs ≤ ~30.3k, iadst4 i32 accumulators ≤ ~1.2e8), making every
`packs`/`adds`/`subs`/`mulhrs` lossless. Over-bound blocks decline to the
scalar fused path.

At 196² cq27 s3: `av1_idct4` 104M + `av1_iadst4` 43.5M + `av1_iidentity4`
2.5M self-Ir eliminated, ZERO declines on encoder-reachable input;
whole-cell Ir 9.523B → 9.344B (−1.9 %). The dispatcher's 16-element
`unsigned_abs` scan costs ~2M Ir — negligible against the ~150M it removes.

Band (24 rounds, rotated arms, same-binary null): **−0.594 % (23/24,
p=2.98e-6, null +0.163 %)**, all arms 40,237 B. The permutation differential
gained a ±4096 gate-edge arm (`coeff_gate` + `inv_spike_gate`: exact bound,
bound+1 decline, partial-block) — the ±2^20 inverse arm had been declining
EVERY 4x4 block, so the accept side of the inverse gate was undriven until
now. The temporary `probe_inv4x4` debug probe was removed; the gate arm
subsumes it.

## `levels: &[u8]` -> `&[u8; TX_PAD_2D]` across the txb context readers — REVERTED

Converting the `levels` params (`get_nz_mag`, `get_br_ctx`,
`get_lower_levels_ctx{,_general}`, `get_nz_map_contexts`,
`two_coeff_cost_simple`, `coeff_cost_eob`, `coeff_cost_general`,
`txb_init_levels{,_scalar}` + SIMD impls) to the array type removes every
bounds check in principle — but Callgrind measured **−2.4M Ir on 9.34B
(−0.026 %)**: LLVM was already eliding every check via inlining context. No
band run; a sub-null Ir delta cannot clear it. Reverted — the change touched
three `pub` signatures for a type-level nicety, not a measurement.

## Forward 8x8 whole-block kernel on i16 lanes — `av1_lowbd_fwd_txfm2d_8x8_sse2` — LANDED −0.951 %

The existing `fwd_8x8_fused` was already fused but ran i32 lanes (every
butterfly widens/narrows); C's whole-block kernel never leaves i16. The new
`fwd_8x8_fused_i16` (`transform/simd/mod.rs`) is the verbatim C dataflow:
`fdct8x8_new_sse2` / `fadst8x8_new_sse2` / `fidentity8x8_new_sse2` over the
`btf_16_sse2` madd butterfly, `round_shift_16bit` for both shifts, one
in-register `transpose_16bit_8x8`, `store_buffer_16bit_to_32bit_w8`
(sign-extend) into the port's column-major `output[c*8+r]`. Lane/register
conventions identical to the i32 form: register = source row pre-transpose,
output column post-transpose; `ud_flip` reverses load order, `lr_flip`
reverses registers after the transpose.

Gate: `max|input| <= 511`, derived by exhaustive sign-vertex simulation of
the exact saturating kernel (every stage is monotone in the inputs, so
vertices bound it): pass-1 outputs <= 11,563 (fdct8 binding), `round_shift`'s
`adds(_,1)` cannot saturate, pass-2 inputs <= 5,782 < fdct8's 5,792
no-saturation input bound (fadst8: 6,420; idtx8 trivially safe). bd8
residuals are <= 255 — every encoder block accepts; out-of-range callers
decline to the i32 fused path, which is always correct. Two gate mechanics
traps: `_mm_abs_epi16(-32768)` yields 0x8000, invisible to SIGNED `cmpgt` —
the accumulate must be `max_epu16` and the test `subs_epu16` + `ptest`
(`_mm_movemask_epi8` under magetypes returned 0 on a nonzero vector —
unverified wrapper, avoided).

At 196² cq27 s3: 392,620 kernel invocations (196,310 transforms x 2 passes),
77.8M Ir; `run_fwd1d` self-Ir 389M -> 276M; whole-cell 9.344B -> 9.255B
(−0.95 %). Band 24 rounds rotated arms + same-binary null: **−0.951 %
(24/24, p=1.19e-7, null +0.011 %)**, all arms 40,237 B. The perm
differential's gate arm now covers ±511 (accept edge for the 8x8) alongside
the 4x4's ±512/±513.

Next of the same shape: `fwd_rect48_fused` (4x8/8x4, ~128k calls / 49M Ir
plus its `run_fwd1d` share) — same [2,-1,0] shift row, reuses `Fwd4` row
kernels + the 8-pt w4 col kernels (`fdct8x4_new_sse2` family) and the rect
sqrt2 store.

## Forward 4x8/8x4 whole-block kernels on i16 lanes — `av1_lowbd_fwd_txfm2d_{4x8,8x4}_sse2` — LANDED −1.343 %

Same construction as the 8x8 landing, mixed widths. One kernel
(`fwd_rect48_fused_i16`, `transform/simd/mod.rs`) handles both shapes: the
4x8 runs the 8-pt column set on four-lane registers — `fdct4x8_new_sse2` /
`fadst4x8_new_sse2` / `fidentity8x8_new_sse2` over `btf_16_w4_sse2`
(`unpacklo` only; `packs(c0,c0)` / `packs(d0,c0)`, don't-care high halves
verbatim) — `transpose_16bit_4x8`, then the 4-pt row set full-width
(`fdct8x4_new_sse2` / `fadst8x4_new_sse2` / `fidentity8x4_new_sse2`). The
8x4 mirrors: 4-pt column pass, `transpose_16bit_8x4` (C runs the padded 8x8
transpose; zeroing the dead registers is equivalent and deterministic),
8-pt w4 row pass. Both end in the rect-scaled store — `scale_round` by
`NewSqrt2` = the scalar's `mul_rshiftv(_, NEW_SQRT2, NEW_SQRT2_BITS)` —
into `output[c*row_n+r]`.

Gate: `max|input| <= 1023`, derived by the sign-vertex simulation over BOTH
pipeline orders and all six kernel kinds. Binding case is any 8-pt pass
feeding `fdct4` (or `fdct4` feeding `fdct8`): pass-1 out <= 23,149 at
4x1023 -> pass-2 in <= 11,575 < `fdct4`'s 11,584 no-saturation input bound.
`fadst8x4`'s `in7` term is a WRAPPING `add_epi16` (not `adds`) — it diverges
from the scalar i32 add past |in| ~16,383, so the wrap is also inside the
proven envelope. bd8 residuals always accept; declines take the i32 fused
path. Perm differential gained ±1023/±1024 edge spikes.

Bug found by callgrind, not by tests — and worth keeping as a pattern: the
w4 store sliced `o[4..]` unconditionally, so `try_from` failed and EVERY
8x4 call declined (54 % of rect48 transforms) while output stayed
byte-exact — a fail-closed gate can never produce a byte diff, so
accept-side coverage must be counted (`calls=` arcs), not just diffed.
After the fix: 130,982/130,982 accepts at 196² cq32 s0, zero i32 fallbacks;
whole-cell 12.439B -> 12.350B Ir/iter (−89.4M); `run_fwd1d` 790.6M ->
731.2M self-Ir, `fwd_rect48_fused_v3` eliminated (399.3M -> 0 self-Ir at
that cell). Band 24 rounds rotated arms + same-binary null: **−1.343 %
(24/24, p=1.19e-7, null −0.133 %, p=0.54)**, all arms 40,237 B.

Next of the same shape: 16x16 and 8x16/16x8 (`fwd_16x16_fused`,
`fwd_rect816_fused` — ~131M + ~110M self-Ir plus their `run_fwd1d` shares)
need 16-pt kernels (`fdct16x16_new_sse2` family, w16 butterflies via two
regs per index) — bigger transcription, same gate derivation.
