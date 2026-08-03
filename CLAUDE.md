# aom-rs — project instructions & durable bug log

Pure-Rust, **bit-exact** reimplementation of libaom ≥ v3.14.1 as a drop-in replacement.
Validated behind differential harnesses against the REAL exported C functions (priority of
evidence: real exported C fn > synthetic-facade-over-real-fn > verbatim transcription —
transcribed oracles can carry shared bugs).

**Module-progress source of truth:** `STATUS.md` (updated per landing by the track agents).
**This file** holds project-level coordination rules + the durable **Known Bugs** log.

## Gates (definition of done)

- **Gate 1 — Decoder:** bit-identical to C across the AV1 conformance corpus (intra scope
  wired in CI: `xtask/conformance.py --fetch --scope intra`; gate = byte-identity + golden MD5).
  **Scope caveat, MEASURED 2026-07-30** (`benchmarks/decoder_corpus_feature_tuples_2026-07-30.tsv`,
  every vector's frame 0 parsed): the intra corpus is a deep sweep of ONE sequence shape —
  **233/235 are 4:2:0 and the other 2 are monochrome** (the subsampling flags read
  4:2:0 on all 235, but 2 carry `mono=1` and so have no chroma planes at all),
  bd8 (169) or bd10 (66), 230/235 SB128, and ZERO carry superres,
  tiles>1, QM, segmentation, `reduced_tx_set`, `disable_cdf_update`, `delta_lf_present`,
  4:2:2, 4:4:4 or 12-bit. Those axes are covered by the port-generated gates instead
  (`real_bitstream`, `config_permutations_decode`), NOT by conformance. Do not read "the
  conformance corpus passes" as breadth across the format.
- **Gate 2 — Encoder:** bitstream bit-identical for every `--cpu-used 0..9`.
  **Scope caveat, VERIFIED 2026-07-30** (`docs/CONFIG_AXIS_INVENTORY_2026-07-30.md`): the port
  never AUTHORS a sequence header — `write_sequence_header_obu`
  (`aom-dsp/src/entropy/header.rs:1046`) has zero call sites in any `crates/*/src` (16 in
  tests), and `SequenceHeaderObu` has no `Default`. Every encoder path parses a seq header
  out of a real aomenc bootstrap stream and emits only an `OBU_FRAME`. So bit depth,
  monochrome, subsampling, profile, SB size and every seq-level `enable_*` bit are
  REPLAYED, not derived: gates asserting "the seq bit equals the knob" are agreement checks
  against libaom's bits, not evidence the port can produce that configuration.
- **Gate 3 — Performance:** user-set acceptance bar ≤ 1.5× C (2026-07-20 directive).
  **Met at the 4K headline cells** (≈1.22× cq20 / ≈1.19× cq40 wall after the bd8 lowbd +
  i16-rows + CDEF find_dir landings); 2K and small-frame cells still exceed it
  (1.66–1.9× at 2K, up to ~2.4× on tiny/entropy-dominated cells) — see
  `benchmarks/gate3_peak_wall_2026-07-25.md` for the committed run, caveats, and the
  ranked remaining levers. The original ≤ 1.20× figure is retained as the stretch target.
- **Gate 4 — Coverage checklist** (+ a zenavif integration gate).

Primary configuration: ALLINTRA (usage=2), speed-0 KEY frame. **Single-frame (KEY-frame)
work must reach byte-exactness across BOTH tracks before inter-frame ("the rest") starts.**

## Decoder desyncs: use the instrumented libaom decoder FIRST (do not bisect by hand)

An arithmetic-decoder desync almost never shows up where it is caused. The cheapest way to
localize one is to compare against the REAL libaom decoder's own instrumentation, which dumps
both C's per-block mode info and C's exact per-symbol sequence. **A build already exists at
`/root/aom-inspect/examples/inspect`** (`CONFIG_INSPECTION=1 CONFIG_ACCOUNTING=1`; rebuild the
same way if it is ever lost):

```
/root/aom-inspect/examples/inspect --limit=<N> -bs -ts -m -r -mm <vector>.ivf   # per-MI block grid
/root/aom-inspect/examples/inspect --limit=<N> -a          <vector>.ivf         # per-symbol accounting
```

The block dump is per-MI (replicated over each block's footprint), so collapse it to block
top-lefts to get C's block list; diff that against the port's walk to find the FIRST structural
divergence. Then read the accounting around that block: it gives every `aom_read_symbol` tagged
by its reading function, so **"the port read the same symbol VALUES but off a different CDF row"**
is directly visible instead of being inferred. That failure mode (a probability-only drift that
desyncs several reads later, with everything looking correct locally) is the dominant one on this
codebase — it is the KB-6 signature on the encoder side and was BOTH inter-decode roots found on
2026-07-19.

Two gotchas, both verified:
- **The JSON emitter drops the first symbol at every new block position** (`put_accounting`,
  examples/inspect.c: on a context change it emits `[x,y]` *instead of* that symbol's triple). So
  `read_skip_txfm` — the first read of most blocks — never appears. Do not conclude a symbol is
  unread; infer it from what follows (e.g. no coeff/var-tx symbols at all ⇒ that block is skip).
- Entries are `[id, bits, samples]` against `symbolsMap`, **not** `[id, value, bits]`; `samples`
  is the number of `aom_read_symbol` calls aggregated at that (tag, position). Reconciling those
  counts against the port's expected read sequence is itself a strong check — e.g. inter-intra:
  40 allowed blocks ⇒ 40 flag reads, +2 wedge flags +1 wedge index = the 43 C records.

## Coverage queue — the named-but-unmeasured axes, ranked (2026-08-03)

**Why this section exists.** Between 2026-07-30 and 08-03 every closed KB was found by
measuring an axis nobody had measured, and every one of those landings NAMED the axes it did
not measure — inside its own KB entry, or in a gate file's doc comment, or in a `Still
unmeasured` bullet. That list is a work queue with a demonstrated hit rate, and it was
scattered across ~15 places. This is the consolidated version. **Keep it current: when you
close an axis, strike it here as well as in the KB entry; when a landing names a new one, add
it here in the same commit.**

**Ranking basis (two factors, in this order).**
1. **Reachability** — a REFUSAL or PANIC on a configuration reachable by default or by one CLI
   flag outranks a byte divergence, which outranks a pinned near-tie. "Reachable" is a
   measurement, not a guess: state the predicate and the sweep that establishes it (KB-34's
   reachability note was wrong by four orders of magnitude in area).
2. **Blast radius** — how much of the encode envelope is wrong when it fires, and how many
   INDEPENDENT landings named the same axis (a proxy for how load-bearing it is).

### Closed on 2026-08-03 (kept for one cycle so the strike-through is visible)
- ~~the screen-content palette arm at `--cpu-used` 8/9~~ → KB-35. Was a FALSE refusal on a
  plain gradient; 22 of 25 measured PANIC rows are byte-identical, 3 are the genuine arm.
- ~~above 1280x720 at speed >= 1~~ → KB-36. Found an unmodelled `is_1080p_or_larger` arm.
- ~~the crop axis at 4:2:2/4:4:4/mono, SB128, bd10, and the 480/720 straddle at cpu 0~~ →
  `s4cov_crop_format_axis.rs`, 58/62 byte-exact; the 4 that are not are KB-27's, proven by
  SB-exact controls.

### T1 — refusals on reachable configurations
| axis | reach (measured) | cost to close |
|---|---|---|
| **lossless (`--cq-level 0`) at ANY speed** | Refused by the **harness**, not the encoder: `aom-bench/src/lib.rs:1153`, measured PANIC at cpu 0/4/8/9 on a 64x64 cell while cq1 is byte-exact at all four. KB-34's "lossless TX_4X4 arm still refused at cpu 8/9" therefore understates it — the e2e path is closed at every speed, and lossless parity rests on KB-5's own driver, which is `let speed = 0i32`. | harness work first (thread a lossless bootstrap through `port_encode_full`), then the encoder's `block_yrd_lowbd`/`_hbd` TX_4X4 arms |
| **multi-tile x `--deltaq-mode` 2/3** | Loud refusal (KB-31 residual a). Needs a frame that REQUIRES a tile split (>4096 px wide, or ~9.44 MP) AND a non-default flag, so reachable but not by default. NOT re-measured 2026-08-03. | `pack_tile`'s running qindex base must carry across tiles instead of restarting |
| **IntraBC blocks > 64x64 (multi-chunk)** | KB-29 residual (a): unreachable from libaom AND from SVT-AV1 (re-checked 2026-08-02, 1,992 streams). A third encoder or a synthetic stream is needed. | unknown; blocked on producing an input |

### T2 — default-reachable axes, no refusal, cheap-to-moderate
| axis | why it matters | cost |
|---|---|---|
| **crops straddling `is_4k_or_larger` (2160)** | KB-28's root at KB-19's boundary; the predicate to check is `default_min_partition_size`'s BLOCK_8X8 arm at e.g. 2154x2160 vs 2160x2160 | ~25-30 min of port encode (KB-19's 2160x2160 speed-0 cell is C ~26 s / port ~195 s) |
| **1440..2160 at speed >= 1** | KB-36 closed 1080..1440; above 1440 only KB-19's single 2160p **speed-0** cell exists | ~20-30 min |
| **the >=1080p band at bd10/12, 4:2:2/4:4:4/mono, SB128, and quantizers other than cq24** | KB-36 swept bd8 4:2:0 SB64 cq24 only | ~10-20 min per format |
| **partial-SB x bd12, and x 4:2:2/mono at high bit depth** | `s4cov_partial_sb_axis.rs` residual; the bd10 arm exists and runs only at cpu {0, 7} | ~2-5 min each, no new machinery |
| **multi-tile at SB128 / bd10-12 / 4:4:4-4:2:2 / mono** | KB-31 residual (c): that whole file is bd8 4:2:0 SB64 | moderate; large frames |
| **multi-tile x the crop straddle** | needs its OWN crop pair at tile-forcing size (e.g. 4090x2154 vs 4096x2160) — it cannot be combined with a 714x720 frame | same cost class as the 2160 arm |

### T3 — blocked on harness work
| axis | blocker |
|---|---|
| **`--dist-metric=qm-psnr` x speed >= 4** | Named twice by KB-21. The C shim already takes `cpu_used` and `dist_metric`; the PORT side is `aom-encode/tests/encoder_gate_tune_iq_e2e.rs`, a hand-built pipeline that hardcodes `let speed = 0i32`, builds `SpeedFeatures::set_allintra(0, ..)` framesize-blind, and passes `uv_lp: speed0_allintra()` / `fs_sf: Default::default()`. Threading speed through it means reproducing the winner-mode two-pass and the framesize/qindex passes — i.e. the KB-26 and KB-22 machinery. Estimate 2-4 h, with a real risk of building a harness that diverges for harness reasons. |

### T4 — measured, pinned, unlocalized (byte divergences, no refusal)
| pin | shape | next step |
|---|---|---|
| `HBD_OPEN` / `b10_64` (`s4cov_qm_axis.rs`, `config_permutations.rs`) | bd10 AND bd12, `--cpu-used` 1..6, LUMA-borne, reaches 4:4:4 + mono, qindex-dependent speed reach | `tx_sf.prune_tx_size_level` was the obvious suspect and is **RULED OUT** 2026-08-03 (INTER-only by assertion, tx_search.c:3438) — see KB-36's audit bullet |
| KB-30 | `cid22_6292444` at cpu6, every quantizer, 1 of 10 real photographs | sibling-C per-block dump (playbook §10); ~half a day |
| `RD_BAND_OPEN` (`kb34_nonsquare_nonrd_leaf.rs`) | 1272x724 cq24 at cpu 2-5 only | adjacent to KB-28's band at a size no gate covers |
| the 17 cpu-8 photographic high-q rows | `benchmarks/nonsquare_leaf_reach_2026-08-02.tsv`, cq 32-63, -24..+13 B, not on in-repo content | needs the content in-repo first |
| `PALETTE_ON_SPEED8_OPEN` (`kb35_nonrd_palette_arm.rs`) | palette ON x cpu8 x screen-detected, 13 rows, -1399..+817 B; the FULL-RD palette leaf, which `rd_close_palette.rs` never crosses (speed 0 throughout) | first-divergent-block on 128x128 cq12 (delta -1) |
| KB-27 / `MONO_S0_OPEN` | monochrome, base_qindex 96, speed 0; **size-independent** (64x64 through 720x720, SB-exact and partial alike, widened 2026-08-03) | its own localizer is in `s4cov_partial_sb_axis.rs` |
| KB-P29, KB-13's 2 cpu3-cq63 cells | 2 palette 128<sup>2</sup> near-ties; 2 real-content cells | sibling-C RD dump (KB-3/KB-7 method) |

**Already closed, do not re-open from stale notes:** KB-10/KB-11's noise-cq63 speed-6/7 pairs
(closed by KB-21 root #2), the whole cpu-4/5 fragile band, and KB-12's estimate-arm residual
(closed 2026-08-02 — it was `aom_hadamard_lp_8x8`'s missing transpose, not a near-tie).

## Known Bugs

Record real bugs here immediately with file:line refs (survives context loss). Do NOT close
an entry by relaxing/excluding a test — only by a landed fix verified on `origin/main`.

### KB-1 — Decoder: recon divergence at base_qindex ≥ 249 (quantizer-62/-63) — REAL CORRUPTION, CI-quarantined
- **Symptom:** decoded RECON diverges from the C oracle at `base_qindex >= 249` — the
  `quantizer-62` / `quantizer-63` conformance vectors. Reproduces at **bd8 AND bd10, luma AND
  chroma**. Divergence is an edge-local ±1 prediction cascade.
- **Root cause (CONFIRMED via isolated C-decoder instrumentation):** NOT an entropy/coeff-value
  bug. The first 311 txb records dump byte-identical (plane, tx, eob, dc_sign_ctx, txb_skip_ctx,
  levels ALL match) — the per-txb entropy decoder + context maintenance are FAITHFUL. The bug is
  the **txb ITERATION ORDER for coding blocks >64×64**: C (`decodeframe.c:929-962`,
  `decode_token_recon_block` intra path) chunks each block into BLOCK_64X64 units and within each
  chunk iterates planes→txbs → **L,U,V interleaved per 64-unit**; the port iterates each plane
  across the WHOLE block (all luma txbs, then all chroma) in `aom-decode/src/lib.rs` (~2235 luma
  loop + separate chroma loop). Identical for ≤64×64 blocks; for 128-sized blocks it desyncs the
  arithmetic decoder and everything cascades (the "edge-local ±1" symptom). Only q62/q63 pick
  partitions >64×64 (flat high-q blocks) → exact q61→q62 threshold. **Fix:** wrap luma+chroma
  reconstruction in the outer 64×64-chunk loop, plane-interleaved, matching C.
  (Earlier "entropy coefficient-decode path" localization was one layer too low.)
- **Fix #1 (VERIFIED, awaiting workspace-compile to land):** the reorder is implemented in
  `aom-decode/src/lib.rs` and proven — b10-q63 now byte-matches C and the port's 328 KEY-frame
  txb reads are byte-identical (up from the record-311 desync). The reorder is correct.
- **Bug #2 = CDEF per-unit strength stamping for >64 blocks (ROOT CAUSE CORRECTED — NOT intra-pred).**
  Exposed by fix #1; b8-q62 / b8-q63 / b10-q62 failed edge-local ±1 (b10-q63 clean). Intra-pred was
  DISPROVEN: the port's predict params for the failing 2nd 64×64 unit match C exactly (DC_PRED,
  n_top=64, n_left=32) and the DC math + left-column extension match C's `build_intra_predictors`
  line-for-line — pred+residual reconstruct the unit correctly. The scattered ±1 across a whole
  64×64 unit is CDEF's signature. C reads the CDEF strength once per 64×64 unit and stores it on the
  block's SHARED MB_MODE_INFO (`decodemv.c` read_cdef, stamped at the unit top-left mi); the frame
  walk reads it back per 64×64 unit top-left mi (`cdef.c:304`). A >64 block shares ONE mbmi across
  all its mi cells, so every covered 64×64 unit reads the same strength. The port
  (`aom-decode/src/frame.rs:1212`) stamped only the block's TOP-LEFT unit → other covered units
  stayed at −1 (CDEF skipped); for the 128-wide mi64,0 the 2nd unit (mi64,16) kept −1 so CDEF ran
  in C but not the port → the ±1. **Fix #2:** stamp `b.info.cdef_strength` on ALL 64×64 units the
  block covers (in-frame h×w extent); sub-64 blocks cover one unit, unchanged. Both bugs are
  >64-only, which is why exactly q62/q63 fail (only very high qindex picks >64 partitions).
- **Fix #1 + #2 VERIFIED GREEN (landing in one commit):** full conformance gate 269 in-scope frames,
  0 failures, WITH q60–q63 present; all four targets (b8/b10 × quantizer-62/63) byte-exact + golden
  MD5, plus 60/61 and everything else (allintra/size/intrabc/cdfupdate...), no ≤64 regression. The
  landing commit reverts the ci.yml q62/q63 rm, adds an explicit q62/q63 × bd8/bd10 regression test,
  and deletes the throwaway scratch. #21 closes only after: on origin, CI green WITH q62/63 restored,
  `merge-base --is-ancestor` confirmed.
- **Encoder cross-check — RESOLVED ✅ 2026-07-18 (SB128 encode landing, PARITY C8).** The encoder
  pack now writes txbs in the C 64×64-chunk **L/U/V-interleaved** order for >64 blocks
  (`av1_write_intra_coeffs_mb`, encodetxb.c:431-472) and the search/re-encode predict+reconstruct
  in the matching `av1_foreach_transformed_block_in_plane` mu-64 chunk order (encodemb.c:560-582).
  Byte-exact vs real `aomenc --sb-size=128` (`aom-bench/tests/sb128_e2e.rs::sb128_coded_128_leaf
  _e2e` — a smooth diag ramp at 256² cq55/cq63, anti-vacuity-checked to actually code a 128-level
  partition; the photographic `quantizer-00` crops in `sb128_natural_e2e` split to ≤64 even at
  cq63, so they exercise only the SEARCH's 128-NONE evaluation, not the pack interleave). NOTE the
  earlier "256×256 cq63 is evidence the
  order is correct" claim was **vacuous** — a 256² SB64 frame has no >64 coding blocks, so the
  >64-block order was never exercised until this landing (the plane-sequential pack was
  coincidentally correct only because ≤64 leaves are a single chunk).
- **CI status (TEMPORARY quarantine):** `.github/workflows/ci.yml:63-64` `rm`s the q62/q63
  vectors after fetch so Gate-1 goes green on the rest. This is a **must-fix corruption bug**
  under the zero-tolerance rule (wrong pixels are a shipping bug, never a known limitation),
  NOT an accepted limitation. The `rm` MUST be reverted in the same PR that lands the fix, and
  the specific q62/q63 vector(s) added as an explicit strong byte-identity case.
- **Tracking:** task **#21** (HIGH). Fix unblock: authorized throwaway reference-*decoder*
  instrumentation to dump the C coefficient + coeff-context/cdf state at the first diverging
  (position, plane, qindex), then revert + rebuild clean (never commit the instrument).
- **Range matters:** q62/q63 is the aggressive end of the quantizer range — exactly the
  web-compression regime this port targets.

### KB-2 — Encoder: `diag+vbars16 256x256 cq62` strong cell — FIXED ✅ (per-block intra edge filter type)
- **FIXED 2026-07-15.** Root cause: the port **never re-derived the intra edge filter type
  (`get_intra_edge_filter_type`, reconintra.c:974) per block** — it carried a frozen SB-level
  `filter_type` (always 0) down into every leaf's `TxfmYrdEnv`/`UvRdEnv`. C re-derives it per
  block from the live mode-info grid: `1` iff the above **or** left neighbour is a SMOOTH mode
  (SMOOTH_PRED=9 / SMOOTH_V_PRED=10 / SMOOTH_H_PRED=11). For the diverging cell, SB(32,32)'s
  VERT_4 strip-1 (16×64 @ mi(32,36)) has a **SMOOTH left neighbour** (strip-0, mode 9), so C
  computes `filter_type=1` while the port used `0`. That flips the intra-edge-filter strength for
  **angled** directional predictions (adj≠0; pure-vertical adj=0 skips the edge filter, which is
  why adj=0 matched exactly and only angled deltas diverged). The port's worse angled prediction
  raised V_PRED adj=−1's **model RD** to 25930 vs C's 24704; the `prune_intra_y_mode`
  `THRESH_BEST=1.5×best_model_rd` (=1.5×17236=25854) then **over-pruned adj=−1** in the port
  (25930>25854, margin 76) where C keeps it (24704<25854). C fully evaluates adj=−1, the ALLINTRA
  variance factor reorders it ahead of adj=0, and C picks adj=−1 → strip winner differs → HORZ_A
  vs VERT_4 → byte divergence. **Fix:** recompute `filter_type` per block from `above_mode`/
  `left_mode` (already read from the grid for the mode-cost context) in `partition_pick.rs`'s
  leaf search, mirroring `get_intra_edge_filter_type`; the `CPick` C-recursion reference in
  `partition_pick_diff.rs` got the identical recompute so the differential stays faithful.
- **Verified:** the cq62 cell now achieves TRUE END-TO-END BYTE MATCH vs real aomenc and is an
  **asserted** case in `encoder_gate_e2e_rich_content_strong_lf` (6/6); full `aom-encode` suite
  green; the port's angled prediction matches C pixel-for-pixel (per-tx-block SATDs identical).
- **Chroma follow-up (#26) — FIXED ✅ 2026-07-15.** The **chroma** `filter_type` (UvRdEnv) was the
  same frozen-at-0 bug on the UV plane: C's `get_intra_edge_filter_type(xd, plane=1)` is `1` iff an
  available above/left chroma neighbour's `uv_mode` is SMOOTH (UV_SMOOTH_PRED=9 / UV_SMOOTH_V=10 /
  UV_SMOOTH_H=11). Fix mirrors the KB-2 luma recompute on chroma: `ModeGrid` now carries a parallel
  `uv_modes` grid (`partition_pick.rs`, stamped alongside luma at every `stamp`/`stamp_grid_from_tree`
  site); `leaf_pick_sb_modes` recomputes the per-block chroma edge `filter_type` from the chroma
  neighbours (chroma-reference mi derivation, av1_common_int.h:1400-1416: `base=(mi_row-(mi_row&ss_y),
  mi_col-(mi_col&ss_x))`, above=`base+(-1,+ss_x)`, left=`base+(+ss_y,-1)`) and feeds it to BOTH the UV
  RD search AND — via the new `LeafWinner::uv_edge_filter_type` — the pack re-encode
  (`encode_b_intra_dry`, encode_sb.rs), which produces the coded chroma bytes. The `CPick`
  C-recursion reference in `partition_pick_diff.rs` got the identical recompute + a parallel `uv_grid`
  (randomized UV neighbours now exercise it as a differential witness). **Verified:** new
  `encoder_gate_444_bd8_chroma_edge_filter_witness` (encoder_gate_chroma_ss_e2e.rs) byte-matches real
  aomenc on all 4 cells WITH the fix and DIVERGES on the 128×128 cq12/cq32 cells with it reverted
  (proven fails-before/matches-after); `partition_pick_diff` passes with randomized smooth UV
  neighbours; full `aom-encode` suite green. Commit: partition_pick.rs + encode_sb.rs +
  partition_pick_diff.rs + encode_sb_diff.rs + the witness.
- **Historical isolation trail (how it was root-caused) below:**
- **Re-verified 2026-07-15 (still diverges), with much sharper isolation:**
  - Facts: qindex **249**, `screen_content=true` (auto-detected — the ONLY screen-content cell in
    the whole encoder suite), port tile **95 bytes vs real 100** (port codes FEWER symbols), port
    derives LF luma **[0,17]** vs real **[1,17]** (a DOWNSTREAM recon symptom, not the cause), first
    payload mismatch at byte 3 (= the header LF-level byte). First **TILE**-byte divergence is at
    **tile-byte 60 of 100** → the first ~60% of the tile is byte-identical, so the divergence is in a
    **MID-FRAME SB, NOT SB(0,0)** (unlike KB-3).
  - **RULED OUT — palette flag** (definitively): the port's RD `try_palette =
    allow_palette(allow_screen_content_tools, bsize)` (partition_pick.rs:589, no `enable_palette`
    gate) is EMPIRICALLY byte-exact — `encoder_gate_e2e_ab_attempt` is the exact
    `enable_palette=0`(standard shim) + `screen_content=1` config and byte-matches WITH it; forcing
    `try_palette=false` REGRESSED that gate. So real includes the palette-Y no-palette flag cost for
    screen-content frames regardless of `--enable-palette=0`, and the port matches. Write side
    (pack.rs:274, `allow_palette` only) matches C (bitstream.c:1042). Palette is fully correct.
  - **RULED OUT — all other screen-content RD effects** (parallel-agent survey of the sibling C,
    verified against build config): at speed-0 / full non-realtime build / ALLINTRA / KEY / qidx249 /
    <720p, there is **zero** screen-content dependence in rdmult (rd.c), quantizer (av1_quantize.c),
    coeff trellis (encodemb.c/txb_rdopt.c), tx-set context, angle-delta / filter-intra / smooth, or
    the partition search — beyond palette (handled) and the header intrabc-present bit (handled: AB
    gate proves the port's header writer emits it). The one latent tx path, `get_default_tx_type`
    forcing DCT_DCT under screen content (blockd.h:1175), is **dormant** because
    `use_intra_default_tx_only=0` in the non-realtime reference build (verified `CONFIG_REALTIME_ONLY
    0` + av1_cx_iface.c:374 default 0). RANK-3 `exhaustive_searches_thresh` differs at speed-0 but is
    inert (no motion search in all-intra). RC is bypassed (fixed AOM_Q, per-block qindex stays 249).
  - **CONCLUSION:** a plain **speed-0 coeff/partition/mode near-tie**, NOT screen-content-specific.
    Same content+generator as the cq**63** cell that byte-matches (strong_lf gate 5/5); cq62 → qidx
    249 tips a near-tie in a later SB. Class-identical to KB-1's "only very-high-qindex flips it".
  - **RD-DUMP DONE (2026-07-15) — root-caused to a single 16×64 leaf's tx/coeff evaluation.**
    Method: re-tailored sibling harness (`/root/libaom-enc-instrument/rd_harness.c`) for
    `diag+vbars16 256×256 cq62 cpu0` and VALIDATED its output == real (117-byte stream, frame OBU
    `32 69` payload = 5 hdr `44 f9 00 51 14` + 100 tile `ff 3b 14 51…`). Then per-SB partition dump
    (port PSB vs sibling C CSB): **15/16 SBs match; SB (mi=32,32) diverges — C picks PARTITION_HORZ_A
    (4), port picks PARTITION_VERT_4 (9).** Per-candidate RD at (32,32): port HORZ_A rate=33741
    dist=8751216 **rdcost=1393344729 == C's HORZ_A EXACTLY**; port VERT_4 rate=23037 dist=8757376
    **rdcost=1307466663 wins**. C's VERT_4 is INVALID: C's 4-way prune allows both HORZ4/VERT4
    (`allowed=[1,1]`, `prune_ext_partition_types_search_level=1` so the level-2 partitioning gate at
    partition_search.c:4202 does NOT fire — not a pruning diff), but C's VERT_4 sub-block search
    **bails at strip 2** (`rd_try_subblock` returns 0: strip-2's own 16×64 mode RD exceeds the
    remaining budget best−cum). **Per-strip VERT_4 at (32,32) (both mono, subsize=BLOCK_16X64=20):
    strip0 (c=32) mode=9 cum_rate=7557 cum_dist=3946048 — MATCHES C exactly; strip1 (c=36) SAME
    mode=1 (V_PRED) in both, but port Δrate=5614/Δdist=933472 vs C Δrate=9980/Δdist=1568992 — port
    UNDER-COMPUTES both.**
  - **EXACT ROOT CAUSE — angle_delta divergence on the strip-1 16×64 V_PRED leaf.** Both pick
    identical `tx_size=TX_16X64 (17)`, `skip=0`, `tt0=DCT_DCT`; the ONLY difference is the intra
    **angle_delta**: **C picks V_PRED `angle_delta=-1`, the port picks V_PRED `angle_delta=0`.** The
    port's adj=0 (rate 5614 / dist 933472) is strictly cheaper on BOTH axes than C's adj=-1 (rate
    9980 / dist 1568992) — so C's OWN adj=0 evaluation must be *worse* than the port's adj=0 (else C
    would pick 0). Both search the full delta range (`use_angle_delta` matches C exactly:
    `bsize>=BLOCK_8X8`, and 16×64=20 qualifies; port `enable_angle_delta=true` at speed 0). ⇒ the
    port's **directional-intra prediction and/or angle-delta RD for this 16×64 (1:4-aspect) leaf is
    wrong** — its adj=0 (or the delta search) is under-costed, so adj=0 wins in the port where adj=-1
    wins in C. (NOT partition pruning, NOT palette, NOT screen-content, NOT tx-size/type/skip, NOT
    #25's speed-1 bugs — this is speed-0.) strip0 (also 16×64, mode=9=D67_PRED-ish non-vertical)
    matching rules out a blanket 16×64 bug — it's specific to V_PRED angle_delta on this leaf.
  - **RESOLVED (see the FIXED block at the top of this entry).** The per-delta dump above was
    slightly mis-framed: adj=0 was **not** under-costed — it matched C exactly. The real mechanism
    is that the port never even *evaluated* adj=−1's full RD: it **model-pruned** adj=−1 at
    `prune_intra_y_mode` because its **model** RD (25930) was inflated by the wrong (0 instead of 1)
    intra edge filter type on the angled prediction, tipping it over `1.5×best_model_rd` (25854).
    The "directional-intra predictor edge/neighbour" guess was on target — it was the per-block
    `get_intra_edge_filter_type` recompute the port was missing. All temp instrumentation and the
    sibling `/root/libaom-enc-instrument` have been removed.

### KB-3 — Encoder: `vgrad 256x256 cq32` cpu-used=1 cell — FIXED (missing speed-1 `use_square_partition_only_threshold` rect-kill)
- **FIXED** (commit pending on origin): the cell now byte-matches; promoted to an asserted winner
  in `encoder_gate_speed1_textured_allintra` (14/14 cpu-used=1 content cells). Root-caused via
  **isolated sibling-libaom encoder instrumentation** (`/root/libaom-enc-instrument`, a throwaway
  copy — never the shared `reference/libaom`) dumping C's per-candidate RD at SB(0,0) 64×64 for
  the exact vgrad-256-cq32 encode. Findings: C's NONE and SPLIT RD matched the port **exactly**
  (NONE rate 36745 / dist 19456 / rdcost 7427690, rdmult 68796); C **never evaluated** the
  rectangular partitions, but the port did, and the port's HORZ (rdcost 7058801) beat NONE → port
  wrongly picked `PARTITION_HORZ`. C disables rect via the "square-partition-only" rect kill
  (`partition_search.c:5749`): `if (bsize > use_square_partition_only_threshold) {
  partition_rect_allowed[HORZ] &= !has_rows; [VERT] &= !has_cols; }`. That threshold is a
  framesize-DEPENDENT ALLINTRA speed feature: sub-480p it is `BLOCK_64X64` at speed 0 (so
  `bsize > 64X64` never holds in a ≤64 SB — why speed-0 never needed it) but drops to
  `BLOCK_32X32` at speed ≥ 1, killing rect on the 64X64 SB. **Fix:** wired the rect-kill into
  `rd_pick_partition_real` (`use_square_partition_only_threshold_allintra`, framesize+speed
  dependent), placed after `partition_rect_allowed` init and before the CNN prune (matching C's
  order). Speed-0 unaffected (threshold `BLOCK_64X64` → no-op); full `cargo test -p aom-encode`
  = 89 passed, 0 failed. NOT a learned-model prune (the CNN/prune_2d/etc. elimination below stands).
- **KB-2 is a SEPARATE root** (do NOT conflate): KB-2's cell runs at **cpu-used=0**, where this
  fix is a no-op (threshold `BLOCK_64X64`). KB-2 needs its own speed-0 root-cause pass.

<details><summary>Original isolation notes (superseded by the fix above)</summary>

Was: `vgrad 256×256 cq32` (base_qindex 128) diverged at byte 5, never re-converging.
- **Symptom:** in `encoder_gate_speed1_textured_allintra`, the `vgrad 256×256 cq32`
  (base_qindex 128) cell does not e2e byte-match aomenc. Diverges at **byte 5** (first
  tile-data byte) and **never re-converges** (`last_common_idx = 4` = last header byte) — an
  early partition/mode cascade at SB(0,0). Excluded (documented) in the winners list of that
  gate; the sibling cells (256×256 cq48, 128×128 cq32/cq48) all byte-match.
- **Isolation COMPLETE — NOT an unported learned-model prune.** The originally-suspected
  `intra_cnn_based_part_prune_level` 0→2 (intra CNN partition prune) is now **fully ported +
  wired** into `rd_pick_partition_real` (commit `a600394`) and its four flags are **bit-exact
  vs C** (`cnn_partition_decision_diff`). For this cell the CNN fires and sets
  `square_split_disabled=true` at every 64×64 SB root — **identically to C** — so it constrains
  port and C the same way and cannot cause a divergence. **Empirically confirmed:** wiring the
  CNN in left byte-5 (157 vs 8) byte-identical. Eliminated candidates (with evidence):
  `prune_2d_txfm_mode` PRUNE_2 (intra path needs `prune_tx_type_est_rd`, which is speed≥4;
  `prune_tx_2D` is `is_inter`-only); `model_based_prune_tx_search_level`,
  `av1_ml_predict_breakout`, `av1_ml_early_term_after_split`, `av1_ml_prune_rect_partition`,
  `simple_motion_search_*` (all `!frame_is_intra_only`); `ml_predict_var_partitioning` (nonrd).
- **Root cause (localized):** a **partition-search RD near-tie** (KB-2 class). The port picks
  `PARTITION_HORZ` for SB(0,0) (two 64×32 DC / TX_64X32 blocks); C picks a different partition.
  A speed-1 RD-cost delta tips the NONE/HORZ/VERT comparison for this specific content+qindex.
- **Next step:** dump the port's per-candidate RD (NONE/HORZ/VERT) at the SB(0,0) 64×64 node vs
  the C reference. Needs an **encode-side RD-dump shim** — but `shim_encode_av1_kf` currently
  lives in the decoder-owned `dec_shim.c` and drives the opaque `aom_codec` API (no `cpi->sf`
  hook), so per-feature C-side toggling / RD dumps aren't reachable from the encoder track
  without a coordinated new shim entry point. Candidate speed-1 RD deltas to bisect once that
  exists: `perform_coeff_opt=2`, `tx_domain_dist_level/thres_level=1`, `adaptive_txb_search
  _level=2`, `top_intra_model_count_allowed=3`.
- **Two LATENT speed-1 bugs found while isolating (NOT this cell's cause — both leave these 8
  cells byte-identical, so no current test exercises them; documented for a future fix + new
  validation cells):**
  1. `part4_prune.rs:234` hardcodes `LEVEL_INDEX = 0`, but C's `ml_4_partition_search_level
     _index = min(speed,3)` (set 0/1/2/3 at `if(speed>=1/2/3)`, speed_features.c:210/237/271;
     default 0 at :2305). Index expr `(LEVEL*3+res_idx)*5+bsize_idx` uses LEVEL **directly**
     (no −1) — the port's `LEVEL_INDEX` == the level. Usage: `av1_ml_prune_4_partition`,
     partition_strategy.c:1507-1510. **CRITICAL caveat (verified 2026-07-15):** at level **3**
     (speed≥3) C flips `ml_model_index = (level<3) == 0` (partition_strategy.c:1359) → a
     **different NN model, no threshold table** (`:1472-1497`, scores vs `max_score−{500,500,200}`).
     So the port's table path is correct ONLY for speeds 0/1/2 (LEVEL 0/1/2). Fix = pass
     `level=min(speed,3)` from `cfg.speed` into `predict_4partition_prune` (caller
     partition_pick.rs:2173) and use it as the table row **only when level<3**; speed≥3 needs the
     alternate (old-NN, tableless) branch = a #10 item, NOT #25. Feeding LEVEL=3 into the table
     would be wrong (that path never runs in C).
  2. `tx_search.rs:1305` `get_search_init_depth_intra_speed0` hardcodes the speed-0
     `intra_tx_size_search_init_depth_rect = 0`, but C uses 1 at speed≥1 (speed_features.c:409);
     `_sqr = 1` for ALL speeds (unconditional at :367). So at speed≥1 BOTH rect and sqr return 1.
     `get_search_init_depth` (tx_search.c:363-383) returns `_rect` when w≠h, `_sqr` when w==h.
     Fix = thread `speed` into `choose_tx_size_type_from_rd_intra` (caller of the init-depth fn,
     tx_search.rs:1356; `TxfmYrdEnv` has no `speed` field yet — add it or pass a param) and return
     `rect = (speed>=1) as i32`, `sqr = 1`.
  Both preserve speed-0 exactly (min(0,3)=0; rect=0 at speed 0). Needs new speed-1 RECT-partition
  test cells to validate — the current speed-1 gates pass WITH the bugs (they don't reach a
  divergent 4-way-prune / rect-tx decision), so exercising cells must be discovered (a speed-1
  e2e harness exists: `encoder_gate_speed1_textured_allintra`).

### KB-4 — Encoder: bd10/bd12 coded-eob divergence (was "RD-decision divergence at high bit depth") — FIXED ✅ (BOTH roots; task #31)
- **FIXED 2026-07-16 (this landing) — OUTPUT_ENABLED tx_type_map copy semantics in `encode_b_intra_dry`.**
  The mono/4:2:0 aggressive-HF divergence (bd10 cq12, bd12 cq8, bd12 cq20 in
  `kb4_bd10_rd_localize.rs`) was NOT a high-bit-depth RD-scaling bug: the port ran C's single
  OUTPUT_ENABLED walk TWICE (the SB-root winner context/recon walk + the pack re-walk) with DRY
  (alias) tx_type_map semantics, so the first walk's `eob==0 → DCT_DCT` resets
  (encodemb.c:770-779, `update_txk_array`) leaked into the pack's re-quant input. A skip-winning
  txb (non-DCT search winner quantizing to eob 0 — exactly what aomenc codes) re-quantized as
  DCT_DCT with eob>0 in the coded bytes (e.g. the bd10 cq12 mi(14,12) BLOCK_16X8/D45 txb5:
  search=ADST_DCT/eob0, coded=DCT/eob1). C's semantics (`av1_update_state`,
  encodeframe_utils.c:217-231): DRY walks **ALIAS** `ctx->tx_type_map` — resets PERSIST into the
  stored winner map (real C behaviour; do NOT "fix" by cloning); OUTPUT_ENABLED **copies** ctx
  into the frame-level map and the resets land THERE, ctx untouched. **Fix:**
  `encode_b_intra_dry`/`encode_sb_dry` take `output_enabled`; the SB-root winner walk
  (partition_pick.rs, C partition_search.c:6010) and the pack walk (pack.rs — the same C walk,
  re-run) use a transient frame-map clone; the mid-candidate propagation (C :3613-3616) and
  non-SB winner walks (C :6023, `should_do_dry_run_encode_for_current_block` :5556 — last SPLIT
  children skipped) keep the alias. The `COracle`/`CPick` differential references mirror the
  split (they had shared the port's mis-model). bd10/12-amplified (larger RD magnitudes make
  non-DCT-eob0 near-tie txbs common) but NOT bd-specific in mechanism: the same leak closed
  KB-6's bd8 `quantizer-00 128×128 cq63` cell.
- **Prior "RD-DECISION layer bd scaling" localization REFUTED (2026-07-16):** per-tx_type
  rate+dist are byte-exact vs the REAL-C leaf chain (`kb4_txb2_probe.rs`); tx-type search order
  matches C (txk_map stays natural `{0..15}` at speed-0 — `prune_tx_2D` reorders only under
  `prune_tx_type_est_rd`, speed≥4); `ref_best_rd` threading and the `adaptive_txb_search` break
  match C, and the break never changed the winner on any divergent txb (with-break == full-eval
  on every one). The kernels were indeed byte-exact — the divergence was PASS-STRUCTURE, not
  arithmetic. (An earlier blanket per-pass-clone attempt regressed 3→5 cells because it also
  cloned C's DRY alias walks and the rd_pick CfL store-luma reencode — both must keep mutating.)
- **Gates:** mono/420 promoted to `kb4_gate_bd10_bd12_mono_hf_byte_match`
  (kb4_bd10_rd_localize.rs) — the full bd10/bd12 × cq8/12/20 × hf/ramp sweep byte-matches real
  aomenc (12/12). Non-420: the other KB-4 witness was FIXED separately by **1ecfafb** (AB HORZ_A
  nested sub-block reuse) — all 4 bd10 non-420 cells (444/422 × 64²/128² cq32) byte-match,
  asserted by `encoder_gate_bd10_non420_e2e_kb4_repro`.

### KB-5 — Encoder: lossless (cq0 / qindex 0) KEY encode — FIXED ✅ (mono + 4:2:0 both byte-exact, hard-asserted; #32 closed)
- **MONO FIXED 2026-07-16.** Mono 64² cq0 (coded-lossless allintra KEY) is now an end-to-end BYTE
  MATCH vs real aomenc, hard-asserted in `encoder_gate_lossless_cq0_e2e_kb5_repro`
  (encoder_gate_chroma_ss_e2e.rs). THREE fixes were required (the two originally localized below,
  plus a third found during landing):
  1. **Harness two-pass (#32):** `run_case` now mirrors the decoder's two-pass lossless probe —
     parse, compute coded_lossless from the probe's quant params (base_qindex==0 && all 5 plane
     q-deltas 0), re-parse with `cfg.coded_lossless/all_lossless=true`.
  2. **Forward WHT (#33):** `av1_fwht4x4` ported into aom-dsp's `transform` module
     (`crates/aom-dsp/src/transform/inv_txfm2d.rs`; bit-exact vs `av1_fwht4x4_c`,
     gated by `fwht4x4_diff`); `QuantParams` gained a `lossless` flag; `xform_quant` (lib.rs) and
     every encoder recon site (encode_intra / tx_search / intra_uv_rd) route coded-lossless TX_4X4
     through WHT/IWHT via `av1_inverse_transform_add(.., eob, lossless)`. The SATD fast model stays
     DCT (`av1_quick_txfm` forces lossless=0 in C — intra_uv_rd.rs:800 unchanged, do NOT "fix" it).
     The differential oracle (tests/common/mod.rs `c_search_tx_type_p` / `c_uniform_txfm_yrd`) uses
     `ref_fwht4x4`/`ref_highbd_iwht4x4_add` for lossless — a faithfulness correction (real C uses
     WHT for lossless, hybrid_fwd_txfm.c:83-86).
  3. **Entropy-context propagation (the actual byte-divergence root, found via decode-both
     localization `kb5_lossless_localize.rs`):** the WRITTEN `txb_skip_ctx`/`dc_sign_ctx` must
     derive from the REAL above/left neighbour entropy context ALWAYS — C's write path
     (`av1_write_coeffs_txb`, encodetxb.c:596-598) is never gated on the trellis; only C's
     trellis-local `ta/tl` fill is (encodemb.c:817-819). The port shared one ta/tl array for both
     uses (encode_intra.rs, luma + chroma arms) and seeded it from the real context only when the
     trellis was on; coded-lossless runs trellis-OFF (USE_B_QUANT_NO_TRELLIS), so a block with a
     coded left neighbour wrote ctx 1/0 instead of the real 3/1 and desynced the decoder. Fix:
     always seed ta/tl from the real neighbour context.
- **420 FIXED 2026-07-16 (mono landed as ba560eb; 420 this landing) — CfL banned at coded-lossless
  in the SEARCH.** The former "≤1-unit chroma RD near-tie" was a search-SPACE gap, not RD math:
  `partition_pick.rs`'s leaf `cfl_allowed` was `!lossless && w<=32 && h<=32`, but C's
  `is_cfl_allowed` (blockd.h) allows CfL at LOSSLESS whenever the partition size equals the
  transform size — `get_plane_block_size(bsize, ssx, ssy) == BLOCK_4X4` (at 420: every
  8×8-and-below chroma-ref leaf). Measured mechanism (instrumented-sibling-C vs port partition
  dumps, faithfulness-gated byte-identical first): at the first 16×16 node NONE matches EXACTLY
  (both 235604, rdmult 52, dist=0 everywhere at lossless) but C's 8×8 SPLIT children pick
  **UV_CFL_PRED** (~16k cheaper rate per chroma-carrying leaf; luma-only 4×4 subs byte-identical)
  → C SPLIT 235256 beats NONE by 348; the port's missing CfL candidates inflated its children and
  starved SPLIT child-3 at the 63759 remaining budget → NONE → desync. **Fix:** route the leaf
  gate through the shared (already-correct, pack.rs already used it) helper
  `aom_entropy::partition::is_cfl_allowed(bsize, env.lossless, ss_x, ss_y)` — expression-identical
  at !lossless, so non-lossless gates are untouched (verified: all chroma-ss/KB-4/KB-6 gates
  unchanged-green). The `CPick` reference in `partition_pick_diff.rs` carried the SAME transcribed
  gate (a shared bug that differential structurally could not catch) — also routed through the
  helper. **Refuted en route (do not re-chase):** the chroma UV RD math at qindex 0 is CLEAN — the
  new `txfm_uvrd_matches_c_walk_lossless_q0` differential (txfm_uvrd_diff.rs; UvRdEnv oracle
  winner-recon taught IWHT-for-lossless in common/mod.rs to match hybrid_fwd_txfm/inverse
  dispatch) proves port==C at qindex 0 across 14 chroma-ref shapes × 8 iters for rate/dist(=0,
  physics-asserted)/sse/winners/recon PLUS strict-`>` budget-boundary agreement at
  min_rd−1/min_rd/min_rd+1. **Gate:** `encoder_gate_lossless_cq0_e2e_kb5_repro` hard-asserts BOTH
  mono AND 420 byte-match (promotion from `assert_open_divergence` per its designed contract).
  The full lossless envelope (coded-lossless cq0 KEY, mono + 4:2:0) is byte-exact; #32 closed.

### KB-6 — Encoder: REAL-content RD divergence at bd8 4:2:0 (PRIMARY config) — FIXED ✅ (all roots landed; real-content map 30/30)
- **FIX #1 LANDED 2026-07-15 (ca2826f) — luma re-encode intra edge filter.** The luma analogue of
  #26 (chroma). `encode_b_intra_dry` — the dry-run re-encode used by BOTH the search's inter-strip
  context propagation (`partition_pick.rs:1054/1338/1914`) AND the pack output (`pack.rs:317`) — froze
  the LUMA intra edge filter at the SB-level `env.filter_type` (always 0) instead of the per-block
  `get_intra_edge_filter_type` (reconintra.c:974). KB-2 fixed only the luma SEARCH RD (leaf y_env); the
  re-encode/stamp stayed at 0. So an angled luma leaf (angle_delta≠0) with a SMOOTH above/left neighbour
  re-encoded its prediction with edge filter 0 not 1 → wrong residual → per-txb eob flip in the coded
  bytes, AND a wrong propagated entropy context that shifted later leaves' RD. **Fix:** carry the
  per-block `luma_edge_filter_type` (already computed in the search, KB-2) on `LeafWinner` and feed it to
  `encode_b_intra_dry`'s y_env. The `CPick` differential reference had to mirror it or diverge on
  smooth-neighbour angled leaves: `CEncPlaneArgs` gained a `filter_type` field so the `COracle`
  propagation re-predicts (ref_hbd_predict_intra 9th arg) with the SAME per-block filter. Localized via
  `kb6_real_rd_localize.rs` (decode-both-streams): first divergent SB was leaf mi(12,12) bsize=BLOCK_4X16
  angled (y_mode=6, angle_delta_y=1), real eob=0 vs port eob=2, ±1 recon at (48,48). Verified: full
  aom-encode suite green; `partition_pick_diff` green with randomized SMOOTH neighbours.
- **CLOSED 2026-07-16 — the REAL-CONTENT MAP IS 30/30 BYTE-EXACT** (was 26/30 after the KB-4
  OUTPUT_ENABLED fix + the partial-SB chunk series; 29/30 after the entropy-stamp/edge-CDF
  landing; the last cell, 196² cq48, closed by the pack write-ctx fix below). Every
  interior-crop cell now matches: size-64×64 all 6 cq (cq5/12/20/48/63 with FIX #1; cq32 with
  1ecfafb — AB HORZ_A nested sub-block reuse); quantizer-64² 6/6, film-64² 6/6, quantizer-128²
  6/6 — the former cq5 low-q cluster and the quantizer-128² cq12/20/32 near-ties cleared with
  the partial-SB chunk series' distortion-clip landings, and **quantizer-128² cq63 + 196×196
  cq63 closed 2026-07-16 by the KB-4 OUTPUT_ENABLED tx_type_map fix** (the port coded DCT-eob1
  where real codes an eob0 skip — the reset-leak signature, present in interior AND edge SB
  rows).
- **DISTINCT SUB-GAP — partial-SB (frame dims not a multiple of 64px) — FULLY FIXED (all 6 cq).** Landed: the CHUNK series (`3167800` CHUNK 0+1 true-frame harness + luma visible
  dist clip, `7c468ee` CHUNK 2 chroma visible clips via `max_block_units`, `4b8b1f1` CHUNK 3
  `set_partition_cost_for_edge_blk`), the KB-4 OUTPUT_ENABLED tx_type_map reset-leak fix
  (`a2dd28e`, closed 196² cq63), and the **frame-edge entropy-stamp tail-zero + frame-init edge
  partition CDF fix** (closed cq12/20/32; map 26/30 → **29/30**). That last root was pinned by a
  full C-vs-port symbol-level bit trace (throwaway instrumented sibling C at `/root/kb6-edge-instr`,
  byte-gate-verified vs real aomenc): the apparent "mi(48,0) 16×8-vs-8×4 over-split" was NOT a
  search decision — the port's search picks C's EXACT tree and every leaf RD matches C to the unit;
  the port's PACK also writes the same symbols. The divergence was a WRITE-side probability defect:
  (a) **`av1_set_entropy_contexts` (blockd.c:29) zeroes the beyond-visible TAIL of an edge txb's
  above/left entropy-context footprint** (`memset(a + above_contexts, 0, txs_wide - above_contexts)`)
  while the port's tile stamp (encode_sb.rs) wrote the cul across the FULL footprint — phantom
  nonzero culs at out-of-frame mi cols (50-51 luma / 25 chroma) fed later edge blocks'
  full-footprint `get_txb_ctx` reads, flipping SB(32,48)'s txb_skip_ctx (1→3 luma, 8→9 U) → same
  symbols on different-probability cdf rows → +3 bits → stream desync at tile-byte 975 → the
  decoded "over-split" artifact; (b) the CHUNK 3 edge partition-cost gather read the SB-adapted
  partition CDF, but C's `set_partition_cost_for_edge_blk` (partition_search.c:3415) reads
  **`cm->fc` — the frame-init table** (measured: C's gather rows == `default_partition_cdf`),
  a shipped-libaom mixed-source quirk (interior costs track the adapting tile state; edge gather
  does not). Note the C encode-path per-txb stamp `av1_set_txb_context` (encodemb.h) is
  full-footprint UNclipped — only the tokenize/persistent stamp clips; the port's local ta/tl
  stamps correctly mirror the former and needed no change.
  **All six 196² cells (cq5/12/20/32/48/63) are asserted byte-match gates** in
  `encoder_gate_real_image_e2e_kb6_repro` (now a FULL 30-cell byte-match gate).
  **cq48 (the LAST cell) FIXED 2026-07-16 — pack WRITE-ctx source (tokenize vs trellis):**
  decode-both + pass-context markers proved the search was ALREADY C-identical at the divergent
  leaf (mi(0,48) 32×64 SMOOTH; both OUTPUT_ENABLED walks requantize txb4 to C's coded
  (tt1, eob37)) — the decoded "(eob4, tt2)" was a desync artifact of the port's own bits. C caches
  the pack's `(txb_skip_ctx, dc_sign_ctx)` in the TOKENIZE walk
  (`av1_update_and_record_txb_context`, encodetxb.c, OUTPUT arm; `av1_write_coeffs_txb` writes the
  CACHED pair) derived from the PERSISTENT entropy arrays — whose within-leaf stamps are
  edge-CLIPPED (`av1_set_entropy_contexts`) — while the TRELLIS uses the encode walk's
  full-footprint local `av1_set_txb_context` stamps; the port used the trellis pair for the write
  too. `txb_skip_ctx` is OR-based (tail-zero inert — why the 29/30 landing sufficed there) but
  `dc_sign_ctx` is SIGN-OF-SUM: at txb blk(8,0) (16×16, vis 8×16) the above tail-zero drops +2
  (C: −4+2 = −2 → ctx 1; port: −4+4 = 0 → ctx 0) → ONE DC-sign symbol on a different cdf row →
  bits diverge at tile byte ~253 with IDENTICAL symbols everywhere. Fix: `encode_b_intra_dry`
  Step 4 (encode_sb.rs, the tokenize-equivalent stamp loop) derives the write pair from the
  persistent arrays per txb — before that txb's clipped stamp, C's exact read point — and
  overwrites the cached `TxbEncode` pair (dcs gated on `qcoeff[0] != 0`, Y+U+V planes); sole
  consumer is `pack_plane_coeffs`. Interior txbs derive identical values (structurally zero-diff
  on the green corpus).
- **MULTI-TILE encode is byte-exact** (commit f6e6319, `encoder_gate_multitile_e2e`): the port's own
  per-tile search+pack byte-matches real aomenc across 2×1/1×2/2×2 grids (4:4:4 128² × cq{12,32,63}).
- **DISCOVERED 2026-07-15 via the new real-image e2e gate** (`encoder_gate_real_image_e2e_kb6_repro`
  in `encoder_gate_chroma_ss_e2e.rs`): decode the first KEY frame of a small conformance vector
  (`av1-1-b8-01-size-64x64`, `av1-1-b8-01-size-196x196`; `01-size` is in CI's intra fetch scope) to
  genuine YUV via the C decode oracle, then run the port's full encode vs real aomenc byte-for-byte on
  those REAL pixels. **Every synthetic e2e gate is byte-exact, but genuine image content diverges
  across the whole quality range.** Map (bd8 4:2:0, cq5..63): the multi-SB **196×196 frame diverges at
  EVERY cq** (e.g. cq20 port tile 1457B vs real 1556B — port codes ~100 FEWER bytes); the 1-SB
  **64×64 diverges at cq5/12/32/48** and byte-matches only at the coincidental cq20/cq63. 2/12 cells
  byte-exact, 10 diverge. (Superseded by FIX #1 above: after the luma re-encode fix + the expanded
  photographic/film crop gate, the map is now 15/30 byte-exact.)
- **Signature = KB-2 class:** the port codes FEWER symbols than aomenc ⇒ it makes different (cheaper)
  partition/mode/tx RD decisions — a near-tie flip, exactly like KB-2 (`get_intra_edge_filter_type`)
  and KB-3 (speed-1 rect-kill), but now on the **PRIMARY bd8 4:2:0 speed-0 KEY** path and on REAL
  content. The hand-tuned synthetic patterns (diag/vbars/vgrad/tex_*) never exercised the diverging
  decision; real photographic/screen statistics do. **This means the "byte-exact regime: bd8 all
  content" note under KB-4 is TRUE ONLY for the synthetic gates — it is FALSE for real content.**
- **Root cause: MULTIPLE KB-2-class near-ties, several roots landed.** FIX #1 (luma re-encode
  edge filter) took real 64×64 from 2/6 to 5/6; 1ecfafb (AB HORZ_A nested reuse) closed 64×64 cq32
  + the 4 bd10 non-420 KB-4 cells; the partial-SB chunk series (distortion visible-clips + edge
  partition cost) cleared the cq5 low-q cluster + the quantizer-128² cq12/20/32 near-ties + 196²
  cq5; the KB-4 OUTPUT_ENABLED tx_type_map fix (2026-07-16) closed quantizer-128² cq63 + 196²
  cq63; the frame-edge entropy-stamp tail-zero + edge partition CDF landing (4567e58) closed 196²
  cq12/20/32; and the pack write-ctx fix (2026-07-16) closed the final cell, 196² cq48 — the
  last three roots were all WRITE-side probability defects (identical symbols on
  different-probability cdf rows), not search decisions.
- **Repro (COMMITTED, CI-green characterization):** `encoder_gate_real_image_e2e_kb6_repro` prints the
  full per-cell MATCH/MISMATCH map, asserts a byte-exact CONTROL (64×64 cq20 — harness-faithfulness +
  regression guard), and asserts the KB-6 divergence is still PRESENT (gates: when the port becomes
  byte-exact on real content the test FAILS → promote it to a full `report_and_assert` byte-match
  gate). Not a weakened test — the correct end state is full byte-identity on real content.
- **Next step: NONE — the real-content map is complete (30/30).**
  `encoder_gate_real_image_e2e_kb6_repro` is promoted to a full byte-match gate over all 30
  cells; any real-content divergence is now a regression, not an open KB-6 axis. (KB-1, KB-5,
  KB-7, KB-8 and the Gate-2 cpu-used sweep remain separate tracks.)
- **Priority note:** KB-6 hits the single most common real-world case (bd8 4:2:0 photographic content
  at web qindex), so it is arguably higher-impact than the bd10/bd12 (KB-4) and lossless (KB-5)
  corners. Sequencing is the coordinator's call.

### KB-7 — Encoder: `--cpu-used=3/4` cq12/cq32 4:2:0 partition flips — FIXED ✅ (TWO speed-feature-port roots; speed-3 AND speed-4 gates 64/64)
- **FIXED 2026-07-16.** All 8 pinned cells (3 at speed-3 + 5 at speed-4) now BYTE-MATCH real
  aomenc; both gates assert FULL 64/64 byte-identity. The "latent chroma-RD near-tie"
  hypothesis was REFUTED by the sibling-C RD dump (throwaway instrumented C, kb7-instr inject
  pattern; validated byte-inert vs the clean build): every leaf RD — NONE/HORZ/VERT, luma AND
  chroma parts, and every SPLIT child total — matched C **to the unit**. The flips were TWO
  partition-search-SPACE / speed-feature-port gaps:
  1. **(speed>=3, closed ALL 3 speed-3 pins) `av1_ml_prune_4_partition`'s OLD-model branch was
     unported.** At `ml_4_partition_search_level_index = 3` (allintra speed>=3) C flips
     `ml_model_index = (level < 3) == 0` (partition_strategy.c:1359) → the old
     `av1_4_partition_nn_*` weight set (LABEL_SIZE=4), **UNnormalized** features,
     `int_score[i] = (int)(100*score[i])`, `thresh = max_score − {500,500,200}` (16/32/64),
     zero-then-set from the label bits (:1472-1497). On these cells it prunes HORZ_4/VERT_4 at
     every 32×32 node (measured: scores like [530,−348,0,−392], thresh=30 → only label 0 ⇒ both
     pruned). The port's `predict_4partition_prune` guarded `level_index >= 3` as a NO-OP, so it
     searched HORZ_4 and found a cheaper 4-way (two-tone 64² cq12: child-0 HORZ_4 rdcost 12.9M vs
     NONE 16.5M) → root NONE→SPLIT. **Fix:** transcribe the OLD weight tables
     (`xtask/transcribe_part4_nn.py` → `part4_nn_weights.rs` `OLD_*`) + the old-branch decision in
     `part4_prune.rs` (normalize skipped, int-score/max−thresh, OVERWRITE-from-zero semantics —
     C can resurrect a pre-ML-cleared flag; the caller re-ANDs only the interior-envelope
     frame-fit guard). Also added the missing `av1_nn_output_prec_reduce` (ml.c:19 — BOTH
     `av1_ml_prune_4_partition` call sites pass `reduce_prec=1`; C's `+ 0.5` is a DOUBLE literal)
     to part4's NN — and the same latent gap in `ab_nn_prune.rs` (the AB NN call :1296 is also
     reduce_prec=1). Witness: `part4_old_nn_diff.rs` — 4000 random-input decisions identical to a
     REAL-`av1_nn_predict_c` oracle on the same OLD tables.
  2. **(speed>=4, closed ALL 5 speed-4 pins) the chroma-HOG force-disable tail was unported.**
     The UNCONDITIONAL tail of `set_allintra_speed_features_framesize_independent`
     (speed_features.c:608-616) zeroes `chroma_intra_pruning_with_hog` whenever
     `prune_chroma_modes_using_luma_winner` is on (allintra speed>=4; this also deadens the
     speed-5/6 `=3/4` settings). Measured: the instrumented C computes ZERO chroma-HOG masks at
     cpu-used=4. The port kept the HOG live at speed 4 and HOG-pruned UV_V_PRED where C evaluates
     and picks it (two-tone 64² cq12 root NONE: C uv=V 58469617 vs port uv=SMOOTH 58779332) →
     different chroma bytes. **Fix:** the tail in `SpeedFeatures::set_allintra` + the inline
     `chroma_hog_level` gate in `partition_pick.rs` (`&& !prune_chroma_luma_winner`); the
     `UvLoopPolicy` build now threads the luma-winner prune independently of the HOG mask
     (they were coupled — dropping the HOG must not drop the luma-winner prune).
- **Verified locally (worktree, rebased over 57d5ce0):** speed-3 gate 64/64, speed-4 gate 64/64
  (both promoted from pinned-residual to full byte-identity asserts), new single-cell asserted
  witnesses `kb7_rd_localize.rs` (cpu3 + cpu4, with decode-both diff on failure),
  `part4_old_nn_diff` 4000/4000, `speed4_allintra_deltas_match_source` corrected to the
  C-source value (`chroma_intra_pruning_with_hog == 0` at speed 4), full `cargo test -p
  aom-encode` **149 passed / 0 failed**. Speed-0/1/2 byte gates unaffected (the old-model branch
  only fires at level 3; the prec-reduce is decision-neutral on those grids — now faithful).

### KB-8 — Encoder: `--cpu-used=4` speed-4 deltas — PORTED ✅ (64/64 after the KB-7 roots; luma was byte-exact at 59/64)
- **Status (2026-07-16): every documented speed-4 delta is PORTED + LIVE — 64/64 cells byte-identical**
  vs real aomenc (`encoder_gate_speed4_textured_allintra`, {64,128}² × cq{12,32,48,63} ×
  {flat,two-tone,vgrad,diag} × {mono,420}), up from 35/64 baseline → 51/64 (chunk 1 series) →
  59/64 (the winner-mode flip) → **64/64 (the KB-7 roots: the level-3 OLD-model 4-way ML prune +
  the speed>=4 chroma-HOG disable tail — see KB-7)**. ALL 32 mono cells were already byte-exact
  at 59/64 (the speed-4 LUMA path); the 5 former 4:2:0 residuals (`diag 128² cq12`, `two-tone
  64² cq12/cq32`, `vgrad 128² cq12`, `vgrad 64² cq12`) were KB-7's two roots, not a missing
  speed-4 delta (confirmed: both are speed-feature gates, one shared with speed 3, one
  speed-4-specific).
- **The full landed chunk series (each verified on origin/main):**
  1. `prune_chroma_modes_using_luma_winner` + NON_DUAL LF search (e8c662f, 51/64).
  2. SATD trellis-skip body `skip_trellis_opt_based_on_satd` (16d4d85) — unit-tested vs REAL C
     (`ref_satd` = exported `aom_satd_c`).
  3. Stage-aware `TxTypeSearchPolicy` derivation (7bd30fb) — MODE_EVAL/WINNER_MODE_EVAL coeff-opt
     + tx-domain columns per `set_mode_eval_params`, validated vs the C tables.
  4. `USE_LARGESTALL` tx-size arm (42bdffc) — `choose_largest_tx_size` demotion tables verified vs C.
  5. `use_default_intra_tx_type` in `get_tx_mask_intra` (96eeb71) + threading (9c6ed2a) —
     differential vs the C shim across use_default × screen sweeps.
  6. Winner-mode two-pass skeleton in `rd_pick_intra_sby_mode_y` (0ee9f97) — `store_winner_mode_
     stats` C-semantics unit-tested; `use_rd_based_breakout` rd_thresh (AOMMIN) in the depth loop.
  7. Est-rd tx-type prune (264bba4) — `av1_cost_coeffs_txb_laplacian` (REAL-C differential across
     15,960 cases) + `prune_txk_type` + txk_map reorder; LIVE on intra in the WINNER pass.
  8. THE FLIP (this landing): `set_allintra(4)` real values (`perform_coeff_opt=5`,
     `tx_domain_dist_thres_level=3`, `fast_intra_tx_type_search=2`, `winner_mode_tx_type_pruning=2`,
     `prune_2d_txfm_mode=PRUNE_3`, `prune_tx_type_est_rd=1`, `enable_winner_mode_for_{coeff_opt,
     use_tx_domain_dist,tx_size_srch}=1`, `multi_winner_mode_type=MULTI_WINNER_MODE_DEFAULT(=2)`);
     `use_rd_based_breakout_for_intra_tx_search=1` at speed>=3 (:460 — speed-3 gate re-verified
     61/64, empirical no-op confirmed); the two-pass wiring in `partition_pick.rs` (per-leaf
     `WinnerModeCfg` derivation); BOTH split-info prunes (`prune_ext_part_using_split_info`:
     the AB `evaluate_ab_partition_based_on_split` at level 2 = speed>=4 — inert at qindex>=128
     by its threshold formula — and the 4-way `prune_4_partition_using_split_info` at level 1 =
     speed>=3, via `split_part_rect_win` rect-win threading through the SPLIT recursion).
- **Key facts for future speeds (verified against source):** `top_intra_model_count_allowed` stays
  **3** at speeds 4 AND 5 — the `=2` drop is **speed>=6** (:533, inside the `if (speed >= 6)`
  block at :527; an earlier note here mis-attributed it to speed>=5); `MULTI_WINNER_MODE_DEFAULT=2`
  / `FAST=1` (speed_features.h:226/230), `winner_mode_count_allowed={1,2,3}`; the AB split-info
  threshold `min(3*(2*(MAXQ-q)/MAXQ),3)` is 3 for q<=127 / 0 for q>=128; C's chroma search runs
  DEFAULT_EVAL (rdopt.c:3659 resets right after the luma two-pass); the winner re-eval
  (`intra_block_yrd`) gets NO ALLINTRA variance factor yet compares vs the factored first-pass
  best_rd (C asymmetry, preserved); C's LARGESTALL arm bypasses `uniform_txfm_yrd`'s rate assembly
  — equivalent to it with `tx_mode_is_select=false` (tx_size_rate=0), which is how the port models
  it.
- **Gate asserts FULL 64/64 byte-identity** — FAILS on any regression.

### KB-9 — Encoder: `--cpu-used=5` speed-5 deltas — PORTED ✅ (64/64 byte-identical, 0 residuals)
- **Status (2026-07-16): every speed-5 delta is PORTED + LIVE — 64/64 cells byte-identical** vs
  real aomenc (`encoder_gate_speed5_textured_allintra`, {64,128}² × cq{12,32,48,63} ×
  {flat,two-tone,vgrad,diag} × {mono,420}). No pinned residuals: the two cells that had been
  KB-7-pinned at speed 4 (`two-tone 64² cq12`, `vgrad 128² cq12` — since closed there by the
  KB-7 roots) byte-match at speed 5 independently, because the AB/4-way disable (below) removes
  the near-tie's partition candidates from the search space entirely.
- **LIVE deltas (each individually witness-verified by bisect during landing):**
  1. `winner_mode_sf.multi_winner_mode_type = MULTI_WINNER_MODE_FAST` (:524): the luma two-pass
     stores/re-evaluates the top-**2** winners (speed 4: top-3) — `winner_mode_count_allowed`
     rdopt_utils.h:236, already parameterized through `WinnerModeCfg::max_winner_count`. Flips
     `two-tone 64² mono cq63` + `420 cq63` on the gate grid (the mono flip proves it luma-side).
  2. `part_sf.ext_partition_eval_thresh`: default BLOCK_8X8 through speed 4; at speed 5 the
     framesize-independent :510-511 sets `screen ? BLOCK_8X8 : BLOCK_16X16`, then the
     qindex-dependent `aggr = AOMMIN(4, speed-2)` == 3 arm (:2947-2962) sets **BLOCK_128X128
     UNCONDITIONALLY for sub-480p frames** (no boosted/intra gate) → `bsize > thresh` never holds
     → **AB + 4-way partitions are never evaluated** on sub-480p KEY frames. Consumers:
     `allow_ab_partition_search` (partition_search.c:4005) + `prune_4_way_partition_search`
     (:4136), both now read `ext_partition_eval_thresh_allintra_key` (partition_pick.rs; the
     other qindex-dep arms are dead on KEY — boosted + intra-only; speed>=6 = BLOCK_128X128 for
     ALL sizes; `ext_part_eval_based_on_cur_best` is GOOD-only, :1013). Flips the 2 cq12 cells (the former speed-4 KB-7 pins).
- **Set-then-overridden:** `chroma_intra_pruning_with_hog = 3` (:515) is zeroed by the :608-615
  final override (chroma HOG off at speed>=4 — the KB-8 entry documents the override fix).
- **Screen-only:** `intra_cnn_based_part_prune_level`: screen arm 0 → 1 (:512-513; non-screen
  stays 2). Wired through the existing CNN prune (`predict_decision` handles level 1's
  `none_disallowed` exemption); byte-inert on the (non-screen) gate grid.
- **Verified INERT on the allintra KEY envelope:** `simple_motion_search_prune_agg=LVL5` (:509,
  motion), `use_coarse_filter_level_search=0` (:517, ALREADY the default — init :2532),
  `disable_wiener/sgr_filter` (:519-520, restoration off), `prune_mesh_search=LVL_2` (:522,
  intrabc/motion), qindex-dep `winner_mode_tx_type_pruning=3` (:3059, `!(intra||screen)` —
  stays 2), qindex-dep `prune_sub_8x8_partition_level=0` (:3070, field only raised at speed>=6),
  qindex-dep `rect_partition_eval_thresh` aggr 0→1 (:2980, `!boosted`). The framesize-DEPENDENT
  setter has NO speed-5 block (:302 jumps 4→6). LF stays NON_DUAL (:496; LPF_PICK_FROM_Q is
  speed>=6), tx/winner tables all carry speed-4 values.
- **Anti-vacuous witness (asserted):** `encoder_gate_speed5_vs_speed4_sf_witness` — the port with
  SPEED-4 features vs real `aomenc --cpu-used=5` DIVERGES (4 cells incl. mono cq63); with speed-5
  features it matches. Gate asserts full 64/64 — FAILS on any regression.
- **Speed-6 prep facts (verified against source while here):** speed>=6 block :527-564 —
  `top_intra_model_count_allowed=2` (:533), `prune_filter_intra_level=2` (:529),
  `intra_pruning_with_hog=4` (:531) + `chroma_intra_pruning_with_hog=4` (:530, still overridden
  to 0), `cfl_search_range=1` (:532), `adapt_top_model_rd_count_using_neighbors=1` (:534),
  `prune_luma_odd_delta_angles_in_intra=1` (:535), `multi_winner_mode_type=OFF` (:561),
  `prune_winner_mode_eval_level=1` (:562), `dc_blk_pred_level=1` (:563), `winner_mode_tx_type_
  pruning=3` + `prune_tx_type_est_rd=0` (:551-552), `prune_intra_tx_depths_using_nn` (:553),
  `perform_coeff_opt=6` + `tx_domain_dist_level=3` (:555-556), `lpf_pick=LPF_PICK_FROM_Q` (:559 —
  **building block LANDED**: `pick_filter_level_from_q` in lf_search.rs, oracle-validated vs real
  cpu-6 header levels by `speed6_prep_lf_from_q_matches_real_aomenc`; needs only the harness flip),
  partition prunes :537-546 (`prune_rectangular_split_based_on_qidx=2`, `prune_rect_part_using_
  4x4_var_deviation/none_pred_mode`, `prune_sub_8x8_partition_level=1`, `prune_part4_search=3`,
  `default_max_partition_size=BLOCK_32X32`!), framesize-dep :304-316 (`use_square_partition_only_
  threshold=BLOCK_16X16` etc.). Substantially new machinery (LPF-from-Q, NN tx-depth prune,
  DC-block prediction, odd-delta-angle prune) — NOT a pure re-parameterization like speed 5 was.
  **All of the above LANDED — see KB-10.**

### KB-10 — Encoder: `--cpu-used=6` speed-6 deltas — PORTED ✅ (64/64 canon; the noise-extension cq63 near-tie CLOSED 2026-07-31 by KB-21 root #2)
- **Status (2026-07-16): the canon gate is 64/64 byte-identical** vs real aomenc
  (`encoder_gate_speed6_textured_allintra`, {64,128}² × cq{12,32,48,63} ×
  {flat,two-tone,vgrad,diag} × {mono,420}) + the anti-vacuous witness
  (`encoder_gate_speed6_vs_speed5_sf_witness`: port with FULL speed-5 features vs
  `aomenc --cpu-used=6` DIVERGES on `vgrad 64² cq32` mono+420; with speed-6 features it
  matches). Speed 0-5 gates all re-verified byte-unchanged. Speed 6 is NEW MACHINERY
  (speed_features.c:527-564 + framesize-dep :304-316 + qindex-dep aggr=4), not a
  re-parameterization — landed as one chunk after the KB-9 prep-facts series:
  1. **`lpf_pick = LPF_PICK_FROM_Q`** (:559): the closed-form KEY LF derivation
     (`lf_search::pick_filter_level_from_q`, chunk-1 building block 5935250,
     oracle-validated vs real cpu-6 headers) replaces the reconstruction search —
     wired in the harness LF derivation at `speed >= 6` (the `non_dual` flag's shape).
  2. **Partition prunes** (bisect: baseline chunks-2+3 took the map 0→54/64):
     `default_max_partition_size = BLOCK_32X32` (:546 — `set_max_min_partition_size`
     min(sf, CLI cap, sb) forces square-split-only at the 64² root),
     `use_square_partition_only_threshold = BLOCK_16X16` (framesize-dep :315),
     `ext_partition_eval_thresh = BLOCK_128X128` for ALL sizes (qindex-dep aggr=4
     else-arm :2963), `prune_rectangular_split_based_on_qidx = 2` (:537, the
     qindex-thirds rect kill), `prune_rect_part_using_4x4_var_deviation` (:539 — arm 2
     of the ALLINTRA var block, `do_rectangular_split = 0` when `var_max - var_min <
     3.0`; also WIDENS the stats computation to sub-16x16 nodes),
     `prune_rect_part_using_none_pred_mode` (:540 — post-NONE mode-class rect prune;
     needs the new `ModeGrid::bsizes` neighbour-bsize stamps for
     `is_neighbor_blk_larger_than_cur_blk`), `prune_sub_8x8_partition_level = 1`
     (:541 — disable splits at 8x8 when either neighbour block is larger),
     `prune_part4_search = 3` (:543 — inert: 4-way is off via the ext threshold).
  3. **Intra mode loop**: `top_intra_model_count_allowed = 2` (:533) +
     `adapt_top_model_rd_count_using_neighbors` (:534 — the neighbour-mode-adaptive
     prune slot; machinery pre-existed in intra_rd.rs, now threaded),
     `prune_luma_odd_delta_angles_in_intra` (:535 — evens-first delta order
     `{-2,2,-3,-1,1,3}` + the even-neighbour rd_thresh prune; pre-existed, now gated
     on), `intra_pruning_with_hog = 4` (:531, luma HOG threshold 0.4),
     `prune_filter_intra_level = 2` (:529 — no filter-intra search,
     intra_mode_search.c:239).
  4. **predict_dc skip-block prediction** (`dc_blk_pred_level = 1`, :563 → per-stage
     `predict_dc_levels[1] = {1,1,0}`): `predict_dc_only_block` (tx_search.c:2011) in
     the DEFAULT_EVAL + MODE_EVAL tx-type searches — `pixel_diff_stats` (DOUBLE-norm
     mse/mean/var over the visible txb) + the low-var/low-mean eob-0 skip fast path.
     KEY QUIRK ported: the skip path's `zero_blk_rate` reads `get_txb_ctx` at the
     BLOCK ORIGIN from the PERSISTENT entropy arrays (C re-derives ctxa/ctxl via
     `av1_get_entropy_contexts` and passes UN-offset pointers — every txb of the block
     shares the origin ctx; threaded as
     `TxTypeSearchInputs::predict_skip_zero_blk_rate`). Bisect: flips 4 canon cells
     (diag 128² mono+420 cq32/48 — mono proves it luma-side); 2384 luma fires on the
     canon grid, chroma fires on the flat-uv extension cells.
  5. **8x8 NN intra-tx-depth prune** (`prune_intra_tx_depths_using_nn`, :553):
     `ml_predict_intra_tx_depth_prune` (tx_search.c:2823) — transcribed weights
     (`xtask/transcribe_intra_tx_nn.py` → `intra_tx_nn_weights.rs`),
     `get_mean_dev_features` (14 features incl. log1pf(source_variance) +
     log1pf(dc_q²/256)), 16-node ReLU + prec-reduce, thresholds ±0.405465 →
     TX_PRUNE_SPLIT (skip smaller depths) / TX_PRUNE_LARGEST (abort largest eval).
     Threaded into `choose_tx_size_type_from_rd_intra`'s largest-depth walk via
     `NnDepthPruneCtx` (needs `TxfmYrdEnv::qindex`, new field). **Differential:
     `intra_tx_nn_diff` — 4000/4000 randomized decisions identical to the REAL
     `av1_nn_predict_c` (ref_nn_predict) on the same tables, all three verdicts
     exercised.** Byte-inert on the canon grid (no 8x8 leaf searches there — probes
     measured 0 calls); LIVE on the noise extension (96 Split verdicts, byte-exact at
     cq32/48).
  6. **Winner-mode restructure**: `multi_winner_mode_type = OFF` (:561) —
     `store_winner_mode_stats` returns immediately (rdopt_utils.h:688; count-1 arm in
     intra_rd.rs) and the re-eval runs ONCE on `best_mbmi` (C's else-arm,
     intra_mode_search.c:1727-1737 — including a filter-intra winner);
     `prune_winner_mode_eval_level = 1` (:562) — `bypass_winner_mode_processing`
     skips the re-eval when `source_variance < 64 - 48*qindex/256`.
  7. **Chroma narrowing**: `cfl_search_range = 1` (:532 — est-only CfL refinement +
     the range-1 invalid/overhead early-outs; machinery pre-existed in intra_uv_rd,
     now threaded via UvLoopPolicy). Bisect: flips 8 canon cells (all 4:2:0 gradient —
     vgrad/diag 64²+128² cq12-48). `prune_smooth_intra_mode_for_chroma` (:528 — prune
     UV_SMOOTH when BOTH chroma planes' per-pixel source variance < 20,
     intra_mode_search.c:850) — consumer wired (pre-existed), currently UNREACHED on
     all grids (the speed>=4 luma-winner mask only admits UV_SMOOTH when the luma
     winner is SMOOTH-family; carried transcription-faithful).
  8. **rd tables**: `perform_coeff_opt = 6` (:555, columns {432,97}/{86,16}) and
     `tx_domain_dist_level = 3` (:556 — types row {2,2,2}: the WINNER pass moves to
     tx-domain distortion); `winner_mode_tx_type_pruning = 3` + `prune_tx_type_est_rd
     = 0` (:551-552 — the est-rd prune turns OFF again; the PRUNE_5/PRUNE_2 stage rows
     are carried but inert on intra with est-rd off).
- **Verified INERT on the allintra KEY envelope:** `mv_sf.use_bsize_dependent_search_
  method = 3` (:548, motion) and `intrabc_search_level = 1` (:549, screen-only intrabc);
  `cdef_pick_method = CDEF_FAST_SEARCH_LVL4` (:558, CDEF off); qindex-dep
  `rect_partition_eval_thresh` (boosted-gated, KEY is boosted); the qindex-dep
  speed>=5 screen sub-8x8 re-zero (screen arm). `chroma_intra_pruning_with_hog = 4`
  (:530) is still zeroed by the :608-616 tail (chroma HOG stays OFF at speed>=4).
- **WAS PINNED OPEN, CLOSED ✅ 2026-07-31 by KB-21 root #2:** `noise 64² cq63` (mono +
  420) on the `encoder_gate_speed6_noise_flatuv_allintra` extension. The old
  localization — the (mi 8,0) 32×32 leaf's WINNER-pass tx-size sweep picking TX_16X16
  over TX_32X32 by a 0.19% rd margin — was a SYMPTOM: the rd it compared was computed
  from a mis-ordered `prune_txk_type` est-rd and an FP-instead-of-B quantized txb (see
  KB-21 root #2). That also explains why no single-feature revert closed it: neither
  defect is an sf field. **The gate is now a full byte-identity assert over all 6 cells**
  (cq32/48/63 × mono/420).
- **Unit locks:** `speed6_allintra_deltas_match_source` (the full sf-block field set +
  stage policies incl. predict_dc columns + the speed-5 regression guard);
  `store_winner_mode_stats_matches_c_semantics` (the OFF count-1 no-store arm);
  `intra_tx_nn_diff` (REAL-C NN differential). The harness `max_partition_size` is now
  sf-driven (`min(default_max, CLI cap, SB)` — BLOCK_64X64 through speed 5, unchanged
  consumer outcomes; BLOCK_32X32 at 6).

### KB-11 — Encoder: `--cpu-used=7` speed-7 VAR_BASED_PARTITION — PORTED ✅ (64/64 canon; the KB-10-twin noise near-tie CLOSED 2026-07-31 by KB-21 root #2)
- **Status (2026-07-17): the canon gate is 64/64 byte-identical** vs real aomenc
  (`encoder_gate_speed7_textured_allintra`, {64,128}² × cq{12,32,48,63} ×
  {flat,two-tone,vgrad,diag} × {mono,420}) + the anti-vacuous witness
  (`encoder_gate_speed7_vs_speed6_sf_witness`: port with FULL speed-6 features vs
  `aomenc --cpu-used=7` DIVERGES on `vgrad 64² cq32` mono+420; speed-7 features match) +
  the deep-tree noise extension (`encoder_gate_speed7_noise_flatuv_allintra`, cq12/32/48
  hard-asserted). Speed 7 is STRUCTURALLY NEW (speed_features.c:569-575) — the partition
  tree is FIXED up front from variance thresholds, no RD partition search:
  1. **`av1_choose_var_based_partitioning` KEY arm** (`var_part.rs::
     choose_var_based_partitioning_key`): 4x4-downsampled variance tree
     (`fill_variance_4x4avg`: `aom_avg_4x4(src) − 128` per 4x4 — [`avg_4x4`]
     differentially locked 4000/4000 vs the REAL exported `aom_avg_4x4_c`,
     `avg_4x4_diff.rs`); `set_vbp_thresholds_key_frame` (`threshold_base = 120 *
     av1_ac_quant_QTX(qindex, 0, bd)`; <720p: t[2]=base/3, t[3]=base>>1; t[0]=t[1]=base;
     t[4]=base<<2); stage-2 force-split (16x16 var > t[3] and 32x32 var > t[2] propagate
     ONLY_SPLIT up; 64x64/128x128 have no key forcing rules but `set_vt_partitioning`'s
     `bsize > BLOCK_32X32 → split` KEY rule caps NONE at 32x32); the assignment descent
     with the sb64 boundary half-fit extensions (`bs_width_check = (w>>1)+1` at the frame
     edge) and edge-fit VERT/HORZ pair stamps; leaves stamped as mi-grid bsizes at block
     top-lefts (`set_block_size`), read back by `get_partition_from_stamps` (= C's
     `get_partition`, av1_common_int.h:1775, ext-partition disambiguation included).
     NOTE: interior rect stamps are reachable only on exact `variance == threshold` ties
     (stage-2 forcing fires strictly-above, NONE strictly-below) — the rect arms' real
     purpose is frame-edge blocks (unit-locked on a 48x48 frame).
  2. **`av1_rd_use_partition`** (`partition_pick.rs::rd_use_partition_real`,
     partition_search.c:1764): the fixed-tree walk running the EXISTING full-RD
     `leaf_pick_sb_modes` per leaf (`use_nonrd_pick_mode` stays 0 until speed 8) with
     C's exact context shape — HORZ/VERT strip-0-then-encode-then-strip-1
     (`encode_b_intra_dry` mid-stage propagation, the rect stage's own pattern), SPLIT
     recursion with `do_recon = i != 3` (last child skips its re-encode), per-node
     save/restore + `if (do_recon) encode_sb` (OUTPUT_ENABLED at the SB root, DRY below).
     Leaf budgets are `invalid_rdc` (INT64_MAX — no early-outs on a fixed tree).
     **Structurally DEAD at allintra speed 7 (verified, documented in the fn docs, NOT
     ported):** the PARTITION_NONE re-eval (:1827) + split-of-NONEs re-eval (:1986) —
     both need `adjust_var_based_rd_partitioning` ∈ {1,2}/{>2}, which is **0 outside
     REALTIME** (init :2288; setters :2002/:2896 are REALTIME-only) → the walk is a pure
     replay, its RD totals decision-inert; `setup_block_rdmult`'s ALLINTRA
     `intra_sb_rdmult_modifier` fold is IDENTITY (only av1_rd_pick_partition's root
     recomputes the modifier, partition_search.c:5715 — the VBP path leaves the per-SB
     reset 128, encodeframe.c:1303) → **`pack_tile` skips the SB rdmult fold at
     speed >= 7** (byte-visible: the fold is live at speeds 0-6).
  3. **sf deltas** (`speed7_allintra_deltas_match_source`): `partition_search_type =
     VAR_BASED_PARTITION` (:571; pack_tile derives the branch inline as `allintra &&
     speed >= 7` per the established pattern) + `default_min_partition_size = BLOCK_8X8`
     (:570 — assertion-only: the KEY tree never stamps below 8x8; the RD-search max/min
     clamps never run on this path). INERT (verified vs source): `cdef_pick_method =
     CDEF_PICK_FROM_Q` (:572, CDEF off in allintra), `rt_sf.mode_search_skip_flags |=
     FLAG_SKIP_INTRA_DIRMISMATCH` (:573 — sole consumer `search_intra_modes_in_interframe`,
     rdopt.c:5824, inter frames only), `rt_sf.var_part_split_threshold_shift = 7` (:574 —
     `set_vbp_thresholds_key_frame` reads it ONLY under
     `rt_sf.force_large_partition_blocks_intra`, which is 0 below speed 8/720p+
     [speed_features.c:327] and in this envelope; carried as a field for provenance).
     Everything else carries the speed-6 set unchanged (incl. LPF_PICK_FROM_Q).
- **WAS PINNED OPEN (the KB-10 near-tie TWIN), CLOSED ✅ 2026-07-31 by KB-21 root #2:**
  `noise 64² cq63` (mono + 420) on `encoder_gate_speed7_noise_flatuv_allintra`. The
  localization by `kb11_speed7_noise_localize.rs` was accurate as far as it went —
  decoded partition trees IDENTICAL, every decoded mode record matching, and the port's
  (mi 8,0) leaf carrying **tx_size TX_16X16 where real keeps TX_32X32** (the tx-plan
  difference desyncing the LARGEST-tx parse; the decoded eob-50 / 420 "(8,8) tree diff"
  were desync artifacts) — but the tx-size choice was downstream of the two KB-21 root-2
  coefficient-path defects, exactly as predicted ("closes both speeds' cells at once").
  Both gates are full byte-identity asserts now; the localizer test is
  `kb11_speed7_noise_cq63_byte_matches` and still prints the structural diff on failure.
- **Unit locks:** `speed7_allintra_deltas_match_source` (+ speed-6 regression guard);
  `avg_4x4_diff` (REAL-C kernel differential); var_part.rs threshold/shape/edge tests;
  `kb11_speed7_noise_cq32_control_matches` (the localization harness's own soundness).
- **Speed-8 prep facts (KB-12 seed, verified against source 2026-07-17):** speed 8 flips
  `use_nonrd_pick_mode = 1` (speed_features.c:578) — the nonrd PICKMODE, the big one:
  - `encode_nonrd_sb` (encodeframe.c:581-663): the SAME
    `av1_choose_var_based_partitioning` (KEY arm already ported, var_part.rs) fixes the
    tree, then **`av1_nonrd_use_partition`** (partition_search.c:2960) — a SINGLE-PASS
    walk: per leaf `pick_sb_modes_nonrd` + `encode_b_nonrd` IMMEDIATELY (dry_run=0 — the
    encode IS the output; NO save/restore, NO mid-strip re-encode, NO root winner walk;
    `set_mode_eval_params(DEFAULT_EVAL)` per node). HORZ/VERT strips: pick+encode strip 0
    then strip 1 (in-frame gated, `bsize > BLOCK_8X8` for strip 1). SPLIT: plain
    recursion. `try_merge` (`nonrd_check_partition_merge_mode = 1`, :580) is
    `!frame_is_intra_only`-gated → INERT on KEY; `nonrd_check_partition_split` stays 0;
    `direct_partition_merging` is `!frame_is_intra_only` too.
  - The KEY-intra leaf search is `hybrid_intra_mode_search` (partition_search.c:756):
    `hybrid_intra_pickmode = 2` (:579) → full-RD `av1_rd_pick_intra_mode_sb` (the
    EXISTING ported search) for `bsize < BLOCK_16X16 && x->source_variance >=
    var_thresh[1] = 101`; else **`av1_nonrd_pick_intra_mode`** (nonrd_pickmode.c:1582) —
    NEW machinery: `intra_mode_list` loop (RTC_INTRA_MODES = DC/V/H/SMOOTH) with
    `intra_y_mode_bsize_mask_nrd` (:583-590: INTRA_DC only >= BLOCK_32X32, INTRA_DC_H_V
    below — mask consumed where? verify: the mask gates the loop in nonrd inter path;
    the intra-frame fn loops intra_mode_list directly), per-mode
    `av1_estimate_block_intra` (foreach-txb SATD/model estimate, not full RD),
    skip_txfm-cost fold + `bmode_costs[y_mode_costs[above_ctx][left_ctx]]`, tx_size =
    min(max_txsize_lookup, biggest for tx_mode) — NO tx search, NO angle deltas, NO
    filter-intra. Palette arm gated `enable_palette && allow_screen_content_tools`
    (`prune_palette_search_nonrd = 1`, :582). CHROMA on the nonrd KEY path: locate it
    (av1_nonrd_pick_intra_mode is PLANE_Y only — uv likely inside encode_b_nonrd's
    encode_superblock or a uv estimate step; UNRESOLVED, first thing to trace).
  - `x->source_variance` IS live at speed 8: choose_var computes it per SB
    (var_based_part.c:1728, `use_nonrd_pick_mode && source_sad_nonrd > kLowSad`;
    content_state_sb.source_sad_nonrd inits kMedSad per SB, encodeframe.c:1289) — but
    verify what `pick_sb_modes_nonrd` re-derives per LEAF before trusting the SB value
    in hybrid_intra's threshold.
  - `var_part_split_threshold_shift = 8` (:581): STILL force_large-gated on KEY
    (`force_large_partition_blocks_intra` rises only at speed>=8 AND 720p+ —
    speed_features.c:326-328) → inert on sub-720p grids; LIVE at 720p+ (the
    `set_vbp_thresholds_key_frame` shift-steps arm + thresholds[2]/[3] shift_val 1 —
    port the arm when a 720p+ speed-8 cell lands).
  - `encode_b_nonrd` (partition_search.c:2100): the single-pass leaf encode
    (av1_update_state-equivalent + encode_superblock with dry_run=0 → tokens + cdf
    updates inline as it walks — the pack IS the walk; the port's search/pack split
    needs rethinking for this path, or model it as search==pack in one pass).

### KB-P29 — Encoder: palette 128² AB/4-way partition near-tie (2 cells) — PINNED (genuine; palette machinery C-faithful)
- **Status (2026-07-17 pickup):** the palette-Y+UV RD search (#29, PARITY Section B) is 5/7
  byte-exact — those 5 cells are now HARD byte-identity asserts in
  `rd_close_palette::palette_y_rd_close_gate`. The 2 remaining CLOSE cells (`ui_420_128_cq32`,
  `text_420_128_cq20`, both 128² 4:2:0) are PINNED as genuine palette-induced AB/4-way partition
  RD near-ties, NOT a palette-cost bug.
- **Decode-both localized** (`decode_diff_palette_close_cells`, the regression guard):
  `ui` diverges at (mi 0,0) BLOCK_32X32 real PARTITION_HORZ_B vs port PARTITION_HORZ_4;
  `text` at (mi 8,20) BLOCK_16X16 real PARTITION_VERT vs port PARTITION_VERT_A.
- **Both cells are BYTE-EXACT with palette OFF** (the localizer's palette-OFF control proves it),
  so partition/mode/tx are correct; the palette contribution alone tips the AB/4-way tie. The
  palette machinery is verified C-faithful: `av1_allow_palette`, `av1_get_palette_bsize_ctx`
  (`num_pels_log2[bsize] − num_pels_log2[8X8]`), `av1_get_palette_mode_ctx` (above+left palette
  count), k-means (rtcd-validated), and mid-search neighbour palette cache/ctx stamping (the
  winner's palette is threaded at every `grid.stamp`) all match C; the byte-exact 64² palette
  cells exercise the same non-square block sizes. Same class as the KB-10/KB-11 pinned near-ties.
- **Next step (deferred, the close move):** sibling-C per-candidate partition-RD dump at the two
  divergent nodes (the KB-2/3/7 method) — compare C's HORZ_B-vs-HORZ_4 / VERT-vs-VERT_A RD with
  palette ON to find whether a specific leaf's palette RD/flag (or a pruning gate) tips it. The
  localizer asserts the divergence PRESENT, so any fix self-promotes the cell into
  `BYTE_EXACT_CELLS`.
### KB-12 — Encoder: `--cpu-used=8/9` nonrd PICKMODE — PORTED ✅, and the estimate-arm residual is CLOSED ✅ 2026-08-02 (speed-8 AND speed-9 **64/64** canon + noise 8/8; the "leaf-mode near-tie" was `aom_hadamard_lp_8x8`'s missing trailing transpose) — GATE 2 (cpu 0-9) COMPLETE
- **Status (2026-07-17, updated 2026-08-02): speed 8 AND speed 9 land, and Gate-2 (cpu-used
  0..9) is byte-complete with NO pinned cells** — the 4 speed-8 near-ties that stood here from
  2026-07-17 closed 2026-08-02 (see the transpose section below).** The nonrd PICKMODE (`use_nonrd_pick_mode = 1`,
  speed_features.c:578): the SAME `av1_choose_var_based_partitioning` KEY tree the speed-7
  gate fixes now drives **`av1_nonrd_use_partition`** (partition_pick.rs `nonrd_use_partition_
  real`) — a SINGLE-PASS walk (NO save/restore, NO mid-strip re-encode, NO root winner walk):
  per leaf `hybrid_intra_mode_search` then `encode_b_intra_dry(output_enabled=true)` (C's
  `encode_b_nonrd`, dry_run=0) immediately; bits via the unchanged `pack_sb` re-walk (same
  search==pack split proven for speeds 0-7). `try_merge`/`direct_partition_merging` are
  `!frame_is_intra_only`-gated → KEY-dead (not modelled).
  - **Leaf search — `hybrid_intra_mode_search`** (partition_search.c:756): `hybrid_intra_
    pickmode = 2` at speed 8 → full-RD `av1_rd_pick_intra_mode_sb` (the EXISTING
    `leaf_pick_sb_modes`) for `bsize < BLOCK_16X16 && source_variance >= var_thresh[1] = 101`,
    else the ESTIMATE arm `av1_nonrd_pick_intra_mode` (nonrd_pickmode.rs, NEW +880): the
    DC/V/H/SMOOTH `intra_mode_list` loop, per-mode `av1_estimate_block_intra` = one txb
    `av1_block_yrd` (LP Hadamard SATD estimate: `hadamard_lp_8x8/16x16` + `quantize_lp` +
    `satd_lp` + `block_error_lp`, all `wrapping`-i16, over the `*_lp_*_transpose` scans),
    skip-cost fold + `bmode_costs`, tx_size = max-square, NO tx search / angle delta /
    filter-intra. Speed 9: `hybrid_intra_pickmode = 0` → EVERY leaf is the estimate arm, plus
    the three estimate-loop prunes (`prune_h_pred_using_best_mode_so_far`, `enable_intra_mode_
    pruning_using_neighbors`, `prune_intra_mode_using_best_sad_so_far`) and `INTERNAL_COST_
    UPD_OFF` (<4k → every SB reads the FRAME-INIT cost tables; `sb_real` becomes an `Option`
    in `pack_tile`/`pack_tile_from_trees`, byte-visible on 128² multi-SB cells).
  - **The nonrd CHROMA path — RESOLVED (the KB-11 flagged unknown):** `av1_nonrd_pick_intra_
    mode` is PLANE_Y only and hard-sets `mi->uv_mode = UV_DC_PRED` (nonrd_pickmode.c:1735, "Keep
    DC for UV since mode test is based on Y channel only"). Estimate leaves code chroma as DC
    via the ordinary leaf encode (`LeafWinner{uv_mode:0}`); CfL never a candidate; full-RD leaves
    keep the existing uv search. Confirmed byte-exact by the mono+420 gate agreement.
  - **`output_enabled = true` (the KEY correctness item, KB-4 class):** C's nonrd walk encodes
    every leaf dry_run=0 (OUTPUT_ENABLED) → tx_type_map COPY semantics (eob-0 → DCT_DCT resets
    go to a transient frame map, the search winner's `w.tx_type_map` survives to `pack_sb`).
    `false` (alias) would re-introduce the KB-4 reset-leak on the full-RD arm. Matches the
    speed-7 SB-root walk (`output_enabled = bsize == sb_size`) + the pack walk (pack.rs:450).
  - **The 2 salvage blockers (fixed):** (1) `pack.rs` `sb_pick_cfg` dangled on the `Option<sb_
    real>` after the cost-upd-off refactor → `match &sb_real { Some => build; None => *pick_cfg }`
    (frame-init fallback = INTERNAL_COST_UPD_OFF); (2) `nonrd_use_partition_real` was DISPATCHED
    FROM NOWHERE → wired the `allintra && speed >= 8` branch into `pack_tile` (mirrors the
    speed-7 VBP dispatch; `speed >= 9` toggles the vbp 16×16 min/max-sub-var split prune, inert
    <720p). Plus the mechanical arity fixes the concurrent palette work introduced
    (`ModeGrid::stamp` + `LeafWinner` gained palette params/fields).
- **Gates (encoder_gate_e2e_byte_match.rs):** `encoder_gate_speed9_textured_allintra` **64/64**
  + `encoder_gate_speed9_noise_flatuv_allintra` **8/8** (cq12/32/48/63) + `encoder_gate_speed9_
  vs_speed8_sf_witness`; `encoder_gate_speed8_textured_allintra` **64/64** (60/64 until
  2026-08-02) + `encoder_gate_speed8_noise_flatuv_allintra` **8/8** +
  `encoder_gate_speed8_vs_speed7_sf_witness`. Speeds
  0-7 re-verified byte-unchanged (full `cargo test -p aom-encode` green). NOTE: the KB-10/KB-11
  noise-cq63 (mi 8,0) TX_16X16-vs-TX_32X32 near-tie does NOT reproduce at speed 8/9 — the
  estimate arm codes tx_size = max-square directly (no winner-pass tx sweep to flip), so the
  speed-8/9 noise cq63 cells byte-match (unlike speeds 6/7).
- **THE "NEAR-TIE" WAS A DROPPED TRANSPOSE — FIXED ✅ 2026-08-02 (`aom_hadamard_lp_8x8`).**
  The four pinned speed-8 `diag` cells, KB-32's whole surviving residual and KB-28's speed-8/9
  rows were ONE root: `nonrd_pickmode::hadamard_lp_8x8` omitted the trailing transpose C
  performs at **`aom_dsp/avg.c:232-236`** (*"Extra transpose to match SSE2 behavior (i.e.,
  aom_hadamard_lp_8x8_sse2)"* — `coeff[i * 8 + j] = buffer2[j * 8 + i]`), so the lowbd estimate
  arm's coefficients were the exact TRANSPOSE of libaom's, and `aom_hadamard_lp_16x16`'s the
  per-64-quadrant transpose. **Not an ISA divergence**: `aom_hadamard_lp_8x8_c` and `_neon`
  agree bit-for-bit over the whole 9-bit residual domain (measured), unlike `aom_hadamard_16x16`
  (LIBAOM_UPSTREAM_NOTES A1 / KB-20 root #4). Fix: one 4-line transpose loop
  (`crates/aom-encode/src/nonrd_pickmode.rs`).
  - **WHY IT WORE A NEAR-TIE'S CLOTHES, which is the transferable part.** Every consumer of
    those coefficients except the EOB is ORDER-INVARIANT: `aom_satd_lp` and `av1_block_error_lp`
    are sums over the whole array, `eob == 0` (the `skippable` flag) is a set property, and
    `eob == 1` can only mean the DC — a transpose fixed point. So rate, distortion and
    skippability were all RIGHT, and the single quantity that moved was `eob` itself, through
    `eob_cost += get_msb(eob + 1)` into `rate += eob_cost << 9`. **Measured: the pre-fix kernel
    changed the eob on 477 of 4,000 correlated 8x8 blocks and changed satd / block-error /
    skippable on ZERO** (`nonrd_block_yrd_lp_diff::lp_hadamard_transpose_is_load_bearing_and_
    only_moves_the_eob`). A defect that perturbs one small additive term inside a four-way RD
    comparison expresses itself as an occasional mode flip and nothing else — which is exactly
    what four separate localization passes read as "a genuine tie at ~0.7 % rdcost". Playbook
    §10's "never infer the mechanism from the SIZE of the delta" has a twin: **never infer it
    from the delta's SHAPE either.** Sign-random, sub-byte-per-superblock, flat in area, only
    at leaves — all four held, and it was still a kernel bug.
  - **THE ROOT CAUSE OF THE ROOT CAUSE: there was no differential.** KB-12 recorded *"the whole
    traced estimate chain (the LP kernels + `quantize_lp` + `block_yrd` structure + the mode
    loop) matches libaom line-for-line"*. That was a READING. The five lowbd kernels were the
    only hand-transcribed kernels in the tree with no lock against their exported C symbol —
    the hbd twin has had `nonrd_block_yrd_hbd_diff.rs` since KB-20. The in-module unit tests
    that did exist (`hadamard_lp_8x8_flat`, `hadamard_lp_16x16_flat`) are transpose-BLIND by
    construction: a flat input puts all the energy at coefficient 0, the transpose's fixed
    point. Playbook §1 in its purest form. New gate:
    **`crates/aom-encode/tests/nonrd_block_yrd_lp_diff.rs`** (5 tests) — every kernel vs the
    exported `_c`, the SIMD tier vs `_c` over the reachable range with the magnitude bound
    asserted, the whole `block_yrd_lowbd` walk vs a C-composed oracle (2,400 walks x skippable
    / coded / edge-clamped coverage asserted), and the teeth above. Plus a golden
    asymmetric-impulse vector in-module (`hadamard_lp_8x8_golden_asymmetric_impulse`) and a
    speeds-0..9 reachability lock that both `hybrid_intra_pickmode` arms reach the kernel
    (`estimate_arm_is_reachable_from_both_hybrid_arms`).
  - **BITE PROOF (playbook §1).** Reverting the transpose ALONE reproduces every recorded
    pre-fix number exactly: the 4 KB-12 `diag` cells (speed-8 gate back to 60/64, speed 9 still
    64/64 at 64²/128² — the prunes really do mask it there); KB-32's ladder 512² +61, 768² −50,
    896² −23, 1024² −168, 2048² +21, 2176² s9 −184; KB-32's localizer back to the SAME first
    leaf and mode pair it recorded (512² cq30 s8 mi(4,108) real SMOOTH / port DC; 2176² cq30 s9
    mi(108,174) real DC / port V); KB-28's map back to −132/+36 at 1280x720 cpu8 cq24/cq40 and
    −8/+4 at cpu9, 1280x704 −1/−6, 1216x768 +1/−8. `nonrd_speed9_area_threshold_byte_identical`
    stays GREEN under the revert, so this root's cell set is disjoint from KB-32 root #1's.
    Speed 7 and below never move: `block_yrd_lowbd` has exactly one caller.
  - **WHAT CLOSED, all measured 2026-08-02 on aarch64-apple-darwin, `--profile test-fast`:**
    `encoder_gate_speed8_textured_allintra` **60/64 → 64/64** (the pinned list is deleted, not
    relaxed); KB-32's four gates all byte-exact including the 2176² cpu9 cell and the whole
    cpu8 size ladder (both promoted from shape-pins to hard byte gates);
    `kb28_crop_dims::vbp_band_crop_dims_byte_match` **18 of 20 open rows → 0** (the 2 that
    remain are KB-32's non-square-leaf HANDOFF *refusal* at speed 9, not a wrong stream);
    `config_permutations::speed_sensitivity_s2` — `SPEED_OPEN_SINGLETONS` is now **EMPTY at
    every speed 0..9** (`(8, "rtxs1")` and `(8, "trel2")` closed), and
    `SPEED_OPEN_COMBINATIONS` is **EMPTY** too, re-measured on the BROADER speed-8 array that
    emptying the singleton list produces.
  - **Also recorded while here, NOT changed (inert, and now asserted so):** the port's
    `hadamard_lp_16x16` combine spells `a0.wrapping_add(a1) >> 1` — truncate-then-shift, which
    is what `_mm_srai_epi16` does in `aom_hadamard_lp_16x16_sse2` (avg_intrin_sse2.c:442) and
    its AVX2 twin — whereas `_c` writes `int16_t b0 = (a0 + a1) >> 1`, where the sum promotes
    to `int` and only the result narrows. They differ only if `|a0 + a1| > i16::MAX`, and
    `block_yrd_lowbd` runs only at bd8 where the residual is 9-bit by construction, so the 8x8
    stage peaks at 16320 and the sum at 32640. `lp_hadamard_tiers_agree_over_the_reachable_
    range` asserts BOTH halves (tiers agree; the grid drives |coeff| past 16000).
  - **Record:** `benchmarks/kb12_lp_hadamard_transpose_2026-08-02.tsv` (every cell in both arms,
    the 477/4000-vs-0/0/0 teeth counts, and KB-32's localizer output under the revert).
    Verified on aarch64-apple-darwin in both dispatch modes + `cargo check --target
    x86_64-apple-darwin` and `--target i686-unknown-linux-gnu`. The x86 arm of the new
    differential calls `aom_hadamard_lp_{8x8,16x16}_sse2` (SSE2 is x86-64 baseline and
    `aom_hadamard_lp_8x8` has no AVX2 tier at all, rtcd_defs.pl:1288), so the tier-agreement
    assertion is real on both CI architectures rather than vacuous off aarch64.
- **HBD (bd10/12) estimate arm + lossless TX_4X4 + palette (screen) arms NOT ported** — asserted
  dead on the 8-bit canon grid (nonrd_pickmode.rs:594/460/784); required before any high-bit-depth
  or screen-content speed-8/9 cell.

### KB-13 — Encoder: REAL-content byte-parity at speed >= 1 — ROOT FOUND (**50/60 byte-exact** as of 2026-07-31; AB mode-cache landed 2026-07-19)

> Headline count corrected 2026-07-31: this line read `41/60` while the entry's own
> body recorded the map at 45 -> 47 (KB-21 root #1) -> **50/60** (KB-21 root #2).
> The map is the source of truth; the headline had simply not been re-read.

- **ROOT FOUND + FIXED 2026-07-19 — `part_sf.reuse_best_prediction_for_part_ab` (the AB
  MODE CACHE) was unmodelled. 24/60 → 41/60 byte-exact (17 cells promoted, 0 regressions).**
  At allintra speed >= 1 (speed_features.c:397; OFF at speed 0 — exactly why KB-6's 30/30
  held), C's `ab_partitions_search` populates a per-sub-block `mode_cache`
  (`set_mode_cache_for_partition_ab`, partition_search.c:3729-3759: HORZ_A = {split[0].none,
  split[1].none, horizontal[1]}, HORZ_B = {horizontal[0], split[2].none, split[3].none},
  VERT_A = {split[0].none, split[2].none, vertical[1]}, VERT_B = {vertical[0], split[1].none,
  split[3].none}; entries valid iff the source ctx has rate < INT_MAX) and `rd_test_partition3`
  sets `x->mb_mode_cache` per sub-block: the luma mode loop then SKIPS every mode != cached
  (intra_mode_search.c:1581 — all angle deltas of the cached mode still run) and the
  filter-intra search runs only when the cache used filter-intra, then only the cached fi mode
  (:254-257, :269-273). The port searched every AB sub-block UNCONSTRAINED → its AB RD was
  systematically <= C's → AB won near-ties and coded fewer bytes (the exact map signature).
  Port fix: `IntraSbyGates.mb_mode_cache`, threaded `rd_pick_ab_part` →
  `rd_pick_rect_partition` → `leaf_pick_sb_modes`; sources retained as `split_none_cache[4]`
  (a new NONE-arm out-param on `rd_pick_partition_real` — the cache is NOT gated on the
  child's final partitioning or uv_mode, unlike the stricter REUSE readiness) +
  `rect_mode_for_cache[2][2]`. Gated `cfg.allintra && cfg.speed >= 1`. Found by a full C-vs-port
  prune-ladder read (2026-07-19) that also verified ~17 other gates faithful (see the agent
  report in STATUS.md).
- **Also landed: the speed-3 qindex arm of `less_rectangular_check_level`**
  (av1_set_speed_features_qindex_dependent, speed_features.c:3032-3034: speed 3 → qindex >= 170
  ? 1 : 2; the port had 2 unconditionally). Cfg-fill site (aom-bench lib.rs). Did not flip the
  remaining cq63 cells — kept as a faithfulness fix.
- **196² "SEPARATE root" was a HARNESS BUG, not an encoder root — CORRECTED 2026-07-24.** The
  gate harness `attempt_case_content_uv_sep` walked `floor(mi/16)` SBs (`n_sb = mi_cols / SB_MI`)
  over an unpadded `h+4`-row source. 196px = 50 mi = 3.0625 SBs → floor 3 **dropped the partial
  edge SB entirely**, coding a short tile the real C decoder REJECTS (this is what the
  `intra_tiebreak_deltas_2026-07-23` "196² emits invalid AV1" rows actually measured — a harness
  artifact, NOT the encoder). Fix: the KB-6 `run_case` partial-SB setup — `ceil(mi/16)` SBs over
  an SB-aligned, border-EXTENDED source (replicate the crop edge into the overhang, matching C's
  `aom_extend_frame_borders`; `av1_get_perpixel_variance`/`av1_subtract_txb` read the full block
  incl. off-frame overhang). The port ENCODER was correct all along — KB-6 proves the same
  `pack_tile` 30/30 byte-exact on 196² at speed 0. After the fix the 196² cq63 cells byte-match
  and the map went 41/60 → **45/60**.
- **MAP (45/60 byte-exact vs real aomenc, gate `encoder_gate_real_content_speed1to4_e2e`):**
  - `01-size-64x64` (1-SB, aligned): **12/12 MATCH**.
  - `00-quantizer-00` 64² crop: **11/12** (open: cpu4 cq32).
  - `00-quantizer-00` 128² crop: **8/12** (open: cpu3 cq63, cpu4 cq12/cq32/cq63).
  - `23-film_grain-50` 64² crop: **10/12** (open: cpu3 cq63, cpu4 cq32).
  - `01-size-196x196` (PARTIAL-SB, multi-SB): **4/12** — cpu1-4 **cq63 MATCH** (promoted into
    `byte_exact`); cpu1-4 cq12/cq32 are ordinary valid-stream near-ties (first-diff at real tile
    bytes 437/830/852 or a ±1 LF level, NOT short-tile rejects).
- **Remaining 15 DIFF cells (all pinned self-promoting):** the 196² cq12/cq32 cluster (8) + 7
  interior near-ties concentrated at cpu3/cpu4 (5 of 7 at cq63/cq32 high-speed corners). Next:
  the interior localizer on the survivors + a sibling-C dump of the first divergent node per the
  KB-3/KB-7 method.

### KB-14 — Decoder: superres single-SB-column coded frame decoded flat — FIXED ✅ (header coded-lossless false-positive; NOT the upscale)
- **FIXED 2026-07-18.** Root cause is the HEADER PARSE, **not** the normative upscale (the original
  "likely locus: `upscale_plane`/`get_upscale_convolve_x0`" guess was WRONG — `superres.rs` is
  byte-faithful to `av1_convolve_horiz_rs_c`; the coded recon fed to it was already flat).
- **Symptom (found 2026-07-18 via the ENCODER superres QTHRESH gate):** a superres KEY stream whose
  CODED frame is **a single 64-wide superblock** (`coded_w <= 64`, one SB column, recon stride 64 —
  first reached at denom 16 / exact-2:1, e.g. 128→64) decoded to a **flat-128 luma plane** (the
  whole first SB row went to the DC no-neighbour default; content cascaded wrong below it). The
  encoder was byte-exact (`port_tu == c_tu`); purely decoder-side.
- **Root cause (decode-both localized):** the two-phase header parse in `parse_frame_header`
  (`aom-decode/src/frame.rs`). The PROBE parses on the full (upscaled-width) mi grid; under superres
  its `tile_info` is OVER-SIZED, so every field the probe reads after it — the quant params included
  — comes off a shifted bit position and is garbage. `coded_lossless` was computed from that garbage
  probe (`frame_coded_lossless(&probe)`), and for these streams it **FALSE-POSITIVED** (a normal
  qindex-128 frame read as lossless). The superres re-parse then set `cfg2.coded_lossless = true`,
  which **drops the loop-filter + CDEF header sections** (`read_uncompressed_header`'s
  `if !coded_lossless` gate), so the header ended ~2 bytes early → `tile_data_off` off by 2 → the
  tile arithmetic decoder started on the wrong bytes → SB(0,0)'s BLOCK_64X64/TX_64X64 read
  `EOB=1`/skip where C reads real coeffs → flat-128. Why only `coded_w<=64`: only there does the
  probe's sb_cols (upscaled) differ from the coded sb_cols such that the misread quant lands on a
  lossless-looking pattern (denoms 9-15 and wider coded frames keep sb_cols matching, so the probe's
  quant was accidentally still right; single-SB-column plain frames re-parse identically and always
  worked). CONTENT/qindex-dependent because whether the garbage looks lossless depends on the exact
  bits.
- **Fix:** derive `coded_lossless` from quant read with the CORRECT (downscaled) mi grid. When
  superres is active, parse a probe on the downscaled `tile_info` FIRST and decide lossless from IT
  (reused as the final header unless it really is coded-lossless). Handles all four cases (±superres
  × ±lossless); non-superres path unchanged. `frame.rs` `parse_frame_header` (~L490-528).
- **Regression gate (Gate-1, asserted, NO graceful skip):**
  `superres_denom16_single_sb_column_byte_identical_kb14` in `aom-decode/tests/superres_diff.rs` — 5
  single-SB-column upscaled widths (128/126/120/116/100 → coded 64/63/60/58/50) at denom 16, each
  decoded byte-identical to the C decoder + a golden per-plane MD5 (shared `tests/common/md5.rs`) +
  an explicit "flat-128" guard. **Proven revert-catching:** stashing only the `frame.rs` fix flips
  every cell to the flat-128 mismatch. The existing `superres_diff` denoms 9/12/16 arms + the full
  Gate-1 conformance corpus + `real_bitstream` (incl. sb128/multi-tile) stay green.

### KB-15 — Encoder: IntraBC (screen content) — SEARCH + skip-arm + wiring LANDED; PINNED on the inter var-tx COEFF ARM
- **Status (2026-07-18):** `rd_pick_intrabc_mode_sb` (`aom-encode/src/intrabc_search.rs`) is WIRED
  (rd_pick.rs step 6 → real, `PickFrameCfg::intrabc` gated on `p.allow_intrabc`) and runs the full
  DV search under the screen-content gate. LANDED: the source-frame hash (chunk 3a) + the **NSTEP
  `full_pixel_diamond`** (site config + `diamond_search_sad` coarse→fine + `UPDATE_SEARCH_STEP`
  num00 collapse) + the **`full_pixel_exhaustive` mesh** (screen `exhaustive_searches_thresh=1<<20`);
  the pixel search ALWAYS runs at `intrabc_search_level 0` (rdopt.c:3570). Geometry unit-locked
  (`nstep_config_matches_c`, `mv_step_param_matches_c`, `diamond_finds_exact_repeat`). The HASH is
  square-gated (mcomp.c:1918); the diamond runs for EVERY bsize (non-square intrabc supported).
  `predict_skip_txfm` (tx_search.c:183) + `set_skip_txfm` hbd sse: the port offers intrabc ONLY in
  the skip regime (luma predict_skip fires AND chroma exact match), where `av1_txfm_search` forces
  `skip_txfm=1` and BYPASSES the coeff arm → the skip RD (`mode+mv+skip1`, `dist=sse`) is byte-exact.
  Full wiring: `LeafWinner`/`RdPickIntraBest` fields, `ModeGrid` DV grid (`dc_screen`, 25 stamp
  sites), `encode_b_intra_dry` intrabc arm (predict-from-recon + skip entropy reset + skip txfm ctx),
  `pack_leaf` (use_intrabc + DV diff, skip tx/coeff), harness (hash from source luma, LF forced 0,
  `ToggleKnobs::enable_intrabc`, `PackCfg::allow_intrabc`).
- **[SUPERSEDED 2026-07-19 — historical, as of 2026-07-18. Every "NOT ported" item below has
  since LANDED; see the three PROGRESS 2026-07-19 entries and the DV-clamp fix (`434b865d`).
  Kept for the measured census only. Do NOT read its remaining-work list as current.]**
  **PINNED — real-content byte-exactness blocked on the inter var-tx COEFF ARM (the L piece).**
  Real screen content codes the MAJORITY of intrabc blocks via the COEFF arm (nonzero quantized
  residual) and as NON-SQUARE shapes: measured on a 196² conformance crop (`intra_only-intrabc-
  extreme-dv` @ (480,180) cq48), C uses **49 intrabc blocks = 39 coeff-arm + 42 non-square, only 10
  skip**. The port codes those blocks as intra and the frame diverges. NOT ported (the remaining
  work): `av1_pick_recursive_tx_size_type_yrd` inter var-tx quadtree (`select_tx_size_and_type` /
  `select_tx_block` / `try_tx_block_no_split`/`_split`) + `prune_tx_2D` + `ml_predict_tx_split` NN
  prunes + the var-tx WRITE path in pack (`write_tx_size_vartx` exists in aom-entropy — post-reorg
  that is `crates/aom-dsp/src/entropy/partition.rs` — unused by
  pack) + `derive_real_costs` inter tx-cost fill (currently a DUMMY zero cdf; source from
  `kf.inter_ext_tx`). The 420 skip subset ALSO needs a chroma-eob-0 check (currently `chroma_sse==0`,
  exact-only). No synthetic content found that makes aomenc use ONLY skip-square intrabc (intra wins
  on synthetic repeats; real content uses the coeff arm) — so the skip-arm alone byte-matches 0 real
  cells today.
- **Gate:** `rd_close_intrabc::intrabc_dv_search_pinned` (aom-bench) — asserts anti-vacuous (C
  genuinely codes intrabc on the crop, via decode census) + PINS the divergence self-promotingly (a
  byte-match fails → promote into `BYTE_EXACT_CELLS`). Envelope UNTOUCHED: non-screen frames
  (`allow_intrabc=0`) are byte-inert — palette gate + `partition_pick_diff` + `rd_pick_intra_sb_diff`
  all green.
- **PROGRESS 2026-07-19 (inter var-tx coeff arm — the CORE is differential-locked; witness still
  PINNED):** landed on origin/main:
  - `44bc51c` — `derive_real_costs` inter ext-tx costs sourced from `kf.inter_ext_tx` (§5 #C
    one-liner; was a zero stub).
  - `db90148` — `crates/aom-encode/src/var_tx.rs`: the inter/intrabc var-tx recursion
    (`pick_recursive_tx_size_type_yrd` → `select_tx_size_and_type` → `select_tx_block` /
    `try_tx_block_no_split` / `try_tx_block_split`) + the inter per-txb leaf
    (`search_tx_type_inter` + `get_tx_mask_inter` + `trellis_rdmult_inter_y` mult-16). The inter
    LEAF is byte-locked vs the REAL C kernels (`var_tx_leaf_diff.rs`: fwd/quant/optimize/cost/
    dist hybrid + is_inter cost + adaptive break, all 19 tx sizes × amp/qidx/bd/reduced).
  - `3b9278f` — the RECURSION is byte-locked vs an independent C transcription
    (`var_tx_recursion_diff.rs`: no-split/split RD, pick-skip, txfm_partition cost, context
    threading + backtracking, adaptive_txb_search_level=1 + txb_split_cap=1, depth-2 splits).
- **PROGRESS 2026-07-19 cont. — BOTH NN prunes now differential-locked + ENABLED in the recursion:**
  - `a40d598` — `ml_predict_tx_split` NN (`ml_tx_split_thresh=8500`, bd8): weights transcribed
    (`xtask/transcribe_tx_split_nn.py` → `tx_split_nn_weights.rs`), `av1_nn_predict` eval +
    prec-reduce, wired into `select_tx_block`'s `try_split` gate (`VarTxEnv.ml_tx_split_thresh`).
    Differential-locked vs real C `av1_nn_predict` (`tx_split_nn_diff.rs`).
  - `5aa145d` + `a77a7d8` — `prune_tx_2D` NN (`TX_TYPE_PRUNE_1`): all 5 helpers
    (`get_energy_distribution_finer` / `av1_get_horver_correlation_full` / `av1_nn_fast_softmax_16`
    + `approx_exp` / `get_adaptive_thresholds` + table / `av1_sort_fi32_8/16`) + the driver ported
    (`prune_tx_2d.rs`), weights transcribed (`transcribe_prune_tx_2d_nn.py`). Differential-locked
    vs a tier-1 real-C shim (`shim_prune_tx_2D` copies the static helpers + driver verbatim,
    calls the exported scalar `av1_nn_predict_c` etc. + real nnconfig maps; `prune_tx_2d_diff.rs`,
    576 cases). WIRED into `search_tx_type_inter`'s multi-type arm (`VarTxEnv.prune_2d`, reorders
    `txk_map`). NOTE: the differential uses the SCALAR `_c` reference — the real encoder's SIMD
    `av1_nn_predict_avx2` gives ULP-different scores that flip near-tie sort ORDER (decision-inert
    for the RD pick over the same masked set) but not the decision-relevant mask.
- **PROGRESS 2026-07-19 cont. — the COEFF ARM IS WIRED END-TO-END; witness still PINNED (not
  byte-exact).** Landed on origin/main: `a33929ca` (winner carry: `inter_tx_size[16]` on
  LeafWinner/RdPickIntraBest + var-tx root tx_size and luma tx_type_map on IntrabcBest),
  `194ae39f` (the inter tx leaf parameterized for CHROMA — `get_tx_mask_inter_uv` pins the
  co-located luma tx type per tx_search.c:1841-1847, the tx-type-cost plane arg, and
  `trellis_rdmult_inter_uv` = `plane_rd_mult_chroma[inter][UV]` = 10 vs luma's 16),
  `9a5fafdc` (`txfm_uvrd_inter`: C's chroma inter arm — UNIFORM tx, EOB-based skip),
  `351143a1` (the integration: `av1_txfm_search` assembly + the `av1_encode_sb` var-tx re-encode
  + the `write_tx_size_vartx` / inter-ext-tx pack + `KfFrameContext::txfm_partition`).
  - **Item 1 (`model_based_tx_search_prune`) is CLOSED BY SOURCE PROOF — no port needed.**
    `rd_pick_intrabc_mode_sb` calls `av1_txfm_search` with `ref_best_rd = INT64_MAX`
    **hardcoded** (rdopt.c:3611 — the incoming `best_rd` is used only for the post-hoc
    `rd_stats_yuv.rdcost < best_rd` compare at :3615). The prune is gated on
    `ref_best_rd != INT64_MAX` (tx_search.c:3562-3565), so it can never fire on the intrabc path.
    The same reasoning voids every other early exit in `av1_txfm_search` (:3811, :3844) and
    `av1_txfm_uvrd` (:3737) for intrabc.
  - **Items 2 + 3 (PACK + INTRABC integration) are DONE and exercised**, including the chroma
    eob-0 skip check (the old `chroma_sse == 0` gate was a strict SUBSET of it).
  - **NOT DONE — byte-exactness.** `rd_close_intrabc::intrabc_dv_search_pinned` still PINS:
    port **1907B vs c 1891B (delta +16), first differing tile byte 646 of 1891**. The port now
    codes MORE bytes than C — the OPPOSITE of the classic "codes fewer symbols" RD-near-tie
    signature — and the first ~34% of the tile is byte-identical.
  - **A/B PROOF THE COEFF ARM IS LIVE (not inert):** re-running the witness with the coeff arm
    gated back off (`if !luma_skip { continue; }`, the pre-landing skip-only behaviour) gives
    **1974B, delta +83, first diff at byte 1113**; with the coeff arm on it is **1907B, delta
    +16, first diff at byte 646**. So the arm cuts the size delta ~5x (it codes residual instead
    of falling back to intra) but moves the first divergence EARLIER — i.e. a coeff-arm intrabc
    block BEFORE tile byte 646 is coded differently from C, while the later blocks the skip-only
    build got wrong are now right. That earlier first-divergence is the sharper localization
    target. The gate now prints that
    signature plus (when the port's stream still decodes) the first block whose mode-info differs;
    today the port's stream does NOT decode past the divergence (bitstream desync), so the
    block-level localization needs the C-side dump method (KB-2/KB-3/KB-7) or a
    truncate-at-first-divergence probe.
    **UPDATE 2026-08-01 — that desync is EXPLAINED AND FIXED: it is KB-29.** This same witness
    cell (`scc_480x180_196_cq48`) was the cheap repro that closed KB-29's six roots; the port's
    stream now decodes cleanly through `aomdec`, `dav1d` AND the port decoder, with C-vs-port
    pixel identity (`crates/aom-bench/tests/armed_tools_decode_gate.rs`, cell
    `scc196_cq48_s0`). So the gate's "when the port's stream still decodes" branch is now the
    live one and block-level localization no longer needs the C-side dump. The BYTE-EXACTNESS
    pin against aomenc is unaffected — KB-29 made the output CONFORMANT, not byte-identical.
  - **Three real bugs found while wiring (all fixed in `351143a1`):** the pack's intrabc branch
    returned early and skipped the neighbour-grid stamp; chroma prediction buffers were sized
    `bw >> ss_x` while the chroma PLANE block is padded to a 4x4 minimum (sub-8x8 luma overran
    them); and the var-tx recursion sized its entropy/txfm context arrays by the frame-edge
    CLIPPED extent while `av1_get_entropy_contexts` fills the FULL plane block (only the txb LOOP
    is clipped) — under-running `get_txb_ctx`'s `a[..w_unit]` read on an edge block. All
    frame-edge extents now route through the validated `max_block_units`.
  - **Envelope verified byte-inert:** full `cargo test -p zenav1-aom-encode` green (97 binaries,
    0 failures) including `encoder_gate_real_image_e2e_kb6_repro` (30/30 real content); both
    var-tx differentials (`var_tx_leaf_diff`, `var_tx_recursion_diff`) still pass, plus
    `prune_tx_2d_diff` / `tx_split_nn_diff`. Non-screen frames construct no intrabc args at all.
  - **ROOT CAUSE #1 — CLOSED ✅ 2026-07-19: UNCLAMPED tile bounds fed into `av1_is_dv_valid`.**
    All THREE previously-ranked suspects are **REFUTED — do not re-chase**: it was NOT the joint
    skip decision / `set_skip_txfm` rate, NOT the chroma inter tx-type inheritance timing, and NOT
    the AVX2-vs-scalar `prune_tx_2D` sort order. The port's `is_dv_valid`
    (`aom-dsp/src/entropy/dv_ref.rs:1369`) is faithful line-for-line **and its constants are
    right** — the defect was in its INPUTS. `partition_pick.rs`'s `IntrabcLeafArgs` passed
    `env.tile_{row,col}_end` **raw**, and those are unclamped sentinels (`1 << 16`, set at
    `aom-bench/src/lib.rs:1381`). C's `av1_tile_set_row`/`av1_tile_set_col` (tile_common.c) CLAMP
    the tile end to the frame (`AOMMIN(row_start_sb[i+1] << mib_size_log2, mi_rows)`). That single
    input blows up `total_sb64_per_row = ((mi_col_end - mi_col_start - 1) >> 4) + 1` from **4** to
    **4096** on this 196² frame, which makes the already-coded-SB64 ordering gate
    `src_sb64 >= active_sb64 - INTRABC_DELAY_SB64` (mvref_common.h:328) stop rejecting anything —
    so the port accepted DVs C rejects and coded intrabc where C codes intra. **Measured at the
    first divergent block** (throwaway instrumented sibling C, byte-inertness re-verified against
    the clean build on every rebuild — the witness reported identical 1907B / first-diff-646 with
    and without the instrument): **mi(16,0) BLOCK_16X4 — C's own search finds the IDENTICAL
    `dv=(-512,0)` with `bestsme=35311`, `inrange=1`, but `dvvalid=0` → `continue`**, while the
    port had `dvvalid=1` with `tile[0,65536,0,65536]`. **Fix:** clamp to `env.mi_rows`/
    `env.mi_cols` at the `DvTileBounds` construction (`partition_pick.rs`) — the SAME clamp
    `pack.rs:1377` already applies to the decoder-facing bounds. `DvTileBounds` is intrabc-only
    (non-screen frames construct no intrabc args at all), so the envelope structurally cannot move.
    **Measured effect: size delta +16 → 0 (port 1891B == C 1891B), first differing byte 646 → 1038.**
  - **RESIDUAL (witness still PINNED) — a partition near-tie at mi(38,4). NOT a KB-15 intrabc
    defect.** At the BLOCK_16X8 node mi(38,4) C picks PARTITION_VERT_4 (four BLOCK_4X8 at mi cols
    4/5/6/7 — c=4 intra mode 10, **c=5 and c=6 themselves intrabc `skip=1 dv=(-1032,-40)`**, c=7
    intra mode 11); the port picks the AB shape **8×8 + 4×8 + 4×8**. **The intrabc COEFF ARM IS
    BYTE-EXACT AT THAT SITE** — for the 8×8 candidate the port and C agree **to the unit**
    (`dv=(-1032,24)`, rate 32747, dist 171120, sse 952608, rdcost 40519646, yrate 18114,
    ydist 170288, ysse 951776, uvrate 6, uvdist 832, uvsse 832, uvskip 1) — and **C ACCEPTS that
    same candidate** (its 40519646 beats C's incoming `best_rd` 43226527). The tip is on the
    **INTRA** side: the intra winner rd feeding the node is **43223116 in the port vs 43226527 in
    C — 3411 low (0.008%)**. So the residual belongs to the **KB-2 / KB-6 speed-0 partition/mode
    near-tie family (KB-13 is the speed≥1 analogue), a SEPARATE root** — at the divergent site the
    intrabc DV search, the DV validity, the coeff arm and the var-tx pack all match C.
  - **THE INTRA-SIDE RESIDUAL — FOUR ROOTS FOUND AND FIXED 2026-07-19.** Method: a byte-inert
    instrumented sibling C (`/root/intra-rd-instr`, throwaway, removed) dumping (a) every
    `pick_sb_modes` leaf's incoming budget + rate/dist/rdcost/mode/ibc/skip/tx, (b) every
    `av1_rd_pick_partition` node's chosen partition + best_rdc, and (c)
    `av1_rd_pick_intra_mode_sb`'s luma/chroma split (`rate_y`/`rate_uv`/`dist_y`/`dist_uv`,
    filter-intra, tx_type, CfL alpha). **Byte-inertness was verified on every rebuild** — the
    instrumented binary reproduces the clean build's stream byte-for-byte (1891-byte frame OBU).
    The KB's "3411 low" number decomposes as **exactly 6 rate units** at this cell's
    rdmult (291065). The roots (in the order they were found):
    1. **`rd_pick.rs` — the intra-budget early exit SKIPPED the intrabc search.**
       `av1_rd_pick_intra_mode_sb` (rdopt.c:3680-3690) sets `rd_cost->rate = INT_MAX` when
       `intra_yrd >= best_rd` and skips the uv search + assembly, but it does **not return**:
       `rd_pick_intrabc_mode_sb` still runs, with `best_rd` left at the INCOMING budget (the
       `best_rd = rd_cost->rdcost` tightening at :3686 is gated on `rate != INT_MAX`), and
       `RD_STATS best_rdstats = *rd_stats` (:3491) carries that INT_MAX in so a winning DV
       overwrites the whole tuple — the `assert(rd_cost->rate != INT_MAX)` at :3690 exists
       precisely for this rescue. The port returned early, dropping the intrabc candidate on
       exactly the leaves intrabc exists for (badly intra-predicted = repeated content).
       Measured at the divergent BLOCK_8X8 mi(38,4): C's VERT sub-1 at mi(38,5) is rescued by
       intrabc, the port's bailed → VERT lost → the port took NONE (the 8×8 intrabc block).
       Effect: first differing byte **1038 → 1120**.
    2. **`partition_pick.rs` — `skip_ctx` hardcoded 0.** `av1_get_skip_txfm_context` is
       `above->skip_txfm + left->skip_txfm`; `pick_sb_modes` zeroes `mbmi->skip_txfm`
       (partition_search.c:910) and the intra path never sets it, so ctx is identically 0 on a
       pure-intra KEY frame — the invariant the hardcode encoded. It BREAKS under intrabc: a
       skip-arm intrabc block has `skip_txfm = 1` (`set_skip_txfm`, tx_search.c:254), so its
       neighbours owe the dearer `skip_txfm_cost[1..2][0]`. Now read from the live DV grid
       (which already carries the per-mi `skip_txfm` projection). Byte-inert without a DV grid.
    3. **`partition_pick.rs` — `allow_intrabc: false` hardcoded in the leaf mode-cost cfg, so
       `intra_mode_info_cost_y` never added `intrabc_cost[use_intrabc]`
       (intra_mode_search_utils.h:563-564).** On an intrabc frame EVERY intra luma candidate pays
       the `use_intrabc = 0` flag (`write_intra_frame_mode_info` writes it for every block), so
       every intra leaf's luma rate was **35 units too CHEAP** on this cell while the PACK still
       wrote the flag — a systematic under-cost of the whole intra side against the intrabc arm
       and against every partition assembled from more leaves. **This is the measured root of the
       "intra RD comes out slightly cheap" signature.** Verified at the first divergent node
       mi(44,20) BLOCK_16X16: port luma rate 131946 → **131981 == C's 131981** exactly (chroma
       already matched at 2897, and after the fix `rate`/`rytok`/`mode`/`tx_size`/`tx_type`/
       filter-intra/CfL-alpha ALL match C to the unit).
    4. **`encode_sb.rs` + `aom-dsp/src/intra/cfl.rs` — `cfl_store_block` was unported.**
       `encode_superblock` (partition_search.c:580-583) runs `if (is_inter_block(mbmi) &&
       !xd->is_chroma_ref && is_cfl_allowed(xd)) cfl_store_block(xd, mbmi->bsize, mbmi->tx_size);`
       and **`is_inter_block` is TRUE for intrabc** (blockd.h:372). So an intrabc block that is not
       a chroma reference must still publish its reconstructed luma to the CfL buffer for the later
       chroma-reference sibling covering it. The port's own docs called this site "dead for intra"
       — TRUE until the intrabc arm landed, FALSE after. Without it, a CfL chroma-ref block whose
       luma footprint contains an intrabc sibling predicted from a STALE luma buffer. Found
       statistically: of 995 leaves where port and C agree on winner+rates, 985 had identical
       distortion and **all 10 divergences were `UV_CFL_PRED`**, 9 of them chroma-only and every
       delta a multiple of 16 (i.e. real SSE). The fix closed the 3 earliest — exactly the ones
       with a coded intrabc block inside their CfL luma footprint. `cfl_store_block` +
       `get_tx_size` are now ported (frame-edge clip via `max_block_wide/high` + align-up, then
       one synthetic tx). The intrabc SKIP arm's `mbmi->tx_size` was also corrected to
       `max_txsize_rect_lookup[bsize]` per `set_skip_txfm` (previously the dead intra winner's
       size; now live because `cfl_store_block` reads it for the edge alignment).
  - **SHARED-ROOT HYPOTHESIS: REFUTED (do not re-chase).** The "intra RD is slightly cheap"
    signature is shared across KB-15 / KB-13 / KB-10 / KB-11 / KB-12 / KB-P29, but the ROOT is
    not. Root 3 — the only one that is a systematic intra under-cost — is gated on
    `av1_allow_intrabc(cm)` and is therefore **structurally zero** on every other member:
    KB-13's cells are decoded photographic content (`allow_screen_content_tools = 0`), KB-10/11/12
    are synthetic noise/diag (non-screen), and the KB-P29 palette gate runs
    `enable_intrabc = 0`. Empirically confirmed: the full encode+bench suite is 340/340 both
    before and after all four fixes, and every one of those entries' pins **asserts DIVERGENCE**
    — a flip to MATCH would have failed the pin. So KB-13 keeps its own hypothesis (a speed>=1
    prune C applies that the port does not) and KB-10/11/12/P29 keep theirs.
  - **ROOT 5 — CLOSED ✅ 2026-07-19: the intrabc DV SEARCH ranked candidates against the RECON
    buffer; C searches the SOURCE frame.** `rd_pick_intrabc_mode_sb` sets `xd->plane[i].pre[0]`
    from `xd->cur_buf` (`av1_setup_pred_block`, rdopt.c:3482) and **`xd->cur_buf = cpi->source`**
    (encoder.c:4121, encodeframe.c:217) — so the hash-candidate costing (`get_mvpred_var_cost`,
    mcomp.c:1966) AND the full-pel diamond+mesh measure variance against the SOURCE frame; only
    the accepted candidate's RD stage (`av1_enc_build_inter_predictor`) predicts from the recon.
    The port passed `recon_y` as the search reference in BOTH arms (`intrabc_search.rs` hash arm
    + `FullPelSearch.refb`). The port's source-clone recon initialization MASKED the bug wherever
    the referenced region was never yet written (recon == source there) — which is why the DV
    search matched C on every previously-analyzed block. At coded regions recon ≠ source:
    measured at **mi(42,22) BLOCK_8X8** (dual byte-inert dump) — dir=1, SAME mv (-17,-39): C sme
    26062 (vs source) / port 40162 (vs recon); dir=0: the port found dv=(-1168,224) (sme 24062
    vs recon; RD 27551753 < intra 28154158 → **intrabc WON**) where C's source-based search finds
    (-116,-23) → dv=(-928,-184) (RD 34841670 → intra wins). The port coded intrabc where C codes
    intra H_PRED — the ACTUAL first divergent coded block (byte 1120 = mi(42,22)'s mode info).
    **The earlier "mi(44,20) first-order" framing is superseded**: it came from the
    CfL-distortion table, which only covered winner-AGREEING leaves; mi(44,20)'s wrong above-row
    pixels were DOWNSTREAM (they are the TR-node mi(40,20) winner re-encode's output, which
    contained the intrabc recon instead of H_PRED's). **Fix:** both search arms reference
    `a.src_y` (intrabc-only code — non-screen envelope structurally untouched). **Measured
    effect: port 1902B → 1895B (delta +11 → +4)**; dir=0 now finds C's exact mv and both sides
    reject intrabc at that node. `first_diff` unchanged at 1120 (the residual below).
  - **ROOT 6 — CLOSED ✅ 2026-07-19: `get_tx_size_context`'s INTER-neighbour override was
    unwired on the ENCODER side (search + pack + CDEF repack).** C's ctx (blockd.h) substitutes
    an `is_inter_block` neighbour's BLOCK dims for its txfm-context byte — intrabc qualifies on
    KEY frames. The DECODER models it (aom-decode lib.rs read_tx_size, with a doc comment
    describing this exact drift); the encoder passed `None, None` at both
    `partition_pick.rs`'s leaf ctx and `pack.rs`'s `write_selected_tx_size` ctx. Mechanism of
    the resulting silent drift: the DEFAULT `tx_size_cdf` rows ctx0/ctx1 are IDENTICAL
    (AOM_CDF2(19968)), so a ctx-row flip next to a coeff-arm intrabc neighbour codes
    byte-identical bits while the ROW STATES diverge; the per-SB cost refresh
    (`derive_real_costs` = C's INTERNAL_COST_UPD_SB `av1_fill_mode_rates`) then reads the
    drifted row — measured as the mi(42,22) "3-rate-unit tx-size-cost gap"
    (`tx_size_cost[cat0][ctx0][depth1]` 179 vs 182 from row 25718-ish vs 25607). Fix: derive
    `above/left_inter_bsize` from the DV grid (`ModeGrid::dv_at`, intra default on non-screen
    frames ⇒ byte-inert there) at BOTH sites + rebuild the grid from the picked trees in
    `pack_tile_from_trees` (the CDEF repack — above/left reads only target already-coded
    positions, so the fully-stamped grid is read-equivalent). **Verified: the per-SB cost-fill
    snapshots now match C at every SB (dual dump; SB(2,1) row 25607→182 both sides), and the
    ENTIRE mi(42,22)+mi(44,20) chain matches C to the unit** (mi(42,22): rate 27934, rdcost
    28155864 == C; the mi(44,20) HORZ_4 strips' tx-size adaptations all match). The witness
    bytes are unchanged (1895B/+4/first-diff 1120) because the Δ3 was decision-INERT at
    mi(42,22) — intra won there either way; byte 1120 was never that node.
  - **REMAINING RESIDUAL (witness still PINNED) — a CHROMA/CfL divergence at the mi(40,28)
    node family (BR 32×32 quadrant), flipping its partition.** C picks VERT (sub0 = coeff-arm
    intrabc 4×8 `dv=(-816,-888)`, sub1 intra); the port picks HORZ (two intra 8×4s) — the
    actual byte-1120 divergence. At every compared leaf of the node family the LUMA matches C
    EXACTLY (16×16 NONE: rate_y 167400, ry_tok 164521, dist_y 525008 — identical) but the
    CHROMA does not: 16×16 `rate_uv` 4921 (port) vs 8216 (C), `dist_uv` 54848 vs 39168; 8×8
    NONE `rate_uv` 3057 vs 3042, `dist_uv` 37776 vs 46032 (uv_mode = CFL both sides). These
    are the "7 remaining CfL distortion divergences" of the four-roots block, now the
    first-order residual. With luma recon identical, the CfL divergence must come from the
    chroma-side INPUT state (chroma recon neighbours feeding the CfL DC / uv modes, or the
    CfL buffer contributions of the preceding TL-16×16 subtree's intrabc blocks —
    `cfl_store_block` / intrabc chroma recon in the winner re-encode). Next: dump the CfL
    buffer + chroma recon neighbourhood at the mi(40,28) 16×16 leaf on both sides (the same
    dual-dump method), find the first differing chroma input pixel, and trace its producer.
    **2026-07-19 chroma-path structural sweep (read-only agent): NO structural gap found — the
    residual is VALUE-level.** Verified-faithful (do not re-read): `is_sub8x8_inter` (intrabc
    + intra-covering ⇒ false; chroma-ref intra block predicts chroma via intra),
    non-chroma-ref chroma skip on both arms, `is_cfl_allowed`, `store_cfl_required`,
    `cfl_store`/`cfl_store_tx`/`cfl_store_block`/`sub8x8_adjust_offset`, the
    `cfl_store_block_for_inter` gate, CfL threading search+pack, the uv-search luma
    re-encode (store_y), `chroma_plane_offset`, chroma availability, uv edge-filter
    inter-neighbour handling, `intra_avail` vs `av1_predict_intra_block` (C has NO
    inter-neighbour case in has_top_right/has_bottom_left — pure geometry), and the chroma
    residual application. Measured detail: the port's divergent 4×4 chroma unit (the HORZ
    8×4 pair at mi(40..41,26..27) with the intrabc top) reconstructs as pure row-flat H_PRED
    from its left column while C's has textured residual rows — with byte-identical coding
    through that region, i.e. an encoder-side value divergence. NEXT CONCRETE STEP: dump
    `cfl.recon_buf_q3` (the SUBSAMPLED luma) right after the intrabc sibling's
    `cfl_store_block` and at the chroma-ref sibling's CfL read, both sides; if it differs
    while full-res recon_y matches, the bug is the CfL subsample input offset/stride for the
    intrabc sibling; if it matches, chase the chroma-neighbour cascade one unit earlier.
  - **DUAL-DUMP DONE 2026-07-20 — ROOT PRECISELY LOCATED + KB's "mi(40,28)" ATTRIBUTION
    CORRECTED. It is a PURE CHROMA RD NEAR-TIE (CfL-vs-directional), NOT a CfL-buffer/subsample
    bug.** Method: env-gated port dumps (`intra_uv_rd`/`encode_sb`/`pack`, all removed) vs the
    C DECODE oracle (`decode_frame_obus_prefilter` gives `tc.recon_u`/`tc.blocks` = ground truth,
    since the port's bytes match C up to the first diff). Findings, in order:
    1. **The TRUE first byte-divergence is at `mi(41,25)` (C coding idx 438), NOT `mi(40,28)`
       (idx 444).** C's BR-32×32 coding order: …438 mi(41,25), 439 mi(40,26), 440 mi(41,26), …
       444 mi(40,28). mi(40,28)'s VERT/HORZ flip is DOWNSTREAM (its CfL neighbours are the swapped
       recon below).
    2. **The port SWAPS the chroma mode between two adjacent sub-8×8 chroma-reference blocks:**
       `mi(41,25)` (chroma-ref for the 4×4 U at chroma cols 48-51) — **C = UV_H_PRED (uvmode 2),
       PORT = UV_CFL_PRED (uvmode 13, alpha_idx 16, signs 4)**; `mi(41,26)` (chroma-ref for the
       4×4 U at cols 52-55) — **C = UV_CFL_PRED (a16,s4), PORT = UV_H_PRED**. The luma (ymode)
       AND partition are IDENTICAL to C at both; only the chroma mode is swapped. Reconstructions
       confirm the swap: cols 48-51 port textured / C flat `[135,135,135,135]`; cols 52-55 port
       flat / C textured `[146,129,151,129]` (both port search-recon AND pack-recon).
    3. **RULED OUT — do not re-chase.** (a) `intrabc_predict_chroma` is **bit-exact vs C's
       `av1_convolve_2d_sr`/`_x_sr`/`_y_sr` for bd8** — worked the 2-tap {64,64} bilinear
       (`av1_intrabc_filter_params`, filter.h:197) through both rounding stages by hand: 2D →
       `(S+2)>>2`, 1D → `(a+b+1)>>1`, both == C (round_0=3, round_1=11, offset_bits=19 / bits=4).
       (b) **Luma recon matches C EXACTLY** at both CfL AC sources — mi(40,24) 8×8 (feeds mi 41,25)
       AND mi(40,26) 8×4 (feeds mi 41,26), all pixels identical to `tc.recon`. (c) `mi(41,26)`'s
       `recon_buf_q3` is CORRECT: rows 0-1 textured (top intrabc luma), rows 2-3 flat (bottom
       luma) — exactly what C's textured-top/flat-bottom chroma implies. (d) `cfl_store_block`
       geometry faithful (cfl.c:421). So the CfL AC source, the DV prediction, the store, and the
       luma are ALL correct; the divergence is the CHROMA MODE RD DECISION alone.
    4. **This is the KB-2/KB-6/#26 chroma-mode near-tie family (as predicted).** The port's chroma
       RD favours CfL where C favours H (and vice-versa) despite identical inputs, so the tip is
       on either the CfL alpha rate/dist or the directional (H) prediction dist.
       **NEXT CONCRETE STEP (the fix): sibling-C instrumented chroma-RD dump at
       `pick_sb_modes`/`rd_pick_intra_sbuv_mode` for mi(41,25) + mi(41,26)** — dump C's per-uv-mode
       rate+dist (UV_DC/V/H/…/CFL incl. cfl alpha search) and diff vs the port's `intra_uv_rd`
       per-mode RD to find which side's rate or dist diverges. Suspect the H_PRED chroma
       edge-filter / the neighbour `uv_mode`-in-grid stamp for the intrabc siblings feeding the
       directional prediction, since a CfL-buffer/subsample/luma bug is now disproven.
  - **DEEPER NARROWING 2026-07-20 (port-side per-uv-mode chroma-RD dump; instrument removed).**
    The port's `rd_pick_intra_sbuv_mode` iterates C's `uv_rd_search_mode_order` (DC, **CFL**, V,
    **H**, … — CFL is index 1, evaluated BEFORE H), rdmult=291066:
    - **mi(41,25)** (BLOCK_4X4, luma H): recorded per-mode this_rd `{DC 4372287, CFL 3357235,
      SMOOTH 4892811, SMOOTH_H 5231194, D203 4886830}`, WIN CFL rate 3708 dist 9760 rd 3357235.
      **H IS reached (passes every skip gate) but `rd_pick_intra_angle_sbuv` returns None** — its
      angle_delta=0 `pick_intra_angle_routine_sbuv` exceeds `ref_best_rd = best_rd + best_rd>>3`
      (= CFL's 3357235 ×1.125 = 3776889), i.e. the port's H rd > 3776889 (>12% worse than the
      port's CFL). So it is NOT a mode-search-space gap (H is searched) and NOT a sub-6-unit tie —
      the port's H is *substantially* worse than its CFL here, yet C picks H (so C's H < C's CFL).
    - **mi(41,26)** (BLOCK_8X4, luma H): per-mode `{DC 5605183, CFL 3942963, H 3484625, PAETH
      5320274}`, WIN H rd 3484625 (beats CFL 3942963); C picks CFL. So here the port's CFL is the
      *loser* by ~458k, where C's CFL wins.
    So the divergence is DIFFERENT-SIGNED at the two blocks (mi41,25: port H too high / CFL too
    low relative to C; mi41,26: port CFL too high relative to C) — which rules out a single
    uniform CFL-cost or H-cost bias and makes the sibling-C per-mode rd+dist dump (above) the only
    way to attribute it. Since ALL prior recon matches C-decode up to mi(41,25) (bits identical),
    the H prediction READS the same neighbours as C — so a wrong port H **dist** would have to be
    a chroma-edge-filter / prediction-kernel difference, while a wrong port **CFL rd** would be in
    the alpha search or the DC-pred; the dump distinguishes them. **Reusable tooling note:** the
    C DECODE oracle (`decode_frame_obus_prefilter` → `tc.recon_u`/`tc.blocks`) is a cheap
    ground-truth for recon+decisions and pinned everything above; only the per-candidate *RD
    scalars* still need sibling-C encoder instrumentation (also required by KB-13).
  - **ROOT FIXED 2026-07-20 (sibling-C dual-dump nailed it) — the intrabc SKIP-arm chroma
    extent bug.** Built a byte-inert instrumented sibling libaom (throwaway `/root/intra2-instr`,
    removed; symlinked in as the workspace `upstream`, C stayed 1891B) and dumped C's per-uv-mode
    chroma RD + `recon_buf_q3`/`ac_buf_q3` + DC-pred + neighbours vs the port. Chain: at mi(41,25)
    the CfL AC was IDENTICAL to C, but the port's CfL **dist was 9760 vs C's 17856** because the
    port's chroma **DC-pred was 137 vs C's 140** — the above-neighbour row read `[140,140,128,128]`
    where C read `[140,140,140,140]`. The `128`s were the unwritten default: `encode_b_intra_dry`'s
    intrabc **SKIP arm** (`encode_sb.rs`) sized the chroma prediction as `(bw>>ss_x, bh>>ss_y)` — a
    **2×4 strip** for the 4×8 intrabc-skip chroma-ref `mi(38,25)` — instead of the **padded
    plane-block `BLK_W_B[plane_bsize]` (4×4)** that C (and the port's COEFF arm) use, leaving the
    right chroma columns as the `128` "island". That corrupted the CfL DC-pred of the block below
    and flipped its CfL/H decision. **Fix:** one line — size the skip-arm chroma extent by
    `plane_bsize` (mirrors the coeff arm; only sub-8×8 intrabc chroma-refs change; non-screen
    frames are `use_intrabc`-gated out). Verified: the port's recon at `(79,50-51)` is now `140`,
    its CfL dist matches C's `17856`, and the search+pack chroma decisions at mi(41,25) (→H) and
    mi(41,26) (→CfL) now match C to the unit. Full `aom-encode` suite green, differentials green,
    envelope byte-inert. **Witness stays PINNED / floor stays 1120:** this fix is output-inert on
    the witness because the byte-1120 divergence is a SEPARATE root — **mi(40,28)'s partition flip
    (C = VERT with a 4×8 coeff-arm intrabc sub, dv=(-816,-888); port = HORZ two intra 8×4s;
    bsize C=1 vs port=2)**, a speed-0 partition/mode near-tie in the KB-2/KB-6 family. NEXT for the
    witness: sibling-C per-candidate PARTITION-RD dump at the mi(40,28) BLOCK_16X8/8X8 node (the
    KB-3/KB-7 method) — the sibling-C tooling is already built and byte-inert.
  - **THREE ROOTS FOUND + FIXED 2026-07-22 (`0cd64bf`) — the mi(40,28) VERT sub0 (a 4×8
    coeff-arm intrabc block, dv=(-816,-888)) SEARCH now matches C to the unit; partition flips
    HORZ→VERT matching C.** Method: byte-inert instrumented sibling-C (`/root/intra3-instr`,
    throwaway) per-candidate dumps at the mi(40,28) BLOCK_8X8 node — partition RD, the 4×8
    intrabc DV search internals, the var-tx leaf/node cost decomposition. The port's 4×8 VERT
    sub0 was picking PAETH intra (rd 32594350) where C picks intrabc dv=(-816,-888) (rd 26166279);
    three independent bugs stacked:
    1. **`error_per_bit` used the FRAME rdmult, not the per-SB `x->rdmult`.** The DV-search
       variance-metric MV cost (`mv_err_cost`, `full_pixel_diamond`'s var_cost pick) took
       `error_per_bit` from the frame-init `ibc.error_per_bit` (465664>>6 = 7276), but C recomputes
       it per block from `x->rdmult`, which carries the per-SB `intra_sb_rdmult_modifier` fold
       (partition_search.c:5710) — on SB(2,1) that scaled 465664→291065, so C's is 291065>>6 =
       4547. Proven by a fixed-point probe: the variance kernel is byte-identical (port==C at every
       mv); only the err_cost term differed, by exactly the epb ratio. Fix: `error_per_bit:
       av1_set_error_per_bit(env.rdmult)` at the `IntrabcLeafArgs` site (`partition_pick.rs`).
       `sad_per_bit` is qindex-based (unaffected) — the mesh already matched.
    2. **NSTEP vs NSTEP_8PT.** The intrabc pixel search uses **NSTEP_8PT** (`av1_get_default_mv_
       search_method` → method 2; sibling-C `search_site` dump: `num_search_steps = 16`, every
       stage 8 points, `tan_radius = radius`), but the port's diamond modelled **NSTEP** (level 0:
       15 stages, 12-point tangent stages `tan = (int)(0.41*radius)` for radius>5). The extra
       tangent points let the port's diamond reach a lower-SAD local optimum (-103,-95) that C's
       8-point search never visits — so the port kept a recon-flavoured DV over C's (-102,-111).
       Fix: parameterize `nstep_stage_sites`/`diamond_search_sad`/`full_pixel_diamond` by an
       `eight_pt` flag + a 16-stage `NSTEP_8PT_RADII`; the intrabc call passes NSTEP_8PT, the inter
       call keeps NSTEP (`intrabc_search.rs`). After this the port's diamond+mesh find C's exact
       (-102,-111) → dv=(-816,-888), which wins the leaf.
    3. **The var-tx `txfm_partition_cost` was a FRAME CONSTANT, not the per-SB adapted value.**
       Even with the DV matching, the intrabc coeff-arm var-tx RD came out 47 rate units low
       (16335 vs C's 16382, same partition/tx_types/recon). Decomposed to the var-tx ROOT
       split-flag cost: both compute ctx 19, but `txfm_partition_cost[19][1]` is port **850** vs C
       **897** — C refreshes it at every INTERNAL_COST_UPD_SB from the *adapting* `txfm_partition`
       CDF (screen frames code split flags as intrabc coeff-arm blocks land), while the port used
       the frame-init `ibc.txfm_partition_costs`. Fix: add `txfm_partition_costs` to
       `RealCosts`/`derive_real_costs` (from `kf.txfm_partition`) and to `SbEncodeEnv` (the per-SB
       refresh); the intrabc leaf reads `env.txfm_partition_costs`. After this the intrabc leaf RD
       is **26166279 == C to the unit** and mi(40,28) picks **VERT** matching C.
    All three are intrabc-only / per-SB-cost-additive, so the intra envelope is byte-inert: full
    aom-encode+aom-bench suite **340/340** (KB-6 30/30 real content, KB-13 speed1-4, speed 0-9
    gates, palette, lossless, bd10, LR, multitile all unchanged; the SbEncodeEnv field is zeroed on
    non-intrabc paths where the var-tx never runs).
  - **REMAINING RESIDUAL (witness still PINNED, floor stays 1120) — PACK symbol coding, NOT the
    search and NOT the coeff re-encode.** With the search C-exact at mi(40,28), the witness is
    **port 1886B vs c 1891B (delta -5), first-diff still 1120** (output-inert to root 3 — VERT wins
    either way — and identical under clean vs instrumented C). RULED OUT by a port PACK re-encode
    dump (`encode_b_intrabc_coeff` → `y_txbs`): the port's re-encoded VERT-sub0 intrabc block is
    **txb[0] eob=12 tx_type=7 txb_skip_ctx=4, txb[1] eob=0 (skip) txb_skip_ctx=3** — identical to
    C's coded tree (2×TX_4X4, tx_types [7,0], dist 93973), so the DV, the var-tx partition, the
    tx-types, the coeffs, and the coeff-coding contexts are all correct. The byte-1120 divergence
    is therefore in the block's remaining PACK **symbol coding** — one of: the DV diff
    (`dv - dv_ref`, `write_intrabc_info`), the `use_intrabc` flag, or the `write_tx_size_vartx`
    txfm-partition split-flag CDF/ctx at pack time (a possible KB-15-ROOT-6-class pack-side
    txfm-context drift distinct from the RD-cost fix in root 3) — NOT the coeffs. NEXT: a symbol-
    level compare of the port's coded bitstream vs C at the mi(40,28) coding position (the
    `/root/aom-inspect` accounting tool, or a C-side write-path dump), focused on the DV /
    use_intrabc / txfm-partition symbols. Do NOT re-chase the DV search, the epb, the NSTEP
    pattern, the txfm_partition COST, or the coeff re-encode — all closed and unit-verified.
    (Aside: the pack re-encode's leaf1 `dc_sign_ctx` reads 0 where the C *search* had 1, but that
    is INERT here — eob 0 codes no dc-sign symbol — and the search/write contexts legitimately
    differ per KB-6; flagged only so a future pass doesn't mistake it for the cause.)
  Working notes: `docs/inter-vartx-coeff-arm-notes.md` (updated with the chroma inter path, the
  encode-vs-write walk-order difference, and the `set_skip_txfm` nonzero-rate detail).

### KB-ARM-FLOAT — aarch64: 15 float C-differentials in aom-encode fail — CLOSED ✅ 2026-07-30 (all four roots; `--workspace` green on aarch64, CI leg widened)
- **Symptom (as first logged):** on `aarch64-apple-darwin` the workspace suite is 755 passed
  / 15 failed. Every
  failure is float-domain and every one is in `zenav1-aom-encode`:
  `cnn_partition_cnn_diff`, `cnn_partition_decision_diff`, `cnn_partition_nn_diff`,
  `curvfit_diff`, `denoise_and_model_diff`, `hog_prune_diff`, `intra_rd_pick_diff`,
  `noise_fft_diff`, `noise_model_diff`, `noise_strength_solver_diff`, `quant_setup_diff`,
  `rd_mult_diff`, `wiener_denoise_diff`. Typical delta is a few ULP, e.g.
  `cnn_buffer[1]: rust=0.5873599 (0x3f165d38) c_scalar=0.58736044 (0x3f165d41)` — 9 ULP.
- **Scope, MEASURED 2026-07-25:** the failure set is BYTE-IDENTICAL to the set at clean
  `4b92e2b` in a sibling worktree (`diff` of both sorted lists is empty), so it is a
  property of the ARM box, not of any landing since. `aom-dsp` (352/352) and `aom-decode`
  are unaffected — this is float-only; every integer differential passes.
- **ROOT #1 — FP CONTRACTION IN THE ORACLE. CONFIRMED + FIXED 2026-07-28 (11 of the 15).**
  The fp-contract hypothesis was correct. Clang defaults to `-ffp-contract=on` for C; on
  aarch64 `fmadd` is baseline so `a*b + c` fuses and rounds ONCE, while Rust never contracts.
  On x86-64 the same source cannot fuse because **no libaom TU is compiled with FMA** — the
  AVX2 object libraries get `-mavx2` only (`cmake/aom_optimization.cmake:57`,
  `av1/av1.cmake:812-817`), and neither clang nor gcc lets `-mavx2` imply `-mfma`. That
  asymmetry is the whole bug. Measured at the instruction level, not inferred:
  `av1_nn_predict_c` in the ARM oracle carried 2 `fmadd`s (the scalar remainder loops; the
  unrolled body was already `fmul.4s` + in-order `fadd`, which is why the deltas were only a
  few ULP) → 0 with the flag. Isolated probe: `float v=0; for(i) v += w[i]*x[i];` at `-O3`
  emits 1 `fmadd` under Apple clang 21 and Homebrew clang 22 on aarch64, 0 with
  `-ffp-contract=off`, and 0 on `-target x86_64-apple-darwin` at the default.
  **Fix:** `ORACLE_FP_CFLAGS = "-ffp-contract=off"` pinned in `crates/aom-sys-ref/build.rs`
  on BOTH oracle compile paths (cmake `CMAKE_C_FLAGS` for libaom, and the shim `cc` line),
  with the build-cache stamp keyed on the flags so stale build dirs rebuild.
  **Measured:** `cargo test --profile test-fast -p zenav1-aom-encode` on aarch64 went
  25 failed → 4 (the 15 KB-ARM-FLOAT tests → 4; the other 10 baseline failures were a
  worktree-local missing `conformance/data`). Fixed outright: `cnn_partition_nn_diff` (2),
  `curvfit_diff` (2), `denoise_and_model_diff`, `noise_fft_diff`, `noise_model_diff`,
  `noise_strength_solver_diff`, `quant_setup_diff`, `rd_mult_diff`, `wiener_denoise_diff`.
- **Why the ORACLE was the thing to pin, not the port** (the judgement call, recorded):
  a differential is only meaningful if "the C answer" is ONE value. Contraction is
  implementation-defined in C (`FP_CONTRACT`), so leaving it at the default makes
  "bit-exact vs libaom" mean different things on different hosts — every KB-1..KB-16
  byte-exactness claim would be host-qualified. Rust has no fast-math, so "make the port
  match" would mean hand-placing `f32::mul_add` exactly where one clang version on one
  target chose to fuse: that specifies nothing and cannot be maintained. Pinning
  `-ffp-contract=off` makes the oracle strictly IEEE-per-operation and host-independent,
  which is the same class of decision as the already-pinned `CONFIG_MULTITHREAD=0`.
  **Honest tradeoff:** a *production* aarch64 libaom build DOES contract, so a real
  ARM-native libaom encoder can make marginally different RD decisions from both this port
  and from x86 libaom. That is a property of libaom-on-ARM; the port matches the x86-64
  build, which is the platform every gate in this repo was established on.
- **x86-64 impact: provably none.** No libaom or shim TU enables FMA on x86-64 (evidence
  above), so the compiler could not contract there before the flag and cannot after — the
  flag is a no-op on the x86 oracle. The x86 AVX2 float kernels use explicit intrinsics
  (`ml_avx2.c` splits mul/add across statements, which `-ffp-contract=on` does not fuse
  anyway). Full x86 confirmation is CI's.
- **ROOT #2 — aarch64 RTCD binds SIMD at COMPILE TIME, so the "C-scalar" oracle isn't scalar.
  CNN HALF FIXED ✅ 2026-07-28 (2 of 3); `hog_prune_diff` remains, and is NOT fixable this
  way — see the hog paragraph.** NEON is baseline on arm64, so libaom's generated
  `config/av1_rtcd.h` emits `#define av1_nn_predict av1_nn_predict_neon` and
  `#define av1_cnn_convolve_no_maxpool_padding_valid ..._neon` — a macro, not the swappable
  function pointer x86-64 gets. `av1_cnn_predict_c` (cnn.c:703) therefore called the NEON
  convolve, so `rd_shim.c`'s `force_cscalar` arm returned a NEON result on ARM while the
  differential believed it was scalar (residual after root #1: `cnn_partition_cnn_diff`
  9 ULP → **1 ULP**, `cnn_partition_decision_diff` downstream of it).
- **CNN FIX: a scalar-bound copy of libaom's own engine in the shim archive.**
  `crates/aom-sys-ref/shim/cnn_cscalar.c` includes `config/av1_rtcd.h` first (so its
  per-target declarations are processed verbatim), then rebinds the CNN's ONE
  RTCD-dispatched primitive to `av1_cnn_convolve_no_maxpool_padding_valid_c` and renames all
  10 symbols `av1/encoder/cnn.c` exports (`shim_cscalar_*`, entry point
  `shim_cnn_predict_img_multi_out_cscalar`), then `#include "av1/encoder/cnn.c"`. Nothing is
  transcribed — it is libaom's source with one binding pinned. `rd_shim.c`'s `force_cscalar`
  arm calls it; the `force_cscalar = 0` arm still calls libaom.a's dispatched
  `av1_cnn_predict_img_multi_out`, so both bars of the differential survive.
  `shim_cnn_force_cscalar_supported()` now returns 1 unconditionally.
  **The rebinding is uniform across targets, deliberately** — the old pointer swap was a
  second, x86-only mechanism that no CI leg could validate against the ARM one; one
  mechanism means the CNN oracle denotes the same thing on every host, the same reasoning as
  `-ffp-contract=off` and `CONFIG_MULTITHREAD=0`. build.rs compiles this one TU at
  `-O3 -DNDEBUG` (`extra_shim_cflags()`) to match libaom's Release flags exactly; absent
  fast-math the optimisation level cannot change float values anyway.
  **Measured (aarch64-apple-darwin, `--profile test-fast -p zenav1-aom-encode
  --no-fail-fast`, conformance/data provisioned): 269 passed / 3 failed →
  271 passed / 1 failed.** `cnn_predict_matches_c_scalar_bit_exact_and_reports_avx2_gap`
  and `predict_decision_matches_c` are bit-exact; no test was skipped, tolerated or
  modified (the two test files are untouched).
  **x86-64 impact, stated plainly: the mechanism changes, the values should not.** The
  forced-scalar path used to be libaom.a's `_c` convolve reached through a swapped pointer;
  it is now the shim's compiled-from-the-same-source `_c` convolve reached through a
  compile-time binding. Verified by compiling both changed TUs with
  `clang -target x86_64-apple-darwin` against a **real x86_64-generated** `config/av1_rtcd.h`
  (`cmake -DAOM_TARGET_CPU=x86_64`, where the primitive is an `RTCD_EXTERN` pointer object,
  `HAVE_AVX2=1`): both compile clean; `nm` shows the object exports exactly the 10 renamed
  symbols (zero collision with libaom.a, which still owns `av1_cnn_predict_img_multi_out`)
  and has **zero undefined CNN references** — i.e. it neither reads the RTCD pointer nor
  calls the AVX2 variant, so the scalar binding resolved entirely inside the TU. Identical
  `nm` shape on the arm64 object (no `_neon` reference). Running the x86 differential is
  CI's. **No CI cache-salt bump is needed** (checked, not assumed): the `libaom-…-v3` caches
  cover only `upstream/build`, and libaom's own cmake flags and the build.rs SHA/FP stamp are
  unchanged — the new flags apply to a shim TU, whose objects live in the cargo target dir
  and are rebuilt because build.rs itself changed plus its `rerun-if-changed` on the new
  `shim/cnn_cscalar.c`. **What DOES need updating is the `test-macos-aarch64` comment block
  in `.github/workflows/ci.yml` (~lines 131-147)**, which still says four tests fail under
  two roots; it is now ONE (`hog_prune_diff`). Widening that leg to `--workspace` remains
  blocked, but by a single inherently-x86 contract rather than by four unresolved failures.
- **hog (`hog_prune_diff::hog_nn_predict_matches_avx2_and_dispatch`) is NOT fixable this way,
  and must not be "fixed" by changing the port.** The port's `hog_nn_predict` replicates the
  **x86 AVX2** kernel's lane math by design (hog.rs module docs), matching neither `_c`
  (sequential) nor `_neon` (4-lane tree). The CNN worked because `..._valid_c` exists on
  every target; `av1_nn_predict_avx2` does **not** exist in an ARM libaom at all (`ml_avx2.c`
  is x86-intrinsic source, never compiled), so no shim or oracle build option can produce the
  AVX2 accumulation order on an ARM box. Changing the port's order would change x86 output —
  the shipping path — and is forbidden. Prior measurement: pointing the ARM arm at
  `av1_nn_predict_c` gives 1 ULP, not 0.
  **New measurement 2026-07-28 (throwaway probe, reverted), 20k cases × 8 lanes × 2 modes:**
  at `reduce_prec = false` **49,918 / 160,000 lanes differ, worst |Δ| = 1.53e-5** (a few
  ULP); at `reduce_prec = true` — what the real caller passes — only **56 / 160,000 differ,
  but worst |Δ| = 1.953e-3 = exactly one 1/512 prec-reduce bucket.** So libaom's
  `av1_nn_output_prec_reduce` is a *near*-equaliser across SIMD variants, not a guarantee:
  ~0.035% of evaluations sit on a bucket boundary where a ULP flips a whole bucket, which can
  flip the `score <= -1.2f` prune mask and hence the encode. That is a property of
  libaom-on-ARM (same shape as the root-#1 contraction finding), and the port matches the
  x86-64 build — the platform every gate here was established on. Consistent with that, the
  three mask-level hog tests (`prune_intra_mode_with_hog_matches_c`, `..._uv_...`,
  `generate_hog_matches_c`) all PASSED on aarch64 throughout; only the ULP-level NN test
  failed — which is itself evidence that the mask, not the ULP, is the encode-relevant layer.
  **RESOLVED 2026-07-30 by STATING the contract, not by relaxing one** (user decision,
  verbatim: *"as long as it matches libaom that is fine"* — the port matches x86-64 libaom,
  which IS matching libaom; what was wrong is that the test did not SAY so, and failed on ARM
  as though something were broken). `crates/aom-encode/tests/hog_prune_diff.rs` now carries
  TWO tests over the SAME shared 20,000-case input corpus (`hog_case_hist`), each with a
  one-sentence contract in its doc comment:
  1. **`hog_nn_predict_matches_avx2_and_dispatch`** — `#[cfg(any(target_arch = "x86_64",
     target_arch = "x86"))]`. **Assertions UNCHANGED** (f32 bit-equality vs
     `av1_nn_predict_avx2` at both `reduce_prec` settings + RTCD-dispatch identity); only the
     label is new. It is now stated as the x86-64 contract it always was.
  2. **`hog_nn_predict_agrees_with_dispatch_within_one_prec_quantum`** — `#[cfg(not(...))]`.
     The non-x86 contract, every clause a hard equality or an exact integer bound:
     (a) the RTCD-dispatched kernel IS the widest SIMD variant libaom has on this target
     (bit-equal — on aarch64 `av1_nn_predict` is `#define`d straight to `_neon`);
     (b) each side's `reduce_prec = true` output is bit-equal to `av1_nn_output_prec_reduce`
     (ml.c:19-25, transcribed in the test) of its OWN `reduce_prec = false` output — which
     both establishes the 1/512 lattice rather than assuming it AND keeps the
     `reduce_prec = false` regime covered instead of dropped;
     (c) both sides land EXACTLY on that lattice (`v * 512` is integral, asserted — not a
     rounding tolerance);
     (d) their lattice indices differ by **at most 1** — the bound is `prec_bits = 9` read off
     ml.c:20, not a chosen epsilon;
     (e) **prune-mask parity**: `score <= th` agrees at every threshold production ever passes
     (`{-1.2, -0.6, 0.0, 0.4, 1.2}` — intra_mode_search.c:1321/1505/961-964; `0.0` is itself
     ON the lattice so it is the most exposed and is deliberately included), with both
     polarities of every threshold asserted to occur >1,000 times so parity is not vacuous;
     (f) the one-quantum divergence is **characterized and pinned**, not tolerated:
     `one_quantum_lanes <= 56` so it cannot grow, and `> 0` so a target where it vanished gets
     routed to the bit-exact x86 contract instead of silently passing this one.
  **Measured on aarch64-apple-darwin (the test prints it):** `lanes=160000
  one_quantum_lanes=56 worst_gap=1 quanta mask_flips=0` — i.e. the 56/160,000 one-bucket
  figure above, and mask parity is EXACT on this corpus. **Teeth verified, not assumed:**
  perturbing `INTRA_HOG_MODEL_BIAS[0]` by 0.002 (~one quantum) makes the ARM test FAIL first
  on clause (e) — *"prune-mask parity broke: 28 flips (pinned max 0) across 160000 lanes x 5
  production thresholds"*, samples showing exactly the predicted straddles
  (`port=-1.1992188 (-614/512, prune=false)` vs `oracle=-1.2011719 (-615/512, prune=true)`).
  Perturbation reverted; `git diff` clean on `src/`.
  **Why this is not the banned `cfg!(target_arch)` skip:** a skip makes a test disappear and
  leaves the kernel unasserted on that target. Here every target has a complete, hard contract
  — the x86 one bit-exact, the ARM one exact-lattice + exact-integer-bound + mask parity — and
  neither is a weakened form of the other; they assert different quantities because different
  quantities are defined. Nothing is `#[ignore]`d and no tolerance was widened.
  **Rejected alternative (still rejected):** making the port's NN kernel pick lane order by
  `cfg!(target_arch)` — it would pass everywhere but make the *port's own output*
  host-dependent, contradicting the root-#1 decision that the port matches the x86-64 build.
- **ROOT #3 — the harness fed `av1_block_error_c` OUTSIDE its defined domain
  (`intra_rd_pick_diff`; NOT a float bug at all) — FIXED ✅ 2026-07-28.**
  `rdopt.c:892`'s `error += diff * diff` multiplies in `int`, so the kernel is DEFINED
  only where that product is representable (`|diff| <= 46340`). Past that it is
  signed-overflow UB and "the C answer" is not one value: measured on aarch64 the oracle
  vectorises as `mul.4s` (32-bit, wrapping — same as x86) but widens with
  **`uaddw`/`uaddw2`, i.e. ZERO-extension** (clang proved the product non-negative from
  the absence of UB), so a wrapped-negative product gains exactly 2³²; the port models the
  x86 sign-extending behaviour. Signature confirmed it: every failing `dist` differed by
  exactly N·2³² pre-shift (port −387033316 vs C 3907933980, N=1), and ONLY the `bd=8`
  arm failed. Byte-identical before and after `-ffp-contract=off`, so fully independent
  of root #1.
- **The harness-vs-production question, ANSWERED WITH ARITHMETIC: only the harness could
  reach it — this is NOT a libaom production bug.** At bit depth `bd`:
  * `coeff` is forward-transform output, bounded to **`bd + 8` signed bits** by
    `av1_gen_fwd_stage_range` (av1_fwd_txfm2d.c:41) over `fwd_txfm_range_mult2_list` +
    `av1_fwd_txfm_shift_ls`. Evaluated across all 19 tx sizes × all 1-D type pairs
    (`final = ((mult2_col[last] + mult2_row[last] + 1) >> 1) + shift[0] + shift[1] + bd + 1
    + shift[2]`), the worst final width at bd=8 is exactly **16 signed bits** ⇒
    `|coeff| <= 2^15`.
  * `dqcoeff` is spec-clamped to `[-(1 << (7 + bd)), (1 << (7 + bd)) - 1]`
    (decodetxb.c:116) ⇒ `|dqcoeff| <= 2^15` at bd=8, and **every** quantizer variant
    writes it with `coeff`'s sign: `(abs_dqcoeff ^ coeff_sign) - coeff_sign`
    (quantize.c:77 / :164 / :231 / :313).
  Same sign + both `<= 2^15` ⇒ `|diff| <= 2^15` ⇒ `diff*diff <= 2^30`; concretely
  `32768² = 1 073 741 824 < INT_MAX = 2 147 483 647`. bd>8 is structurally immune —
  `av1_highbd_block_error_c` (rdopt.c:919) uses `int64_t diff` and explicit `(int64_t)`
  casts. This is also the domain **libaom's own unit test declares** for the kernel
  (`test/error_block_test.cc`: `msb = bit_depth + 8 - 1`, with the comment that coeff and
  dqcoeff "always have at least the same sign"), and the bound
  `aom-dsp/tests/block_error_diff.rs` already pinned for its lowbd generator (~14-bit
  magnitudes) — which is why that differential always passed on ARM.
- **What actually left the domain: an unrealizable QUANTIZER PARAMETER TRIPLE, not a
  coefficient magnitude.** `intra_rd_pick_diff` drew `quant = 65536/dq` and
  `quant_shift = rng.range(8000, 32767)` **independently**, but AOM_QUANT_B consumes the
  two as ONE reciprocal —
  `tmp32 = ((((tmp*quant) >> 16) + tmp) * quant_shift) >> (16 - log_scale + AOM_QM_BITS)`
  (quantize.c:70) — and libaom builds them together in `invert_quant`
  (av1_quantize.c:582): `quant = 1 + (1 << (16 + msb(d)))/d - 65536`,
  `quant_shift = 1 << (16 - msb(d))`. For the harness's `d ∈ [16, 800)` the REAL
  `quant_shift` is 128..4096, never up to 32767. With the real pair
  `abs_dqcoeff ≈ tmp ≈ |coeff|`; with the independent pair
  `abs_dqcoeff ≈ tmp·d·quant_shift / 2^16`, i.e. an overshoot factor of
  `d·quant_shift/2^16` = 2× … ~400×, reaching **~1.3e7 against the spec's 32768 cap** —
  hence `|diff| ~ 1.3e7` and a `diff*diff` that wraps many times.
- **Fix (`crates/aom-encode/tests/intra_rd_pick_diff.rs`, harness only — no port change,
  no tolerance, no `#[ignore]`, no `cfg!(target_arch)` skip):** derive the AOM_QUANT_B
  `(quant, quant_shift)` from `dequant` exactly as `av1_build_quantizer` does; the Fp arm
  keeps `65536/dq`, which IS libaom's `*_quant_fp` (av1_quantize.c:628), so the bd=12
  arm's inputs are unchanged. Plus a **domain guard** asserting every bd=8
  `(coeff, dqcoeff)` pair keeps `diff*diff` and `coeff*coeff` representable in int32, so a
  future generator change fails loudly at the offending value instead of reporting a
  target-dependent UB result as a port bug. The `argmin_spread >= 30` non-vacuity guard
  still passes, so the argmin is still a real decision on the constrained inputs.
  **Measured:** aarch64 `cargo test --profile test-fast -p zenav1-aom-encode
  --no-fail-fast` 268 passed / 4 failed → **269 passed / 3 failed**; the 3 residuals are
  all root #2 and were untouched.
- **Why none of this is "just relax the test":** STATUS's own rule is that the float decision
  helpers stay scalar *because* float reassociation shifts RD decisions. A few-ULP oracle
  disagreement on ARM is the same hazard wearing a different hat — it means an ARM-built
  encoder can make different partition/mode choices than the x86 one. Diagnose, do not widen
  the tolerance.
- **Blast radius today: none.** No decoder path, no integer path, and no *port* source line
  changed for any of the four roots — roots #1/#2 moved the ORACLE's build/binding, root #3
  and this one moved HARNESSES. The port's shipping bytes are byte-for-byte what they were.
- **CI (2026-07-30):** `test-macos-aarch64` is **widened from `-p zenav1-aom-dsp` to
  `cargo test --profile test-fast --workspace --no-fail-fast`**, both dispatch modes, with the
  Gate-1 conformance corpus provisioned/cached exactly as the linux legs do (the workspace
  suite fail-loud-asserts on an empty `conformance/data`) and `timeout-minutes: 90` matching
  the linux legs. The scoping comment in `.github/workflows/ci.yml` is rewritten to record the
  closure and to answer, in place, its own prior "do NOT resolve this by relaxing or skipping"
  instruction.
- **Verified before widening, on aarch64-apple-darwin (this box), `--profile test-fast
  --workspace --no-fail-fast`, both dispatch modes:** default dispatch **850 passed / 0 failed
  / 6 ignored**; `AOM_FORCE_SCALAR=1` scalar pin likewise green. Per-crate,
  `-p zenav1-aom-encode` went **271 passed / 1 failed → 272 passed / 0 failed** (the +1 is the
  new non-x86 test; the x86 test compiles on this host — checked by temporarily forcing its
  `cfg` on, zero errors — but is correctly not selected here).

### KB-20 — Encoder: bd10/bd12 x `--cpu-used>=8` PANICKED (unported hbd nonrd estimate arm) — FIXED ✅ 2026-07-30 (roots 1-3 aarch64-measured; root 4 = the x86 `aom_hadamard_16x16` tier split, source-derived + CI-measured)
- **Found 2026-07-30** by the speed axis of the config-permutation gate (`9a996b9`).
  `crates/aom-encode/src/nonrd_pickmode.rs` carried a hard `assert!(env.bd == 8, "HANDOFF: hbd
  estimate arm (av1_quantize_fp + fp scans) not ported")`. Speeds 8 and 9 use the nonrd
  PICKMODE path (KB-12), so **every bd10/bd12 encode at cpu-used >= 8 panicked** — on a stream
  real aomenc produces without complaint. Worse than a divergence: a panic on valid input.
  PARITY.md §A listed cpu-used 8/9 byte-identical AND bd10/bd12 byte-identical, each on its own
  grid, never crossed; the panic sat in the gap between two green rows.
- **FIXED — bd10 and bd12 x `--cpu-used` {8,9} are now BYTE-IDENTICAL to real aomenc**, 24 cells
  (bd{10,12} x cq{5,12,20,32,48,63} x speed{8,9}, `av1-1-b10-00-quantizer-00` 64x64 crop; bd12
  by the `pix << 2` promotion `deltaq_mode3_e2e.rs` established). Gate:
  `config_permutations::speed_nonrd_hbd_byte_identity`; the `speed_envelope_stock_map_is_pinned`
  b10_64 row flips `s8/s9` from `panic` to `ok`.
- **THREE things were bd8-specific, and the assert named only one of them.**
  1. `av1_block_yrd`'s hbd arm (`nonrd_opt.c:199-215` + `update_yrd_loop_vars_hbd` :92) —
     `aom_hadamard_{8x8,16x16}` (32-bit `tran_low_t`), `av1_quantize_fp`, `aom_satd`,
     `av1_highbd_block_error`, over `default_scan_8x8_transpose`/`av1_default_iscan_8x8_transpose`
     and `default_scan_fp_16x16_transpose`/`av1_default_iscan_fp_16x16_transpose`. The 16x16 **fp**
     scan pair is NOT the **lp** pair the lowbd arm uses (`aom_hadamard_16x16_c` carries an extra
     AVX2-matching column shift `aom_hadamard_lp_16x16_c` does not), and libaom requires each
     scan/iscan pair be used together. Port: `nonrd_pickmode::block_yrd_hbd` — every kernel was
     already in `aom-dsp` (`dist::hadamard::{hadamard_8x8,hadamard_16x16,satd}`,
     `dist::highbd_block_error`), so this really was the deltaq-mode-3 shape: reuse, don't invent.
  2. the speed-9 SAD prune in `av1_estimate_block_intra` (`nonrd_opt.c:629`) —
     `fn_ptr[bsize].sdf` is bound by `highbd_set_var_fns` to `aom_highbd_sadWxH_bits{8,10,12}`,
     i.e. the raw SAD `>> (bd - 8)` (`MAKE_BFP_SAD_WRAPPER`, encoder_utils.h:158). NOT named by
     the assert. **Measured decision-inert on the 24-cell grid** (the prune is a ratio test, so an
     equal shift on both sides only changes rounding) — but the WRONG form `2 * (bd - 8)` diverged
     on 1 of the first 12 cells, which is how the right one was found. Kept because it is what
     libaom does; do not read the inertness as permission to drop it.
  3. **`av1_quantize_fp` is ISA-CONDITIONAL in this regime — the real root of the last 3 cells.**
     Every SIMD tier of `av1_quantize_fp` is a 16-bit kernel that narrows `tran_low_t` on load and
     computes `dqcoeff` in 16 bits. `aom_hadamard_16x16`'s 4-way combine reaches **+-65534**, so
     the hbd nonrd estimate leaves the `int16` range routinely and the tiers stop agreeing with
     `av1_quantize_fp_c` **and with each other**: `av1_quantize_fp_neon`
     (`arm/quantize_neon.c:57/76`) narrows with `vmovn_s32` (**TRUNCATING**), no per-coefficient
     dequant threshold, `vmulq_s16` for `dqcoeff`; `av1_quantize_fp_avx2`
     (`x86/av1_quantize_avx2.c:194/224`) narrows with `_mm_packs_epi32` (**SATURATING**) and gates
     a 16-lane group on `abs > (dequant >> 1) - 1`. **Measured on this aarch64 reference build:
     NEON model 12/12 byte-identical, saturating (x86) model 9/12, `av1_quantize_fp_c` 9/12.**
     Port: `nonrd_pickmode::quantize_fp_dispatched`, `cfg(target_arch)`-selected (aarch64/arm =
     NEON model, MEASURED; x86/x86_64 = AVX2 model, TRANSCRIBED but not runnable on this host;
     otherwise `_c`). This is the same class of ISA-conditional truth as the `-ffp-contract` note
     in `reference/BUILD_CONFIG.md`. **The "+-65534" premise holds on `_c`/NEON ONLY** — root #4
     below is why it is false on x86, which made this model's x86 arm inert rather than wrong.
     The old "if the KB-20 gate fails on x86, check the SSE2 tier first" hint pointed at the
     wrong suspect: `av1_quantize_fp_sse2`'s `thr = dequant >> 1` (no `- 1`) and 8-lane grouping
     do differ from AVX2, but neither is reachable at this call site once root #4 is modelled.
- **Unit gates:** `aom-encode/tests/nonrd_block_yrd_hbd_diff.rs` — the walk composition vs a
  pure-C oracle (`ref_hadamard`/`ref_quantize_fp`/`ref_satd`/`ref_highbd_block_error`) over 4,800
  bd10/bd12 cells, run at 8-bit residual magnitude where every tier and `_c` provably agree (the
  precondition is ASSERTED, not assumed); `quantize_fp_dispatched` reduces exactly to
  `av1_quantize_fp_c` inside `int16`; and a teeth test that it genuinely differs outside it. The
  scan tables are pinned as mutually inverse permutations, and fp != lp.
- **Teeth (all re-verified, then reverted):** restoring the assert makes the 24-cell gate panic
  with the original HANDOFF message; swapping `quantize_fp_dispatched` for the `_c` semantics
  makes 3 cells diverge (bd10 cq12 s9 710 vs 709 B; bd12 cq63 s8 47 vs 46 B; bd12 cq63 s9 44 vs
  42 B).
- **Still out of envelope at every bit depth** (unchanged, both arms): lossless TX_4X4
  (`unimplemented!`) and the screen-content palette arm (`debug_assert`).
- **The first x86 CI run (30595796744) FAILED the 24-cell gate — and the failure exonerated
  root 3 and named a FOURTH ISA-conditional kernel.** Both `quantize_fp_dispatched` unit teeth
  (`..._reduces_to_c_inside_int16`, `..._differs_from_c_outside_int16`) PASSED on x86 while
  `speed_nonrd_hbd_byte_identity` + the `speed_envelope_stock_map_is_pinned` b10_64 row failed
  (identically under the scalar pin, as expected — the arm is `cfg`-selected, so
  `AOM_FORCE_SCALAR` cannot reach it). That combination localises the divergence UPSTREAM of the
  quantizer. Measured x86 table: bd10 s8 MATCH at cq{5,12,20,32} and MISMATCH at cq{48,63};
  bd10 s9 MATCH at cq5 and MISMATCH at cq{12,20,32,48,63} (port 710/487/293/120 vs C
  708/490/295/119) — i.e. bigger `dequant` (higher cq) -> more coefficients out of `int16`.
- **ROOT #4 — `aom_hadamard_16x16` is ISA-conditional TOO, and it runs first. FIXED.**
  The 4-way combine is `tran_low_t` (**int32**) in `aom_hadamard_16x16_c` (`aom_dsp/avg.c:249`)
  and `int32x4_t` (`vhaddq_s32`/`vaddq_s32`) in `aom_hadamard_16x16_neon`
  (`aom_dsp/arm/hadamard_neon.c:188`) — but **`int16` with wrapping** in
  `aom_hadamard_16x16_avx2` (`aom_dsp/x86/avg_intrin_avx2.c:144`, `_mm256_add_epi16` +
  `_mm256_srai_epi16`) and `aom_hadamard_16x16_sse2` (`aom_dsp/x86/avg_intrin_sse2.c:442`),
  whose `store_tran_low` then sign-extends the wrapped `int16` back to `tran_low_t`. libaom's own
  comments bound the input at 9-bit `src_diff` and the output at "[-32640, 32640]" — inside
  `int16`, so at bd8 every tier agrees and the split is invisible (which is why upstream's
  cross-tier tests never see it). The hbd estimate feeds an 11-/13-bit residual, the combine
  reaches +-65534, and x86 wraps where `_c`/NEON do not — changing the coefficients BEFORE
  `av1_quantize_fp`, `aom_satd` and `av1_highbd_block_error` ever see them. Port:
  `nonrd_pickmode::hadamard_16x16_dispatched`, `cfg(target_arch)`-selected. **This arm is
  tier-INDEPENDENT on x86-64** (AVX2 and SSE2 make the same `int16` choice, and SSE2 is baseline),
  unlike root 3 — so the AVX2-vs-SSE2 worry recorded above is moot here.
- **Root 3's model was RIGHT; its premise was wrong on x86.** With root #4 fixed, every
  coefficient reaching `av1_quantize_fp` on x86 is `int16`-valued, so `_mm_packs_epi32` is inert
  and the AVX2 group threshold `abs > (dequant>>1) - 1` never separates from `_c` at this call
  site. `quantize_fp_dispatched` is unchanged.
- **Blast radius checked: `block_yrd_hbd` is the ONLY exposed call site.** The port's other
  `hadamard_16x16` caller (`tx_search.rs:2626`) is the `(hbd == false, TX_16X16)` arm — the hbd
  arm one line down calls `highbd_hadamard_16x16` instead, and `aom_highbd_hadamard_16x16` is
  32-bit in every tier (`aom_dsp/x86/avg_intrin_avx2.c:419`). So no other path can feed
  out-of-`int16` values into an `int16` x86 combine.
- **The three neighbouring kernels were checked the same way and are NOT ISA-conditional:**
  `aom_hadamard_8x8` is `int16_t` in `_c` (`hadamard_col8`, avg.c:149), SSE2 and NEON alike;
  `aom_satd_{avx2,sse2}` accumulate `abs_epi32` in 32 bits like `_c`; `av1_highbd_block_error_avx2`
  subtracts in `epi32` and squares with `_mm256_mul_epi32` into 64-bit accumulators like `_c`.
- **Unit gates for root #4** (`nonrd_block_yrd_hbd_diff.rs`, both arch arms assert something):
  `hadamard_16x16_models_agree_with_c_at_bd8_magnitude` pins BOTH models to the real exported
  `aom_hadamard_16x16_c` at 9-bit residual magnitude on every host;
  `hadamard_16x16_dispatch_is_isa_conditional_at_hbd_magnitude` pins the `_c` model to the real C
  fn at bd10 magnitude, requires the dispatched model to DIVERGE on x86 and to be byte-identical
  off x86, and asserts the x86 output is `int16`-valued (the fact that makes root 3's narrow
  inert). Its grid is deliberately CORRELATED: the first version used uniform white noise and
  self-reported `0 of 500` out-of-`int16` blocks — the combine only leaves `int16` when the four
  8x8 quadrants agree in sign, i.e. on the smooth content an intra residual actually produces.
- **Root #4 CONFIRMED FROM THE AARCH64 BOX, before CI answered** — throwaway instrumentation
  (added, run, reverted; never committed) counted, per KB-20 cell, how many `aom_hadamard_16x16`
  outputs leave `int16` on the `_c`/NEON arm. That predicate is *exactly* the x86 mismatch set:
  all 7 x86-divergent cells have out-of-`int16` coefficients (bd10 cq12/s9 4 blocks, cq20/s9 3,
  cq32/s9 3, cq48/s8 4, cq48/s9 3, cq63/s8 16, cq63/s9 16) and all 15 x86-matching bd10/bd12
  cells below cq63 have **zero**. 7/7 recall, 22/24 cells predicted. The two misses are bd12
  cq63 s8/s9 (11 and 7 out-of-range blocks, MATCH on x86) — false negatives only, and expected:
  the estimate is a *decision* input, so a numeric difference need not flip the winning mode,
  least of all at cq63 where nearly everything skips. That also explains the otherwise odd shape
  of the x86 table (all 12 bd12 cells matched while half of bd10 diverged): the bd12 residual
  reaches the overflow band in fewer blocks on this crop, not more.
- **VERIFICATION SPLIT — say which half is measured where.** aarch64 (this box): the 24-cell gate
  and all 6 unit gates are green, the non-x86 arm is byte-identical to before this fix (it is the
  same call), `-p zenav1-aom-encode/-bench/-dsp` = 280/171/361 all 0 failed. x86: the model rests
  on the C source cited above plus the CI legs — **no x86 code can be executed on an
  aarch64-apple-darwin box** (Rosetta is not installed here, so even a cross-compiled AVX2 probe
  cannot run). The `cargo check --target {x86_64,i686}-unknown-linux-gnu --lib` legs prove only
  that the x86 arm compiles.

### KB-21 — Encoder: cpu-4/5 is the fragile band — CLOSED ✅ (bd8). ROOT #1 FIXED 2026-07-30 (`early_term_after_none_split` was unported); ROOT #2 FIXED 2026-07-31 (two coefficient-path defects in the speed>=4 tx-type search); ROOT #3 FIXED 2026-07-31 (the QM x speed>=4 axis root #2 predicted — `av1_setup_quant`'s qmatrix NULLing inside the SATD trellis-skip helper)
- **Found 2026-07-30** by the speed axis (`9a996b9`). Not a knob-combination bug: these
  diverge with **every knob at its default**. 3 of 5 contexts are byte-identical at speeds
  0-3 AND 6-9 but diverge at 4 and 5; the nonrd speeds (8/9) are clean.
- **The `multi_winner_mode_type` framing in the original entry was WRONG.** The 4->5 delta
  is that field, but the 4/5 BAND is what matters, and exactly two sf fields are non-default
  in the band (measured by diffing `SpeedFeatures::set_allintra(s)` for s=0..9):
  `prune_tx_type_est_rd` (true at 4,5; false everywhere else) and `multi_winner_mode_type`
  (2/1 at 4/5; 0 everywhere else). Neither turned out to be root #1.
- **ROOT #1 — `part_sf.early_term_after_none_split` was UNPORTED. FIXED.**
  `speed_features.c:477` sets it inside `if (speed >= 4)` of
  `set_allintra_speed_features_framesize_independent`, i.e. it is on at ALLINTRA speeds 4-9;
  the port carried it in the speed-4 doc block as *"(C) INERT on this path (byte no-op,
  verified) — NONE always yields a valid rd on textured content, so it never triggers here"*.
  That premise is FALSE on real photographic content.
  - **Localization (decode-both + ar-swapped sibling-C dump per HANDOFF-TOGGLES.md / KB-3).**
    Repro cell `av1-1-b8-00-quantizer-00` cropped 64x64@(64,64), **monochrome** (so the root
    is proven luma-side), cq32, `--cpu-used=4`, stock knobs: port 504 B vs real aomenc 498 B.
    First divergent node **mi(4,12) BLOCK_16X16: real = PARTITION_NONE, port = PARTITION_SPLIT**;
    everything before it byte-identical.
  - **The two decisions side by side.** The NONE arm is not the problem — port and C agree to
    the unit: `rate=132534 dist=85648 rdcost=22231958`, winner `H_PRED / angle_delta -3 /
    TX_16X16` in BOTH. So is the whole SPLIT accumulation, child for child
    (`sum_rate 88376 / sum_dist 47952 / sum_rdcost 13652233`, remaining budget 8,692,897
    entering child 3 — identical in both). The divergence is entirely inside SPLIT **child 3**,
    the BLOCK_8X8 at mi(6,14):
    * **C**: `pick_sb_modes` returns `rate = INT_MAX` (the remaining budget covers no mode) =>
      `part_none_rd = INT64_MAX`; `do_square_split == 0` at BLOCK_8X8 so the SPLIT stage never
      runs => `part_split_rd = INT64_MAX`; `early_term_after_none_split` fires
      (`partition_search.c:5851-5856`) => `terminate_partition_search = 1` => the rect / AB /
      4-way stages are SKIPPED => `av1_rd_pick_partition` returns `found_best_partition = false`
      => the parent invalidates `sum_rdc` and breaks => SPLIT loses => **NONE**.
      Instrumented-C witness line:
      `CTERM (6,14) bs=3 sf=1 none_rd=INT64_MAX split_rd=INT64_MAX mfvp=0 term=1`.
    * **Port**: no early-term, so it fell through to the rect stage, found a valid
      `PARTITION_HORZ` at mi(6,14), completed the SPLIT at `sum_rdcost = 21662749 <` NONE's
      `22231958`, and picked **SPLIT**.
    It fires at **6 nodes** in that single 64x64 frame — the "inert" claim was off by a lot.
  - **FIX** (`crates/aom-encode/src/partition_pick.rs`, `rd_pick_partition_real`): track
    `part_none_rd` (set POST-pt_cost at `:4474`, not gated on NONE winning) and `part_split_rd`
    (`:4619`, INT64_MAX when the stage was skipped or a child aborted), then apply C's arm
    between the SPLIT and rect stages. `terminate_partition_search` was a `let ... = false`
    constant and is now a real variable; the rect/AB/4-way stages already read it.
    `x->must_find_valid_partition` stays modelled as always false (the established
    simplification; instrumented C confirms `mfvp=0` at every node of the repro frame).
    `SpeedFeatures` gained `early_term_after_none_split` + the shared derivation
    `early_term_after_none_split_allintra(speed) = speed >= 4`, unit-locked by
    `early_term_after_none_split_is_allintra_speed4_up` (speeds 0..9) and asserted in
    `speed4_allintra_deltas_match_source`.
  - **What it closed** (all re-pinned in the same commit; every one is a self-promoting pin
    that FAILED in the closing direction):
    * `speed_envelope_stock_map_is_pinned`: `q00_64` **cpu-5** and `q00_128` **cpu-5** stock
      encodes now byte-match (2 of the 6 pinned stock divergences).
    * `speed_sensitivity_s1`: the cpu-4 open singletons went **{dir0, rtx0, flip0} -> {flip0}**.
    * `combinations_t3_speed4_*`: the three pinned cpu-4 combination rows are gone. NOT a
      like-for-like count — unpinning `(4,dir0)`/`(4,rtx0)` also lets those levels back into
      the covering array (`remap_open_levels`), so the executed rows changed; on the broader
      array **2** cpu-4 rows are open, both carrying `dir0`.
    * The **5 cpu-8 combination rows are unchanged**, which is the expected answer, not a
      surprise: speed 8 is the nonrd PICKMODE path and never enters `rd_pick_partition`, so
      this root structurally cannot reach it.
    * KB-13 (`encoder_gate_real_content_speed1to4_e2e`): **+2 cells promoted** —
      `av1-1-b8-00-quantizer-00 420 128x128@64,64 cpu4 cq12` and
      `av1-1-b8-23-film_grain-50 420 64x64@96,64 cpu4 cq32` (map 45/60 -> 47/60).
  - **Bite proof:** with the arm reverted to a no-op,
    `speed_envelope_stock_map_is_pinned` fails with *"q00_64 cpu-used=5: stock encode is now
    `diverge` (pinned `ok`)"* + the same for `q00_128`, and
    `encoder_gate_real_content_speed1to4_e2e` fails with *"regression: graduated real-content
    cell `av1-1-b8-00-quantizer-00 420 128x128@64,64 cpu4 cq12` must byte-match real aomenc"*.
  - **Envelope (this box, `--profile test-fast`):** `-p zenav1-aom-encode` **274 passed /
    0 failed** (was 272/1 with `hog_prune_diff` red; that closed on `main` at `faeaf50`, and
    this landing adds the 1 new unit lock + the 1 re-pin), `-p zenav1-aom-bench` **170/0/5
    ignored**, `-p zenav1-aom-dsp` **361/0/3 ignored**.
- **ROOT #2 — FIXED ✅ 2026-07-31. TWO defects in `search_tx_type`'s coefficient path, both
  reachable only at ALLINTRA speed >= 4.** The narrowing below was right about the layer
  (coefficient-level, not search-space) and right that no single sf flip explains it.
  - **Localization method (the deliverable — reusable).** Per-txb dump on BOTH sides, aligned
    into (entry, per-candidate, winner) groups and diffed by group index. C side: the
    HANDOFF-TOGGLES ar-swap — one instrumented `av1/encoder/tx_search.c` TU compiled with the
    flags from `upstream/build/CMakeFiles/aom_av1_encoder.dir/flags.make`, `ar r`'d into
    `upstream/build/libaom.a`, `cargo clean -p zenav1-aom-sys-ref` to force the relink, and the
    archive restored from a pristine copy + `cmp`-verified afterwards. Port side: the same three
    dumps around `search_tx_type_intra`. Both filtered by `mi(row,col)`. **All instrumentation
    was removed before landing.**
  - **FIRST DIVERGENT TXB — group 11 of 51 at mi(2,4)**: BLOCK_8X8, WINNER_MODE_EVAL,
    `DC_PRED / angle_delta 0 / filter_intra 0`, TX_4X4, txb `blk_row=0, blk_col=1` (top-right
    4x4). Everything entering it is IDENTICAL on both sides: allowed tx mask `0x080f` (5 types),
    `block_sse 6480`, `block_mse_q8 6480`, `qstep 22`, `rdmult 43534`, `ref_best_rd 3165954`,
    `skip_trellis 0`, `perform_block_coeff_opt 1`, `use_transform_domain_distortion 0`.
    * **C evaluates exactly ONE candidate and stops**: `tx_type=11 (H_DCT), eob=0,
      rate=1612, dist=6480, rd=966504` — winner, `skip_txfm=1`. It stops because
      `tx_type_search.skip_tx_search && !best_eob` breaks the loop on the FIRST candidate
      (tx_search.c:2352).
    * **The port evaluated five**, in the order `3, 0, 2, 1, 11`, and its first
      (`ADST_ADST`) kept coefficients through the trellis: `eob=3, rate=5802, dist=2960,
      rd=872209` — cheaper than the skip's 966504, so ADST_ADST won and the search never
      terminated early.
    * The whole difference is the ORDER `prune_txk_type` wrote into `txk_map`.
  - **DEFECT 2a — `prune_txk_type`'s est-rd added the tx-type cost to `eob == 0` candidates.**
    At this txb ALL FIVE allowed types quantize to `eob = 0` under the prune's B-quant, so C's
    `av1_cost_coeffs_txb_laplacian` returns `txb_skip_cost[ctx][1] = 1612` for every one
    (txb_rdopt.c:742-744, the early return) and the sort is decided purely by tx-domain
    distortion: `11:6458 < 2:6493 < 0:6501 < 3:6525 < 1:6529` => `txk_map[0] = H_DCT`. C only
    reaches `get_tx_type_cost` inside `warehouse_efficients_txb_laplacian` (txb_rdopt.c:674),
    i.e. on the `eob > 0` path. The port's caller added it unconditionally
    (2807/3103/3011/2440/3309), so the five sorted by tx-type SIGNALLING cost instead:
    `3 < 0 < 2 < 1 < 11` => `txk_map[0] = ADST_ADST`. **Fix:** gate the caller's
    `get_tx_type_cost` on `xq.eob > 0` in `prune_txk_type_intra`
    (`crates/aom-encode/src/tx_search.rs`) — the same gate the main loop's no-trellis arm
    already had. NOTE the DSP-level differential `cost_coeffs_diff.rs` was green throughout:
    it validates `cost_coeffs_txb_laplacian` alone against a shim that puts the tx-type cost
    out of scope, so the defect lived in the CALLER's composition — a playbook §7 "gap between
    two green rows".
  - **DEFECT 2b (exposed once 2a closed) — the SATD trellis-skip did not switch the
    QUANTIZER.** Next divergence: mi(0,6) BLOCK_8X16, MODE_EVAL, `H_PRED / angle_delta 1`,
    TX_8X16, DCT-only mask. Both sides agreed the SATD gate fired (`satdskip=1`, `optb=0`) and
    both produced the SAME forward coefficients (`sse` 503325 identical on both), but the
    dequantized ones differed: `C rate=54833 dist=57169` vs `port rate=63012 dist=48017`.
    `skip_trellis_opt_based_on_satd` does not merely return a flag — it re-runs
    `av1_setup_quant` (tx_search.c:2002-2007) with
    `skip_block_trellis ? (USE_B_QUANT_NO_TRELLIS ? AV1_XFORM_QUANT_B : AV1_XFORM_QUANT_FP)
    : AV1_XFORM_QUANT_FP`, and `USE_B_QUANT_NO_TRELLIS` is 1 (blockd.h:34), so that tx type
    quantizes with **B**. The port carried the block-level `kind` into both arms.
    **Fix:** a per-tx-type `kind_this`, and — the part that matters — a per-tx-type
    `QuantParams` to go with it. `QuantParams::from_plane_rows` BAKES the facade's table
    choice (`quant_fp_QTX`/`round_fp_QTX` vs `quant_QTX`/`round_QTX`), so switching only the
    kind ran B's algorithm on FP's tables and made it WORSE (dist 48017 -> 152297 against C's
    57169). Both `qp_fp` and `qp_b` are now materialised once and selected together with the
    kind.
  - **Byte-inert below speed 4, structurally:** `skip_trellis_opt_based_on_satd` short-circuits
    on `coeff_opt_satd_threshold == UINT_MAX` (tx_search.c:1986), which is every eval stage's
    value at ALLINTRA speeds 0..3, and then `skip_trellis_this == skip_trellis` so `kind_this`
    reproduces the old `kind` exactly. Defect 2a is likewise unreachable below speed 4
    (`prune_tx_type_est_rd` is speed 4/5 only). Unit-locked over speeds 0..9 by
    `satd_trellis_skip_arm_is_allintra_speed4_up` (`speed_features.rs`), which asserts BOTH
    directions — no stage has a finite SATD threshold at speeds 0..3 (inertness) AND at least
    one does at speeds 4..9 (non-vacuity: without the second half the lock would pass on a
    port that never enables the arm).
  - **Verification.** On the repro cell the port's per-txb stream is **4346/4346 groups
    identical to instrumented C** frame-wide, and the encode is a byte match (498 == 498 B).
  - **What it closed** (every one a self-promoting pin that FAILED in the closing direction,
    all re-pinned in this landing):
    * `speed_envelope_stock_map_is_pinned`: `q00_64` cpu-4, `q00_mono64` cpu-4 AND cpu-5,
      `q00_128` cpu-4 — **all three bd8 contexts are now byte-identical at every speed 0..9**,
      so the "fragile band" is a bd10-only statement now.
    * `speed_sensitivity_s1`: `(4, flip0)` and `(5, minp16)` closed — **no RD-search speed
      (0..7) has an open singleton left**; the only survivors are nonrd (speed 8).
    * `combinations_t3_speed4_s0`/`_s1`: both open cpu-4 combination rows closed — **0 open of
      31 and 0 of 32**, so `SPEED_OPEN_COMBINATIONS` has no cpu-4 entry at all. (Re-measured
      AFTER removing the two singletons, since `remap_open_levels` reads that table and the
      executed rows change with it.) The 5 cpu-8 rows are unchanged, as expected — nonrd
      PICKMODE never enters this tx-type search.
    * `encoder_gate_speed6_noise_flatuv_allintra` + `encoder_gate_speed7_noise_flatuv_allintra`
      + `kb11_speed7_noise_cq63_pinned_open`: the KB-10 and KB-11 pinned-open cq63 near-tie
      pairs (mono + 4:2:0) now byte-match. Both gates are full byte-identity asserts now and
      the KB-11 localizer is renamed `kb11_speed7_noise_cq63_byte_matches`.
    * KB-13 (`encoder_gate_real_content_speed1to4_e2e`): **+3 cells, map 47/60 -> 50/60**
      (`quantizer-00 420 64x64@96,64 cpu4 cq32`, `quantizer-00 420 128x128@64,64 cpu4 cq32`
      and `cq63`). Its per-cell `assert!` was restructured to accumulate both directions first
      — it used to stop at the first promotion and hide the rest.
  - **Bite proof (each defect reverted ALONE, everything else in place):**
    * revert 2a (drop the `eob > 0` gate) ->
      `speed_envelope_stock_map_is_pinned` fails with *"q00_128 cpu-used=4: stock encode is
      now `diverge` (pinned `ok`)"* and `encoder_gate_real_content_speed1to4_e2e` fails with
      *"regression: graduated real-content cells must byte-match real aomenc:
      [\"av1-1-b8-00-quantizer-00 420 128x128@64,64 cpu4 cq32\",
      \"... cpu4 cq63\"]"*.
    * revert 2b (carry the block-level kind/qp into both arms) -> the envelope fails on ALL
      FOUR cells (`q00_64` cpu-4, `q00_mono64` cpu-4 + cpu-5, `q00_128` cpu-4), plus
      `encoder_gate_speed6_noise_flatuv_allintra` and `..._speed7_...` with
      *"diverging: [\"64x64 mono cq63\", \"64x64 420 cq63\"]"* and
      `kb11_speed7_noise_cq63_byte_matches`.
    The two reverts fail DIFFERENT cell sets, which is the evidence they are two roots and
    not one. Note the unit lock passes in both reverted states by design — it guards the
    *inertness* claim (which speeds can reach the arm), not the fix; the byte gates guard
    the fix.
  - **Envelope (this box, `--profile test-fast`, both dispatch modes):**
    `-p zenav1-aom-encode -p zenav1-aom-bench` **452 passed / 0 failed / 5 ignored** with
    default dispatch AND with `AOM_FORCE_SCALAR=1`.
  - **ROOT #3 (the QM x speed>=4 quirk this landing named but did not fix) — CONFIRMED BY
    MEASUREMENT AND FIXED 2026-07-31.** `av1_setup_quant` also NULLs `qparam->qmatrix` /
    `iqmatrix` (encodemb.c:367-368) as its last two statements, and
    `skip_trellis_opt_based_on_satd` CALLS it (tx_search.c:2001-2006) whenever it does not
    take its early return (:1988, `skip_trellis || coeff_opt_satd_threshold == UINT_MAX`) —
    including when `skip_block_trellis == 0`. That call sits INSIDE `search_tx_type`'s
    per-tx-type loop, between the `av1_setup_qmatrix` that installs the matrix (:2204-2207)
    and the `av1_quant` that would use it (:2221), so with QM enabled at speed >= 4 C
    quantizes — and measures `dist_block_tx_domain` (:2248/:2265, handed
    `quant_param.qmatrix`) — with NO quant matrix for every tx type of every txb. The
    quantize facades branch on `qm_ptr != NULL && iqm_ptr != NULL` (av1_quantize.c), so NULL
    means the flat kernels. The port kept the matrix attached.
    - **PREDICTED, THEN MEASURED (playbook §4).** New e2e gate
      `aom-bench/tests/kb21_qm_speed4.rs::qm_speed_map_byte_matches` — 3 real-content 4:2:0
      cells (`quantizer-00` 64x64@64,64 and 128x128@64,64, `film_grain-50` 64x64@96,64, all
      cq32) x `--cpu-used` 0..7, QM on via `--enable-qm=1 --qm-min=4 --qm-max=10`
      (`EncodeCell::c_encode_qm`, the `shim_encode_av1_kf_qm` path). **The split landed
      exactly on the predicted boundary: speeds 0-3 12/12 MATCH, speeds 4-7 12/12 MISMATCH**
      (e.g. `q00 128x128 cpu4` port 1722 B vs real 1751 B; `cpu5` 1751 vs 1806). Per-cell
      byte deltas are 1-55 B — playbook §10: that says nothing about mechanism, and here the
      mechanism is a systematic quantizer difference at every txb.
    - **Reachability is exactly the finite-satd rows of `coeff_opt_thresholds`**
      (speed_features.c:88-98 + :2804-2809). ALLINTRA `perform_coeff_opt` is 1/2/3 at speeds
      0/2/3 (:383/:415/:433) — rows whose satd column is `UINT_MAX` in EVERY MODE_EVAL slot —
      5 at speeds 4-5 (:493) and 6 at speeds >= 6 (:555), whose DEFAULT_EVAL / MODE_EVAL satd
      columns are finite (97 / 16). `WINNER_MODE_EVAL` is `UINT_MAX` in every row, so it never
      drops the matrix. Speeds 8-9 are nonrd PICKMODE and never enter this search.
    - **FIX, in two halves — and the second is what makes the first correct.**
      1. `crates/aom-encode/src/tx_search.rs`: the `QUANT_PARAM`-level matrix is now
         `qparam_qm_level_in_search(coeff_opt_satd_threshold, skip_trellis, frame_qm_level)`
         (new pub helper next to `satd_trellis_skip_arm_runs`), i.e. `None` whenever the SATD
         helper runs its body. Both `qp_fp`/`qp_b` and the `dist_qmatrix` the tx-domain arms
         read are built from it.
      2. The TRELLIS must NOT lose its matrix: `av1_optimize_txb` selects its own straight
         off the frame state, `av1_get_iqmatrix(&cpi->common.quant_params, ..)`
         (txb_rdopt.c:344-349), never through a `QUANT_PARAM`. So on this band C quantizes
         flat and then trellises with the real inverse matrix. New
         `crate::xform_quant_optimize_split` takes the quantizer's and the trellis's
         `QuantParams` separately (`xform_quant_optimize` = both the same, unchanged for
         every other caller). **This is the KB-21 root-#2 lesson again on a new axis: the
         two matrix sources are independent in C, and modelling only one of them makes a
         DIFFERENT set of cells wrong** (see the bite proofs).
      Also `prune_txk_type` is correct as-is and was NOT touched: its `av1_setup_quant` runs
      BEFORE its loop (tx_search.c:1334-1335) and the in-loop `av1_setup_qmatrix` (:1348) is
      the last word, so the prune's est-rd genuinely uses the matrix.
    - **Byte-inert below speed 4, structurally**, and the pre-existing speed-0 QM gate
      (`encoder_gate_qm_on_e2e`, 24 cells) is untouched — `satd_trellis_skip_arm_runs` is
      false at speeds 0-3, so `qparam_qm_level == frame_qm_level` and every `qp` is what it
      was. Unit-locked over speeds 0..9 x BOTH QM states by
      `aom-encode/tests/kb21_qm_satd_arm_lock.rs::qparam_qm_is_dropped_exactly_from_allintra
      _speed_4`, which asserts both directions (inertness at 0-3, at-least-one-stage
      reachability at 4-9 so the drop cannot be dead code), that QM-off stays flat everywhere,
      that `skip_trellis == true` keeps the matrix at every speed (the first term of C's early
      return), and that `WINNER_MODE_EVAL` never fires.
    - **Bite proofs (each half reverted ALONE, everything else in place):**
      * revert half 1 (`qparam_qm_level = inp.qm_level`) -> `qm_speed_map_byte_matches` fails
        on the SAME 12 cells with the SAME byte counts as the pre-fix map; the unit lock stays
        green (it guards the derivation, not its consumption).
      * revert half 2 (trellis shares the qparam block) -> the gate fails on a **different,
        smaller set of 7** (`cpu4` x3, `cpu5` x2, `cpu7` x2) — the evidence these are two
        things and not one.
      * stub `qparam_qm_level_in_search` to return `frame_qm_level` -> the unit lock fails
        with *"speed 4 DEFAULT_EVAL: QM must be dropped iff the SATD arm runs, left: Some(8)
        right: None"*.
    - **Verification.** `qm_speed_map_byte_matches` 24/24 byte-exact in BOTH dispatch modes
      (default and `AOM_FORCE_SCALAR=1`); envelope `-p zenav1-aom-encode -p zenav1-aom-bench`
      **455 passed / 0 failed / 6 ignored** in both modes (was 452/0/5 before KB-21 root #2's
      landing plus this landing's 2 new tests + 1 new ignored e2e). No new clippy findings
      (29 pre-existing errors on `main`, 29 after).
    - **Harness addition:** `ToggleKnobs::qm: Option<(qm_min, qm_max)>` (PORT side; the C side
      is `EncodeCell::c_encode_qm`, same pattern as `deltaq_mode2/3` and `enable_intrabc`).
      `port_encode_full` derives `qmatrix_level_{y,u,v}` via `aom_get_qmlevel_allintra` and
      CROSS-CHECKS them against the levels the bootstrap header signalled, so a wiring error
      fails before any byte comparison.
    - **Still open (stated plainly):** only 4:2:0 bd8 cq32 cells were run. bd10/bd12 QM at
      speed >= 4, monochrome, 4:4:4/4:2:2, and the qindex extremes (cq5 / cq63, which move the
      derived QM level) are unmeasured on this axis; so is `--dist-metric=qm-psnr` (tune=IQ /
      SSIMULACRA2), where `use_qm_dist_metric` makes `dist_block_tx_domain`'s NULL matrix
      observable directly rather than only through `av1_quant`.
    - **MEASURED 2026-08-01** (`aom-bench/tests/s4cov_qm_axis.rs`, 18 cells × `--cpu-used` 0..9 ×
      both QM states = 360 encode pairs; record `benchmarks/s4cov_axes_2026-08-01.tsv`). All of
      the above EXCEPT `--dist-metric=qm-psnr` is now swept, and the answer is clean:
      * **bd8 is 80/80 byte-exact in BOTH QM states** across 4:4:4, 4:2:2, monochrome and the
        cq5/cq63 extremes, at every speed 0..9 (the nonrd speeds included — "QM × nonrd" was
        itself unmeasured and is byte-exact);
      * **QM never changes a verdict on any of the 180 rows.** Every high-bit-depth cell agrees
        between QM-on and QM-off, so the QM path adds no divergence anywhere on the extended axis.
      * The only divergences are the QM-OFF controls at bd10/bd12, `--cpu-used` 1..6 — the
        pre-existing pinned `b10_64` band of `speed_envelope_stock_map_is_pinned`. **This sweep
        widens the known shape of that band**: it is not 4:2:0-specific (4:4:4 diverges
        identically), not bd10-specific (bd12 too), **not chroma-borne (MONOCHROME diverges
        identically, which puts the root on the LUMA path)**, and its speed reach is
        qindex-dependent (cq5 reaches 1..6 like cq32; cq63 only reaches cpu6). Speeds 0, 7, 8, 9
        are clean at every bit depth. Pinned as `HBD_OPEN` in `s4cov_qm_axis.rs`.
      * Caveat found while building the cells: the (0,0) 64×64 corner of
        `av1-1-b10-24-monochrome` codes an 18-22 byte all-skip frame at every speed, so a
        byte-match there proves nothing — the hbd mono cells drop the chroma of the textured
        `av1-1-b10-00-quantizer-00` crop instead. Switching to it is what exposed the mono row
        above; the flat crop had reported "mono is clean".
      * `--dist-metric=qm-psnr` × speed >= 4 REMAINS unmeasured: it needs `ref_encode_av1_kf_tune`
        (whose port-side pipeline lives in `encoder_gate_tune_iq_e2e.rs` and hardcodes
        `let speed = 0`), not `ToggleKnobs`, which carries no dist-metric field.
- **PARITY.md §A lists `--cpu-used=4` and `=5` as 64/64 byte-identical** — on the textured
  synthetic grid those rows were established on. These are different contexts, so the §A rows
  are not falsified; what is falsified is reading them as speed-4/5 coverage in general.

### KB-18 — Encoder: SB128 x `--max-partition-size=32` performs a `restore_context` that C skips — FIXED ✅ 2026-07-30
- **Found 2026-07-30** by the size axis of the config-permutation gate (`fc44646`).
- **Root cause — the port asserted what C treats as a condition.**
  `crates/aom-encode/src/partition_pick.rs` carried
  `debug_assert!(bsize <= cfg.max_partition_size || bsize == env.sb_size)` with the comment
  *"always true here"*, then restored unconditionally. It is NOT always true: C's SPLIT-stage
  restore is gated *conditionally* on exactly that predicate —
  `if (bsize <= x->sb_enc.max_partition_size || bsize == cm->seq_params->sb_size)
  av1_restore_context(x, x_ctx, mi_row, mi_col, bsize, av1_num_planes(cm));`
  (`partition_search.c:4645-4646`, with C's own comment naming the two cases: block sizes at
  or under the max-partition cap get a dry-run encode, and the SB-sized block gets the final
  encode). Where the predicate is false C SKIPS the restore and the port performed it. In
  release builds (`debug_assert` compiled out) that was a silent behavioural divergence, not
  a crash.
- **Reachable only at SB128.** At SB64 the window between a 32px cap and the 64px SB size is
  empty, so the predicate cannot be false — which is exactly why 2,617 cells at SB64 never saw
  it. It took adding SB128 size classes to reach.
- **Fix:** the restore is now taken inside `if bsize <= cfg.max_partition_size || bsize ==
  env.sb_size`; the assert and its "always true here" comment are gone.
- **Verified.** `SizeCtx::skip_reason` is now EMPTY (it kept the mechanism, not the entry), so
  every SB128 size context runs `--max-partition-size=32` at full strength, and
  `size_axis_open_divergences_pinned`'s Finding A is promoted from "assert that the port
  panics" to a byte-identity gate: **port 1968 B == C 1968 B** (`S_SB128_128`, mono,
  cq63, `--sb-size=128 --max-partition-size=32`). Bite proof: reverting to the unconditional
  restore makes that cell **port 1963 B vs C 1968 B DIVERGE**, and the gate fails with
  *"KB-18 REGRESSED: --sb-size=128 x --max-partition-size=32 is no longer byte-identical to
  real aomenc"*. Envelope unmoved (see the KB-17 entry's counts — the same run covers all
  three fixes).

### KB-19 — Encoder: `default_min_partition_size`'s >=2160p arm was UNMODELLED — FIXED ✅ 2026-07-30
- **Found 2026-07-30** by the same size-axis cross-check, in the direction that matters: it
  compared libaom's framesize-dependent derivations against the port's BOTH ways, rather than
  only checking that ported thresholds were right.
- libaom sets `default_min_partition_size = BLOCK_8X8` at >=2160p — `if (is_4k_or_larger)
  sf->part_sf.default_min_partition_size = BLOCK_8X8;` (`speed_features.c:187-189`, where
  `is_4k_or_larger = AOMMIN(cm->width, cm->height) >= 2160`, :172). It is inside
  `set_allintra_speed_feature_framesize_dependent` and unconditional on speed, so it also
  applies below speed 7 (which sets the same value framesize-independently at :570). The port
  left the field `BLOCK_4X4` unconditionally — no framesize arm.
- **Every other framesize-dependent derivation was verified either correctly ported or dead on
  the allintra-intra envelope** (auto_max_partition / ml_* breakouts / use_downsampled_sad /
  partition_search_breakout are inter-only or speed-gated; the full table is in
  `docs/CONFIG_PERMUTATION_DESIGN_2026-07-30.md`). This was the one real omission.
- **Fix, in two halves** (the second is why the first bites):
  1. `SpeedFeatures::apply_allintra_framesize_dependent(w, h)` — a new method holding the
     modelled arms of `set_allintra_speed_feature_framesize_dependent`, currently just the
     `is_4k_or_larger` one. `set_allintra` stays framesize-blind by design; the framesize arms
     are applied afterwards from the frame's real dimensions, which is the shape the port
     already used for `prune_tx_type_using_stats` (the `is_480p_or_larger` arm). Called from
     `aom-bench/src/lib.rs`'s `port_encode_full` right after `set_allintra`.
  2. `ToggleKnobs::min_partition_bsize` now takes `sf.default_min_partition_size` and AOMMAXes
     with it, matching `set_max_min_partition_size` exactly (`partition_strategy.h:224-226`:
     `min_partition_size = AOMMIN(AOMMAX(sf->part_sf.default_min_partition_size,
     dim_to_size(oxcf px)), sb_size)`). It previously dropped the sf term with a comment
     claiming the field is `BLOCK_4X4` "at every allintra speed" — false since speed 7 (:570),
     and false at >=2160p at any speed. The `max` is an identity on the whole gated envelope
     (speed 0..6 sub-2160p), which is why this half is byte-inert there.
- **Verified — and be precise about what is and is not proven.**
  - DERIVATION, default tier: `aom-encode`'s `framesize_dependent_min_partition_size_4k_arm`
    unit test (speeds 0..9 x {64², 1920x1080, 4096x2159, 2159x4096} must NOT take the arm x
    {2160², 3840x2160, 2160x3840, 7680x4320} must, plus a whole-struct equality check that the
    arm moves no other field). Bite proof: stubbing the arm body out fails it with *"speed 0
    2160x2160 must take the 4k arm (BLOCK_8X8)"*.
  - E2E, `--ignored` tier: `aom-bench/tests/kb19_min_partition_4k.rs::
    min_partition_4k_arm_e2e_byte_match` (renamed from `..._pinned` on 2026-07-31 when KB-22
    closed and it became a hard byte gate) — mirror-tiled `av1-1-b8-00-quantizer-00` at
    2160x2160 bd8 4:2:0 cq32 speed-0. **Measured A/B, same binary, only the arm toggled:**

    | port build | port bytes | C bytes | delta |
    |---|---|---|---|
    | WITHOUT the arm | 440,347 | 431,724 | **+8,623 (+2.00%)** |
    | WITH it (shipped) | 431,574 | 431,724 | **-150 (-0.035%)** |

    The arm closes 98.3% of the byte gap, so it is heavily load-bearing at this frame size —
    NOT a paper fix. Wall on the reference box: C ~26 s, port ~195 s.
  - **The residual 150 bytes were a SECOND unmodelled arm, not a near-tie — closed 2026-07-31,
    see KB-22 below.** The cell is now byte-identical and the e2e test is a hard `assert_eq!`
    byte gate (it was a self-promoting pin until then, and it is what fired to report the
    close).

### KB-22 — Encoder: `av1_set_speed_features_qindex_dependent`'s speed-0 >=720p arm was UNMODELLED — FIXED ✅ 2026-07-31 (the 2160p cell is byte-exact)
- **The 2026-07-30 framing was WRONG and is corrected here.** The entry reasoned that 150 bytes
  over ~1,156 superblocks (0.035% of payload) "argues near-tie, not a missing tool" — while
  stating plainly that this was not established. It is now established, and it was **an
  unmodelled arm**, KB-19-shaped. Scale arguments about byte deltas are not evidence about
  mechanism; the localization is.
- **Localization (decode-both, the KB-6 recipe — `decode_diff_multisb.rs` retargeted).** Encode
  the cell with real aomenc and with the port, splice the port's frame-OBU payload back into the
  reference stream, decode both with the (bit-exact vs C) decoder, replay both partition trees:
  **the FIRST divergence is node 1 — SB(0,0)'s first `BLOCK_32X32` node at mi(0,0): real picks
  `PARTITION_VERT_B`, the port picks `PARTITION_SPLIT`.** First divergent recon pixel is luma
  (0,0) (real 81, port 89). Whole-frame decoded shape: real 33,371 tree nodes / 40,403 blocks vs
  port 29,105 / 38,398. A divergence at the very first 32x32 node of the very first superblock,
  with a frame-wide shape difference, is NOT a late near-tie — it is a systematic
  search-configuration difference.
- **ROOT CAUSE: the port never modelled `av1_set_speed_features_qindex_dependent`
  (speed_features.c:2873) at all** — C's THIRD speed-feature derivation pass, run after both
  `set_allintra` cascades (`encoder.c:3114`, `encoder_utils.c:1280`). Its `speed == 0` block
  (:2904-2937) has an arm gated on `is_720p_or_larger && base_qindex <= 128` (:2914) that sets:
  - `rd_sf.perform_coeff_opt = 2 + is_1080p_or_larger` (:2915) + the `coeff_opt_thresholds`
    memcpy (:2916) → at 2160x2160 that is **1 → 3**, moving the DEFAULT_EVAL trellis dist gate
    from row 1's **3200 to row 3's 864** (`coeff_opt_thresholds`, speed_features.c:88-98);
  - `tx_sf.intra_tx_size_search_init_depth_rect = 1` (:2923) → **0 → 1**, raising the
    rectangular intra tx-size search floor in `get_search_init_depth_intra`;
  - `tx_sf.model_based_prune_tx_search_level = 0` (:2924) → byte-INERT here: its only C consumer
    is `av1_pick_recursive_tx_size_type_yrd` (tx_search.c:3563), which asserts `is_inter_block`.

  The cell sits exactly on the qindex boundary: cq32 → `base_qindex` 128 → `<= 128` holds.
- **Why no green test could have caught it:** the arm is unreachable below 720p, and every
  pre-existing encoder gate encodes at most 640x640 (the config-permutation size axis tops out
  at 640; the KB-6 real-content map is 196x196). Same structural blind spot as KB-19 — this is
  the playbook §7 thesis (bugs live in the gap between two individually-green rows), on the
  frame-size axis.
- **FIX:** `SpeedFeatures::apply_allintra_qindex_dependent(width, height, base_qindex, speed)`
  (`crates/aom-encode/src/speed_features.rs`), modelling the `speed == 0` block — both the
  >=720p arm and the sub-720p `base_qindex <= 70` arm (C's `boosted` term is
  `frame_is_kf_gf_arf`, true on every KEY frame, the only frame type this port authors). Called
  from `aom-bench/src/lib.rs`'s `port_encode_full` immediately after
  `apply_allintra_framesize_dependent`, matching C's pass order. The method's doc comment lists
  every field of that function it deliberately does NOT model, with the reason.
- **Verified.**
  - DERIVATION, default tier: `aom-encode`'s `qindex_dependent_speed0_hd_arm` unit test — the
    arm fires only at speed 0, only at `min(w,h) >= 720`, only at `base_qindex <= 128`
    (719/720 and 128/129 boundary pairs both directions), the 1080p step in `perform_coeff_opt`,
    the sub-720p threshold of 70, a whole-struct equality check that nothing else moves, and the
    derived `TxTypeSearchPolicy` (3200 → 864 dist threshold, rect init depth 1). Bite proof:
    stubbing the arm body (`if false && ...`) fails it with *"speed 0 1280x720 q128 must take
    the >=720p qindex arm, left: (1, 0) right: (2, 1)"*; the rest of the `speed_features` suite
    stays green (13/14 → the asymmetry playbook §1 asks for).
  - E2E, `--ignored` tier: `aom-bench/tests/kb19_min_partition_4k.rs::
    min_partition_4k_arm_e2e_byte_match`, now a hard `assert_eq!` byte gate.
    **Measured A/B on the same box, same cell (C reference 431,724 B, ~26 s):**

    | port build | port bytes | delta vs C | port wall |
    |---|---|---|---|
    | KB-19 arm only (2026-07-30) | 431,574 | -150 (-0.035%) | 125 s |
    | + the KB-22 qindex arm | 431,724 | **0 — BYTE IDENTICAL** | 83 s |

    The 34% wall drop is the arm doing real work (a higher `perform_coeff_opt` row and a deeper
    rectangular tx-size init depth both cut search). Confirmed in BOTH dispatch modes (default
    and `AOM_FORCE_SCALAR=1`). Record: `benchmarks/kb22_qindex_arm_2026-07-31.tsv`.
  - The self-promoting pin worked as designed: the old `assert_ne!` fired with *"KB-22 HAS
    CLOSED"* rather than letting the fix pass unnoticed (playbook §5).
- **Both follow-ups CLOSED 2026-07-31** (record: `benchmarks/kb23_partial_sb_2026-07-31.tsv`,
  gates: `crates/aom-bench/tests/kb22_hd_arms.rs`):
  1. **The 720p..2159p isolation cell — DONE, byte-identical.**
     `kb22_hd_arms::qindex_arm_720p_isolation_e2e_byte_match` encodes mirror-tiled
     `av1-1-b8-00-quantizer-00` at **1280x720 cq32 (base_qindex 128) speed 0**: port
     **85,441 B == C 85,441 B** (C 8.6 s, port 25.6 s). The cell is live for the KB-22 arm
     ALONE — `min(w,h) >= 720` fires it, `< 2160` keeps KB-19's `is_4k_or_larger` arm dead,
     and `< 1080` makes `perform_coeff_opt` resolve **2**, a `coeff_opt_thresholds` row
     (DEFAULT_EVAL dist gate 1600) no other cell in the suite reaches (the 2160p cell resolves
     3 / gate 864; every sub-720p cell resolves 1 / gate 3200). Bite proof: stubbing the arm
     body to `if false && is_720p_or_larger` fails this cell at **+226 B (85,667 vs 85,441)**,
     and the asymmetry playbook §1 asks for holds — **61 of the 62 `aom-encode` lib tests stay
     green**, the one failure being that arm's own derivation test
     (`qindex_dependent_speed0_hd_arm`, *"left: (1, 0) right: (2, 1)"*).
  2. **The `min_lr_unit_size` / `max_lr_unit_size` prediction — MEASURED AND REFUTED. The
     "still unmodelled" claim above was WRONG.** They carry no field in
     `aom_encode::SpeedFeatures` because the port's LR search takes its whole `LrSearchSf`
     from its caller, and that caller (`aom_bench::lr_search_sf_allintra` / `..._good`)
     already transcribes speed_features.c:3080-3108 **including both framesize arms**.
     Measured at 1280x720 speed 1 with `--enable-restoration=1`, `base_qindex` on both sides
     of the `<= 96` threshold (`kb22_hd_arms::lr_unit_size_hd_speed1_e2e`): the port codes
     exactly C's restoration unit size — cq24/q96 → `[128,128,128]` both sides, cq25/q100 →
     `[256,256,256]` both sides — and the coded size IS the derived bound, so the fields
     reach the bitstream correctly. Two structural notes that make the prediction wrong on
     independent counts: for ALLINTRA at `speed >= 1` the `speed >= 1` framesize block
     (:3085-3093) is entirely **overwritten** by the `ALLINTRA && speed >= 1` block
     (:3095-3107), so the only surviving framesize term is `is_1440p_or_larger` — **720p is
     not a size at which the allintra bounds move at all**; and the whole derivation is now
     locked over speeds 0..9 × the 719/720, 1439/1440 and 96/97 boundaries in both directions
     (allintra AND good) by `kb22_hd_arms::lr_unit_size_bounds_track_c`. The residual byte
     delta those cells showed was **identical with restoration ON and OFF** (-3/-3 at cq24,
     +31/+31 at cq25), i.e. entirely in the base encode — that is **KB-23**, and with it fixed
     all four pairs are byte-identical.
  3. **`config_permutations.rs`'s size-axis headline** ("at SPEED 0 every framesize-dependent
     SPEED FEATURE below 2160p is either inert on the all-intra KEY path or gated on speed >= 1")
     surveyed `set_allintra_speed_feature_framesize_dependent` only. It is true of that gate's
     own cells (all <= 640x640), but it does NOT cover `av1_set_speed_features_qindex_dependent`,
     whose speed-0 arm is live from 720p up. Read it as scoped to sub-720p. **Additional
     scoping (KB-23):** at `speed >= 1` the size axis is not only a geometry axis — whether a
     frame is an exact multiple of the 64-px superblock changes which speed features can fire
     inside its edge superblocks. Every size that axis uses (64, 128, ..., 640) is SB-exact,
     which is why it never saw KB-23.

### KB-23 — Encoder: the intra-CNN partition prune fired inside FRAME-EDGE superblocks (C's `cnn_output_valid` latch was unmodelled) — FIXED ✅ 2026-07-31
- **Found 2026-07-31 while testing KB-22's loop-restoration prediction**, from the control
  rather than the subject: the `--enable-restoration=1` cell at 1280x720 speed 1 diverged, but
  so did the `--enable-restoration=0` control at the identical cell, **by exactly the same byte
  delta** (-3/-3 at cq24, +31/+31 at cq25). A divergence that is unchanged by turning the
  feature under test off is not that feature's.
- **Localization — the size axis, with SB-exact and partial-SB sizes INTERLEAVED so the two
  candidate explanations were separable by the result pattern** rather than by intuition
  (`kb22_hd_arms::kb23_partial_sb_size_and_speed_axis`, cq24 / speed 1 / mirror-tiled
  `av1-1-b8-00-quantizer-00`):

  | size | multiple of 64? | verdict |
  |---|---|---|
  | 132², 196², 480², 720², 1280x720 | no | **DIVERGE (5/5)** |
  | 192², 256², 448², 512², 640², 704² | yes | MATCH (0/6) |

  The split is total, and it is **not** a framesize bucket: 480x480 diverges while 512/640/704
  all match, so no `is_480p_or_larger` arm can explain it. 132x132 diverges at an IDENTICAL
  total length (2,407 B both) — a different tile, not a size effect. The speed sub-sweep then
  showed the same partial-SB sizes are byte-exact at **speed 0** and divergent at 1/2/3, i.e.
  the frame-EDGE path at `speed >= 1`.
- **ROOT CAUSE: `cnn_output_valid`.** C runs the intra-CNN partition prune through a cached
  output, not per block. `intra_mode_cnn_partition` (partition_strategy.c:142) COMPUTES the CNN
  only when `bsize == BLOCK_64X64 && !part_info->cnn_output_valid` (:160-224) and every smaller
  node returns at `if (!part_info->cnn_output_valid) return;` (:227);
  `init_partition_search_state_params` INVALIDATES it at every BLOCK_64X64 node
  (partition_search.c:3340-3343). The compute is additionally gated on the block being
  whole-in-frame (`av1_is_whole_blk_in_frame`, partition_strategy.c:1784). So in a superblock
  whose 64x64 root is NOT whole-in-frame, **C computes nothing and prunes NOTHING anywhere
  inside it** — including the 32x32/16x16/8x8 sub-blocks that ARE whole-in-frame. The port had
  only the per-block whole-in-frame gate (`partition_pick.rs`), so inside every frame-edge
  superblock it CNN-pruned where C does not. Live from speed 1 up because
  `intra_cnn_based_part_prune_level` is 0 at speed 0 (speed_features.c:387-388), which is why
  KB-6's speed-0 partial-SB series was unaffected.
- **FIX:** `cnn_root_whole_in_frame` in `partition_pick.rs` — the containing 64x64
  (`(mi/16)*16`) must be whole-in-frame before the CNN block runs. Correct under SB64 and
  SB128 alike because C's reset is per-64x64, not per-superblock; byte-inert on SB-exact
  frames, where the containing-64 test is implied by the per-block one.
  **SUPERSEDED 2026-08-02 by KB-PERF-1**: that predicate is now DELETED, because the real
  `cnn_output_valid` latch it was standing in for is implemented, and under a real latch
  the containing-64 condition is emergent exactly as it is in C (a frame-edge 64x64 never
  reaches the compute branch, so nothing inside it prunes). KB-23's RESULT is unchanged and
  its whole grid still gates it — see KB-PERF-1's "RESULT" for the measurement.
- **Verified — and note which halves are asymmetric.**
  - The size axis goes **5/5 divergent → 0/6**, while the SB-exact sizes stay 0/6 both ways.
  - The speed sweep goes 6/8 divergent → 0/8 over speeds 0..3 on the partial-SB sizes.
  - **The speed-4 deltas are BYTE-IDENTICAL pre and post** (-1 at 192², -25 at 256²; all four
    sizes stayed divergent). That is the evidence the fix does not reach into KB-21's cpu-4/5
    band. (Rebasing onto **KB-21 root #3, `8a0faa7`** then took speed 4 to 0/4 divergent on
    this grid — that is KB-21's result, not KB-23's, and the gate deliberately does not assert
    that band from here.)
  - `encoder_gate_real_content_speed1to4_e2e`'s self-promoting pin fired and **6 cells
    graduated on KB-23 alone**: `av1-1-b8-01-size-196x196 420 cpu{1,2,3} cq{12,32}` (196x196
    is 3.0625 SB — a partial-SB frame). That took the 196² block 4/12 → 10/12 and the gate
    47/60 → 53/60. They had been attributed to KB-13's interior AB/SPLIT near-tie class; that
    attribution was wrong — the whole cpu1-3 block closed at once. **Rebasing onto KB-21 root
    #3 (`8a0faa7`) then closed the last two, cpu4 cq{12,32}** — measured still-divergent with
    KB-23 alone, so that pair is the COMBINATION's, not KB-23's. The 196² block is now 12/12
    and the gate 58/60; the two remaining divergences (`quantizer-00 128x128@64,64 cpu3 cq63`,
    `film_grain-50 64x64@96,64 cpu3 cq63`) are SB-EXACT crops and therefore not KB-23's shape.
  - Full `aom-encode` + `aom-bench` suites green in BOTH dispatch modes (default and
    `AOM_FORCE_SCALAR=1`).
  - Bite proof: reverting `cnn_root_whole_in_frame` restores exactly the pre-fix table above.
- **Why no green test could have caught it:** the config-permutation speed axis runs 64x64 and
  128x128 only (`benchmarks/config_perm_speed_axis_2026-07-30.tsv`) — **both exact multiples of
  64**; the KB-6 real-content partial-SB map (196x196) is speed 0, where the CNN level is 0;
  and the one place the two DID cross — 196² at cpu1-4 — was sitting in a pinned near-tie list
  under a different root's name. Playbook §7 thesis again (bugs live in the gap between two
  individually-green rows), on the (frame-alignment × speed) crossing.
- **Record:** `benchmarks/kb23_partial_sb_2026-07-31.tsv` (three states kept distinct: base
  `999d295`, +KB-23, and rebased onto KB-21 root #3). **Residual, stated plainly:** none on this
  grid after the rebase — but the grid is one content source at cq24, bd8 4:2:0, SB64, speeds
  0..4. KB-21 root #2 remains open on its own terms, and the partial-SB × speed crossing has
  not been swept at other bit depths, subsamplings, or SB128.
- **Residual RESOLVED 2026-08-01** by `aom-bench/tests/s4cov_partial_sb_axis.rs`: the crossing is
  now swept at 4:4:4 (32/32), 4:2:2 (32/32), monochrome (28/32 — its 4 failures are speed-0 and
  belong to KB-27, not to this axis), bd10 (8/8 at the speeds where bd10 is readable) and
  **SB128 (48/48)**. Getting there took two fixes, KB-24 and KB-25. KB-24 in particular is this
  entry's own thesis one level up: KB-23's fix keyed `cnn_output_valid` off the containing 64×64
  correctly, but the SIBLING piece of the same C statement — `quad_tree_idx`, reset by the same
  two lines of `init_partition_search_state_params` — was still keyed off the superblock.
- **The 250×250 row: SETTLED 2026-08-02 (KB-28), and it was HALF a different thing.** KB-28's
  entry claimed 250×250 "names the same gap"; the honest split is:
  * the row exercises the intra-CNN **window** (`extract_intra_cnn_window`), which is **inert**
    whether it clamps to the crop or to the mi extent. C does not clamp at all — it reads
    `x->plane[0].src.buf - stride - 1` out of the border-extended source
    (partition_strategy.c:205-220) where everything past the crop is the replicated edge pixel,
    so both clamps produce the identical 65×65 window. Measured both ways: 250×250 was
    byte-identical before KB-28's change and after it
    (`kb28_crop_dims::cnn_window_clamp_is_replication_inert`, and the row here). It is a control,
    not a witness.
  * the row **cannot reach** the crop-dependent CNN consumer — the res-tier threshold select
    (partition_strategy.c:311-312) — because `min(250,250)` and `min(256,256)` are both below
    480. That consumer WAS KB-28's root, and the rows that reach it are 474×480 (mi 480×480) and
    714×720 (mi 720×720), now gated at cpu 1..6 in
    `kb28_crop_dims::rd_band_min_dim_tiers_byte_match` (0/12 with the predicates reverted).
  So: same underlying gap (the walk had no crop dims), **different effect** — one arm of it was
  observable and one was provably not. Recorded per playbook §1 / the KB-24 precedent for saying
  so rather than claiming two fixes.

### KB-24 — Encoder: the intra-CNN `quad_tree_idx` was anchored at the SUPERBLOCK, not at the 64×64 — `--sb-size=128` PANICKED at every RD speed 1..6 — FIXED ✅ 2026-08-01
- **Found 2026-08-01** by extending KB-23's (partial-SB × speed) crossing to SB128, which its own
  residual note named as unmeasured. Symptom: `index out of bounds: the len is 4 but the index
  is 4` in `crates/aom-encode/src/cnn_partition/decision.rs:146` —
  `quad_to_linear_1[quad_tree_idx - 1]` at a BLOCK_32X32 node. **36 of 48 SB128 cells panicked**
  (every size × every speed 1..6); speed 0 was clean because `intra_cnn_based_part_prune_level`
  is 0 there (`speed_features.c:387-388`), and speed 7 because the VAR_BASED walk never enters
  `rd_pick_partition_real`. Frame alignment was irrelevant — 128² and 256², both exact multiples
  of 128, panicked exactly like 132²/192²/196²/320².
- **ROOT CAUSE.** C maintains `x->part_search_info.quad_tree_idx` **per BLOCK_64X64**, in two
  places that must be read together:
  * `init_partition_search_state_params` resets it (with `cnn_output_valid`) at every
    `bsize == BLOCK_64X64` node — `partition_search.c:3339-3343`;
  * the SPLIT recursion advances it to `4*idx_parent + idx + 1` and restores it afterwards
    **only when `bsize <= BLOCK_64X64`** — `partition_search.c:4571-4575` / `:4590-4592`.
  Together those make the index a position WITHIN one 64×64, which is what
  `intra_mode_cnn_partition`'s per-bsize feature selection indexes with (`quad_to_linear_1[4]` /
  `_2[16]` / `_3[64]`, `partition_strategy.c:268/283/298`). The port seeded 0 at the SUPERBLOCK
  root (`pack.rs:1846`, comment: *"quad_tree_idx: 0 at the SB (64×64) root"* — true under SB64,
  false under SB128) and advanced it unconditionally. Under `--sb-size=128` the 64×64 children of
  the 128 root therefore carried 1..4 instead of 0, and their 32×32 children 5..20, walking a
  4-entry table off its end.
- **FIX** (`partition_pick.rs`, `rd_pick_partition_real`): re-anchor `quad_tree_idx` to 0 at
  entry when `bsize == BLOCK_64X64`, and gate the SPLIT-child advance on `bsize <= BLOCK_64X64`.
  Byte-inert under SB64 (the root already arrives with 0 and no descendant is BLOCK_64X64; every
  node in the recursion is `<= 64X64` so the guard is always true).
- **Result:** `s4cov_partial_sb_axis::sb128_partial_sb_speed_axis_byte_matches` goes **11/48
  byte-exact with 37 panics → 48/48 byte-exact, 0 panics** (the 37th panic was KB-25's).
  Every one of the 36 previously-panicking cells now BYTE-MATCHES real
  `aomenc --sb-size=128` — the fix did not merely stop the crash.
- **Bite proof, and an honest note on it.** Reverting BOTH halves restores 36 panics on the same
  SB128 cells while `mono`/`444`/`422`/`bd10` stay exactly as they were (playbook §1 asymmetry).
  But reverting each half ALONE leaves the grid at **48/48, 0 panics** in both directions — on
  this grid the two halves are two spellings of one fix, not two roots, because the only >64×64
  node an SB128 frame has is the 128 root itself. Both are kept because both are in C and the
  pair is what makes the invariant ("the index is a position inside a 64×64") true by
  construction rather than by the shape of the current grid; the honest claim is one root, not
  two. Playbook §1's warning about identical cell sets is exactly this case, investigated.
- **Why no green test could have caught it:** `sb128_e2e.rs` runs SB128 at **speed 0 only**
  (5 tests, all `speed = 0`), where the CNN prune does not exist; the config-permutation speed
  axis runs SB64 at 64×64/128×128. Playbook §7, on the (superblock size × speed) crossing.

### KB-25 — Encoder: the speed-7 VAR_BASED walk PANICKED on a frame-edge single-strip rect — FIXED ✅ 2026-08-01
- **Found 2026-08-01** on the same sweep: `not implemented: frame-edge single-strip VERT at
  (20,48) bsize 6: out of the interior-envelope SbTree rect representation`
  (`partition_pick.rs`, `rd_use_partition_real`). Fires at 196×196 cq24 `--cpu-used=7` on
  monochrome, 4:4:4, bd10 and `--sb-size=128`, and at 132×132 cpu7 on bd10 — i.e. wherever the
  speed-7 variance tree happens to pick PARTITION_VERT on a block whose right half is out of
  frame. The bd8 4:2:0 grid KB-23 was closed on never picked one.
- **The `unimplemented!()`'s stated reason was wrong.** It read *"the SbTree Horz/Vert variants
  carry both winners ... an edge single-strip rect is representable only once that envelope
  lifts"*. In fact **all four consumers of slot 1 already gate it on the identical frame
  predicate** the constructor was refusing on: `encode_sb.rs::encode_sb_dry`
  (`if mi_row + hbs < env.mi_rows` / `if mi_col + hbs < env.mi_cols`), `pack.rs::pack_sb_tree`,
  `partition_pick.rs::stamp_grid_from_tree`, `lf_search.rs::stamp_tree_lf`. The representation
  supported the shape; only the constructor did not build it. This is playbook §9 in a new
  costume — a comment asserting a limitation, checked against the code and found stale.
- **FIX:** build `SbTree::Horz`/`Vert` with sub 0 and a **poisoned** clone of it
  (`bsize = usize::MAX`) in slot 1. The poison is what makes the unreadability enforced instead
  of argued: both `encode_sb_dry` and `pack_sb_tree` run `debug_assert_eq!(s1.bsize, subsize)`
  before touching slot 1, so a future consumer that drops its frame gate fails immediately under
  `--profile test-fast` (CI's profile, `debug-assertions` on). The `rate == INT_MAX` half of the
  original guard is a genuinely different case (sub 1 IS in frame there, so slot 1 would be
  read) and is kept as a hard `assert!` rather than folded in.
- **Bite proof:** restoring the `unimplemented!()` fails **a different cell set from KB-24's** —
  `bd10 132² cpu7`, `mono 196² cpu7`, `444 196² cpu7`, `sb128 196² cpu7`, across four different
  tests — while KB-24's 36 SB128 speed-1..6 cells stay green. Different cell sets, different
  roots (playbook §1).

### KB-26 — Encoder: LARGE FRAMES (`min(w,h) >= 480`) diverge at `--cpu-used >= 4` — FIXED ✅ 2026-08-01 (framesize-derived sf dropped by the winner-mode stage derivation)
- **ROOT (one line, one root).** `partition_pick.rs`'s speed>=4 winner-mode two-pass
  (`wm_parts`) builds its MODE_EVAL and WINNER_MODE_EVAL tx policies from a **fresh
  `SpeedFeatures::set_allintra`**, and that constructor is **framesize-BLIND by design** (its own
  doc says so; the framesize arms are applied afterwards from the frame's real dimensions). So
  `tx_sf.tx_type_search.prune_tx_type_using_stats` — which is set ONLY by
  `set_allintra_speed_feature_framesize_dependent` (`is_480p_or_larger`,
  `speed_features.c:261-263` / `:299-301`) — arrived as **0** in both stage policies on every
  frame. C reads it straight off `cpi->sf` inside `get_tx_mask` (`tx_search.c:1887`): one
  frame-level value, identical in all three mode-eval stages. Result: the ENTIRE luma tx search
  ran with the stats prune off on every >=480p frame at `--cpu-used >= 4`, while speeds 0..3 —
  which use the caller's already-resolved policy directly and never enter `wm_parts` — kept it
  on. That is exactly the measured "load-bearing at 2-3, provably INERT at 4-5" split below.
- **FIX:** `TxTypeSearchPolicy::carry_frame_level_tx_sf` (`aom-encode/src/tx_search.rs`) — copies
  the frame-level, stage-INDEPENDENT tx-search inputs (`oxcf.txfm_cfg.enable_flip_idtx`,
  `oxcf.txfm_cfg.use_intra_dct_only`, and now `sf.prune_tx_type_using_stats`) from the resolved
  policy onto each derived stage policy. `partition_pick`'s `wm_parts` calls it in place of the
  two hand-copied oxcf lines it already had.
- **How it was found (playbook §10, "diagnose to the decision"):** temporary counters around
  `get_tx_mask_intra`'s three arms on the 512² cq24 cpu4 cell. The named suspicion — that the
  multi-type arm was not reached at speed>=4 — was **wrong and the counter said so immediately**:
  the arm is reached **140,154 times**, but the stats-prune body inside it ran **0 times**. That
  moved the question from "which arm" to "why is the sf zero here", i.e. from the kernel to its
  caller — playbook §12's shape exactly (the `tx_mask_diff` differential sweeps
  `prune_tx_type_using_stats ∈ {0,1,2}` and was green throughout; it licenses the kernel and
  nothing else). Post-fix the same counters read 140,516 stats-prune runs, changing the mask on
  1,174 of them.
- **MEASURED (`benchmarks/kb26_large_frame_speed4_2026-08-01.tsv`):**
  `hd_speed_axis_byte_matches` **13 open rows → 0**; the gate is now **26/28 byte-exact**, the
  only two non-matches being KB-28's speed-7 1280×720 panics (KB-28 closed 2026-08-02 and it is
  now **28/28** with `LARGE_FRAME_OPEN` empty). `large_frame_speed4_size_ladder`
  **7/7 byte-exact** (was 512²/576²/640² divergent at −28/+29/−33 B).
  `tx_stats_prune_ab_across_the_480p_boundary`: the prune is now load-bearing at **4/4** of
  speeds 2,3,4,5 on 512² and still inert at 448² — every prune-on cell byte-matches real aomenc.
- **Gates (all three re-pinned; two PROMOTED from open-divergence probes to hard byte-match
  gates):** `LARGE_FRAME_OPEN` held only the two KB-28 panics (empty since 2026-08-02) and the
  clean-speed assertion widened from 1..3 to 1..6 (to 1..7 once KB-28 closed);
  the size ladder asserts 7/7 byte-exact with a reach assertion that
  it straddles 480; the A/B asserts `(load_bearing_at_4_5, inert_at_4_5) == (2, 0)` — i.e. it
  fails the moment the framesize-derived sf goes dead again. Unit lock:
  `tx_search::kb26_frame_level_sf_tests` sweeps **`--cpu-used` 0..9 × {448², 512²}** and asserts
  in BOTH directions (sub-480p must stay 0 at every speed; >=480p must reach 1 at speed 2-3 and
  2 at speed >= 4, in BOTH stages), plus a lock that `set_allintra` really is framesize-blind so
  the carry cannot silently become redundant.
- **Bite proof (playbook §1, asymmetric form).** Removing the single carry line and nothing else:
  `kb26_frame_level_sf_tests` 2/2 FAIL, `large_frame_speed4_size_ladder` FAILS reproducing the
  original deltas exactly (512² −28, 576² +29, 640² −33, with 256²..448² still clean), and
  `tx_stats_prune_ab_across_the_480p_boundary` FAILS with *"KB-26 REGRESSED"* (inert 2/2) — while
  the pre-existing suites stay green in that same state. ONE root, ONE change, so there is no
  per-root revert split to report.
- **Regression envelope, same box** (aarch64-apple-darwin, `--profile test-fast`,
  `AOM_CONFORMANCE_DIR` provisioned): `-p zenav1-aom-encode --no-fail-fast` **285 passed /
  0 failed**; `-p zenav1-aom-bench --no-fail-fast` **174 passed / 0 failed / 17 ignored**;
  `AOM_FORCE_SCALAR=1` reproduces both KB-26 e2e gates identically; `cargo check --target
  x86_64-apple-darwin -p zenav1-aom-encode` clean. The fix is inert outside
  (`allintra && min(w,h) >= 480 && speed >= 4`), and no other ignored gate has a cell in that
  class — every one of them is sub-480p or speed <= 3.
- **The general lesson, worth more than the byte count:** `SpeedFeatures::set_allintra` models
  only C's framesize-INdependent cascade. Anything derived from the frame's dimensions or qindex
  (`apply_allintra_framesize_dependent`, `apply_allintra_qindex_dependent`, and the harness's own
  `prune_tx_type_using_stats` wiring) lives OUTSIDE it — so **any code path that re-derives an sf
  from `set_allintra` silently resets every framesize/qindex-resolved field to its default.**
  `wm_parts` was the only such path reaching the tx policies; the palette derivation at
  `partition_pick.rs:1048` re-derives too but reads only framesize-independent fields. Audit that
  invariant before adding another internal `set_allintra` call.
- **THE AUDIT IS DONE — 2026-08-01, independently, from upstream source.** Enumerating every
  `SpeedFeatures::set_allintra(` call site: production ENCODER code has exactly **two**
  (`partition_pick.rs:975` = `wm_parts`, fixed above; `partition_pick.rs:1050` = the palette
  derivation). Everything else is `speed_features.rs`'s own unit tests, `tx_search.rs`'s tests, or
  the `aom-bench` harness (`lib.rs:1450`, `lib.rs:2332`, `config_perm.rs:1527`) which resolves the
  later passes itself.
  The palette site reads exactly two fields, and **every assignment to either one in upstream is
  inside a `*_framesize_independent` function** — verified, not taken on report:
  `prune_palette_search_level` at `speed_features.c:402`/`:456`
  (`set_allintra_speed_features_framesize_independent`), `:1204`/`:1329`
  (`set_good_..._framesize_independent`), `:1956`/`:2431` (`set_rt_..._framesize_independent`);
  `prune_luma_palette_size_search_level` at `:362`/`:403` (allintra fs-independent), `:1957`/`:2432`
  (rt fs-independent). Neither field is touched by
  `set_allintra_speed_feature_framesize_dependent` or `av1_set_speed_features_qindex_dependent`.
  **So the palette site is safe, and KB-26's class has no second instance in the current tree.**
  The invariant still binds any FUTURE internal `set_allintra` call — prefer
  `TxTypeSearchPolicy::carry_frame_level_tx_sf`'s shape (carry the resolved frame-level value)
  over re-deriving.
- **Historical isolation trail (how the boundary was found) below:**
- **Found 2026-08-01** by `aom-bench/tests/s4cov_hd_speed_axis.rs`, extending the speed axis above
  640×640 for the first time (nothing had run `--cpu-used >= 1` above 640² except KB-22's two
  1280×720 speed-0/1 cells). bd8 4:2:0 real content, stock knobs:
  * **speeds 1, 2, 3: byte-exact** at 640×640 AND 1280×720, at cq24 and cq40 both;
  * **speeds 4, 5, 6: DIVERGE** at both framesizes and both quality points (deltas −111..+152 B
    on frames of 17–137 KB);
  * speed 7: diverges at 640² cq24, matches at 640² cq40, and PANICS at 1280×720 (KB-28 — that
    panic is gone since 2026-08-02; the whole grid is 28/28).
- **It is NOT a `is_720p_or_larger` arm.** The 640×640 control — deliberately included for this
  purpose — diverges exactly as the 1280×720 rows do. `large_frame_speed4_size_ladder` then
  walks SB-EXACT sizes at cpu4/cq24 and lands the boundary precisely on the OTHER framesize
  predicate: **448² MATCH, 512² DIVERGE (−28 B)**, with 576²/640² divergent and 256²/320²/384²
  clean. `is_480p_or_larger` is `AOMMIN(w,h) >= 480` (`speed_features.c:169`).
- **The one `is_480p_or_larger` × `speed >= 4` setting in C, A/B'd — and the result is an
  anomaly, not a fix.** `set_allintra_speed_feature_framesize_dependent` has exactly one:
  `tx_sf.tx_type_search.prune_tx_type_using_stats = 2` (`:299-301`; the same predicate sets **1**
  at `speed >= 2`, `:261-263`). Everything else 480p-keyed there is speed-independent or
  `is_720p_or_larger`-keyed, and 640² is below 720p.
  `tx_stats_prune_ab_across_the_480p_boundary` forces the port's derived value to 0 with
  `ToggleKnobs::disable_tx_stats_prune` and measures:

  | cell | prune on | prune forced off | verdict |
  |---|---|---|---|
  | 448² cpu2..5 (sub-480p) | MATCH | MATCH | inert, correctly |
  | 512² cpu2 | MATCH | DIVERGE −100 | **load-bearing** |
  | 512² cpu3 | MATCH | DIVERGE −61 | **load-bearing** |
  | 512² cpu4 | DIVERGE −28 | DIVERGE −28 | **byte-identical ⇒ INERT** |
  | 512² cpu5 | DIVERGE +106 | DIVERGE +106 | **byte-identical ⇒ INERT** |

  So the port's stats prune changes the bitstream at speeds 2-3 and provably does not at 4-5, on
  the same frame. On a lone KEY frame `update_type == KF_UPDATE == 0` and
  `thresh_arr[0][0] == thresh_arr[1][0] == 10` (`tx_search.c:1887-1891`), so the 1→2 level change
  is EXPECTED to be a no-op — which makes *"the prune stopped mattering at all"* the thing to
  explain, not the level.
- **The named next probe was the right instrument and the wrong hypothesis.** It read: *"at
  `speed >= 4` the MODE_EVAL stage takes the single-type `use_default_intra_tx_type` arm
  (`tx_search.c:1871`), which never reaches the stats prune, so the question is whether
  WINNER_MODE_EVAL is reaching it."* WINNER_MODE_EVAL **was** reaching it — 140,154 times on that
  one cell. Both stages were reaching it and both had the sf zeroed. Keep the probe (a reach
  counter answers "is this arm live" in one run); do not keep the inference.

### KB-27 — Encoder: MONOCHROME at `--cq-level 24` (`base_qindex` 96), speed 0 — a single-point near-tie — OPEN, pinned
- **Found 2026-08-01** by adding monochrome to KB-23's (partial-SB × speed) grid. First read as
  a multi-superblock effect (132²/192²/196²/256² all divergent at cpu0 while every speed 1..7 was
  byte-exact); `mono_speed0_size_qindex_localize` reduced it much further:
  **64×64 monochrome — ONE superblock — diverges too, at cq24 and at NO other cq in 18..30, and
  at speed 0 only.** Its 4:2:0 control on the identical crop and cq is byte-exact at every speed.
- **What it is not:** not loop-restoration (diverges identically with LR off, LR on, and via
  `c_encode_ctrls(&[])`), not multi-SB, not partial-SB, not a qindex band (a single cq).
- **What it looks like:** the KB-2/KB-6 near-tie signature. The port codes 5 fewer bytes of 320
  and the payload first-differs at byte 2 — the loop-filter-level field, a value the port derives
  from its OWN reconstruction, so the header delta is downstream of a tile difference rather than
  the cause of it. Content-specificity was measured, not assumed: across 5 crops × 5 quality
  points × {mono, 4:2:0} = 50 cells, exactly ONE diverges
  (`av1-1-b8-00-quantizer-00` crop (64,64)@(0,0), mono, cq24, cpu0).
- **Why no green test saw it:** `config_permutations.rs::speed_envelope_stock_map_is_pinned` runs
  the same 64×64 mono content at every speed 0..9 — at `SPEED_CQ = 32`, one quality point away.
- **Minimal repro:** `EncodeCell::real_content(_, "av1-1-b8-00-quantizer-00", Some((64,64,0,0)),
  24, 0)` with `mono = true` and the chroma planes cleared. Pinned in
  `partial_sb_speed_axis_chroma_formats_byte_match::MONO_S0_OPEN` plus the localizer, both
  directions.
- **SHAPE WIDENED 2026-08-03** by `s4cov_crop_format_axis::crop_straddle_speed0_byte_matches`:
  the class also reaches **474×480 (−150 B), 480×480 (−211), 714×720 (+160) and 720×720 (+105)**,
  all monochrome cq24 cpu0, all with byte-exact 4:2:0 twins. So it is **size-independent**
  (which the 64×64 localizer already implied and nothing had confirmed above 256²), and it does
  NOT depend on a partial superblock — 480×480 and 720×720 are SB-exact. That pair is also what
  ACQUITS KB-28's crop root for these rows: the SB-exact controls sit at the crops' own
  mi-aligned extents, where KB-28's fix is a literal no-op, and they diverge by MORE than the
  crops do. Pinned in the new file too, with the controls as the attribution.

### KB-28 — Encoder: the framesize predicates read the mi-aligned extent, not `cm->width`/`cm->height` — an EXACTLY 1280×720 frame REFUSED to encode at `--cpu-used` 7, 8 AND 9 — FIXED ✅ 2026-08-02
- **Found 2026-08-01, fixed 2026-08-02.** `pack.rs` needed `cm->width * cm->height` to select
  `set_vbp_thresholds`' sub-720p bucket (var_based_part.c:667 → `..._key_frame` :547) for the
  VAR_BASED partitioning, but `pack_tile` was given only **mi-aligned** extents. Rather than
  guess it asserted, refusing across the window where "the mi-aligned area and the up-to-3px-
  smaller crop could land on opposite sides of 1280*720". An exactly-1280×720 frame is inside
  that window (`mi_px == 921600` is not `< 921600` while `1277*717 == 915609` is), so the most
  ordinary HD frame panicked with *"VBP threshold resolution arm is crop-ambiguous at 1280x720
  mi-aligned: thread the true crop dims"*. **Title corrected: the refusal fired at speeds 7, 8
  and 9**, not 7 alone — the entry said speed 7 because `hd_speed_axis_byte_matches` stops at 7.
- **The mi grid rounds UP to 8 px, not 4.** `av1_get_MBs` (alloccommon.c:30-33) sets
  `mi_cols = ALIGN_POWER_OF_TWO(width, 3) >> MI_SIZE_LOG2`, so `mi_cols * 4` is up to **7** px
  larger than the crop. The guard's `mi - 3` was therefore too narrow, and that had a
  consequence worse than the refusal: **8,776 crops (both mi extents ≤ 4096) took the wrong
  threshold arm with NO refusal at all** — e.g. 1274×722 → mi 1280×728, where
  `(1280-3)*(728-3) = 925,825 ≥ 921600` so the guard stayed silent while the true crop area
  919,828 is below it. Measured pre-fix at speed 7: **+775 B on 1274×722 and +1196 B on
  954×962**, silently. The window and the hole are enumerated and pinned in
  `kb28_crop_dims::refusal_window_is_characterised` (369 in-window mi shapes; of the crops that
  would take the WRONG arm, 19,071 were refused loudly and 8,776 were not refused at all).
- **SIX consumers, one root — all of them re-derived a framesize predicate from `env.mi_*`:**
  `pack.rs`'s VBP `num_pixels` (var_based_part.c:667→:547);
  `partition_pick.rs`'s `use_square_partition_only_threshold` (speed_features.c:175-316),
  intra-CNN res-tier thresholds (partition_strategy.c:311-312),
  `ext_partition_eval_thresh` (speed_features.c:510-511), and
  `av1_ml_prune_4_partition`'s `res_idx` (partition_strategy.c:1349-1352);
  plus `extract_intra_cnn_window`'s clamp (inert — see the KB-23 note below).
  The last two carried in-tree comments naming the gap and leaving it open (playbook §9 again:
  a correct citation with a conclusion true only of the envelope that had been run — every
  size any gate encoded was either SB-exact or far from 480/720).
- **FIX:** `SbEncodeEnv` gains `frame_width` / `frame_height` (= `cm->width`/`cm->height`) plus
  `frame_min_dim()` / `frame_num_pixels()` / `assert_crop_dims_match_mi()`; all 22 construction
  sites pass the real crop. `pack.rs`'s `fs_sf` checks (KB-32's caller guard) become **exact**
  instead of "unambiguous only ≥ 3 px clear of a boundary", and the crop-ambiguity refusal is
  deleted — not relaxed: it is replaced by the value it was refusing for lack of.
- **PER-ROOT BITE PROOF, disjoint cell sets (playbook §1).** Reverting the VBP half alone
  (`num_pixels` back to the mi area + the old assert) fails `vbp_band_crop_dims_byte_match`
  only — 4 PANICs with the original message at speed 7 + the two silent divergences above —
  while `rd_band_min_dim_tiers_byte_match` stays **12/12**. Reverting the four
  `partition_pick.rs` predicates alone fails `rd_band_min_dim_tiers_byte_match` **0/12**
  (474×480 −69/−178/−86/+13/−94/+9 and 714×720 −416/−555/−338/+317/−42/+30 at cpu 1..6) while
  the whole VBP band stays green. Different consumers of one root, provably separate bands.
- **VERIFIED: `hd_speed_axis_byte_matches` is now 28/28** (was 26/28) — its self-promoting pin
  fired unprompted and `LARGE_FRAME_OPEN` is **empty**. New gates:
  `aom-bench/tests/kb28_crop_dims.rs` — `refusal_window_is_characterised` (default tier, pure
  arithmetic + the reach assertions for every encoded cell), `vbp_band_crop_dims_byte_match`
  (9 shapes × cpu 7/8/9), `rd_band_min_dim_tiers_byte_match` (474×480 / 714×720 × cpu 1..6),
  `cnn_window_clamp_is_replication_inert`. Unit locks:
  `partition_pick::kb28_crop_dim_locks` (3 tests — speeds 0..9 × both sides of 480 and 720 in
  both directions, plus the "crop and mi readings must DISAGREE on the gated crops"
  non-vacuity assertion) and `var_part::tests::kb28_num_pixels_is_the_crop_area_not_the_mi_area`.
- **RESIDUAL, stated plainly — the refusal was HIDING a divergence. CLOSED ✅ 2026-08-02.**
  1280×720 was byte-identical at cpu 7 (both cq) and **NOT** byte-identical at cpu 8/9: −132/+36
  (cq24/cq40) at cpu 8 and −8/+4 at cpu 9. That was **KB-12's nonrd estimate arm** — and it was
  not a near-tie but `aom_hadamard_lp_8x8`'s missing trailing transpose (KB-12, fixed
  2026-08-02). All 18 open rows of `vbp_band_crop_dims_byte_match` are byte-exact now; the 2
  that remain are KB-32's non-square-leaf HANDOFF *refusal* at speed 9. The attribution
  evidence stands as recorded: a
  partial-superblock explanation was tested and REFUTED (the SB-exact controls 1280×704 and
  1216×768 diverge too, −1/−6 and +1/−8), the cells with `cm->width == mi_cols * 4` on both
  axes — where this fix is a literal no-op — carry byte-identical deltas in all three arms
  (post, revert-VBP, revert-min-dim; `benchmarks/kb28_crop_dims_2026-08-02.tsv`), and every speed-8 residual is
  inside KB-32's pinned `< 1.0 B/SB` shape (worst 0.550 at 1280×720 cq24). Also: two speed-9
  cells reached KB-32's non-square-leaf HANDOFF refusal, which KB-32 measured as reachable only
  on its 108 MP cell — **0.9 MP frames reach it too**, which was the first contradiction of that
  claim and is what started KB-34. Both closed 2026-08-02 when the estimate arm learned the
  non-square leaf: `vbp_band_crop_dims_byte_match` is **30/30 byte-exact** and
  `NONRD_ESTIMATE_ARM_OPEN` is **empty**.
- **Is there an analogous window at another `set_vbp_thresholds` bucket? NO — checked in
  source.** `set_vbp_thresholds` compares `num_pixels` against `RESOLUTION_288P/480P/720P/
  1080P/1440P`, but on the KEY path it delegates to `set_vbp_thresholds_key_frame` and
  **returns** (var_based_part.c:660-664) before `tune_base_thresh_content`,
  `tune_thresh_based_on_resolution` and `tune_thresh_based_on_qindex` — where every other
  bucket lives. `..._key_frame`'s only bucket is `RESOLUTION_720P` (:547). The file's other
  `cm->width * cm->height` reads are `chroma_check` (:1004, returns immediately on a key frame)
  and two `!is_key_frame`-gated arms (:1344/:1358, :1821). The MIN-DIM axis has three boundaries
  (480/720/2160, speed_features.c:169-172) and all three are covered by the same change — the
  frame-level SF resolver already read the true crop, only the in-walk re-derivations did not.
- **Still unmeasured:** the crop axis at bd10/12, 4:2:2/4:4:4, monochrome, SB128 and multi-tile;
  crops straddling the `is_4k_or_larger` (2160) predicate; and the 480/720 straddle at
  `--cpu-used 0` (where `use_square_partition_only_threshold`'s base tier still moves).
- **MEASURED 2026-08-03** — `aom-bench/tests/s4cov_crop_format_axis.rs`, 62 rows, record
  `benchmarks/s4cov_crop_format_2026-08-03.tsv` + `.meta`. **Four of those seven are now swept,
  and the crop read holds on every one of them:**
  * **chroma formats 36/36 byte-exact** — monochrome, 4:4:4 and 4:2:2 at both crops across
    KB-28's own `--cpu-used 1..6` band;
  * **SB128 12/12 byte-exact**, with the C stream verified CHANGED vs `--sb-size=64` on 12/12
    rows. A predicate written into that arm was FALSE and the first run refuted it: it asserted
    474×480 would be downgraded to SB64 by `av1_select_sb_size`'s `AOMMIN(w,h) <= 480` rule —
    the rule KB-34's entry quotes — but that rule is in the `AOM_SUPERBLOCK_SIZE_DYNAMIC` branch
    and an **explicit** `--sb-size=128` returns `BLOCK_128X128` from the top of the function
    (encoder_utils.c:961-963). Both crops are genuine SB128;
  * **bd10 8/8 byte-exact** at `--cpu-used` {0, 7} — the only speeds where bd10 is byte-exact on
    SB-exact content (the pinned `b10_64` band owns 1..6), each crop run beside an SB-exact
    control at **its own mi-aligned extent**, which is the sharpest available A/B: same tier
    under the mi reading, different tier under the crop reading;
  * **`--cpu-used 0` 2/2 byte-exact at 4:2:0** — the named gap where
    `use_square_partition_only_threshold`'s base tier moves. The monochrome rows at cpu0 DIVERGE,
    and that is **KB-27's class, not this root**: measured, not argued — the SB-exact monochrome
    controls at 480×480 / 720×720 diverge too (see KB-27's widened shape).
  Still unmeasured, with costs, at the tail of the gate file: **bd12** (~2 min, the same four
  cells as the bd10 arm), **crops straddling 2160** (~25-30 min of port encode — KB-19's
  2160×2160 speed-0 cell is C ~26 s / port ~195 s; the predicate to check is
  `default_min_partition_size`'s `BLOCK_8X8` arm at e.g. 2154×2160 vs 2160×2160), and
  **multi-tile × the straddle** (same cost class; it needs its own crop pair at >4096 px or
  ~9.44 MP, e.g. 4090×2154 vs 4096×2160, because a tile split cannot coexist with a 714×720
  frame).

### KB-29 — Encoder: the IntraBC-armed encode produced a NON-CONFORMANT bitstream (`Invalid intrabc dv`) — FIXED ✅ 2026-08-01 (5 roots), + the general decode-side gate it was missing
- **Found 2026-08-01** by the cross-encoder still-picture benchmark
  (`benchmarks/xbench_2026-08-01.md` stage 2, raw
  `benchmarks/xbench_stage2_aom_screentools_2026-08-01.tsv`). Repro: `EncodeCell::port_encode_with`
  on real screen content (`codec-corpus/gb82-sc/terminal.png`, 1024×1024 native centre crop,
  8-bit 4:2:0), `cpu-used 6`, `cq 50`, `ToggleKnobs { enable_intrabc: true, ..default() }` with a
  `c_encode_defaults` bootstrap. Driver: `benchmarks/xbench/drv-aom` with `XBENCH_AOM_INTRABC=1`.
  **The same defect reproduces in ~2 s on a 196² crop of the corpus vector**
  (`av1-1-b8-16-intra_only-intrabc-extreme-dv` @ (480,180), the KB-15 witness cell) — no 1 MP
  encode needed, which is what made the close tractable.
- **Symptom:** the stream was REJECTED by both reference decoders — `aomdec` (libaom 3.14.1):
  *"Corrupt frame detected / Invalid intrabc dv"*; `dav1d` 1.5.4: *"Invalid argument"*.
  **`Invalid intrabc dv` was a DOWNSTREAM symptom, not the root: no DV validity constraint is
  violated.** The port's own `is_dv_valid` (`aom-dsp/src/entropy/dv_ref.rs:1578`) is
  differential-locked against the real `static inline av1_is_dv_valid`, and at every diverging
  site its inputs (tile bounds, `mib_size_log2`, `is_chroma_ref`) and the coded DV diff matched
  the decoder's exactly. What actually happened is a **tile-payload desync**: a defect in the
  IntraBC coefficient pack moved the bit position, and some later block's garbage
  `use_intrabc` + DV diff then failed validity, which is the first check libaom happens to
  hard-error on. **Do not re-chase the wavefront / 256-px lag / reference-area / tile-clamp
  constraints — all measured clean.** (KB-15 root #1 was a genuine tile-bounds bug; this is NOT
  a recurrence of it.)
- **Method that closed it** (reusable): instrument BOTH sides of the same stream — the encoder's
  `pack_leaf` and the decoder's `decode_block` — with a per-block `(mi_row, mi_col, bsize,
  use_intrabc, skip)` dump plus a running arithmetic-symbol count and `od_ec_*_tell` / range.
  Align the two walks and find the first block whose **symbol delta** differs (a desync moves the
  count) or whose **range** differs at equal count (a CDF-index/state divergence at equal symbol
  count). Then bisect inside the block with checkpoints after mode-info / after tx-size /
  after coefficients. Every root below was localized to one block in one pass this way.
- **ROOT 1 (encoder, `pack.rs`) — the IntraBC coeff arm never wrote CHROMA coefficients for any
  block with a 4-px side.** `write_inter_txb_coeff` (`bitstream.c:1414-1421`) recomputes the
  chunk extent from the **chroma-plane** 64×64 unit — `get_plane_block_size(BLOCK_64X64, ss)` =
  `BLOCK_32X32` at 4:2:0, so 8 mi units — and offsets by `row >> ss_y`. The port instead
  subsampled the LUMA chunk bound: `(row + mu_h) >> ss_y`. For a block spanning ONE mi unit in a
  subsampled dimension (BLOCK_4X4 / 4X8 / 8X4 / 4X16 / 16X4 at 4:2:0) that truncates `1 >> 1` to
  **0**, so the chroma loop ran zero times and the block's U and V `all_zero` symbols were never
  emitted. Measured at the first divergent block, `mi(39,10)` BLOCK_8X4: encoder wrote 26 symbols,
  the decoder read 28. Those shapes are the MAJORITY of real IntraBC blocks (KB-15 measured 42 of
  49 non-square on this crop), so essentially every screen-content IntraBC encode was corrupt.
- **ROOT 2 (encoder, `pack.rs`) — an IntraBC coeff leaf that does not SIGNAL its tx size never ran
  `set_txfm_ctxs`.** `write_modes_b`'s else arm (`bitstream.c:1554-1556`) stamps the
  txfm-partition contexts from the derived tx size whenever the tx-size branch is not taken —
  which for an IntraBC coeff leaf means `block_signals_txsize(bsize)` false (BLOCK_4X4),
  TX_MODE_LARGEST, or lossless. Neither port arm covered it: the var-tx write is gated on the
  same predicate, and `encode_b_intrabc_coeff` returns before `encode_b_intra_dry`'s step-6
  stamp (correctly — the var-tx write owns the stamp when it runs). The context stayed STALE, so
  a later leaf's `txfm_partition_context` picked a different CDF row than the decoder's: **same
  symbol count, divergent arithmetic range**, desync a few blocks later. Measured at speed 2
  cq12: a BLOCK_4X4 IntraBC coeff leaf at `mi(40,30)` left `above_tctx[30]` at 8 where every
  conforming decoder derives 4; the visible failure was 5 blocks later at `mi(46,40)`, whose
  var-tx symbol split the range (enc rng 38168 vs dec 45916 at an identical symbol index and an
  identical CDF slot `[31223, 30352, 28283, 27407]`).
- **ROOT 3 (encoder, `partition_pick.rs`) — a PALETTE leaked onto an IntraBC/inter winner.** C
  zeroes both palette sizes inside the DV loop (`memset(&mbmi->palette_mode_info, 0, ...)`,
  `rdopt.c:3592`, immediately before `mbmi->use_intrabc = 1`; the inter path does the same at
  `:4804-4805`), so a winning `best_mbmi` never carries one. The port carried the intra
  candidate's `palette_y`/`palette_uv` through onto the IntraBC winner, and `pack_leaf` then
  emitted the colour-map tokens for a block whose mode info wrote none — `write_mb_modes_kf`
  RETURNS right after `write_intrabc_info` when the block is IntraBC
  (`bitstream.c:1289-1291`), so the decoder reads no palette syntax at all. Observed directly:
  encoder `mi=(40,28) ibc=true paly=Some(6)`, decoder `mi=(40,28) ibc=1 paly=0`. This is the root
  of the **palette + IntraBC** arm specifically (each tool alone was fine at that cell), i.e. the
  "both" row of the xbench table. Fixed with the same `non_intra` gate the `mode` / `angle_delta`
  / `use_filter_intra` / `uv_mode` fields at that site already used.
- **ROOT 4 (decoder, `aom-decode/src/lib.rs`) — the 64×64 chunk walk (which contains the CHROMA
  read) was gated on `do_uniform`**, so a split IntraBC var-tx block never read its U/V
  coefficients. `write_tokens_b`'s inter arm writes them regardless (`bitstream.c:1463-1468`
  breaks only on `!is_chroma_ref`). Only the LUMA sub-loop should be skipped when the leaf arm
  has already consumed it.
- **ROOT 5 (decoder) — the leaf-vs-raster var-tx walk was selected on "leaf SIZES differ", not on
  "the quadtree was READ".** Equal leaf sizes do not make the raster loop equivalent: a
  BLOCK_16X8 split all the way to TX_4X4 has eight same-size leaves whose DFS order is
  `(0,0)(0,1)(1,0)(1,1)(0,2)(0,3)(1,2)(1,3)` — not raster — so the per-txb `txb_skip_ctx`
  sequence differs and the decode desyncs at the third txb. Bite: at the 196² crop, speed 6
  cq12, C accepts the stream and the port decoder rejects it.
- **ROOT 6 (decoder) — the leaf arm was missing the CfL luma store** that the raster arm runs
  (`predict_and_reconstruct_intra_block` tail, `store_cfl_required`): a NON-chroma-reference
  block always stores, because a later member of its shared chroma group may pick `UV_CFL_PRED`
  over a footprint containing it, and `is_inter_block(mbmi)` is TRUE for IntraBC
  (`blockd.h:372`) — the exact mirror of KB-15 root #4 on the encoder side. Symptom: luma
  byte-identical to the C decoder, chroma off by 253 of 9604 U samples (first at chroma (80,44)).
  Only became reachable once root 5 routed more blocks through the leaf arm.
- **TEETH, per root** (playbook §1 "revert each ONE ALONE and compare which cells fail" — every
  root fails a DIFFERENT cell, which is what proves they are separate roots):
  | reverted root | failing cell | failure |
  |---|---|---|
  | 1 (chroma chunk extent) | `scc196_cq48_s0/enable_intrabc` | C decoder REJECTS (1886 B) |
  | 2 (`set_txfm_ctxs`) | `scc196_cq12_s2/enable_intrabc` | C decoder REJECTS (6138 B) |
  | 3 (palette leak) | `scc196_cq12_s2/enable_palette+enable_intrabc` | C decoder REJECTS |
  | 4 (chunk-walk gate) | `scc196_cq12_s2/enable_intrabc` | port decoder rejects a stream C accepts |
  | 5 (walk selection) | `scc196_cq12_s6/enable_intrabc` | port decoder rejects a stream C accepts |
  | 6 (CfL store) | `scc196_cq48_s0/enable_intrabc` | chroma differs C-vs-port, 253/9604 U |
  End-to-end control on the ORIGINAL repro shape (196² crop, `enable_intrabc`, speeds {0,3,6} ×
  cq {12,32,48,63}, streams written out and run through the real binaries): **pre-fix 9 of 12
  rejected by BOTH `aomdec` and `dav1d`; post-fix 0 of 12** (the 3 cq63 cells code no IntraBC
  coeff leaf with a 4-px side and were already clean — a negative control). A wider post-fix
  sweep (speeds 0..7 × 8 quality levels × {intrabc, palette+intrabc} = 144 cells) is clean on
  all three decoders with pixel identity. **The LITERAL headline repro re-run post-fix**
  (`drv-aom`, `terminal.png` 1024² native centre crop, cpu-used 6, cq 50, one encode each):
  IntraBC-only 10 031 B and palette+IntraBC 10 082 B, **both accepted by `aomdec` AND `dav1d`**
  (were both FAIL); table re-measured in `benchmarks/xbench_2026-08-01.md`. The −49 % IntraBC
  byte win is therefore REAL compression — the benchmark's "that −49 % is not achievable" line
  was wrong and is corrected there. What remains open on that arm is **speed**: 70.0 s for 1 MP
  = 0.015 MP/s, ~43x the default path, which is a separate performance item, not a correctness
  one.
- **THE STRUCTURAL FIX — `crates/aom-bench/tests/armed_tools_decode_gate.rs` (new).**
  **Byte-identity to a reference proves conformance only where a reference exists AND is
  asserted equal.** Every `ToggleKnobs` arm aomenc cannot be driven into from
  `ToggleKnobs::c_ctrls` is a configuration the port can PRODUCE and that nothing ever decodes —
  the port could emit arbitrary garbage there and the whole suite would stay green. The gate
  encodes each such arm and asserts (1) the REAL `aom_codec_av1_dx` accepts it — the authority,
  (2) the port decoder accepts it, (3) **both produce identical pixels**, and (4) `dav1d`
  accepts it when `AOM_DAV1D_BIN` is set (wired in `just gate-armed-decode`, so the skip decision
  is the caller's, never inside the test body). 4 cells × 8 arms = 32 decode round-trips, ~26 s.
  - **Coverage is DERIVED, not asserted by name.** `single_knob_arms()` lists every
    one-knob-off-default `ToggleKnobs` value (kept exhaustive by a no-`..` destructure, so a NEW
    field is a COMPILE error), and the gate recomputes the unguarded set as "flipping this knob
    emits no `aome_enc_control_id`". **The measured answer, 2026-08-01 — 7 of 31 knobs:**
    `enable_palette`, `enable_intrabc`, `disable_tx_stats_prune`, `delta_lf_mode`, `qm`,
    `deltaq_mode2`, `deltaq_mode3`. IntraBC was simply the one we tripped over. Adding a knob
    without a C control now fails `armed_arm_coverage_is_complete` until it gets a decode arm.
  - The destructure earned its keep on the first run: it caught `deltaq_mode2`/`deltaq_mode3`,
    which the hand-written list had missed.
- **ENVELOPE (aarch64-apple-darwin, `--profile test-fast`, `AOM_CONFORMANCE_DIR` provisioned):**
  `-p zenav1-aom-encode -p zenav1-aom-decode --no-fail-fast` **354 passed / 0 failed**;
  `-p zenav1-aom-bench --no-fail-fast` **177 passed / 0 failed / 17 ignored** (the 17 pre-date
  this change); the decoder conformance corpus is **274 frames OK / 0 failed**, including both
  frames of `av1-1-b8-16-intra_only-intrabc-extreme-dv`, which is what exercises roots 4-6.
  All three encoder roots are inside `if winner.use_intrabc` / `non_intra` branches and all three
  decoder roots inside `if info.use_intrabc != 0`, so a non-screen frame is byte-inert by
  construction — confirmed by the unchanged byte-exactness suite.
- **RESIDUAL / open:** (a) the >64×64 multi-chunk non-uniform IntraBC block still reads all luma
  leaves before any chroma where C interleaves L,U,V per 64×64 chunk — not reachable from this
  port's encoder and strictly better than the pre-fix "no chroma at all", noted at the site.
  **Re-checked 2026-08-02 against a DIFFERENT encoder and still not reachable**: across 1,992
  real SVT-AV1 v4.2.0 screen-content streams, SVT never emits an IntraBC block larger than
  BLOCK_64X64 and emits no 128-px block size at all, so closing (a) needs an SB128
  screen-content encoder (IntraBC on BLOCK_128X128 / 128X64 / 64X128). Still open, still
  untested;
  (b) KB-15's byte-exactness pin against aomenc is untouched — this work makes the port's
  IntraBC output CONFORMANT, not byte-identical to libaom's.
- **CONFIRMED CAUSAL FOR GitHub #5, 2026-08-02.** Roots 4 and 5 are what closed the
  cross-encoder rejection issue: reverting EITHER one alone makes a real SVT-AV1 stream fail
  with the exact issue-#5 message (`corrupt frame: intrabc DV failed validity
  (non-conformant stream)`), with the DV code untouched. See KB-33.

### KB-30 — Encoder: `cid22_6292444` at `--cpu-used=6` diverges at EVERY quantizer (1 of 10 real photographs) — OPEN, not localized
- **Found 2026-08-01** by the libaom-C arm of the cross-encoder benchmark
  (`benchmarks/xbench_2026-08-01.md`, "The same control for the aom port"). The measurement is a
  whole-stream sha256 of `drv-aom` against `drv_libaom` over the RD corpus at MATCHED
  `--cpu-used 6` — both drivers running the same `shim_encode_av1_kf_defaults` config, so the
  only difference is which encoder coded the frame
  (`benchmarks/xbench_aom_byteidentity_2026-08-01.tsv`, reproduce with
  `python3 scripts/xbench.py byteid --a zenav1-aom --b libaom-c --preset-a 6 --preset-b 6`).
- **The map: 127/182 cells byte-identical.** photo-hr 28/28. photo 99/112 — seven of the eight
  CID22 512² images are 14/14 across the whole quantizer grid, and **`cid22_6292444` is 1/14**
  (identical only at cq62, where both encoders emit the same 1619 bytes). screen 0/42, which is
  a different and understood thing (`ToggleKnobs::default()` has palette/IntraBC off while
  libaom's ALLINTRA defaults have them on) — that half is KB-15/KB-29 territory, not this entry.
- **Shape of the divergence.** At every cq from 10 to 58 the port emits **0.25-2.52 % FEWER
  bytes** at an equal-or-slightly-lower SSIMULACRA2 (−0.00 to −2.46), widening monotonically
  toward the aggressive end: −0.27 % at cq10, −1.56 % at cq50, −2.52 % at cq58. Net BD-rate
  cost +0.37 pp (+18.25 % for the port vs +17.88 % for C, ssim2 vs `svt-c` p1). Fewer bytes for
  slightly less quality is the classic near-tie-won-by-an-underestimated-RD signature — the same
  shape KB-13's AB mode-cache root had before it was found — so the first suspect is a search
  the port runs LESS constrained than C, not a coefficient-path defect.
- **Why no green test saw it:** every real-content byte-parity gate in the tree runs the AV1
  conformance vectors (KB-13's 5-vector map, speeds 1-4) or synthetic/diagnostic content. CID22
  photographs at cpu-used 6 are not in any gate, and cpu-used 6 is outside KB-13's 1-4 band.
  Playbook §8 applies literally: coverage here was derived from a corpus name, and the corpus
  does not reach this cell.
- **Minimal repro** (no harness needed):
  `benchmarks/xbench/target/drv_libaom 512 512 50 6 <cid22_6292444.yuv> c.obu 0 1` vs
  `benchmarks/xbench/target/release/drv-aom 512 512 50 6 <same.yuv> p.obu 0 1`, then `shasum`.
  The `.yuv` is produced by `xtool prep <CID22-512/validation/6292444.png> out.yuv square:512`.
- **Not localized.** No first-divergent-block walk has been run. Per playbook §10 the next step
  is the sibling-C dump with a per-block symbol count on this exact cell, NOT reasoning from the
  byte delta — which is small (0.25-2.52 %) and, per KB-22's lesson, says nothing about whether
  the cause is a near-tie or a whole unmodelled pass.

### KB-31 — Encoder: every frame big enough to REQUIRE more than one tile PANICKED (`single-tile envelope only`) — FIXED ✅ 2026-08-01 (TWO roots: a driver gap and a real frame-header PARSE defect)
- **Reported as GitHub issue #6** from zensysbench: `EncodeCell::port_encode` exits 101 at
  5472x3648 (20 MP) and 12000x9000 (108 MP), across `--cpu-used` {3,6,9} and cq {30,50},
  bd8 4:2:0 allintra; 512x512 fine. The panic text was swallowed by the reporting harness.
  It is `crates/aom-bench/src/lib.rs:1138` (pre-fix line):
  *`assertion left == right failed: single-tile envelope only; left: 1, right: 0`*,
  frame `<aom_bench::EncodeCell>::port_encode_full`.
- **The governing property is NOT the issue's guess.** The issue noted both failing widths
  are `64k+32` (a partial-superblock column) and suspected a partial-SB path. Measured, with
  SB-EXACT sizes deliberately interleaved so the two candidate explanations were separable by
  the result pattern rather than by intuition:

  | size | multiple of 64? | MP | verdict (pre-fix) |
  |---|---|---|---|
  | 512x512, 4032x64 | yes | 0.26 / 0.26 | OK |
  | 3072x3072 (exactly 2304 SB64s) | yes | 9.44 | OK |
  | **4096x64** | **yes** | **0.26** | **PANIC** |
  | 4096x3072, 3136x3072 | yes / no | 12.6 / 9.6 | PANIC |
  | 4160x2048, 5472x3648 | no | 8.5 / 20 | PANIC |

  4096x64 panics while 3072x3072 — 36x its area and equally SB-exact — does not, so it is
  neither alignment nor "large". It is libaom's **tile requirement**, and there are TWO
  independent triggers (`av1_get_tile_limits`, `av1/common/tile_common.c:31-50`):
  * **width** — `min_log2_cols = tile_log2(MAX_TILE_WIDTH >> sb_size_log2, sb_cols)`; and the
    ENCODER's own bound is stricter still, `set_tile_info` (`av1/encoder/encoder.c:385-390`)
    re-raises `log2_cols` with a **`<=`** loop, so **64 SB columns is already 2 tiles** rather
    than 1 — `mi_cols >= 1009`, i.e. width >= 4033 px (measured: 4032 -> 1 tile, 4096 -> 2);
  * **area** — `min_log2 = tile_log2(MAX_TILE_AREA >> 2*sb_size_log2, sb_cols*sb_rows)`,
    i.e. more than 2304 SB64s ~ **9.44 MP**.
  With libaom's uniform-spacing default (`--tile-columns` unset) that resolves
  `log2_cols + log2_rows >= 1`, which the harness refused. 12 MP was never in doubt: 4000x3000
  would have panicked too (3000 SB64s).
- **ROOT #1 (port-only SCALING defect, no C counterpart): the frame driver only ever packed
  one tile.** `port_encode_full` asserted `tiles_log2 == 0` before any speed-dependent code —
  which is why the panic was speed-invariant — and then called `pack_tile` once over the whole
  frame. Nothing was missing from the ENCODER: `pack_tile` already takes
  `(mi_row0, mi_col0, n_sb_rows, n_sb_cols)` and `SbEncodeEnv` already carries tile bounds; the
  per-tile walk with tile-edge isolation is byte-proven by
  `aom-encode/tests/encoder_gate_multitile.rs`, and the derived multi-tile header + tile-group
  assembly by `aom-encode/tests/obu_assemble_multitile_diff.rs`. **The fix is composition**: a
  per-tile loop in raster order with a fresh `KfFrameContext` + `OdEcEnc` per tile (C's
  `av1_init_tile_data`), FRAME-level shared recon, the tile mi ends clamped like
  `av1_tile_set_row`/`_col` (`AOMMIN(.., mi_rows)`, tile_common.c:124-140), the per-tile SB
  trees re-assembled into frame raster for `build_lf_mi_grid`, the real tile SB spans handed to
  the loop-restoration search (which already resets its delta-coding refs per tile), the LR
  repack looped identically, and `assemble_multitile_frame_obu_payload_derived` for
  `tiles_log2 > 0`.
- **ROOT #2 (port-FIDELITY defect, decoder-side too — and the composition is what exposed it):
  `read_tile_info_max_tile` never ran `av1_calculate_tile_cols` between the column and row
  reads.** C calls it there unconditionally (`av1/decoder/decodeframe.c:2180`) and it
  RE-DERIVES the row bound from the just-read column count:
  `tiles->min_log2_rows = AOMMAX(tiles->min_log2 - tiles->log2_cols, 0)`
  (`av1/common/tile_common.c:73`). The port used the caller's `min_log2_rows`, which is
  `av1_get_tile_limits`' composition `max(min_log2 - min_log2_cols, 0)` — equal only when the
  stream codes `log2_cols == min_log2_cols`. libaom's encoder breaks that tie routinely via the
  `<=` loop above. Measured at 4096x3072: the row unary started at 1 instead of 0, so
  `log2_rows` read as 1 instead of 0 (a 2x2 grid where C coded 2x1) and
  `context_update_tile_id` — whose width is `log2_cols + log2_rows` bits — consumed one bit too
  many, **desyncing the entire rest of the frame header**: `base_qindex` parsed as 240 instead
  of 120 and the port coded 77,639 B where aomenc coded 1,336,439. Fixed by restructuring the
  reader into C's order (read cols -> `av1_calculate_tile_cols` -> read rows), recovering
  `min_log2` as `min_log2_cols + min_log2_rows` (exact, since `av1_get_tile_limits` ends with
  `min_log2 = AOMMAX(min_log2, min_log2_cols)`, `:49`). One fix repairs BOTH directions: the
  writer's unary is relative to `t.min_log2_rows`, so a re-serialized header was equally wrong.
  **The DECODER had this too** — `aom-decode/src/frame.rs:183`'s `tile_limits` composes
  `min_log2_rows` identically — so any conformant stream with `log2_cols > min_log2_cols` and
  `min_log2 > 0` mis-parsed. No in-repo decoder gate reached it: every multi-tile decode fixture
  is small enough that `min_log2 == 0`, where the re-derivation is an identity.
- **Verified.**
  * NEW `aom-bench/tests/kb31_mandatory_tiles.rs`. Default tier (0.15 s):
    `mandatory_tile_split_encodes_byte_identical` — 4032x64 (1x1, the negative control),
    **4096x64 (2x1)**, 4160x64 (2x1), 4160x128 (2x1) at cq30 cpu9, all BYTE-IDENTICAL to real
    `aomenc --allintra`, with the coded tile grid read back off the reference stream
    (`decode_frame_obus_prefilter`) rather than derived (playbook §8). 4096x64 is the SMALLEST
    frame that reproduces issue #6 — 0.26 MP — which is why this sits in the default tier.
    `--ignored` tier: `area_forced_tile_split_byte_identical` (4096x2368 -> 2x1 and 4032x2368 ->
    1x2, ~9.6 MP at cpu7, both byte-identical — the only cells with a tile ROW boundary and the
    only ones that reach root #2); `mandatory_tile_split_byte_identical_across_speeds` (4160x64
    tiled + 4032x64 control at every speed 0..9); `issue6_reported_sizes_encode` (the issue's own
    5472x3648 and 12000x9000).
  * **Bite proof, per root, with different cell sets (playbook §1).** Restoring the
    `tiles_log2 == 0` assert alone fails ONLY `mandatory_tile_split_encodes_byte_identical`, with
    the issue's exact text, while config_permutations (87) + toggles_rd_close (5) + sb128_e2e (25)
    stay green. Reverting the parse re-derivation alone fails a DIFFERENT pair —
    `rb_diff::read_tile_info_inverts_write` (*"uniform log2, left: (2, 2) right: (2, 0)"*) and
    `kb31::area_forced_tile_split_byte_identical` (*"real aomenc coded 2x2 tiles"*) — while the
    width-predicate default gate stays green, because those cells have `min_log2 == 0`.
  * **Byte-inertness of the refactor, measured not argued.** The single-tile path now passes real
    clamped tile bounds where it passed a `1 << 16` sentinel. Same-binary A/B via `git stash`:
    4032x64, 1024x1024, 2048x2048 and 3072x3072 at cpu8 and cpu9 produce **byte-identical port
    payloads before and after** (13,224 / 111,581 / 436,835 / 977,820 B at cpu8). Full suites:
    `-p zenav1-aom-dsp -p zenav1-aom-decode` 15 files 0 failed; `-p zenav1-aom-encode -p
    zenav1-aom-bench` **463 passed / 0 failed / 19 ignored**. Both dispatch modes (default and
    `AOM_FORCE_SCALAR=1`); `cargo check --target x86_64-apple-darwin -p zenav1-aom-dsp` clean.
  * `rb_diff`'s uniform tile fixture drew `min_log2_rows` INDEPENDENTLY of `log2_cols` — a state
    C cannot produce, and precisely what hid this root. It now derives both row bounds from one
    `min_log2` (the reader's arg `max(min_log2 - min_log2_cols, 0)`, the writer's
    `max(min_log2 - log2_cols, 0)`) and asserts the re-derived `g.min_log2_rows`. That is a
    TIGHTENING, not a relaxation.
  * The issue's own cells, `--cpu-used=9`, cq30 (aarch64-apple-darwin): 5472x3648 -> **2x2 tiles**,
    2,124,645 B vs aomenc's 2,121,452 (+3,193, +0.15%), 1.6 s, 0.79 GB peak RSS; 12000x9000 ->
    **4x4 tiles**, 11,548,497 B vs 11,520,317 (+28,180, +0.24%), 8.5 s, **3.11 GB peak RSS** (whole
    process, `/usr/bin/time -l`). Neither OOMs; 108 MP is the memory ceiling to expect from this
    harness. The residual sub-percent deltas are **KB-32's**, not tiles' — proven by the
    single-tile controls (see below), which is why `issue6_reported_sizes_encode` pins a verdict
    plus a `< 1%` bound rather than byte-identity.
- **Why no green test could have caught it:** the widest frame any pre-existing gate encodes is
  2160 px (KB-19's 2160x2160 = 34 SB columns, 1,156 SBs), so no cell in the tree came near
  either predicate and none had `tiles_log2 > 0` at all. `encoder_gate_multitile.rs`
  DOES encode multi-tile frames byte-exactly — but it drives them with explicit
  `AV1E_SET_TILE_COLUMNS`, never through `EncodeCell::port_encode`, and at sizes where
  `min_log2 == 0`. Playbook §7 and §12 together: two individually-green rows (the tile machinery,
  the frame driver) with nothing crossing them, and a green unit differential that licenses a
  kernel and not its caller.
- **RESIDUAL, stated plainly.** (a) Multi-tile x per-SB delta-q (`--deltaq-mode` 2/3) is REFUSED
  with a loud assert, not modelled: `pack_tile`'s running qindex base restarts per tile while the
  harness's frame-raster replay loops (`dq3_present`, `stamp_lf_delta_lf`) carry one base across
  the frame; no delta-q gate encodes a frame large enough to need tiles, so the arm is unreached
  today. (b) `av1_calculate_tile_cols` also re-derives `max_height_sb` from the widest tile
  (`tile_common.c:82-95`); that is deliberately NOT modelled and the caller's value is still used,
  because it is read only by the NON-uniform row read and libaom emits non-uniform spacing only
  under `--tile-width`/`--tile-height`, so there is no real-stream gate to prove a change against.
  Noted at the site. (c) Nothing here is swept at SB128, bd10/12, 4:4:4/4:2:2 or monochrome —
  the whole file is bd8 4:2:0 SB64.

### KB-32 — Encoder: `--cpu-used` 8 (every size >= 512²) and `--cpu-used` 9 (>= ~1 MP) diverged on real content — BOTH BANDS FIXED ✅ 2026-08-01 (two roots) + the attributed residual CLOSED ✅ 2026-08-02 (KB-12's third root)
- **Reported as GitHub issue #7**, found 2026-08-01 while separating KB-31's roots (the tiled
  cells' single-tile CONTROLS diverged too, which is what proved the residual was not tiles').
  Measured (cq30, `c_encode_defaults`, mirror-tiled `av1-1-b8-00-quantizer-00`, bd8 4:2:0,
  aarch64), port bytes minus real aomenc, **before -> after**:

  | size | SBs | cpu8 before | cpu8 after | cpu9 before | cpu9 after |
  |---|---|---|---|---|---|
  | 512² | 64 | +61 | +61 | +0 | +0 |
  | 768² | 144 | +152 | -50 | — | — |
  | 896² | 196 | +253 | -23 | +0 | +0 |
  | 1024² | 256 | +581 | -168 | +613 | **+0** |
  | 2048² | 1,024 | +2,576 | **+21** | +2,311 | **+0** |
  | 2112² | 1,089 | — | — | (never run) | **+0** |
  | 2176² | 1,156 | — | — | (never run) | -184 |
  | 3072² | 2,304 | +5,643 | — | -1,883 | -325 |
  | 5472x3648 (20 MP) | 4,902 | — | — | +3,193 | **+339 (0.016 %)** |

- **ROOT #1 — `rt_sf.force_large_partition_blocks_intra` was UNMODELLED, and it is BOTH bands.**
  `set_allintra_speed_feature_framesize_dependent` sets it at `speed >= 8 && is_720p_or_larger`
  (`speed_features.c:326-328`). Its ONLY consumer anywhere in libaom is
  `set_vbp_thresholds_key_frame` (`var_based_part.c:535-560`) — the KEY variance partitioner this
  port runs from speed 7 up — where it has TWO arms, and the two reported bands are the two arms:
  * `threshold_base <<= (var_part_split_threshold_shift - 7)` (`:539-544`). The shift is **8** at
    speed 8 (`:581`) and back to **7** at speed 9 (`:601`, *"intentionally lower than speed 8's"*),
    so `shift_steps` is 1 at speed 8 and 0 at speed 9. **That is the cpu8 band**, and it is why
    cpu8 had no threshold: every frame at least 720 px on its short side takes it, and the effect
    grows with the number of superblocks;
  * `shift_val = 1` instead of 2 inside the `num_pixels >= RESOLUTION_720P` arm (`:552-554`).
    `RESOLUTION_720P` is `1280 * 720` **pixels of AREA** (`rd.h:65`) — 921,600, which falls
    between 896² = 802,816 and 1024² = 1,048,576. **That is the cpu9 threshold, exactly**, and
    that arm is live at BOTH speeds.
  The port's `var_part` module doc asserted *"`force_large_partition_blocks_intra` ... is 0 on
  this path"* and dropped both arms — playbook §9 in its purest form: true of the envelope it was
  written against (nothing above 640 px had ever run `--cpu-used >= 8`), false of the format.
- **The threshold is AREA, not the short side, and the gate proves it by RESULT PATTERN.**
  `nonrd_speed9_area_threshold_byte_identical` holds the short side at 768 across the area
  threshold (884,736 -> 933,888 px) and then at 704 — below 720, so the speed feature must NOT
  arm — across the same threshold. Measured before the fix: 768x1216 **+498**, 704x1408 +2
  (the residual only). A port keyed on the short side passes the first pair and fails 704x1280;
  one keyed on area alone fails it too.
- **ROOT #2 — the speed-9 cost-update level was hardcoded OFF, dropping `INTERNAL_COST_UPD_SBROW`
  at 4k.** The framesize-INdependent cascade sets `coeff_cost_upd_level` /`mode_cost_upd_level` to
  `INTERNAL_COST_UPD_SBROW` at speed 9 (`speed_features.c:593-594`) and the framesize-dependent
  pass demotes them to `INTERNAL_COST_UPD_OFF` **only below 4k** (`:648-651`,
  `if (!is_4k_or_larger)`). `pack.rs` carried this as a written HANDOFF ("4k+ frames keep
  INTERNAL_COST_UPD_SBROW — out of the canon envelope, unmodelled"); KB-31's 20/108 MP cells and
  this ladder are what made it reachable. Fix: `refresh = c == 0` at `is_4k_or_larger` (C's
  `skip_cost_update`'s `mi_col != tile_info->mi_col_start` early return,
  `encodeframe_utils.c:1556-1564`), with the derivation carried across the SB row.
- **FIX SHAPE (playbook §13 / KB-26): carry the resolved value, do not re-derive it.**
  `SpeedFeatures` gains `force_large_partition_blocks_intra` + the speed-8/9
  `var_part_split_threshold_shift` steps; `apply_allintra_framesize_dependent` gains a `speed`
  argument and the `>=720p` arm; a new `partition_pick::FrameSizeSf { vbp: var_part::VbpSf,
  is_4k_or_larger }` carries both resolved facts from `aom-bench`'s resolver through
  `PickFrameCfg` into `pack_tile`'s `VbpFrame`. **`pack_tile` deliberately does not compute the
  predicates itself — it only has mi-ALIGNED dimensions**, which would be wrong for any crop
  within 3 px of a boundary; it asserts them instead wherever the mi-aligned value is unambiguous,
  so a caller that leaves `fs_sf` at `Default` on a big frame fails loudly. That check is the
  structural fix for how both roots hid.
- **PER-ROOT BITE PROOF, different cell sets (playbook §1).** Reverting root #1 alone (`if false
  && sf.force_large_partition_blocks_intra` at both arms) fails **all four** KB-32 gates and
  reproduces the original ladder EXACTLY (768² +152, 896² +253, 1024² +581, 2048² +2,576;
  768x1216 +498). Reverting root #2 alone fails **exactly one** —
  `nonrd_speed9_4k_cost_upd_sbrow`, at exactly -2,599 on 2176² — while the cpu9 area gate, the
  cpu8 ladder and the localizer stay green. Two roots, two disjoint cell sets.
- **THE RESIDUAL IS KB-12'S, AND THE ATTRIBUTION IS NOW ESTABLISHED (it was explicitly flagged as
  NOT established).** Decode-both localization (playbook §10) on every surviving cell:
  **partition trees agree EXACTLY** — 45,780 nodes at 2176², 3,496 at 512², 892 at 256² — and the
  first divergence is a leaf `y_mode`, both sides inside `av1_nonrd_pick_intra_mode`'s four-mode
  `intra_mode_list` {DC, V, H, SMOOTH}, with `tx_size`, `uv_mode`, angle delta and filter-intra
  all equal. Examples: 512² cq30 s8 mi(4,108) BLOCK_8X8 real SMOOTH / port DC; 2176² cq30 s9
  mi(108,174) BLOCK_8X8 real DC / port V. **Two corrections to KB-12 fall out**: the class is
  broader than the V-vs-H it recorded (DC-vs-SMOOTH, V-vs-SMOOTH, DC-vs-H, DC-vs-V all occur),
  and it is **not speed-8-only** — 512² cq48 and 2176² cq30 diverge at speed 9, so speed 9's
  three estimate-loop prunes MASK it on 64²/128² rather than removing it. Gate:
  `kb32_nonrd_size_bands::estimate_arm_residual_is_a_leaf_mode_near_tie` (asserts both the
  tree agreement and the mode set, self-promoting in both directions).
  Shape after the fix: sign-random and flat in area (per superblock 512² 0.95 -> 0.95 [the arm is
  unreachable below 720], 768² 1.06 -> 0.35, 896² 1.29 -> 0.12, 1024² 2.27 -> 0.66, 2048²
  2.52 -> 0.02) where the pre-fix ladder ROSE monotonically.
- **THE RESIDUAL IS CLOSED ✅ 2026-08-02, and it was NOT a near-tie** — `hadamard_lp_8x8`
  dropped the trailing transpose at `aom_dsp/avg.c:232-236`, so the nonrd estimate arm's `eob`
  (its only order-sensitive output) drifted; full record in **KB-12**. Every cell in this entry
  is byte-identical now: the cpu8 ladder 512²/768²/896²/1024²/2048² all **0**, and 2176² cpu9
  **0** (was -184). All four gates are hard byte gates —
  `nonrd_speed8_size_ladder_residual_is_bounded` and `nonrd_speed9_4k_cost_upd_sbrow` were
  PROMOTED from their shape/bound pins, and `estimate_arm_residual_is_a_leaf_mode_near_tie`
  keeps its decode-both localizer as the DIAGNOSTIC that runs if a cell returns.
  **The methodological lesson is worth more than the fix:** this residual's shape —
  sign-random, sub-byte-per-superblock, flat in area, partition trees identical to 45,780
  nodes, every leaf field equal except `y_mode`, all four candidates inside the estimate arm's
  own `intra_mode_list` — was read by three separate sessions as proof of a genuine tie. It was
  a dropped transpose in a kernel that had no differential. Playbook §1 + §10.
- **A real consequence of the fix, since CLOSED as KB-34: 12000x9000 at cpu9 REFUSED instead of
  encoding.** Correct thresholds are LARGER thresholds, which lets `set_vt_partitioning`'s
  HORZ/VERT pair arms win on 108 MP of extremely smooth mirror-tiled content — and the nonrd
  ESTIMATE arm could not code a non-square leaf: for `BLOCK_16X8` etc. `max_txsize_lookup`
  (`common_data.h:105`) gives the square tx of the SHORT side, so
  `av1_foreach_transformed_block_in_plane` visits TWO txbs and `nonrd_pick_intra_mode` was written
  around a single-txb invariant. **The refusal's own reachability claim — "of 18 large cells
  probed at speeds 8 and 9 (768² through 5472x3648) NONE reach a non-square leaf; the 108 MP cell
  is the only one in the tree that does" — was FALSE, and is playbook §9 written by the same
  session that had just been bitten by §9.** KB-34's sweep found **609 of 884 partial-superblock
  rows reaching it, the smallest a 100x100 thumbnail**, and 768² (the first size the claim's own
  range covers) reaches at cq32 cpu9. The arm landed 2026-08-02 (KB-34) and the 108 MP cell is
  byte-identical.
- **Gates.** New `aom-bench/tests/kb32_nonrd_size_bands.rs`:
  `nonrd_speed9_area_threshold_byte_identical` (default tier, 4 cells, hard byte asserts +
  non-vacuity that the grid straddles both predicates), `nonrd_speed9_4k_cost_upd_sbrow`
  (`--ignored`: 2112² byte-exact control + 2176², **promoted to a hard byte gate 2026-08-02**,
  was pinned open with a 1,000 B bound), `nonrd_speed8_size_ladder_residual_is_bounded`
  (`--ignored`: the 5-cell ladder, **now a hard byte gate at every size**; it was pinned on the
  SHAPE — worst armed-cell residual < 1.0 B/SB against a pre-fix 1.06-2.52 and rising),
  `estimate_arm_residual_is_a_leaf_mode_near_tie` (`--ignored`, the localizer above, now
  asserting byte-identity and kept for the diagnosis it prints if a cell returns).
  Unit locks: `speed_features::tests::kb32_force_large_partition_blocks_intra_arm` (speeds 0..9 x
  the 719/720 boundary on both axes, both directions, whole-struct check) and
  `var_part::tests::kb32_force_large_intra_threshold_arms` (both arms x both sides of
  `RESOLUTION_720P`, plus `kb32_shift_steps_floor_is_asserted` for C's `assert(shift_steps >= 0)`).
  Re-pinned: `kb31_mandatory_tiles::issue6_reported_sizes_encode` (20 MP bound tightened
  0.01 -> 0.001; the 108 MP refusal pinned by message, and re-pinned the moment it encodes),
  `mandatory_tile_split_byte_identical_across_speeds` (`OPEN` unchanged — those cells are 64 px on
  the short side, below the speed feature's 720),
  `config_permutations::speed_class_inventory_is_pinned` + `speed_axis_teeth_are_real` (the
  ALLINTRA speed-feature class partition moved from `{0}..{6} {7,8,9}` to `{0}..{6} {7,9} {8}` —
  speed 8 now stands alone because it is the only speed whose `var_part_split_threshold_shift`
  is 8), and `speed_features::tests::framesize_dependent_min_partition_size_4k_arm`'s
  whole-struct expectation (2160² is also >= 720, so the new arm fires there at speed >= 8).
- **Regression envelope, same box** (aarch64-apple-darwin, `--profile test-fast`,
  `AOM_CONFORMANCE_DIR` provisioned): see the commit message for the counts, in both dispatch
  modes.

### KB-17 — Encoder: `use_screen_content_tools` was hardcoded `false`, so `--use-intra-default-tx-only=1` diverged on ALL screen-detected content — FIXED ✅ 2026-07-30
- **Root cause (found 2026-07-30, one line):** `crates/aom-encode/src/speed_features.rs`'s
  `tx_type_search_policy_for_stage` hardcoded `use_screen_content_tools: false` with the
  comment *"Non-screen textured envelope; screen-content would thread the real
  cpi->use_screen_content_tools here."* The rest of the
  plumbing is CORRECT and already present — `TxTypeSearchPolicy::use_screen_content_tools`
  (`tx_search.rs:100`) exists and `get_default_tx_type_y` (`:161`) consumes it faithfully — the
  caller just pinned it false. So where C resolves `--use-intra-default-tx-only=1` through
  `get_default_tx_type(PLANE_TYPE_Y, xd, tx_size, cpi->use_screen_content_tools)`
  (`tx_search.c:1806-1808`) and searches `DCT_DCT`, the port searched the mode-derived type.
- **FIX:** `SpeedFeatures` carries `allow_screen_content_tools` — the flag `set_allintra`
  ALREADY took as an argument (it branches `intra_cnn_based_part_prune_level`,
  `prune_rectangular_split_based_on_qidx`, `prune_sub_8x8_partition_level` on it) and which
  every caller sources from the parsed frame header's `allow_screen_content_tools`.
  `tx_type_search_policy_for_stage` now hands that to the policy.
  **Why the frame header is a faithful source for `cpi->use_screen_content_tools`:** in the
  encoder the two are always equal. Every site that writes one writes the other —
  `estimate_screen_content` (`encoder.c:2096`), the detection-mode-2 variant (`:2419`),
  `screen_content_tools_determination` (`encoder_utils.c:1173/1180`, whose two branches set
  both together, and whose caller `av1_determine_sc_tools_with_encoding` returns early when
  the flag is already set, so it can only flip false→true). The ONE branch that would
  desynchronise them — `av1_set_screen_content_options`'s `force_screen_content_tools != 2`
  early return (`encoder.c:2442-2446`) sets `features->allow_screen_content_tools` without
  touching `cpi->use_screen_content_tools` — is unreachable from the encoder, because
  `seq->force_screen_content_tools` is hard-set to 2 at `encoder.c:598/607`. This reasoning is
  recorded on the field's doc comment.
- **VERIFIED (measured 2026-07-30, all counts from the same tree):**
  - `combinations_screen_dtxo_verdict_set_pinned` (t=2 array x `dtxo=1` forced on, `scr_ibc_b8`
    cq32): **12 of 17 rows diverged → 0 of 17.** `SCREEN_DTXO_DIVERGENT_ROWS` is now empty (the
    pre-fix row list is retained in its doc comment so a regression is recognisable by shape).
  - `CONTENT_DIVERGENT_CELLS`: **9 cells → 1.** All 8 `dtxo=1` cells across `scr_mono_b8`,
    `scr_mono_b10` and `scr_ibc_b8` at cq12/32/63 are byte-identical.
  - `mono_vector_open_divergences_pinned` (`av1-1-b10-24-monochrome`): `dtxo` at cq12/32/63 now
    **exact at 623 B / 240 B / 15 B** (was DIVERGE 623/623, 229/240, 14/15).
  - `dtxo` is PROMOTED out of `pin_dtxo_default` (deleted) — the screen covering arrays now run
    all 21 axes at full strength, so `dtxo x anything` on screen content carries the same t-way
    guarantee as every other axis.
  - Bite proof: restoring `use_screen_content_tools: false` fails **five** pinned tests —
    `combinations_screen_dtxo_verdict_set_pinned` (*"the screen-content
    --use-intra-default-tx-only divergence set MOVED. It is EMPTY since KB-17 was fixed, so any
    row here is a REGRESSION..."*), `mono_vector_open_divergences_pinned`, and all three
    `content_sensitivity_screen_*` shards (*"...any `dtxo=1` cell reappearing here means the
    screen-content tx-type flag ... stopped being threaded"*).
- **ENVELOPE (the acceptance gate for all three of KB-17/18/19, one run, aarch64-apple-darwin,
  `--profile test-fast`, `AOM_CONFORMANCE_DIR` provisioned):** `-p zenav1-aom-bench
  --no-fail-fast` **146 passed / 0 failed / 3 ignored** (unchanged); `-p zenav1-aom-encode
  --no-fail-fast` **272 passed / 1 failed** — the pass count is 271 + the new KB-19 unit test,
  and the single failure is `hog_prune_diff::hog_nn_predict_matches_avx2_and_dispatch`
  (2 ULP, `-0.9569025` vs `-0.9569026`), the named residual of KB-ARM-FLOAT root #2, not mine;
  `-p zenav1-aom-dsp` **361 passed / 0 failed** (unchanged). The full
  `config_permutations` gate is **62 passed / 0 failed** with the re-pins.
- **RESIDUAL, pinned open, PRE-EXISTING (not caused by this fix):** one t=4 covering-array row
  on `scr_ibc_b8` cq32 still diverges —
  `p140-minp16-maxp64-smth0-diag0-flip0-dtxo1-txss0-cdf0`, **port 108 B vs C 79 B**. It only
  became visible because the fix let `dtxo` out of its pin. Direct A/B on the same binary
  toggling only this flag: **23 of 63 open without the fix (this row at 109 B vs 79 B), 1 of 63
  with it** — so the fix closed 22 of 23 and this row was already divergent. Its `dtxo=0`
  sibling is exact, so it IS a `dtxo x <something>` interaction (the row also carries `txss0` =
  `--enable-tx-size-search=0` → TX_MODE_LARGEST, `maxp64`, `minp16`, `smth0`, `diag0`, `flip0`,
  `cdf0`). The 29-byte gap is far outside the KB-10/KB-12 near-tie signature, so treat it as a
  real second defect on the screen tx-type path, not a tie. Pinned self-promoting in
  `SCREEN_ARRAY_OPEN_ROWS` (`config_permutations.rs`).
- **SCOPE CORRECTION.** This entry originally read "the corpus's native monochrome vector" —
  that was the accidental discovery context, not the phenomenon. The real trigger is **any
  content on which `estimate_screen_content` fires** (`encoder.c:2042-2100`, the STANDARD
  detector — `screen_detection_mode` defaults to `AOM_SCREEN_DETECTION_STANDARD` at
  `av1_cx_iface.c:405`; the AA-aware variant is TUNE_IQ/SSIMULACRA2-only). Reproduced on
  `av1-1-b8-24-monochrome` (bd8 4:0:0) and `av1-1-b8-16-intra_only-intrabc-extreme-dv`
  (**bd8 4:2:0, NOT monochrome**), so it is neither a bd10 nor a monochrome phenomenon.
  The original isolation work was sound — its mono-ised *natural* controls simply do not trip
  the detector, so they were byte-exact for the right reason.
- **The detector is a countable statistic, not a vibe:** fraction of full 16x16 luma blocks
  with 2..4 distinct `pix >> (bd-8)` values >= 10%. It is bit-depth independent by
  construction (`av1_count_colors_highbd`, `intra_mode_search.c:352-357`, down-converts to the
  8-bit domain before binning), which is what makes this CONTENT rather than smuggled FORMAT.
  Counter-intuitive and worth keeping: **DC-flat content does NOT trigger it** (0/16 blocks) —
  quantisation ringing keeps colour counts high, so perceptual flatness is not libaom's
  colour-count statistic.
- **`--enable-diagonal-intra=0` is NOT part of this class — CONFIRMED by the fix.** It was in
  the original entry as a second diverging knob; measurement showed one cell only
  (`scr_mono_b10` cq32) and the bd8 twin of the same clip does not reproduce it. **Re-measured
  on the fixed tree (2026-07-30): it did NOT move — still port 225 B vs C 231 B**, while every
  `dtxo` cell on the same content, same run, flipped to exact. That settles it as a near-tie
  coincidence of the KB-10/KB-12 "cheaper RD decision" family, not a screen-content-tools
  consequence. It stays pinned open (the sole remaining entry of `CONTENT_DIVERGENT_CELLS`).
- **Scale of the blast radius:** measured across a 936-cell content grid (12 content probes x
  3 quality x 26 axis levels), **exactly 1 of the 21 knob axes was content-sensitive** — this
  one. The other 19 (+ the diag near-tie) show zero divergences across bd8/bd10, 4:2:0/4:0:0,
  natural/noise/flat/detail/screen. Stock (all-default) encodes stayed byte-exact, and the knob
  is off by default, so the shipped envelope was never affected — which is why this was a
  correctness debt on a reachable configuration rather than a shipping-path regression.
- **Pinned self-promoting** in both directions by `check_content_shard` +
  `combinations_screen_dtxo_verdict_set_pinned` (`crates/aom-bench/tests/config_permutations.rs`).
  Do not "fix" by widening a tolerance or by excluding screen content from the matrix.
- **Related gap, larger than this bug:** palette and intrabc are forced OFF by
  `EncodeCell::c_encode_ctrls`, so the *biggest* consequence of screen detection sits outside
  the permutation matrix entirely. `EncodeCell::c_encode_screen` and `ToggleKnobs::enable_palette`
  already exist; the port's palette RD search needs gating into `port_encode_with`.

### KB-16 — INTER-ENCODE rung 1 ✅ (the port's OWN search codes the zero-MV P byte-exact, single-SB) + two pinned follow-ups
- **LANDED 2026-07-23.** The inter RD loop is WIRED end-to-end: `PickFrameCfg::inter` →
  per-leaf `InterLeafArgs` (`leaf_pick_sb_modes`: `find_inter_mv_refs` + intra_inter/single-ref/
  skip ctx from the DV grid) → the `rd_pick.rs` step-6b inter arm (inter wins RD ties — C
  evaluates inter first per `av1_default_mode_order`; the intra candidate pays
  `ref_cost_intra_in_inter` on P frames) → `encode_b_intra_dry` inter recon arm (co-located ref
  copy, skip resets, padded plane-bsize chroma extents, `SbEncodeEnv::ref_frame`) → `pack_leaf`
  inter branch (`write_inter_leaf_mode_info`; loud-fail on intra-in-P) with `InterFrameCdfs`
  threaded through `pack_tile_lr`/`pack_sb` + the per-SB INTERNAL_COST_UPD_SB inter cost refresh.
  **GATE (hard byte asserts): `aom-bench/tests/inter_e2e_search.rs`** — frame-1 payload ==
  aomenc at 64² × cq{20,40,60,63} 4:2:0 + cq{20,60} mono through the port's own search, plus
  decode-both pixel identity. KEY envelope byte-inert by construction (`inter: None` everywhere
  pre-existing). Harness: `MultiFrameEncodeCell::port_encode_inter_p`.
- **MEASURED (instrumented sibling C, removed): the §3 GOOD stream codes SB128** at every speed —
  a 64×128 frame is ONE column-cropped 128×128 SB whose root codes a gathered 2-way SPLIT/VERT
  symbol. The old "two-superblock SB64" model in `inter_pack_tile_diff.rs` could never match and
  is superseded (tombstone in file); 64×64 frames are walk-degenerate (why single-SB always
  matched). `port_encode_inter_p` drives the declared SB size.
- **PINNED (`zero_mv_p_own_search_64x128_cropped_sb128_pinned_divergent`) — the port models no
  SWITCHABLE interp-filter rate on inter leaves.** During encoding the frame filter is
  SWITCHABLE (the coded REGULAR is post-hoc); `use_more_sharp_interp=1` (speed_features.c:1139,
  GOOD base every speed, non-boosted) gives SHARP a mul=90 discount in `interpolation_filter_rd`,
  so C's 64×128 VERT candidate lands on SHARP (rs 3931 vs REGULAR 109 at ctx 3) and busts the
  budget at `av1_txfm_search`'s skip guard (250,728,413 > 247,851,864) → C picks SPLIT; the
  port's rs=0 leaf picks VERT (`[d7,a0]` vs `[f2,24,80]`). Fix = the CURVFIT model-rd
  (`MODELRD_TYPE_INTERP_FILTER = MODELRD_CURVFIT`, model_rd.h:31 — NOT the ported lapndz) +
  `av1_get_pred_context_switchable_interp` + switchable-interp costs + the reduced zero-MV interp
  compare; then rs joins every inter leaf's mode_rate (C's 64×64 leaves pay 109 each too).
- **PINNED (`good_usage_key_frame0_pinned_divergent`) — GOOD-usage (usage=0) KEY encode is NOT
  byte-exact** (64² cq60: header 2 bytes longer + mid-tile divergence). Every landed KEY byte
  gate is ALLINTRA; chunk-0's "frame-0 control" was decode-side. A GOOD-vs-ALLINTRA speed-feature/
  header gap, independent of the inter wiring; blocks full-STREAM identity of the 2-frame clip.
- **Correction: TX_MODE_LARGEST is a POST-encode demotion** (`txb_split_count==0`); the SEARCH
  runs TX_MODE_SELECT, so a coeff-arm rung needs the var-tx machinery even under a LARGEST header.
- **INTERP-RATE MODEL LANDED (same session)** — `crates/aom-encode/src/interp_rd.rs`: the
  CURVFIT model-rd core `av1_model_rd_curvfit` (tables via `xtask/transcribe_curvfit.py`;
  **differential-locked bit-exact vs the REAL exported C fn**, `curvfit_diff.rs`, 281k cases),
  `model_rd_with_curvfit`, `SwitchableInterpCosts` (DEFAULT_SWITCHABLE_INTERP; the §3 frame
  writes no filter symbols so the frame-init table is the per-SB refresh fixpoint), and
  `pick_interp_filter_zero_mv` (the reduced zero-MV filter search incl. the SHARP mul=90
  accept). Wired: per-leaf `get_pred_context_switchable_interp` from the DV grid (which now
  stamps each inter block's winner filter — `DvCell::interp_filter`), rs joins the inter leaf's
  mode_rate + skip-guard rd. GOTCHA fixed en route: `SWITCHABLE_INTERP_RATE_FACTOR` is **1**
  (rd.h:58), not 2. **The 64x128 cropped-SB128 cell FLIPPED to BYTE-MATCH and is promoted**
  (`zero_mv_p_own_search_64x128_cropped_sb128_byte_exact`, + the cq20 64x128 cell); mono cq40
  also closed. Gate map now: 7 hard byte cells {64² cq63 420; 64² cq40/60/63 mono; 64² cq60
  420; 64×128 cq60 420} + measured-parity of per-block filter/rs at cq20/cq60 vs instrumented C.
- **PINNED (`zero_mv_p_low_cq_term_none_prune_pinned_divergent`) — the next missing C model:
  `av1_simple_motion_search_term_none`** (partition_strategy.c:809; a LINEAR model over
  simple-motion-search + NONE-rd features setting `terminate_partition_search` after NONE;
  GOOD `!frame_is_intra_only`, LIVE at speed 0). Measured at 64² cq20: C terminates after NONE
  (rd 2,509,154, SHARP rs in the NONE rate) and never searches SPLIT; the port's SPLIT (cheap
  ctx-0 REGULAR children) wins → 3 low-cq cells diverge {420 cq20, 420 cq40, mono cq20}. These
  cells matched BEFORE the rs model — a two-wrongs-cancel (rs=0 everywhere also kept NONE on
  top), so the pre-model "pass" hid BOTH missing C models. Next rung: the sms feature
  extraction (`simple_motion_search_prune_part_features` over the ported full-pel search) + the
  per-bsize linear term/prune models (also covers the measured-firing
  `simple_motion_search_prune_rect` at interior nodes).
  Full detail + next-rung plan: `INTER-CHUNK2-HANDOFF.md` §SESSION 2026-07-23.

### KB-33 — Decoder: conformant IntraBC streams from a NON-libaom encoder were rejected — FIXED ✅ (by KB-29 roots 4+5), re-measured 37/100 → 0/1992, gated (GitHub #5)
- **Reported 2026-07-23** (GitHub #5): **37 of 100** real SVT-AV1 v4.2.0 C screen-content
  encodes were rejected by `aom-decode` with `corrupt frame: intrabc DV failed validity
  (non-conformant stream)` (`assign_and_validate_dv`) while real libaom 3.14.1 decoded every
  one of them clean. Gate 1 could not see it: **every intrabc conformance vector is
  libaom-encoded**, and libaom's own encoder never emits the neighbour configurations that
  tripped it.
- **RE-MEASURED 2026-08-02 on `main` (`5925983`): 0 of 1,992 rejected, 0 pixel differences**
  — 10 gb82-sc sources × 50 square crops + 28 geometry cells × presets {0,1,2,3,6,9} ×
  quantizers {15,20,30,35,45,48,58} × 1x1 AND {2x2, 4x1, 4x2} tile grids, all real SVT-AV1
  v4.2.0 C encodes at `screen_content_mode = 1`. Non-vacuity measured with the REAL libaom
  `inspect` (`CONFIG_INSPECTION=1`, `-ibc -bs`): **1,328 of 2,042 decodes carry IntraBC,
  3,100,958 IntraBC mi units**, shapes spanning BLOCK_4X4/4X8/8X4/4X16/16X4 through
  BLOCK_64X64. Record: `benchmarks/svt_interop_2026-08-02.{md,tsv,meta}`.
- **THE ISSUE'S NAMED HYPOTHESIS IS REFUTED — do not re-chase it.** It was NOT a `dv_ref`
  derivation divergence. `find_dv_ref_mvs` / `find_ref_dv` / `is_mv_valid` / `is_dv_valid` and
  the `assign_dv` composition (`aom-dsp/src/entropy/dv_ref.rs`) were read line-for-line against
  `mvref_common.h:267-338` + `decodemv.c:677-731` and transcribe exactly — tile bounds, the
  sub-8x8 chroma case, `total_sb64_per_row`, the `INTRABC_DELAY_SB64` ordering gate, the
  wavefront gradient, and C's truncate-toward-zero division. **The cause was a tile-payload
  desync from missing/misordered COEFFICIENT reads** — KB-29 decoder roots 4 (the 64×64 chunk
  walk, containing the chroma read, gated on `do_uniform`) and 5 (the leaf-vs-raster var-tx
  walk selected on "leaf SIZES differ" instead of "the quadtree was READ"). Reverting EITHER
  alone reproduces the issue's message verbatim on a real SVT stream with the DV code
  untouched. **Third time this exact trap has been sprung** (KB-29, this) — see
  `DIFFERENTIAL_PLAYBOOK.md` §10: an intrabc-DV rejection means "the stream went wrong at or
  before here", never "the DV is what is wrong".
- **Class: port-only, decoder-side.** libaom is right; the port's walk over a correct C
  algorithm was wrong. Not a port-fidelity defect in the DV math.
- **THE GATE — `crates/aom-bench/tests/svt_interop_decode_gate.rs` (new), `just
  gate-svt-interop`.** Four committed REAL SVT-AV1 C encodes (8.7 KB total, in
  `crates/aom-bench/tests/fixtures/svt_interop/`), each asserted: length+FNV-1a-64 pinned to
  the committed IntraBC census, accepted by the REAL `aom_codec_av1_dx`, accepted by the port,
  **pixel-identical between the two**, and carrying the tile grid the table claims; plus an
  optional `dav1d` leg wired in the justfile (caller owns the skip). The fixtures cover
  BLOCK_64X64 IntraBC (the 64×64 chunk boundary), non-square 16X8/32X16/16X32, 4-px-side
  BLOCK_4X4/8X4, and a **4x2 tile grid** — `av1_is_dv_valid` reads `tile->mi_col_start/end`
  directly, so multi-tile is load-bearing coverage, not decoration.
  **This is the coverage class the whole suite was missing: a conformant stream from an
  encoder that is not libaom and not this port.**
- **TEETH (verified, `benchmarks/svt_interop_2026-08-02.md`):** gate FAILS with KB-29 root 4
  reverted, FAILS with root 5 reverted, PASSES on `main`. Root 6 (the leaf-arm CfL luma store)
  is **NOT covered** — its `!chroma_ref || uv_mode == UV_CFL_PRED` condition is not satisfied
  by any fixture's IntraBC blocks, though the leaf arm itself IS exercised (instrumented count:
  70 IntraBC blocks, 40 with `do_uniform == false`). Stated, not hidden.
- **Two SVT facts that BOUND the result** (re-verify before reading it as breadth): SVT emits
  IntraBC only at **presets ≤ 3** (presets 6/9 produce zero IntraBC blocks at every cell
  measured — those streams are vacuous for this bug), and SVT never emits an IntraBC block
  larger than **BLOCK_64X64** here, so KB-29 residual (a) (>64×64 multi-chunk) stays
  unreachable and untested.
- **Discrepancy with the issue's stated repro, recorded:** it names "a 512² crop of
  `gb82-sc/windows95.png`", but that source is **640x480** — no 512² crop of it exists. The
  original artefacts (`/root/aom-rs-oracle-reject-repro-2026-07-23/`) are a Linux path absent
  from this machine, so this is a REGENERATION, not a replay of the original bytes. That is the
  one honest gap.
- **Harness (committed, reusable): `scripts/svt_interop/`** — `svt_scc_encode.c` (SVT-AV1 C
  still-picture driver with `screen_content_mode` + `SVT_TILE_COLS`/`SVT_TILE_ROWS`),
  `build.sh` (out-of-tree, never modifies the `zenav1-svt-c` sibling), `gen_corpus.sh`
  (`TILESET=square|geom`). Probe: `crates/aom-bench/examples/svt_interop_probe.rs` (dual-decode
  + pixel compare over a manifest). `xtool prep` gained an additive `at:WxH+X+Y` explicit-offset
  crop mode; `native` / `crop:WxH` / `square:N` are byte-unchanged.

### KB-34 — Encoder: the fastest preset REFUSED ordinary images — the nonrd estimate arm could not code a NON-SQUARE leaf — FIXED ✅ 2026-08-02 (two roots), 108 MP cell byte-identical
- **Symptom.** `crates/aom-encode/src/nonrd_pickmode.rs:1135` panicked with *"HANDOFF: nonrd
  estimate arm at non-square leaf bsize {bsize} — `max_txsize_lookup` gives a tx smaller than the
  leaf, so `av1_foreach_transformed_block_in_plane` visits more than one txb and
  `nonrd_pick_intra_mode`'s single-txb invariant does not hold (KB-32)"*, so **`--cpu-used 9`
  could not encode a frame that reached one at all**. Introduced by KB-32 (correct thresholds are
  LARGER thresholds, which is what lets `set_vt_partitioning`'s HORZ/VERT pair arms win).
- **ITS OWN REACHABILITY COMMENT WAS FALSE, and that is the most transferable part.** It read
  *"REACHABILITY, MEASURED 2026-08-01: of 18 large cells probed at speeds 8 and 9 (768² through
  5472x3648), NONE reach a non-square leaf. The only cell in the tree that does is issue #6's
  12000x9000 at cpu9."* Playbook §9 in its purest form — and written by the session that had just
  been bitten by §9 twice (the `var_part` module doc, then the "stamps squares only" claim). It
  was already contradicted twice before this landing (KB-28's two 0.9 MP cells; the encoder
  hotspot profile's 1024² cq44). Measured properly:
- **THE TRUE REACHABLE SET, as a shape (2,012 sweep rows,
  `benchmarks/nonsquare_leaf_reach_2026-08-02.tsv` + `.meta`).** The predictor is neither size nor
  quality; it is whether the frame has a **partial superblock**. `set_vt_partitioning` fits a
  candidate by `mi_col + bs_width_check <= tile->mi_col_end` and at the frame's right/bottom edge
  (SB64 only) relaxes the two checks ASYMMETRICALLY — `bs_width_check` to `(block_width >> 1) + 1`
  but `bs_width_vert_check` to `(block_width >> 2) + 1` (var_based_part.c:164-173) — so an edge
  node's NONE candidate stops fitting while its VERT/HORZ pair still does, and a rect gets
  stamped. `av1_select_sb_size` (encoder_utils.c:958) picks which superblock: 64x64 at
  `min(w,h) <= 480`, 64x64 again at allintra speed 9 below 4k, 128x128 otherwise.

  | frame class | rows reaching | smallest reaching |
  |---|---|---|
  | mi-aligned extent NOT a whole number of SBs | **609 / 884 = 68.9 %** | **100x100 = 10,000 px** |
  | mi-aligned extent IS a whole number of SBs | 18 / 1088 = 1.7 % | 589,824 px (768²) |

  So the refusal was wrong by **four orders of magnitude in area** and ~600 cells, and wrong in a
  way no size threshold would have caught: **a 100x100 thumbnail reaches it while 512², 1024² and
  2176² mostly do not.** Any frame whose dimensions are not a whole number of superblocks —
  1920x1080, 1280x720, essentially every non-multiple-of-64 crop — is in the reaching class. The
  1.7 % SB-exact column is the second, rarer route: a genuine interior variance win, needing
  locally flat content (only the photographic source produced it here). **A `min(w,h) >= 720`
  hypothesis fitted the first sweep exactly and is FALSE** — 1272x716 reaches 22 leaves at cq24
  cpu9; recorded because it is the same mistake one sweep later.
  Exactly four shapes occur — BLOCK_8X16 / 16X8 / 16X32 / 32X16 — which is what
  `set_vt_partitioning` predicts (`bsize > BLOCK_32X32` returns 0 on a key frame, :205-209, so no
  64X32/32X64; `bsize == bsize_min` offers only NONE-or-split, :186-199, so nothing below 8x8).
  **No RD speed reaches it**: `nonrd_pick_intra_mode` has one dispatch site,
  `pack.rs:1917`'s `allintra && speed >= 8` — measured, not asserted.
- **ROOT #1 — the txb walk (`nonrd_pickmode.rs`).** `nonrd_leaf_tx_size` is now
  `max_txsize_lookup[]` verbatim at every `bsize`, and `nonrd_pick_intra_mode` runs C's real
  `av1_foreach_transformed_block_in_plane` (encodemb.c:536-585) instead of one inlined visit.
  Three details of C's walk that a naive txb loop gets wrong, each modelled and cited in the code:
  * each visit predicts into `pd->dst` **before** the next visit reads its neighbours out of that
    same buffer (`av1_predict_intra_block_facade`'s `ref == dst`, reconintra.c:1622) — so txb 1 of
    a BLOCK_8X16 predicts from txb 0's *prediction*, there being no residual on this arm;
  * `av1_block_yrd` is handed `bsize_tx = txsize_to_bsize[tx_size]` (nonrd_opt.c:658) so its
    `num_4x4_w/h` are the TXB's — but `xd->mb_to_right_edge` is still the LEAF's, so the frame-edge
    clamp at nonrd_opt.c:141-144 subtracts the leaf's overhang from EACH txb's extent (and can
    clamp a txb to zero rows, which C then codes as rate 0 / dist 0 / skippable). The WALK's own
    clamp is a different formula (`max_block_wide`, av1_common_int.h:1567, which reduces to
    `mi_cols - mi_col`); at a square leaf the two coincide, at a rect they do not;
  * `args->skippable` is **assigned, not accumulated**: `av1_block_yrd` ends
    `this_rdc->skip_txfm = *skippable = temp_skippable` (nonrd_opt.c:327) with `temp_skippable`
    restarting at 1 each call, so a multi-txb leaf's flag is the LAST txb's, not the AND. Rate and
    dist DO accumulate (:667-668).
  The SAD prune stays single-txb by construction — C gates it on `bsize == tx_bsize`
  (nonrd_pickmode.c:1600). Byte-INERT at every square leaf.
- **ROOT #2 — the frame-edge single-strip rect constructor (`partition_pick.rs`), exposed the
  moment root #1 landed.** `nonrd_use_partition_real`'s rect arm carried
  `unimplemented!("frame-edge single-strip nonrd rect ...")` under *"the SbTree rect variants carry
  both winners"* — **the exact claim, in the same words, that KB-25 had already deleted from the
  speed-7 walk on 2026-08-01**. It survived only because every cell that could reach it hit root
  #1's panic first, in the leaf pick one line above. Same fix as KB-25: slot 1 gets a POISONED
  clone of sub 0 (`bsize = usize::MAX`) so the four consumers' `debug_assert_eq!(s1.bsize,
  subsize)` fires if any of them ever drops its frame gate. The OTHER half of C's guard —
  sub 1 in frame but `bsize == BLOCK_8X8` (partition_search.c:3046/:3070) — stays a hard refusal,
  because there the consumers' gate PASSES and a poison would be a wrong stream; it is unreachable
  (the KEY tree offers no rect at `bsize == bsize_min`).
- **PER-ROOT BITE PROOF, ordered cell sets (playbook §1),
  `benchmarks/nonsquare_leaf_reach_bite_2026-08-02.tsv`.** Four arms over one 26-cell list:
  pristine **14 PANIC**, both fixes **0**, revert-root-1-alone **14** (the same 14), revert-root-2-alone
  **9** — a strict subset. The five cells that separate them are the SB-EXACT frames reaching an
  INTERIOR rect (768² cq32, 896² cq28, 1024² cq24/cq36 and the encoder profile's 1024² cq44, all
  cpu9): both strips are in frame there, so they need the txb walk and NOT the rect constructor.
  Every non-panicking byte count is identical across all four arms.
- **TEETH.** Every gate cell panics on the pristine tree with the exact message quoted above (or
  root #2's, for 1920x1080 cq24 cpu8), and `250x250 cq24 cpu9` — mi-aligned to 256 px, i.e.
  SB-exact — MATCHES in both arms, which is the negative control that the harness is not simply
  reporting "panic" for everything.
- **BYTE-IDENTITY on every newly-encodable cell.** `nonsquare_leaf_reach` sweep: **1,859 MATCH /
  2,012 rows, and 0 of the 627 reaching rows is anything but MATCH**. Named cells: KB-28's `1272x724 cq24 cpu9`
  and `954x962 cq24 cpu9`; `1920x1080` cq24 cpu8 and cq48 cpu9; `1280x720` diag cq24/cq48 cpu9;
  `196x196` cq24 cpu8/cpu9; the encoder profile's `1024x1024 cq44 cpu-used 9` photograph (**4,728
  B, delta 0** — the cell `benchmarks/encoder_hotspot_profile_2026-08-02.md` reported as refusing;
  the GATE carries that size/quantizer/speed on in-repo content instead, the mirror-tiled
  `av1-1-b8-00-quantizer-58` decode, 42 leaves, byte-identical — an SB-exact INTERIOR-rect cell,
  found by the `NSQ_VECTOR_SCAN=1` pass);
  and **issue #6's `12000x9000` 108 MP cell, 11,520,317 B, delta 0** (both issue-#6 sizes together:
  14.2 s, 2.69 GB peak RSS).
- **Gates.** New `aom-bench/tests/kb34_nonsquare_nonrd_leaf.rs`:
  `partial_superblocks_are_what_reaches_it_not_frame_size` (**default tier**, 4 cells at
  10k-62k px: 100x100 and 196x196 reach, 128x128 and 250x250 do not — the shape, and the proof the
  counter can read zero), `nonsquare_leaf_cells_byte_match` (`--ignored`, 7 cells, byte-identity +
  a per-cell "this cell reached the arm" non-vacuity assert + a shape-coverage assert that only
  the four KEY rect bsizes ever appear), `rd_speeds_never_reach_the_estimate_arm` (`--ignored`,
  cpu 0..7 + a cpu9 control). Non-vacuity instrument:
  `nonrd_pickmode::multi_txb_leaf_counts()` / `reset_multi_txb_leaf_counts()`, a per-`bsize`
  relaxed-atomic counter bumped only on the multi-txb path. Unit locks:
  `nonrd_leaf_tx_size_is_the_largest_square_tx_that_fits` (derives `max_txsize_lookup`'s MEANING
  from `MI_W`/`MI_H` rather than re-typing the table; also pins that single-txb means "square AND
  <= 64 px", not "square" — BLOCK_128X128 is four txbs) and
  `kb34_key_rect_leaves_are_two_txbs_each`. Re-pinned: `kb31_mandatory_tiles::
  issue6_reported_sizes_encode` (its self-promoting `assert_ne!` fired — the 108 MP refusal branch
  is deleted and both sizes are hard byte gates) and `kb28_crop_dims::vbp_band_crop_dims_byte_match`
  (**28/30 → 30/30**, `NONRD_ESTIMATE_ARM_OPEN` now empty).
- **FOUND WHILE HERE, NOT THIS LANDING'S, pinned rather than smoothed over.** Two open divergence
  classes the sweep turned up, both measured byte-for-byte identical with this landing's two hunks
  stashed, and both on cells that reach ZERO non-square leaves:
  * **`RD_BAND_OPEN`** (pinned in `kb34_nonsquare_nonrd_leaf.rs`): 1272x724 cq24 diverges at
    `--cpu-used` 2/3/4/5 by −14/−104/−167/−189 B and matches at 0, 1, 6, 7, 8, 9. Adjacent to
    KB-28's `rd_band_min_dim_tiers_byte_match` band (474×480, 714×720 at cpu 1..6, 12/12) but at a
    size no gate covers;
  * **the cpu-8 photographic high-q class**: 17 rows of the sweep DIVERGE, all `photo` content, all
    `--cpu-used 8`, all cq 32-63, by −24..+13 B (512²/768²/896²/1024²/2176²). Not on any in-repo
    content, so not reachable by any existing gate; recorded in the sweep TSV.
  Gate 2 (the canon `--cpu-used 0..9` grid) is unaffected and still has **zero pinned cells**.
- **Still refused at `--cpu-used` 8/9, unchanged and pre-existing:** the screen-content palette arm
  (`av1_search_palette_mode_luma`, `nonrd_pickmode.rs`'s `allow_screen_content_tools` debug-assert)
  — all 136 remaining PANIC rows in the sweep, fired by libaom's own screen-content detection on
  the synthetic gradient and on low-q real content. Also unchanged: the lossless TX_4X4 arm
  (`block_yrd_lowbd`/`_hbd`'s `unimplemented!`) and the HBD estimate arm's own gaps.
  **CORRECTED 2026-08-03 — those 136 rows were NOT the palette arm.** The guard tested
  `allow_screen_content_tools`, one of the four terms of C's `try_palette`, and the C oracle on
  every one of those cells passes `--enable-palette=0`, so libaom provably never entered the
  palette search there. See KB-35: the refusal is now C's predicate, 22 of the 25 measured
  PANIC rows are byte-identical, and the 3 that remain are the genuine arm. The lossless TX_4X4
  and HBD sentences stand — **with one measured correction to how the lossless one reads**:
  `--cq-level 0` never reaches the encoder's `block_yrd_lowbd`/`_hbd` `unimplemented!` through
  this harness at all, because `EncodeCell::port_encode_full` refuses it one layer earlier
  (`crates/aom-bench/src/lib.rs:1153`, *"lossless cells are out of this harness's scope"*).
  Measured 2026-08-03 on a 64x64 real-content cell: cq0 PANICs with the HARNESS message at
  `--cpu-used` 0, 4, 8 AND 9, while cq1 is byte-exact at all four. So the lossless gap is not
  a speed-8/9 property — the e2e path is closed at every speed, and lossless byte-parity is
  proven only by KB-5's own driver (`aom-encode/tests/kb5_lossless_localize.rs`), which runs
  `let speed = 0i32`. Reaching the lossless x nonrd crossing means extending the harness, not
  just the encoder.

### KB-35 — Encoder: the nonrd estimate arm's palette refusal fired on the FRAME FLAG, one of four terms of C's `try_palette` — `--cpu-used 8` REFUSED a plain gradient at every size >= 1024x1024 and every quantizer — FIXED ✅ 2026-08-03
- **Found 2026-08-03** by working the "still unmeasured / still refused" queue KB-34 left behind
  (its closing bullet: *"Still refused at `--cpu-used` 8/9, unchanged and pre-existing: the
  screen-content palette arm ... all 136 remaining PANIC rows in the sweep"*). Those 136 rows
  were not a residual of KB-34's landing; they were their own bug, and a bigger one.
- **Symptom.** `crates/aom-encode/src/nonrd_pickmode.rs:1602` (pre-fix line) refused every
  estimate-arm leaf whenever the frame carried `allow_screen_content_tools`:
  *"HANDOFF: av1_search_palette_mode_luma (palette.c) not ported — required before any
  screen-content (allow_screen_content_tools=1) speed-8 cell"*. Measured reach: **136 of the
  2,012 rows of `benchmarks/nonsquare_leaf_reach_2026-08-02.tsv`** — `EncodeCell::synthetic_diag`,
  a SMOOTH GRADIENT, at `--cpu-used 8`, at 1024x1024 / 1272x724 / 1280x720 / 2176x2176, at
  every quantizer from cq2 to cq63.
- **C's `try_palette` (nonrd_pickmode.c:1698-1710) is a conjunction of FOUR terms and the guard
  tested one.** The others are `cpi->oxcf.tool_cfg.enable_palette`, the ordinal size bounds of
  `av1_allow_palette` (blockd.h:1503-1510), and — at `prune_palette_search_nonrd > 0`, which is
  **1 at every speed that dispatches this arm** (speed_features.c:582) — `bsize <= BLOCK_16X16
  && source_variance > 200` plus a SAD term. **The C oracle's own `shim_encode_av1_kf` passes
  `--enable-palette=0`** (`dec_shim.c:614`), so on every one of those 136 cells term 1 is FALSE
  and libaom provably never enters the palette search at all. The port refused where C does
  nothing. This is playbook §9's shape with a twist worth naming: not a comment claiming
  inertness, but a GUARD claiming a superset — an over-broad refusal is exactly as wrong as an
  over-broad inertness claim, it just fails loudly instead of silently.
- **Why it looked like a speed axis and was not** (the part that would have misdirected a
  size/speed-shaped hypothesis, as in KB-34): speed 9 never refused, so the map read "speed 8
  only". The reason has nothing to do with palette — `av1_set_screen_content_options`
  (encoder.c:2466-2470) turns screen-content DETECTION off entirely when
  `use_nonrd_pick_mode && !hybrid_intra_pickmode`, which is speed 9's combination
  (`hybrid_intra_pickmode = 0`) but not speed 8's (`= 2`). At speed 9 the frame flag is 0, so
  the old guard was vacuously satisfied.
- **FIX.** `nonrd_palette_arm_is_live(enable_palette, allow_screen_content_tools, bsize,
  prune_palette_search_nonrd, prune_mode_based_on_sad, best_sad_norm, source_variance)` — C's
  predicate term for term — and the refusal fires on it. Two new `NonrdIntraLeafCtx` fields
  carry the missing inputs: `enable_palette` (from `cfg.palette_costs.is_some()`, the same term
  `PickFrameCfg` already models for the full-RD arm) and `prune_palette_search_nonrd`. **The
  refusal was also STRENGTHENED, not relaxed**: it was a `debug_assert!`, so a release build
  would have silently coded a non-palette winner where C searches palette; it is now a hard
  `assert!`. Both `bsize` comparisons are ORDINAL on the `BLOCK_SIZES_ALL` index, matching C —
  `BLOCK_4X16`/`BLOCK_16X4` satisfy `>= BLOCK_8X8` despite a 4-px side, and are excluded by
  `<= BLOCK_16X16` instead (the zenavif/zenrav1e ordinal-vs-dimensional trap, avoided here by
  construction).
- **MEASURED, both arms, 102 rows** (`benchmarks/kb35_palette_arm_2026-08-03.tsv` + `.meta`;
  the `pristine` arm is this tree with the two hunks `git stash push`-ed):

  | class | rows | pristine | fixed |
  |---|---|---|---|
  | smooth gradient >= 1024x1024, cpu8, screen flag ON, palette OFF | 8 | PANIC | **MATCH** |
  | screen content, cpu8, palette OFF both sides | 9 | PANIC | **MATCH** |
  | smooth gradient <= 512x512, cpu8 (flag off) — the control | 8 | MATCH | MATCH |
  | every cell at cpu9 — the control | 16 | MATCH | MATCH |
  | screen content, cpu8, palette ON both sides | 9 + 4 | PANIC | **DIVERGE** (pinned) |
  | screen content 1024x1024, cq >= 60, cpu8, palette ON | 3 | PANIC | PANIC (**genuine**) |

  **25 pristine PANIC rows -> 3, and all 22 that closed are byte-identical.**
- **THE NARROWING HAS TEETH — the refusal is still reachable, and it was found by measurement,
  not argued.** Screen content at **1024x1024, cq >= 60, cpu8, `--enable-palette=1`** refuses
  with the new message at `bsize == BLOCK_16X16, source_variance = 3140`. The predicate: the
  estimate arm only ever sees `bsize >= BLOCK_16X16` at speed 8 (`hybrid_use_rdopt` sends
  everything smaller to the full-RD leaf), and C's prune caps it at `<= BLOCK_16X16`, so the
  live set is **exactly `bsize == BLOCK_16X16` with `source_variance > 200`**; and what makes
  the QUANTIZER the axis is that `set_vbp_thresholds` scales with qindex, so only at a high
  quantizer does a 16x16 that textured survive the variance split undivided. cq58 on the same
  content at the same size reaches ZERO such leaves — that asymmetry is the gate's control.
- **FOUND WHILE HERE, NOT THIS LANDING'S — `PALETTE_ON_SPEED8_OPEN`, a divergence class the
  over-broad refusal was HIDING.** With `--enable-palette=1` on both sides, screen-detected
  content at cpu8 diverges at every size and quantizer tried (13 rows, -1399..+817 B; the
  -1399 and +817 are cq63/cq40 at 512x512 and 1024x1024). It is **not** this arm: the new
  `palette_gate_reach()[2]` instrument reads **0** on every one of those cells, so no
  estimate-arm leaf satisfies `try_palette` there. That leaves the FULL-RD palette leaf at
  speed 8 (`hybrid_use_rdopt` dispatches `av1_rd_pick_palette_intra_sby` for every
  `bsize < BLOCK_16X16` with `source_variance >= 101`) — a speed crossing of the palette search
  that no gate covers, because `rd_close_palette.rs` is `speed: 0` throughout. Pinned
  self-promoting; its speed-9 twin MATCHes, which is the control that isolates it to the
  speed-8 crossing. Per playbook §10 the next step is the sibling-C per-block dump on the
  smallest divergent cell (128x128 cq12, delta -1), NOT reasoning from the deltas.
- **Gates.** `aom-bench/tests/kb35_nonrd_palette_arm.rs` (3 tests, `--ignored`, 3.1 s total):
  `speed8_screen_detected_cells_byte_match` (26 cells; the screen flag is read out of the REAL
  stream header per row and the grid must straddle it),
  `estimate_arm_palette_refusal_is_reachable_and_loud` (the teeth; asserts the refusing set is
  exactly `{cq60, cq62, cq63}` and that the reach counter and the predicate AGREE on every
  row), `palette_on_speed8_screen_content_is_pinned` (the open class + its speed-9 controls).
  Unit locks in `nonrd_pickmode::tests`: `palette_arm_liveness_is_c_try_palette` (each of the
  four terms varied alone, incl. the ordinal-vs-dimensional rows and the SAD threshold's
  `> 1 ? 100 : 20` level dependence) and `b_log2_matches_c_lookup_tables` (against C's literal
  `b_width_log2_lookup`/`b_height_log2_lookup`, nonrd_opt.h:114-119). Instrument:
  `nonrd_pickmode::palette_gate_reach()` / `reset_palette_gate_reach()`, thread-local for the
  same reason as `multi_txb_leaf_counts`.
- **BITE PROOFS.** (1) Stubbing `nonrd_palette_arm_is_live` back to the frame flag alone fails
  the unit lock with *"bsize 0 is < BLOCK_8X8"* while the rest of the suite stays green.
  (2) Stashing the two hunks makes 25 of the gate's rows PANIC with the OLD message, while
  the 23 control rows (<= 512x512 gradient, and every cpu9 row) still MATCH — the fix is not
  "everything passes now".
- **VERIFIED.** `-p zenav1-aom-encode -p zenav1-aom-bench` **489 passed / 0 failed / 31 ignored**
  in both dispatch modes (default and `AOM_FORCE_SCALAR=1`). Gate 2 is unaffected and still has
  zero pinned cells.
- **Still refused at `--cpu-used` 8/9, unchanged:** the genuine palette arm above
  (`av1_search_palette_mode_luma` is not ported for the estimate arm — the RD-path
  `palette_search::rd_pick_palette_intra_sby` is what it needs to be wired to), the lossless
  TX_4X4 arm (`block_yrd_lowbd`/`_hbd`'s `unimplemented!`), and the HBD estimate arm's own gaps.

### KB-36 — Encoder: `default_min_partition_size`'s >=1080p arm was UNMODELLED — every >=1080p frame at `--cpu-used 6` searched 4x4 partitions C had stopped at 8x8 — FIXED ✅ 2026-08-03
- **Found 2026-08-03** by working the axis `s4cov_hd_speed_axis.rs` names in its own first
  paragraph and then does not cross: it raised the "RD speeds above 640x640" ceiling to
  **1280x720** and stopped. Between 1280x720 and KB-19's 2160p cell sits `is_1080p_or_larger`
  (`AOMMIN(cm->width, cm->height) >= 1080`, speed_features.c:171), and nothing in the tree had
  ever encoded a frame in that band at ANY speed other than 0.
- **The arm.** Inside `set_allintra_speed_feature_framesize_dependent`'s `if (speed >= 6)` block:
  `if (is_1080p_or_larger) sf->part_sf.default_min_partition_size = BLOCK_8X8;`
  (speed_features.c:311-313). The port's `apply_allintra_framesize_dependent` carried the
  `is_4k_or_larger` arm on the SAME field (KB-19) and the `speed >= 8 && is_720p_or_larger`
  force-large arm (KB-32), and not this one — so `ToggleKnobs::min_partition_bsize`'s
  `AOMMAX(default_min_partition_size, dim_to_size(--min-partition-size))`
  (`set_max_min_partition_size`, partition_strategy.h:224-226) resolved BLOCK_4X4 where C
  resolves BLOCK_8X8.
- **MEASURED, both arms, 60 rows** (`benchmarks/kb36_above_720p_2026-08-03.tsv` + `.meta`;
  4 sizes x `--cpu-used` 1..7 each arm, plus the fixed tree's speed-8/9 rows):

  | cell | pristine | fixed |
  |---|---|---|
  | **1920x1080 cq24 cpu6** | **DIVERGE -127 B** | MATCH |
  | **2560x1440 cq24 cpu6** | **DIVERGE +79 B** | MATCH |
  | 1920x1072 cq24 cpu6 (8 px shorter, same content) | MATCH | MATCH |
  | every other cell of the 4x7 grid, and speeds 8/9 | MATCH | MATCH |

- **THE WINDOW IS ONE SPEED WIDE, and that is the whole reason nothing caught it.** Speed 7 sets
  the same field framesize-INdependently (`speed_features.c:570`), so speeds 7, 8 and 9 cannot
  show it; below speed 6 the enclosing block does not run. **A speed sweep at any size under
  1080 is green, and a size sweep at any speed other than 6 is green** — it needs the crossing,
  which is the same shape KB-19, KB-22, KB-26 and KB-28 all had. The 1920x1072-vs-1920x1080
  pair is eight pixels of the same mirror-tiled content.
- **This is the SECOND arm found on `default_min_partition_size`**, and KB-19's entry named the
  queue without meaning to: its fix introduced `apply_allintra_framesize_dependent` holding
  *"the modelled arms of `set_allintra_speed_feature_framesize_dependent`, currently just the
  `is_4k_or_larger` one"*. Accurate as written; read as coverage it was not. Two of the three
  arms that function now carries were found by later sweeps (KB-32's, then this one).
- **FIX:** four lines in `SpeedFeatures::apply_allintra_framesize_dependent`. No other consumer
  changes — `ToggleKnobs::min_partition_bsize` already AOMMAXes with the field (KB-19 half 2).
- **Gates.** `aom-bench/tests/kb36_above_720p_speed_axis.rs` (2 tests, `--ignored`, 343 s):
  `is_1080p_arm_straddle_byte_matches` (the 1072/1080 razor at speeds **5, 6, 7** — 5 and 7 are
  what prove the window, so a wrongly-gated fix that fired at every speed would fail here where
  a speed-6-only gate would pass) and `above_720p_speed_axis_byte_matches` (1920x1080 +
  2560x1440 x `--cpu-used` 1..9, 18/18 — the band the old ceiling was hiding, incl. speeds 8/9
  which cross KB-32's force-large arm at a size no gate had used). Unit lock:
  `speed_features::tests::framesize_dependent_min_partition_size_1080p_arm` (speeds 0..9 x both
  sides of 1080 incl. the 1-px row 1920x1079, plus a whole-struct equality check that the arm
  moves no other field).
- **TWO PRE-EXISTING UNIT LOCKS HAD TO BE RE-SCOPED, and neither was weakened.**
  `framesize_dependent_min_partition_size_4k_arm` asserted `1920x1080` "must not take the 4k
  arm" — true of the 4k arm, and now false of the FIELD, so that row became `1920x1079` (one px
  under the other boundary) and the two 2159-short-side rows fold the 1080p expectation in and
  are noted as distinguishing the 4k arm only at speeds 0..5.
  `kb32_force_large_partition_blocks_intra_arm`'s whole-struct isolation size moved
  `1280x1280 -> 1024x1024`, still >= 720 on both sides (its own premise) and now below 1080 as
  well as below 2160, so it isolates its field again.
- **BITE PROOFS.** (1) `if false && speed >= 6 && ...` fails the new unit lock with
  *"speed 6 1080x1080: expected default_min_partition_size = 3"* AND the re-scoped 4k lock,
  while the other 77 unit tests stay green. (2) The e2e bite is the pristine arm of the TSV: the
  same 4x7 grid, two cells divergent, the 8-px-shorter twin byte-exact.
- **VERIFIED.** `-p zenav1-aom-encode -p zenav1-aom-bench` **489 passed / 0 failed / 31 ignored**
  in both dispatch modes. Gate 2 unaffected and still has zero pinned cells.
- **THE WHOLE FUNCTION WAS THEN AUDITED, arm by arm, and the other two survivors are comment
  bugs rather than code bugs — corrected in place (playbook §9).** Every framesize-conditioned
  assignment in `set_allintra_speed_feature_framesize_dependent` (speed_features.c:166-345) was
  enumerated against the port. All are modelled or provably dead **except** two whose in-tree
  inertness note gave a REASON that does not survive, while the conclusion does:
  * **`tx_sf.prune_tx_size_level`** (:184 / :263 / :265 / :289, four assignments across three
    framesize tiers) was dismissed as *"gated on `use_hbd`, false here"*. `use_hbd` is
    `cpi->oxcf.use_highbitdepth` — **TRUE on every bd10/bd12 encode**. Worse, the field's shape
    (a LUMA tx-SIZE prune, hbd-only, framesize-conditioned, live from speed 2) is a near-perfect
    fit for the long-pinned `b10_64` / `HBD_OPEN` band (bd10/bd12, speeds 1..6, LUMA-borne,
    reaching 4:4:4 + mono + bd12), so the wrong reason reads like a lead. It is not one: the
    field's ONLY consumer is `select_tx_block` (tx_search.c:2629-2635), reached solely from
    `select_tx_size_and_type`, which opens `assert(is_inter_block(xd->mi[0]))` (:3438). INTER-only
    **by assertion**, at every bit depth and framesize. So `HBD_OPEN` stays open and this is
    ruled out as its root.
  * **`part_sf.max_intra_bsize`** (:285) was dismissed as *"only `init_mode_skip_mask`'s INTER
    ref-frame mask, rdopt.c:4217"*. It has **three** consumers, not one — that plus
    `skip_intra_modes_in_interframe` (rdopt.c:6000) and `av1_nonrd_pick_inter_mode_sb`'s
    intra-check gate (nonrd_opt.c:795). All three are inter paths, so the conclusion holds; the
    citation was one-third complete and is now enumerated.
  Both notes are rewritten at their sites. The lesson is the one KB-32 and KB-34 already paid
  for, in a milder form: a conclusion can be right while its stated reason is wrong, and the
  wrong reason costs the next session the same audit.
- **Still unmeasured on this axis:** above 1440p at speed >= 1 other than KB-19's single 2160p
  speed-0 cell (a 2160p cpu-1 cell is ~250 s per pair, so the band 1440..2160 at speeds 1-5 is
  ~20-30 min); the >=1080p band at bit depths above 8, at 4:2:2/4:4:4/mono, and at SB128; and
  any quantizer other than cq24 in this band.

### KB-PERF-1 — Encoder: the intra-mode CNN is recomputed ~10x per superblock (C computes it ONCE and caches) — FIXED ✅ 2026-08-02

**Not a correctness bug — a 10x throughput bug, and the largest single measured
item in the whole project.** Full profile, method, control band and ranked
levers: `benchmarks/encoder_hotspot_profile_2026-08-02.md` (+ `.meta`, five TSVs).
**The fix and its measurement: `benchmarks/encoder_cnn_cache_2026-08-02.md`
(+ `.meta`, `.control.tsv`, `.breadth.tsv`) — see "RESULT" at the end of this
entry.**

libaom runs the intra-mode partition CNN **once per BLOCK_64X64 node** and caches
the 1636-float multi-out buffer in `x->part_search_info.cnn_buffer`
(`av1/encoder/partition_strategy.c:160` `if (bsize == BLOCK_64X64 &&
!part_info->cnn_output_valid)`; invalidated per 64x64 at
`partition_search.c:3342`; every smaller node returns at
`if (!part_info->cnn_output_valid) return;` on :227 and runs only the small
branch DNN). The port's `partition_pick.rs:2805` calls
`cnn_partition::decision::predict_decision`, which at `decision.rs:232` calls
`cnn::cnn_predict` **unconditionally**, at every 64x64 / 32x32 / 16x16 / 8x8 node.
`extract_intra_cnn_window` (`partition_pick.rs:2493`) snaps its origin to the
containing 64x64, so **every one of those calls convolves the identical window
and returns the identical buffer.** Output is byte-correct; the work is discarded.

KB-23 modelled the *gating* half of that latch (the frame-edge
`av1_is_whole_blk_in_frame` implication) and is correct; it did NOT model the
*caching* half.

MEASURED, 1024x1024 photo / cq44 / cpu-used 6 (the mode xbench's +15.9 %
photographic BD-rate deficit is measured at), Apple M4 Pro:

* **2558 CNN cascade runs per frame vs libaom's 256** — 9.99 per 64x64 SB vs 1.00.
  Counted exactly by a counting `GlobalAlloc` with exact-size call-site counters
  (`crates/aom-bench/examples/eprof_alloc.rs`).
* **144.5 µs per run** vs libaom's `_c` 125.2 µs (so the Rust transcription is
  within 15 % of the C it copies) and vs libaom's dispatched NEON 16.4 µs
  (`crates/aom-bench/examples/eprof_cnn_bench.rs`).
* The two factors multiply to **85-90x** on that stage, which is
  **74.7 % of the port's entire encode** and **81.5 % of the whole 10.7x gap
  to libaom**. Remove it and the port is **~3x** libaom, not ~10x — confirmed
  independently at `cpu-used` 7 and 8, where the port takes the
  VAR_BASED_PARTITION path, runs the CNN **zero** times, and measures **2.69x**
  and **3.45x**.
* Armed at **every allintra `cpu-used >= 1`** (`partition_pick.rs:2759`), so this
  is not one preset's quirk. Redundancy is worse at smaller frames (26.4/SB at
  256²) and lower quantizers (22.4/SB at cq 10, where the ratio hits 15.2x).

Fix shape: hoist + memoize the cascade at the 64x64 node exactly as
`cnn_output_valid` does, leaving the branch DNN per node. **Byte-identical by
construction** — a byte-identity gate over the existing corpus is the whole
verification. Projected from the measurement: 511.7 ms → 167.6-178.9 ms, i.e.
10.6x → **3.5-3.7x**, which is enough to move the port's `>= 1 MP/s` qualifying
mode from `cpu-used 6` to `cpu-used 5` (measured 1191 ms = 0.88 MP/s today →
936 ms = 1.12 MP/s). Nothing was changed in that session — the numbers above are
measurement plus arithmetic, not a before/after.

Two adjacent findings from the same profile, both recorded in the .md:

1. **`cpu-used 9` REFUSED the 1024²/cq44 cell at `0953fa7`** — panicked in
   `crates/aom-encode/src/nonrd_pickmode.rs:1135` with the KB-32 handoff message
   ("nonrd estimate arm at non-square leaf bsize 4"). `cpu-used 8` on the same
   cell was fine. `xbench_2026-08-01.md` publishes a cpu-used 9 throughput number
   for this exact cell, measured at `ea3bed3`. **DIAGNOSED AND FIXED 2026-08-02 —
   see KB-34**; that exact cell now encodes byte-identically to real aomenc
   (4,728 B), and the refusal turned out to be reachable on ordinary content from
   100x100 up, not only at 108 MP.
2. **libaom's own NEON CNN kernel is not bit-identical to libaom's own `_c`
   kernel** — 906/1636 output floats differ, max |Δ| 5.28e-6 (the
   `docs/LIBAOM_UPSTREAM_NOTES.md` divergence class). The port matches `_c`,
   libaom ships `_neon`, and the frames still come out byte-identical — the CNN's
   threshold comparisons are robust to ~5e-6 *on this corpus*, which is an
   observation, not a proof. Any SIMD rewrite of the port's `conv_valid` must be
   gated on the full byte-identity corpus.

Second-largest lever from the same profile: the port runs the **highbd** forward
transform + intra predictors where libaom runs `lowbd_fwd_txfm2d_*_neon` /
`av1_dr_prediction_z2_neon` — +38.9 ms, i.e. the SAME structural lever that
`benchmarks/gate3_decode_profile_2026-07-19.md` Finding 1 ranked #1 on the
decode side. Third: 870 167 allocator calls and 559.7 MB per 1 MP encode
(3 399 per superblock), +24.1 ms — the encode-side twin of the per-txb allocation
the decoder fixed with `ReconScratch` / `InvTxfmScratch`.

**RE-PROFILED at the post-fix baseline, same day —
`benchmarks/encoder_hotspot_reprofile_2026-08-02.md` (+ `.meta`, six TSVs).**
Control band **159.78 ms vs 47.80 ms = 3.343x** (9 interleaved invocations/arm,
spreads 2.98 %/3.69 %, ratio spread 3.9 %, both arms 4472 B). **The gap is now
DIFFUSE**: CNN 30.0 %, dsp:transform 21.8 %, allocation 20.9 %, intra pred+RD
12.3 % of a 118.37 ms gap — nothing over 30 %, and the port's profile now has
the same shape as libaom's (top symbol 23.9 % vs 22.1 %, top ten spanning six
stages vs seven). Every non-CNN ABSOLUTE gap from the pre-fix table is confirmed
unchanged to within ±0.5 ms; only the shares moved. Load-bearing corrections and
new findings:

* **The bd8 lowbd lane path is now the single biggest PROGRAMME, at 34.1 % of
  the gap** — bigger than the CNN. Forward transform 6.16x/+19.63 ms, intra
  pred+RD 4.21x/+14.54 ms, inverse 8.55x/+6.15 ms. Source-verified both sides:
  libaom's fdct/fadst NEON kernels take `const int16x8_t *`
  (`upstream/av1/encoder/arm/av1_fwd_txfm2d_neon.c:401-1476`); the port's
  `fwd_col_pass`/`fwd_row_pass` (`aom-dsp/src/transform/simd/mod.rs:574,:685`)
  are `i32x8` and `widen16()` on load — **half the lane width**, and
  `lowbd16.rs` has INVERSE kernels only. Of the port's encode-side inverse,
  **79 % still runs the wide i32 path** (`lowbd16.rs:69-76` — only the DCT
  family passed the i16 audit).
* **The port's `os/setjmp` stage is allocator bookkeeping, not setjmp** —
  `mach_absolute_time` is 98.7 % under `xzm_free`; on libaom the same stage is
  97.6 % genuine `setjmp`. **Corrected allocation class: 27.63 ms vs 2.87 ms =
  9.63x, +24.76 ms.** Retroactive: the pre-fix alloc gap was +26.71, not +24.05.
* **A cheap named sub-lever: 4.5 % of the gap in four lines.**
  `fwd_col_pass`/`fwd_row_pass` open with a flat `[i32x8::zero(t); 64]` x2 =
  **4 KiB memset per forward transform, at every size**. The *inverse* passes in
  the same file are tiered {8,16,64} (`mod.rs:430-443,:784-797,:994-1000`) and
  `lowbd16.rs:132` says why. Those two are the **top two allocator/memset
  callers** (19.2 % of the class, 5.30 ms).
* Allocation re-census: **854 053 calls / 448.8 MB / 3 336 per SB** (was
  870 167 / 559.7 / 3 399) — the cache removed 16 114 calls and 110.9 MB,
  landing within 0.1 % of what that profile projected. CNN cascade **256 runs,
  1.00/SB**, per-call cost unchanged at 146.2 µs vs libaom NEON 17.1 µs (8.57x).
* **`cpu-used 9` is no longer a refusal (KB-34) and is the WORST ratio measured
  anywhere: 5.64x** (22.86 ms vs 4.05 ms, CNN never runs). cpu-used 4 is 7.76x.
  **Neither is profiled** — do not carry the ranking to them.
* Projected floor (arithmetic on measured self-costs, NOT a forecast): CNN-NEON
  + the fwd-scratch tiering + the i16 forward path → **2.21x** (2.42x at
  half-credit on the transform); levers 1-4 at their ceilings → 1.50x.

**RESULT — FIXED 2026-08-02. MEASURED 10.66x → 3.36x, zero bytes moved.**
Record: `benchmarks/encoder_cnn_cache_2026-08-02.md` + `.meta` + `.control.tsv`
+ `.breadth.tsv`.

- **FIX.** `cnn_partition::decision::PartitionSearchInfo` is C's
  `x->part_search_info` (`block.h:391-398`): `cnn_output_valid` +
  `cnn_buffer[1636]` + `log_q`. `rd_pick_partition_real` threads it and calls
  `invalidate_cnn()` at the SAME `BLOCK_64X64` re-anchor that already resets
  `quad_tree_idx` (literally the same two lines of C,
  `partition_search.c:3339-3343` — KB-24's sibling, now both modelled);
  `pack_tile` builds a fresh one per superblock (`encodeframe.c:692`);
  `decision::intra_mode_cnn_partition` computes on the
  `bsize == BLOCK_64X64 && !valid` branch only and returns `None` where C
  returns at `:227`. The 65×65 window is a CLOSURE, so
  `extract_intra_cnn_window` runs only on the computing path.
- **KB-23's `cnn_root_whole_in_frame` predicate is DELETED** — with a real latch
  it is emergent, exactly as in C: at a 64×64 node the per-block
  `av1_is_whole_blk_in_frame` test IS the containing-64 test (blocks are aligned
  to their own size), so a frame-edge 64×64 never computes and everything inside
  it prunes nothing. Verified, not argued: the whole partial-SB axis
  (`s4cov_partial_sb_axis`, `kb22_hd_arms`, `kb28_crop_dims`) is unchanged.
- **MEASURED**, 1024×1024 photo / cq44 / cpu-used 6, 3 arms interleaved over 9
  invocations each (`scripts/eprof_control_ab.sh`, NEW — the N-arm form of
  `eprof_control.sh`): base **520.93 ms** (spread 3.31 %) → cache **164.03 ms**
  (4.90 %) vs libaom-c **48.88 ms** (2.95 %); **10.657x → 3.356x**, −356.9 ms,
  paired cache/C ratios 3.25-3.46 against paired base/C 10.48-11.15. All three
  arms emit the same 4472-byte stream.
- **Cascade runs 2558 → 256 = 1.00 per 64×64, i.e. exactly libaom's 256**
  (`eprof_alloc`, all five exact-size CNN counters). The window copy came with
  it: `extract_intra_cnn_window`'s 4225-byte allocation also 2558 → 256, so the
  profile's "3.70 ms + 2558 allocations, no libaom counterpart" symbol is 10x
  smaller for free. (Removing the copy ENTIRELY — C passes
  `src.buf - stride - 1` in place — was NOT done: it needs `cnn_predict` to take
  a strided u16 source plus the crop clamp, which is not a clean drop-in, and it
  is now ~0.37 ms of a ~164 ms encode.)
- **The cached value is PROVEN equal to a recomputation, not assumed.**
  `decision::set_cnn_cache_verify(true)` makes every cache READ re-extract its
  window, re-run the cascade and assert bit-identity. On the profile cell:
  **256 computes + 2302 reads = 2558, and 2302/2302 reads bit-identical**
  (`examples/eprof_cnn_verify.rs`). Gated in CI at four cells by
  `aom-bench/tests/cnn_cache_identity.rs`, which also pins
  `computes == whole-in-frame 64×64 nodes` — **9, not 16, at 196×196**, which is
  KB-23 restated as an assertion.
- **Breadth** (`eprof_breadth.sh`, 12 cells, `port_bytes == c_bytes` at every
  one): 256² 13.88x → **3.32x**, 1024² 10.88x → **3.33x**, 2048² 9.85x →
  **3.27x**, cq10 15.23x → **2.90x** (was the worst ratio measured anywhere),
  cq26 10.51x → 3.06x, cq58 10.92x → 3.34x, cpu5 5.68x → **4.16x**, cpu4 9.39x →
  7.95x, cpu3 8.18x → 6.88x. **`cnn_per_sb` is 1.00 at every armed cell** (2.00
  at cpu 3/4, where the SB walk runs twice — the before-data shows the identical
  doubling, 15.78/15.77 vs cpu5's 7.89). **cpu-used 7 and 8 are the negative
  control**: CNN runs 0 both ways, 2.69x → 2.66x and 3.45x → 3.39x, inside the
  control spread.
- **Qualifying mode moves 6 → 5, measured both sides**: cpu6 497.67 ms
  (2.11 MP/s) → 159.32 ms (**6.58 MP/s**); cpu5 1191.11 ms (0.88 MP/s, fails
  the bar) → 905.08 ms (**1.16 MP/s**, clears it).
- **Bite proof.** (a) Neutering the latch (compute at every node) restores 2558
  cascade runs, 2558 window extractions and 565.69 ms — bytes still 4472, which
  is a third independent statement that cached == recomputed. (b) Deleting
  `invalidate_cnn()` takes `s4cov_partial_sb_axis::
  sb128_partial_sb_speed_axis_byte_matches` from 48/48 to **12/48** (36 DIVERGE =
  every size × cpu-used 1..6, KB-24's cell set) and makes `cnn_cache_identity`
  panic with *"intra-CNN cache read differs from a recomputation at bsize_idx 1
  quad_tree_idx 0"*, while the three SB64 tests in the same binary stay green.
  **Honest scoping (KB-24 precedent):** (b) bites under `--sb-size=128` only —
  under SB64 a superblock IS a 64×64, so `pack_tile`'s per-SB construction is
  already the per-64×64 reset and the explicit call is inert there (measured:
  SB64 1024²/256²/2048² stay byte-identical with it deleted). Both are kept
  because both are in C.
- **Gates:** full workspace **945/945** with `--run-ignored all` (the nightly
  tier included), plus the scalar-pinned (`AOM_FORCE_SCALAR=1`) full run and
  `cargo check --target x86_64-apple-darwin --workspace --all-targets`. Gate 2
  keeps **zero pinned cells** across cpu-used 0..9. The `--run-ignored all` pass
  did surface two failures, and neither was this change: see the KB-31 re-pin
  note below.
- **Two self-promoting KB-31 pins fired in the GOOD direction, and they are
  KB-12's, not this change's.** `kb31_mandatory_tiles::mandatory_tile_split_
  byte_identical_across_speeds` (OPEN `[(4032,8),(4160,8)]` → observed `[]`,
  20/20 byte-identical) and `::issue6_reported_sizes_encode` (5472x3648 20 MP
  now byte-identical, was +339 B). Both are `#[ignore]`d nightly tests that had
  not run since 2026-07-30 — 74 commits, including **KB-12 `0953fa7`** (the
  dropped Hadamard transpose in the nonrd estimate arm), which both tests' own
  doc comments already named as the owner of those cells. Both are cpu-used 8/9,
  where the intra-CNN prune runs **zero** times (measured), so the cache cannot
  reach them, and **both reproduce on the pristine baseline** — verified by
  re-running the file with the change stashed. Re-pinned to the tighter
  expectation (empty OPEN set / hard `assert_eq!`), which is what each test's own
  failure message asked for; nothing was relaxed. Same story for the three
  regenerated `benchmarks/config_perm_*_2026-07-30.tsv` evidence sweeps.
- **Was still open, unchanged by this, and has since CLOSED (KB-34, same day):**
  `cpu-used 9` refused the 1024²/cq44 cell (`nonrd_pickmode.rs:1135`, KB-32) —
  pre-existing, separately tracked, and now byte-identical. The
  residual 3.36x was NOT re-profiled; the profile's levers 2-5 are arithmetic on
  the OLD build's self-costs, so re-profile before ranking them again.

### KB-PERF-2 — Encoder: per-txb allocation churn + the un-tiered forward-pass scratch — LANDED ✅ 2026-08-02, and the projection was 18x optimistic

Levers **3a** and **3** of `benchmarks/encoder_hotspot_reprofile_2026-08-02.md`.
Record: **`benchmarks/encoder_alloc_scratch_2026-08-02.md`** (+ `.meta`,
`.ab.tsv`, `.alloc.tsv`, `.callers.tsv`).

**⚠ EVERY MILLISECOND BELOW IS AN APPLE M4 PRO MILLISECOND against APPLE'S
allocator, and lever 3's rank does NOT survive the platform change.** Measured
2026-08-02 on both Windows targets — the first time this encoder was ever RUN on
Windows, since `ci.yml`'s `windows-11-arm` job is `portability` and executes
nothing — same cell, same harness, five interleaved arms including a
same-binary null (record: **`benchmarks/winperf_windows_2026-08-02.md`** + `.meta`
+ 8 TSVs; job: **`.github/workflows/winperf.yml`**, `workflow_dispatch`-only):

| lever 3 alone, paired-median | value | share of the combined landing |
|---|---:|---:|
| Darwin M4 Pro, n=24 | −0.49 % | 21 % |
| `windows-11-arm`, n=16 | **−2.38 %** | **86 %** |
| `windows-latest` x86-64, n=16 | **−2.54 %** | **99 %** |

nulls +0.08 % / −0.20 % / −0.05 %; replicated in a second dispatch at n=12.
**The allocator CALL COUNTS are byte-identical on all three platforms**
(835 638 → 488 750, and the same bytes and per-SB figures), so the platforms do
identical work and the whole difference is **cost per call** — Microsoft's heap
charges more for one than Apple's. Lever **3a** goes the other way (−2.02 % on
Darwin vs −0.50 %/−0.77 % on the runners): it removes `memset` bytes, not calls.
**So: on Darwin the memset lever is senior and allocation is a 16.4 % residual;
on Windows the allocator lever is essentially the entire win.** Any future
ranking of allocation work must carry a platform. Also measured: **content moves
lever 3 by more than 2x on ONE box** (−1.26 % photo / −0.49 % `detail` /
+0.31 % `smooth`), and the landing's photo numbers REPRODUCE at load ~2
(−4.51 ms vs the recorded −4.641 ms), so background load is not the variable.
Linux is still unmeasured.

**KB-PERF-HARNESS (2026-08-03): winperf's contents are fitted per AXIS, there
are now THREE, and the workflow takes a `contents` input.** Record
**`benchmarks/winperf_content_census_2026-08-03.md`** + `.meta` + 7 data files.
`detail`/`smooth` were fitted on ALLOCATOR CALL COUNT and are right for
KB-PERF-2/3; they are structurally blind to coding-mode levers (`detail`:
directional intra = **0.15 %** of predicted pixels, no 4x4 forward transform, no
rectangular leaf; `smooth`: **never splits below 32x32**, 1024 leaves = four per
SB). **`aom_dsp::census`** (DEFAULT-OFF `census` feature, hooks on
`predict_intra_high` / `av1_fwd_txfm2d_into` / `pack_leaf`) +
`examples/content_census.rs` make that a committed table instead of an argument;
`examples/content_fit.rs` fitted a third content, **`Content::Photo`**, to the
photograph's MODE distribution over 467 candidates (objective declared first:
L1 over the intra-class pixel-share vector — no lever's delta entered it).
Intra-class L1 to the photograph: `detail` 47.43 pp / `smooth` 15.18 →
**`photo` 5.72**; directional pixels 0.15 % → **17.92 %** (reference 20.78);
leaf-size L1 54.10 → 20.16. **`photo` does NOT replace `detail`** — its
allocator traffic fell to 73 % of the reference where `detail`'s is 95 %; run
the content whose census reaches your lever. Runner floors, n=24, measured:
`windows-11-arm` **MDE 0.07-0.18 %**, `windows-latest` x86-64 **0.50-0.86 %**
(its round-to-round sd is 7-12x worse; it could not resolve a 0.2 % effect at
any content). `windows-11-arm` also has a within-round POSITION drift of
+0.19/+0.28 % between successive IDENTICAL arms — `pre` is always first — so
read `prepost` bands with **`scripts/winperf_prepost_stats.py`** (pools the
copies on each side) and difference against a control content, not with
`--vs pre`.

**KB-PERF-HARNESS-2 (2026-08-03, same day): the census now names EVERY tool
family, the corpus was censused against it, and the "cannot see" list had three
different causes conflated.** Record
**`benchmarks/winperf_family_census_2026-08-03.md`** + `.meta` + 4 data files.
`aom_dsp::census` gained `Leaf` (filter-intra, palette Y+UV, intraBC, UV mode
⇒ CFL, tx_size, both angle deltas, skip/inter/chroma-ref — all at the
bitstream writer), `note_plane_intra_pred` (the per-plane split of
`predict_intra_high`, annotated at the 8 ENCODER call sites since the DSP entry
point gains no argument; **`plane_total() == intra_total_calls()` is asserted**,
which is what proves no site was missed) and `note_cfl_predict`.
`Counts::since` destructures with **no `..`** so a new counter breaks the build.
The three causes:
- **SPEED-gated:** filter-intra is 0.00 on EVERY source at cpu-used 6
  (`prune_filter_intra_level = 2` ⇒ `rd_pick_filter_intra_sby` never called);
  same content reads **10.46 %** of leaves at cpu-used 5. Also measured on that
  sweep: the photograph's much-quoted **20.78 % directional is a speed-6
  number** — 56.61 % at cpu-used 5 on identical content (`intra_pruning_with_hog
  = 4`), so quote directional shares WITH the speed.
- **KNOB+HEADER-gated:** palette / intraBC need `--enable-palette` /
  `--enable-intrabc` (default off) AND `allow_screen_content_tools`, which real
  aomenc sets from its own detection. New content **`Content::Screen`** (flat
  few-colour panels + a repeating glyph alphabet + an `image_q8` share of
  photographic panels; integer-only) reaches both; `gb82-sc` corpus reaches
  palette **6-33 %** and intraBC **0.25-59 %** of leaves. intraBC's share GROWS
  with frame area (0.25 % at 512x384 → 6.5 % at 1 MP, same source).
- **CONTENT-gated:** CFL is **0.00-0.29 %** of chroma-ref leaves on all three
  old contents, **4.59 %** on the photograph, **23.02 %** on a CLIC image — no
  knob involved. Chroma is **28-47 % of intra-predictor calls** and was
  previously uncounted, so earlier `intra_calls` totals are luma+chroma sums.
**The differential corpus was never blind**: `EncodeCell::real_content`
(KB-13's cells, cq32/cpu-used 0) reaches filter-intra **21-31 %** of leaves,
directional **51-54 %** of px, rect leaves **74-82 %**, 4-pt tx **67-75 %**.
Check `real:`/`yuv:` before generating content. **GATE:
`just census-gate`** (`crates/aom-bench/tests/content_family_census.rs`, no
oracle, **6.1 s**, wired into CI `portability`) pins each family's share with a
floor AND a ceiling, pins the known zeros WITH their stated cause, and pins the
screen source's two structural properties (≤ 8 colours per 16x16 block, ≥ 20 %
exactly-repeated 8x8 blocks) on the SOURCE PIXELS with the photographic
contents measured as comparators. Open observation: the port's **intraBC DV
search is ~200-400x content-dependent** — `gb82-sc/imac_dark` does not finish a
census in workable time (>10 min at 512x384) where `codec_wiki` takes 2.8 s;
that is why the gate's screen row runs at 256x256
(`winperf::SCREEN_GATE_CELL`).

**MEASURED, 1024×1024 photo / cq44 / cpu-used 6, 5 arms interleaved over 12
rounds** (`scripts/eprof_ab.sh` + `scripts/eprof_ab_stats.py`, NEW — the N-arm
form of `eprof_control.sh`, since a perf landing has to compare four port
builds and §6 forbids comparing separately-taken medians on this box):
base **159.594 ms** (spread 3.51 %) → **154.953 ms** (3.04 %) vs libaom-c
**47.701 ms** (4.77 %); **3.3457x → 3.2484x**, −4.641 ms (paired-median
−3.05 %), paired final/C ratios 3.126-3.284. All arms emit the same 4472 bytes.
The base arm reproduces the re-profile's published 3.343x to 0.1 % on a
different day, so the deltas are read against a live control.
**RE-TAKEN 2026-08-03 WITH THE ARM ORDER ROTATED — SURVIVES**
(`benchmarks/encoder_rotate_reverify_2026-08-03.md` §2): that band was FIXED
order, which confounds arm with position (§6), and `ROTATE=1` did not exist
yet. Rebuilt from `578653f` → `99a10ab` at 5 arms × **50 rotated rounds** with
nulls both sides: paired-median **−2.986 % / −3.004 %** against −3.05 %
(0.06 pp), the two post copies agreeing to 0.018 pp, 46/50 and 45/50 faster,
p < 0.0001, null −0.050 %; ratio 3.3537x → 3.2517x/3.2522x.

- **Allocator census: 854 053 → 512 557 calls (−40.0 %), 448.8 MB → 267.5 MB,
  3 336 → 2 002 per superblock, peak live UNCHANGED at 27 705 399 B** — it was
  churn, exactly as the re-profile said, and nothing here reduces footprint.
- **THE LESSON, and it invalidates a whole class of ranking: `alloc/libc` is a
  LEAF CLASS MATCHED BY SYMBOL NAME, and most of its mass is `memset`/`memcpy`,
  not allocator bookkeeping.** In the re-profile's own numbers `_platform_memset`
  was 5.36 % of the port window and `_platform_memmove` 2.68 % against
  `xzm_free`'s 2.34 % — so crediting the scratch-reuse lever with the whole
  27.63 ms class was never sound. Projected +24.76 ms; delivered **−1.34 ms**.
  Removing 341 496 malloc/free pairs does not remove the bytes those buffers
  still have to be zeroed with. **Split the allocator symbols from
  `_platform_memset`/`_platform_memmove` before crediting any future lever with
  that stage total — and split them per PLATFORM, because the two halves are
  priced differently by different heaps** (the Windows measurement above is that
  same split arriving from the other side). (The re-profile flags this in "Attribution limits" item 4
  and its ranked table contradicts it — read the limits, not the table.)
- **Per-lever bite proof, and the two levers bite on DIFFERENT axes** (§1):
  3a alone **−3.128 ms / 0 allocator calls** (its scratch is a stack array, so
  it cannot appear in a census — proven, not asserted: the `l3` census and the
  `final` census are identical to the digit); lever 3 alone **−1.339 ms /
  −341 496 calls**. Additive to 3.9 %, inside the control spread.
- **3a** = tier `fwd_col_pass`/`fwd_row_pass` `{8,16,64}` exactly as the INVERSE
  passes in the same file already were (`mod.rs:430-443`, `:784-797`,
  `:994-1000`; `lowbd16.rs:132` states why). The flat `[i32x8; 64]`×2 was 4 KiB
  of memset per forward transform at every size, and those two were the top TWO
  callers of the class — **both are now absent from the top 22** (`.callers.tsv`).
- **3** = the decoder's `ReconScratch`/`InvTxfmScratch` pattern encode-side:
  `FwdTxfmScratch` + `av1_fwd_txfm2d_into`, `XformQuantScratch` +
  `xform_quant_into`/`xform_quant_optimize_split_into`,
  `{TxWalk,TxSearch,IntraTx}Scratch` + `search_tx_type_intra_into` /
  `dist_block_px_domain_into`. Every public entry point kept its signature by
  delegating to the `*_into` form over a fresh scratch.
  **OWNERSHIP IS THE WHOLE LEVER**: owned by `rd_pick_intra_sby_mode_y` (luma
  mode loop), `rd_pick_intra_sbuv_mode` (chroma mode loop) and `PaletteRdState`.
  An earlier revision owned it inside `txfm_rd_in_plane_intra` and returned only
  **−6.4 %** of the calls (854 053 → 799 552) — most walks are a single
  transform block, so a per-walk scratch has nothing to reuse.
- **The `memset` is deliberately KEPT.** Every buffer refills with `clear()` +
  `resize(n, 0)`, byte-identical to the `vec![0; n]` it replaces *by
  construction*. A skip-the-re-zero variant was built (all three
  `XformQuantScratch` buffers are provably fully overwritten — `av1_fwd_txfm2d`
  writes every `coeff[..full]`; all twelve quantizers open with
  `qcoeff[..n].fill(0)`) and **measured inside the control band** (16 rounds,
  −3.34 % vs −2.40 %, each inside the other's spread), so it was reverted.
- **Gates: 950/950 with `--run-ignored all`, 0 skipped, in BOTH dispatch modes**
  (SIMD live and `AOM_FORCE_SCALAR=1`) + `cargo check --target
  x86_64-apple-darwin --workspace --all-targets`. Gate 2 keeps **zero pinned
  cells** across cpu-used 0..9. A fourth gate fell out for free: the
  `config_permutations` sweep regenerates three evidence TSVs, and across
  **616 rows** (5 contents × 10 `--cpu-used` × every singleton knob axis, port
  vs real `aomenc`) the regenerated files differ in the **timing column only** —
  every `exact`/`port_len`/`c_len` unchanged, so they are left as committed
  rather than re-pinned from a load-31 box.
- **Residual: allocation is still 16.4 % of the (now 107.25 ms) gap, and the
  part that is left is the part scratch reuse CANNOT reach** — `finish_grow`
  (Vec growth in `aom_encode`, now rank 1 of the class at 8.85 %), the
  `qcoeff`/`dqcoeff` that `encode_intra_block_plane_{y,uv}` legitimately RETAIN
  per transform block into `TxbEncode`, and the memset bytes themselves.
  Attacking it further means changing what the encoder *stores*, not where.

### KB-PERF-3 — Encoder: the forward transform ran at HALF libaom's lane width — LANDED ✅ 2026-08-02, and the projection was 5x optimistic

Lever **2** of `benchmarks/encoder_hotspot_reprofile_2026-08-02.md`
(+19.63 ms / 16.6 % of the gap projected), and the **cross-platform** half of
the bd8 lane-width programme — i16 lanes help NEON and AVX2 alike, unlike that
ranking's ARM-only rank 1. Record:
**`benchmarks/encoder_i16_fwd_2026-08-02.md`** (+ `.meta`, `.ab.tsv`,
`.halfbatch.tsv`, `i16_fwd_audit_2026-08-02.txt`).

**The defect, read from source rather than the profiler:** libaom's
`fdct*/fadst*_neon` take `const int16x8_t *`
(`upstream/av1/encoder/arm/av1_fwd_txfm2d_neon.c:401-1476`) while the port's
`fwd_col_pass`/`fwd_row_pass` were `i32x8` and `widen16()`d on load — half the
lane width for the same work — and `lowbd16.rs` had INVERSE kernels only.

**MEASURED, 1024x1024 photo / cq 44 / cpu-used 6, 7 arms interleaved over 24
rounds** (`scripts/eprof_ab.sh`, nulls on BOTH sides): base **154.474 ms** →
**150.630 ms** vs libaom-c 47.187 ms; **3.2737x → 3.1922x**, −3.843 ms
(paired-median −2.56 %). Nulls −0.06 % (`baseB`) / +0.16 % (`bothB`); the
paired per-round ratios do not overlap (base 3.244-3.305, both 3.172-3.221).
Column pass alone −1.700 ms, row pass alone −1.665 ms, and they compose
**super-additively** to −3.843 (+0.48 ms, 3x the floor, unexplained).
**RE-TAKEN 2026-08-03 WITH THE ARM ORDER ROTATED — LEVER SURVIVES, MAGNITUDE
MOVES ~0.5 pp** (`benchmarks/encoder_rotate_reverify_2026-08-03.md` §3): that
band was FIXED order (§6) and `ROTATE=1` did not exist yet. Rebuilt from
`590e525` → `7976c0f` at 5 arms × **50 rotated rounds**: paired-median
**−1.893 % / −2.163 %** (mean of the two copies −2.03 %) against −2.56 %;
47/50 and 48/50 faster, p < 0.0001, null −0.086 %, MDE95 0.403. **How much of
the 0.5 pp is protocol is NOT established** — the two copies of the same post
binary differ 0.27 pp inside that band, and its fixed-order twin read
−2.541/−2.344. The ratio moves least: 3.2232x → 3.1604x/3.1517x. Quote
−2.56 % as the fixed-order reading, −2.0 to −2.2 % as the rotated one.

- **THE AUDIT IS THE WORK, and it is a different question from the inverse
  one.** The inverse kernels `clamp_value` at every stage, so
  `audit_i16_safety.py` asks a DOMAIN question. The forward kernels carry **no
  clamp at all**, so nothing bounds a value except the input, and the new
  `xtask/audit_i16_fwd.py` asks a BOUND question: it propagates each value's
  EXACT linear form (`sum c_i*input_i + e`, coefficients as exact `Fraction`s)
  and reports `M*`, the largest `|input|` keeping every value inside i16.
  **Tight, not triangle-inequality** — the sign vertex attains `M*sum|c_i|`,
  and the loose bound is 1.5-2x larger and would REJECT fdct32's column pass,
  which is provably safe. `sum|c|` comes out at exactly `N*sqrt(2)/2` for the
  whole fdct family, so `M*(fdctN) = floor(46340/N)`; that closed form was not
  put in, it came out, and it is a check on the propagator.
- **11 of 12 kernels have an i16 form.** M* (min over cos_bit 10..13): fdct4
  11583, fdct8 5791, fdct16 2895, fdct32 1447, fdct64 723, fadst8 6419,
  fadst16 3214, fidentity4/8/16/32 23167/16383/11583/8191. **REJECTED:
  fadst4** — it works in a PRE-SHIFT domain (stage-1 values are `sinpi[j]*x`
  held UNSHIFTED), `sum|c|` peaks at 21901 and `M*` is 1-11. No bound was
  widened to admit it.
- **The gate is taken at RUNTIME on the actual block** (`max|input| << shift0
  <= M*` / `max|buf| <= M*`), so the path is sound for any caller of the public
  `av1_fwd_txfm2d`, not only bd8 — out of range it declines and the i32 pass
  runs. `try_fwd_col_pass`/`try_fwd_row_pass` additionally gate on the SHIFT
  domains the two shift recipes are proved on (`shift0 in {0,2}` because the
  input scale is written `v+v; d+d`; `shift1/2_bit in 0..=4` because that is
  `rshift_mul`'s proof range), so a future `FWD_SHIFT` table cannot walk off
  either proof — pinned inert by `the_shift_table_stays_inside_the_gated_domains`.
- **THE CEILING WAS 5x OPTIMISTIC, and the mechanism says why** (playbook §14
  again, one landing after KB-PERF-2's 18x). Two measurements, not an argument:
  (a) the +19.63 ms row was taken BEFORE lever 3a, which already took −3.13 ms
  out of these same two functions; (b) a temporary counter in both pass entry
  points shows **only 51.6 % of column-pass calls and 55.1 % of row-pass calls
  are even eligible** — 39 816 of 82 203 column calls are 8-wide and 33 671 of
  74 938 row calls are 8-tall, which a 16-lane batch cannot fill. The honest
  form of the lever is "half the vector width plus a cheaper `half_btf`, on the
  half of the calls whose vectorized dimension is >= 16", and named that way it
  is a ~2.5 % lever.
- **The half-batch extension was BUILT, measured NULL, and reverted.** Running
  the 8-dim blocks as half-idle 16-lane batches is not a silly idea — an
  `i16x16` and an `i32x8` are both 256 bits, so the op count is equal while
  `fbtf16` (widening madd, stays in i32) is much cheaper than `prims::hb`
  (widens to i64 and back). Full implementation, differentials extended,
  reach 81 → 129 column cells: **`half` vs `both` = −0.009 ms (−0.006 %)
  against a same-binary null of +0.08 %.** Reverted — added surface and risk on
  well-tested cells for zero measured benefit. Preserved at
  `~/tmp/i16fwd/lowbd16_fwd_halfbatch.rs`; re-test it on x86 before believing
  the null generalises.
- **Gates: 957/957 with `--run-ignored all`, 0 skipped, in BOTH dispatch modes**
  + `cargo check --target x86_64-apple-darwin --workspace --all-targets`.
  Gate 2 keeps **zero pinned cells** across cpu-used 0..9; the three
  `config_perm_*` evidence sweeps regenerate identical apart from the timing
  column. Every arm (5 port builds + the C oracle) emits the same `.obu` **by
  sha**.
- **Non-vacuity, both directions.** `gate_bite::the_bound_is_load_bearing`
  pins that every kernel genuinely DIVERGES outside `M*` (observed first
  divergence 1.06x-2.97x M* — `M*` is sound but not attained, since `E` is a
  worst case no single input realises), and
  `reach::the_gate_fires_across_the_bd8_grid` pins that it FIRES on the real
  domain (81/81 shape-eligible column cells, 70/73 row; the 3 declines are
  TX_64X64 / TX_32X64 / TX_64X32 DCT_DCT, exactly what the audit predicts).
  `txfm2d_simd_perm_diff` needed a **new bd8 residual arm**: its existing
  forward arm is full-range i16, which is over `M*` for every kernel, so
  without it the integration test could not reach this code at all (playbook
  §1).
- **THE CROSS-PLATFORM CLAIM IS MEASURED, and the lever is worth MORE on
  Windows than on Darwin** — the opposite of KB-PERF-2's allocation lever
  needing a platform caveat, and for the reason playbook §6b gives (this one's
  mechanism is arithmetic, not a call into the platform). `winperf.yml` gained
  a generic **`arms: prepost`** mode for it (base_sha vs HEAD, nulls on BOTH
  sides, usable by any landing); run 30788058276, 16 rounds x 2 contents x 2
  runners, bands committed as `.win_<runner>_<content>.tsv`:
  **`windows-11-arm` −7.43 % `detail` / −7.22 % `smooth`** (nulls +0.08..+0.34 %),
  **`windows-latest` x86-64 −2.75 % / −4.32 %** (nulls −0.80..+0.55 %, but raw
  spreads 5.9-8.0 %, so read it as "−2.8 to −4.3 % on a noisy runner"), against
  Darwin's −2.49 %. **The allocator census is identical to the digit on every
  arm on both runners** except `peak_live` by ONE byte out of 17.4 MB — proof
  this moves arithmetic and not allocation. CAVEAT: winperf's sources are
  integer-generated `detail`/`smooth` (it cannot ship the study photograph), so
  platform and content are confounded — the cross-platform ORDERING is sound,
  the 3x ARM-vs-Darwin RATIO is not.

### KB-PERF-4 — Encoder: the DIRECTIONAL intra predictors had no vector path at any bit depth — LANDED ✅ 2026-08-03, and the ranked row was re-scoped BEFORE the work

Lever **4** of `benchmarks/encoder_hotspot_reprofile_2026-08-02.md`
("bd8 lowbd intra predictors", **+14.54 ms / 12.3 %** of the gap projected).
Record: **`benchmarks/encoder_intra_dir_i16_2026-08-03.md`** (+ `.meta`,
`.ab.tsv`, `.ab_split.tsv`, `.control.tsv`, `.stage.tsv`, `.census.txt`).

**THE RANKED ROW WAS RE-MEASURED FIRST, AND THAT IS THE MAIN RESULT.** Playbook
§14 fired twice in a row (KB-PERF-2 at 18x, KB-PERF-3 at 5x), so this session
re-profiled before writing a line. Total gap **118.37 → 104.94 ms** since that
profile. Three readings of the SAME row, which reconcile
(7.58 + 6.30 + 1.52 = 15.40 vs 15.39):

| reading | port | C | ratio | gap | % of gap |
|---|---:|---:|---:|---:|---:|
| the ranked row (`dsp:intra-pred` + `intra-mode-rd` combined, as the re-profile instructs) | 19.75 | 4.36 | 4.53x | +15.39 | **14.7 %** |
| — the encoder's intra **RD drivers** (`intra_rd`/`rd_pick`/`intra_uv_rd`/`encode_intra`) | 8.12 | 1.82 | 4.46x | +6.30 | 6.0 % |
| **the intra PREDICTOR class, like for like** | **9.91** | **2.33** | **4.25x** | **+7.58** | **7.2 %** |
| CFL (the two arms file it in different stages) | 1.72 | 0.21 | 8.34x | +1.52 | 1.4 % |

**As a STAGE the row has GROWN (12.3 → 14.7 %) because the denominator fell; as
a LEVER it is half that, 7.2 %.** Inside the predictor class: directional
z1/z2/z3 **+3.13 ms (9.6x)**, smooth+paeth +2.63 (8.7x), edge filter +0.74,
DC/V/H fills +0.71, and **edge assembly + mode routing at 1.26x — near parity**,
so there is no per-block plumbing overhead to remove; the gap is all kernels.

**The brief's framing was half wrong and the profile says so.** For the
directional predictors there is no highbd-vs-lowbd choice to make: `dir.rs`'s
`z1_high`/`z2_high`/`z3_high` **and** their lowbd `z1`/`z2`/`z3` twins were
**pure scalar**, while libaom dispatches `av1_dr_prediction_z{1,2,3}_neon`
(`aom_dsp/arm/intrapred_neon.c:1290-1482`) / `_avx2` / `_sse4_1`. For
smooth/paeth the lane-width framing IS right (port `i32x8` vs libaom `u8`/`u16`)
and those are untouched — the named follow-up.

- **THE FIX** is `crates/aom-dsp/src/intra/dir_simd.rs`, ONE
  `#[magetypes(define(i16x16, u16x16), v3, neon, -scalar)]` body (NEON and AVX2
  from one source — the cross-platform half of the programme), computing
  libaom's re-association `a0*(32-s) + a1*s == (a0<<5) + (a1-a0)*s` in i16 lanes.
- **THE AUDIT IS ONE TIGHT BOUND.** With `shift ∈ [0,31]` every intermediate is
  inside i16 **iff every tap `<= 1023`** (`32M + 16 <= 32767`); at `M = 1024`,
  `a0 << 5` is exactly `-32768`. Taken at RUNTIME over the `O(bw+bh)` edge span
  (against `O(bw·bh)` of work), so it is a per-block scan. It admits **bd8 AND
  bd10**, declines bd12 — the gate is on the DATA, not on `bd`. No bound widened.
- **REACH, COUNTED (census committed).** 85 423 predictor calls / 19 151 632
  predicted px per frame = **18.3x the frame's pixels**, **100 % bd8** (no
  bit-depth loss, unlike KB-PERF-3's 51.6 %). Only contiguous runs vectorize:
  z1 `up==0` 87.4 %, z2 above-half 49.8 % (its left half is a true gather —
  `base_y` is not affine in `c`), z3 `up==0` 85.1 %. Pixel-weighted **68.4 % of
  directional px addressable → 2.39 ms of the 3.50**.
- **MEASURED**, 1024×1024 photo / cq 44 / cpu-used 6, 36 rounds × 6 interleaved
  arms with nulls on BOTH sides: base **149.188 ms → 148.062 ms** vs libaom-c
  46.455; **3.2115x → 3.1872x**, −1.126 ms, paired-median **−0.75 %**. Nulls
  −0.01 % (`baseB`) / −0.01 pp (`allB`). **Independently replicated** by an
  earlier 24-round 7-arm band (−0.58 %, paired −0.81 %) whose per-half split was
  z1+z3 −0.20 % / z2 −0.45 % — roughly equal halves, NOT separately resolvable
  against that band's −0.17 % null (which is why the 36-round band was run).
  All arms emit the same 4472-byte `.obu` by sha.
- **RE-TAKEN 2026-08-03 WITH THE ARM ORDER ROTATED — SIGN SURVIVES, the
  −0.75 % MAGNITUDE IS NOT RE-VERIFIED**
  (`benchmarks/encoder_rotate_reverify_2026-08-03.md` §4). Both published bands
  were FIXED order (§6) and `ROTATE=1` did not exist yet — and the position
  drift it corrects is worth **45 % of this lever's whole effect**, which is why
  this lever was re-taken first. Rebuilt from `0279544` → `71c924a` at 5 arms ×
  50 then × **150 rotated rounds**. n=150: paired-median **−0.648 % /
  −0.623 %** (the two post copies agreeing to 0.025 pp) against a null of
  **+0.095 %**; **115/150 and 108/150 rounds faster, p < 0.0001**, null 71/150
  p = 0.57; ratio 3.1785x → 3.1592x/3.1588x (−0.020 vs the published −0.024).
  **BUT the box was heavily contended (raw spreads to 212 %) and NEITHER band
  met the pre-registered gate — MDE95 1.155 and 1.475 pp against a required
  0.375**, so the magnitude is unmeasured to that precision, not refuted. Six
  independent post-vs-base comparisons across three bands are all negative.
  **Open, ~20 min: one 150-round rotated band on an idle box.**
- **13x optimistic against the ranked row; 2.1x against the sub-lever's own
  addressable cost.** Both numbers are in the record because the first is what a
  ranking table produces and the second is what a named mechanism produces.
- **The 8-wide half-batch is worth it HERE and was not in KB-PERF-3 — same
  shape, opposite verdict, because the baseline differs.** There a half-idle
  `i16x16` competed against a full `i32x8` and measured −0.006 %; here it
  competes against a scalar loop, so 8 live lanes is still 8x.
- **A micro-variant was built, measured and REJECTED**: replacing the kernel's
  per-lane store loop with `bitcast_u16x16` + `copy_from_slice` (one memcpy, the
  a-priori better shape) measured **0.15 pp WORSE** against a 0.01-0.06 pp null.
  Reverted; why it loses is not established.
- **Gates.** `dir_simd_diff` — dispatch vs the never-dispatched `*_scalar` cores
  at every token permutation (25 run), full TX grid, every signalled angle
  through the real `dr_intra_derivative` table, bd 8/10/12, tight + padded
  stride. **`upsample` is DERIVED via `edge::use_upsample`, not swept** —
  sweeping it freely walks the 160-entry edge buffers off their ends **in the
  SCALAR kernel too**, which is how that was caught. Probes are asymmetric on
  purpose: a FLAT edge makes `a1-a0 == 0` and is invariant under exactly the new
  term (KB-12's lesson). In-crate `dir::reach` pins **16/19 shapes admitted for
  z1 and z3** (the `bw==4` / `bh==4` ones decline on the run-length floor),
  **19/19 for z2**, `up==1` never, 1023 admitted / **1024 declined**, bd12
  declined everywhere. `the_tap_bound_is_load_bearing` pins that the bound
  genuinely bites — and it **failed the scalar-pinned leg on its first full
  run**, correctly: under `AOM_FORCE_SCALAR=1` the dispatch entry IS the scalar
  core, so it cannot diverge from itself. Its divergence half is now conditioned
  on `dispatch::scalar_forced()` while the gate's own 1024-rejects /
  1023-accepts half stays UNconditional. Nothing relaxed — an implicit
  precondition made explicit, which is the mirror of playbook §1: **a test that
  cannot pass in one dispatch mode is as much a defect as one that cannot
  fail.** Run BOTH modes before believing a new gate.
  **Bite proofs with the asymmetry**: dropping the `* shift`
  fails the kernel differential alone; transposing z3's scatter fails
  `dir_simd_diff` + `dir_highbd_diff` (vs the real C symbol) + `intra_lowbd_diff`
  while `intra_simd_diff`, `highbd_diff` and `build_nd_diff` stay green.
  Gate 2 keeps **zero pinned cells**.
- **WINDOWS: RESOLVED 2026-08-03 ON `windows-11-arm`, ONCE THE HARNESS HAD
  CONTENT THAT REACHES IT — and still NOT resolvable on `windows-latest`.**
  Re-measured on the new oriented content (run 30798500036, 24 rounds, `photo`
  AND `detail` in the same job on the same VM, pooled-copy statistic):
  **`photo` −0.332 % vs `detail` +0.167 %, difference −0.499 pp**, against
  Darwin's **−0.356 / +0.131, difference −0.487 pp** on the same two binaries
  the same day — **agreement to 0.012 pp**, exactly as §6b predicts for a lever
  whose mechanism is integer lane arithmetic rather than a platform call. Sign
  test on `windows-11-arm` `photo`: 19/24 rounds faster, p = 0.0066, with
  `detail` significantly the OTHER way. `windows-latest` x86-64 stays
  unresolvable and now has a number for why: MDE 0.50-0.86 % at n=24 against a
  0.4 % effect. **Also isolated for the first time: the lever COSTS ~+0.15 % on
  content its kernel never reaches** (+0.167 % arm / +0.131 % Darwin, same sign
  and size on two CPUs) — the runtime gate's decline cost, which the landing had
  folded into the reported win. Record:
  `benchmarks/winperf_content_census_2026-08-03.md` §5. Superseded below is the
  2026-08-03 morning reading:
- **WINDOWS (first attempt): MEASURED, NOT RESOLVED — and the harness was why,
  which is the durable finding.** `winperf.yml` `arms: prepost` (run 30792984795, 16 rounds,
  both runners, three nulls per band): every `post − pre` is at or under that
  band's own same-binary nulls (paired medians: `windows-11-arm` +0.35 %
  `detail` / +0.05 % `smooth` against nulls of 0.00-0.22 %; `windows-latest`
  unusable at raw spreads 3.2-31.4 %). The effect is 3-10x smaller than KB-PERF-3's,
  which DID clear these runners. **But the bigger reason was checked, not
  assumed**: the same census re-run on winperf's own sources gives directional
  predicted-pixel share **photograph 20.8 %, `smooth` 13.2 %, `detail` 0.15 %**
  — on `detail`, z1 fires SIX TIMES in the whole frame. **`winperf`'s two
  synthetic sources were tuned to bracket the photograph's ALLOCATOR CALL COUNT
  (`winperf.rs:63-71`), not its MODE DISTRIBUTION**, which made them right for
  KB-PERF-2/3 (both touch every block regardless of mode) and structurally
  vacuous here. **Any future lever scoped to a mode family (directional intra,
  palette, filter-intra, CFL) must run this census BEFORE reading a winperf
  band, or it will read a structural zero as a platform result.** The census is
  now committed and the census-driven content exists — see KB-PERF-HARNESS
  above; the cross-platform claim, which at the time rested on the mechanism
  (integer lane arithmetic, no platform call — playbook §6b) and on the AVX2
  tier passing its differential rather than on a Windows timing, has since been
  measured directly.
- **Untouched and named**: the lowbd `u8` twins (so the DECODER's bd8 path gets
  nothing — this is an encoder lever, and the encoder holds planes as `u16` at
  every bit depth), z2's left half, `up==1` runs, `w==4` blocks, and the
  +4.45 ms of non-directional predictor class.

### KB-PERF-5 — Encoder: SMOOTH ran at HALF libaom's lane width — LANDED ✅ 2026-08-03; PAETH's half built, measured NULL four times, REVERTED

KB-PERF-4's named follow-up ("smooth + paeth, +2.63 ms, already `i32x8` where
libaom is `u8`/`u16`"). Record:
**`benchmarks/encoder_intra_smooth_paeth_2026-08-03.md`** (+ `.meta`, `.audit.txt`,
`.census.txt`, `.control.tsv`, `.stage.tsv`, and FIVE band TSVs).

**THE PROJECTION HELD — the first one in this sequence that did.** Re-measured
first, per §14: **+2.63 ms projected, +2.852 ms measured** (8 % apart) against
KB-PERF-2's 18x, KB-PERF-3's 5x and KB-PERF-4's 13x. **The difference is that
this projection came from a NAMED MECHANISM measured like-for-like, not from a
profiler's ranked stage** — two port symbols against ten C ones, nothing else in
the row. Split: **SMOOTH +1.986 ms (11.5x), PAETH +0.865 ms (7.3x)**, and that
split turned out to be the whole story.

- **THE FIX** is `crates/aom-dsp/src/intra/simd16.rs`, three
  `#[magetypes(define(u16x16), v3, neon, wasm128, -scalar)]` bodies gated at
  runtime; `super::simd`'s `i32x8` kernels stay as the decline path.
  **`wasm128` is in the tier list deliberately** — the kernels it shadows carry
  it, and omitting it would have silently DE-VECTORIZED WASM for every bd8
  block, a regression no gate in the repo can see.
- **MEASURED: −0.38 %, 3.1613x → 3.150x**, 60 rounds, **arm order ROTATED**,
  **51/60 and 52/60 rounds faster on two copies of the shipped binary
  (p < 0.0001 each, paired means agreeing to 0.005 pp)** against a same-binary
  null of −0.009 % at 30/60 (p = 1.00). Three earlier bands replicate at
  −0.28 / −0.31 / −0.36 %. ~200 interleaved rounds in total.
- **THE AUDIT IS EXHAUSTIVE ENUMERATION, not an inequality** (`xtask/audit_nd16_lanes.py`):
  every intermediate is a function of ≤ 4 small scalars, so the whole product
  space is swept. **`M* = 255`, TIGHT** (`(256-w)*b` is exactly 65536 at
  M = 256), on the DATA not on `bd`, so bd10/bd12 decline. SMOOTH's numerator
  does NOT fit `u16`, so the halves combine through libaom's **TRUNCATING**
  halving add; `((A+B)>>1 + 128)>>8 == (A+B+256)>>9` is verified over **every
  reachable sum (0..130560)** and the ROUNDING form is shown wrong at A+B=255.
  No bound widened.
- **PAETH: BUILT IN FULL, MEASURED NULL IN FOUR BANDS, REVERTED.** Own bound
  (`M* = 16383`, admitting bd8/10/12 — wider reach than the shipped half), full
  differential, reach + bite pins. Paired medians **+0.08 / −0.01 / +0.09 /
  +0.04 %**, sign-test **p = 0.65 / 1.00 / 0.29 / 0.39**, across two store
  shapes. In the **position-balanced** band the composed binary and the
  smooth-only binary are indistinguishable (−0.007 %, 25/49, p = 1.00).
  **Mechanism: PAETH is STORE-bound, not arithmetic-bound** (6 vector ops per
  chunk against the same 2-bytes-per-pixel store into a `u16` plane), which the
  store-shape A/B corroborates — `bitcast_u16x16` + `copy_from_slice` measured
  +0.14 % against the lane loop, p = 0.13. Preserved at
  `~/tmp/smooth/simd16.withpaeth.rs`; its bound is still derived by the audit.
- **HARNESS: a fixed-order interleave CONFOUNDS ARM WITH POSITION, and it is
  worth as much as the effect.** Two copies of ONE binary at round positions 5
  and 6 came out **0.34 pp apart** while the copies at positions 1 and 2 agreed
  to 0.11 pp. `scripts/eprof_ab.sh` gains **`ROTATE=1`** (default off) + a
  `position` column; `eprof_ab_stats.py` parses both TSV shapes. Pooled over all
  arms the drift is **1.7 % across a round**, and it vanishes under rotation
  (positions 1-5 within 0.1 % on the headline band). The same drift is on record
  for `windows-11-arm`, where it is handled by pooling after the fact — rotation
  removes it by construction. **Use `ROTATE=1` for every new band.**
- **CENSUS RUN BEFORE THE BAND, and this time it said GO.** `aom_dsp::census`
  gains `nd_mode_tx_calls` / `nd_mode_tx_px` and `content_census.rs` the
  `nd_mode_x_tx` / `nd_mode_x_width` tables — **a column-vectorized predictor is
  priced by BLOCK WIDTH** and the committed census could not report that.
  SMOOTH+PAETH is **43.6 %** of predicted px on the study photograph, **50.6 %
  on winperf `detail`**, 39.1 % on `photo`, 38.3 % on `smooth` — every content
  reaches this lever, the opposite of KB-PERF-4. Eligible fraction **~100 %**
  (100 % bd8; `bw >= 16` is 92.8 % of SMOOTH's pixels), so the delivered 27 % of
  the row is NOT an eligibility loss — it is the loads, the per-row splats and
  the 32-byte stores into a `u16` plane (twice libaom's `u8` bytes) that the
  lane width does not touch.
- **Gates**: **986/986** with `--run-ignored all` in BOTH dispatch modes on the
  pushed tree (972/972 on the pre-rebase tree it was measured on; the profile
  cell encodes to the same 4472-byte stream by sha256 across the rebase), gate 2
  zero pinned cells, `cargo check` for `x86_64-apple-darwin` and
  `wasm32-unknown-unknown`. **The bite proofs are what prove the INTEGRATION
  differentials reach the new code**, and the asymmetry is the result: the
  rounding-halving-add perturbation fails `intra_simd_diff` / `build_nd_diff` /
  `intra_lowbd_diff` / `predict_intra_diff` while **`highbd_diff` (bd10/12 only,
  where the gate declines) stays green**, and the `u8` and directional suites
  stay green through both perturbations.
- **WINDOWS: RESOLVED on `windows-11-arm` on BOTH contents, and the effect ORDERS
  WITH THE CENSUS SHARE** — run 30819647374, `arms: prepost`, `base_sha` =
  the immediately-preceding commit, 24 rounds:
  **`detail` −0.961 % at 2.66x that band's own noise floor, `photo` −0.512 % at
  1.70x, 22/24 rounds each, p < 0.0001.** `detail` carries 50.6 % of the lever's
  mode family and `photo` 39.1 %, measured on the same VM in the same job — the
  census's prediction, made before the run, confirmed by it. Darwin's −0.38 %
  (photograph, 43.6 %) sits between them on a different CPU. **Allocator
  censuses identical to the digit on every arm on both runners** (`peak_live`
  differs by 1-2 bytes of 17.4 MB) — arithmetic, not allocation.
  **`windows-latest` x86-64 NOT resolvable**: `detail` −0.058 % (14/24,
  p = 0.54) at 0.14x floor, but `photo` **+0.362 %** (4/24 faster, p = 0.0015)
  at 0.85x floor — the wrong sign, under the floor. Named rather than buried:
  if the AVX2 tier were slower the RICHER content would show it and `detail` is
  a flat null, so this reads as the runner — but that is an argument, and a
  higher-`rounds` re-run on that runner is the open item.
- **Untouched and named**: PAETH (+0.87 ms), the edge filter (+0.81), the
  DC/V/H fills (+0.74 — already memset/memcpy slice ops, so that row is a
  DISPATCH-AND-CALL cost and needs a different lever), CFL (+1.40), and the
  lowbd `u8` `intra::predict` path. **SMOOTH_V / SMOOTH_H ship with a
  correctness gate and NO timing evidence** — the census says the encoder picks
  them ZERO times at this cell on all four sources.


## Encoder single-frame primary envelope (VERIFIED against reference/libaom)

Primary config = ALLINTRA (usage=2), speed-0 KEY frame. libaom's own allintra tuning
(`av1/av1_cx_iface.c:3065`) sets these **defaults** — so matching them, NOT the base defaults,
is what "single-frame exact" means:

- **CDEF: OFF** by default in allintra ("CDEF has been found to blur images, so it's disabled
  in all-intra mode"). Only `--enable-cdef` turns it on.
- **Loop-restoration: ON** by default in allintra (speeds 0-4). CORRECTED 2026-07-18 (the prior
  "OFF by default" claim was WRONG — verified first-hand against reference/libaom):
  `default_extra_cfg.enable_restoration = 1` (`av1_cx_iface.c:286`, the `!CONFIG_REALTIME_ONLY`
  build), NOT cleared by the `:3065` allintra override (which only touches CDEF / screen-mode /
  qm_min/max), and kept for non-realtime at `:1273-74` (`usage != REALTIME` → stays 1). So a
  plain `aomenc --allintra` runs `av1_pick_filter_restoration` and emits the seq/frame
  restoration syntax (even when every unit resolves RESTORE_NONE → different header bits from
  `--enable-restoration=0`). At **speed >= 5** C disables both Wiener+SGR
  (`speed_features.c:519-520`) → `enable_restoration &= 0` (`:2754`) → restoration OFF (the seq
  bit is 0). PARITY C2 was correct all along. The port's byte-exact LR search is now wired into
  the DEFAULT path (`aom-bench::EncodeCell::port_encode` derives the LR stage from the frame's
  `enable_restoration` = C's `is_restoration_used`); default parity is gated by
  `lr_default_parity::port_default_matches_plain_aomenc_allintra` (port default byte-matches a
  no-flags `aomenc --allintra`).
- **QM: OFF** by default in allintra. CORRECTED 2026-07-15 (the prior "QM: ON" claim was WRONG —
  it conflated the qm_min/max override with `enable_qm`). The allintra override at
  `av1_cx_iface.c:3065` sets `qm_min=4`/`qm_max=10` but does NOT assign `enable_qm`, which stays
  at its base default `0` (`:290/447`); `using_qm = enable_qm` (`:1310`). qm_min/max are INERT
  unless QM is turned on by `--enable-qm` (`:2076`) or `tune=IQ`/`SSIMULACRA2` (`:1946`).
  Empirical proof: the passing `encoder_gate_e2e_*` gates byte-match the port with `qm=None` —
  impossible if the reference allintra encodes were QM-on.
- screen_detection_mode = ANTIALIASING_AWARE.

**What the encoder track has byte-matched (`encoder_gate_e2e_*`):** own-search partition / mode /
tx / coefficients + LF-level derivation, in a **CDEF-off + restoration-off + QM-off** reference
encode (`shim encode_av1_kf`, cdef/restoration/qm passed as explicit params). This envelope
matches the allintra defaults for CDEF and QM (both off) but codes restoration **off** — a
NON-default config for **speeds 0-4** (the true default is restoration ON; see the corrected note
above). Those `encoder_gate_e2e_*` gates stay valid as `--enable-restoration=0` config tests
(and at speed >= 5 restoration-off IS the true default). The DEFAULT (restoration-on) path is
now separately wired + gated: `lr_restoration_gate` (the LR search, 8/8 byte-exact) +
`lr_default_parity` (port default == plain no-flags `aomenc --allintra`, restoration on). The
frame HEADER is still bootstrapped from the real parse (qindex, tile info, cdf-update, ...) —
only LF-level (and, on the default path, the restoration decision) is port-derived.

**Remaining for single-frame-PRIMARY exactness (blocks "all single frame exactly"):**
- **KB-2 (#22) cq62 speed-0 — FIXED ✅ (74fb582)**: per-block `get_intra_edge_filter_type`
  recompute in `partition_pick.rs` (a SMOOTH neighbour was not raising the angled-prediction edge
  filter → model-RD over-pruned V_PRED adj=−1 → flipped SB(32,32) partition). cq62 byte-matches +
  asserted in `encoder_gate_e2e_rich_content_strong_lf`. See the KB-2 FIXED block above.
- **#25 two latent speed-1 bugs — DONE ✅** (verified 2026-07-15): both are fixed in source
  (parameterized, no longer hardcoded 0) — `part4_prune.rs` takes a `level_index` param
  (`min(speed,3)`, with the `>=3` alternate-branch guard) and `tx_search.rs` takes an
  `intra_tx_size_init_depth_rect` field — and the asserted per-feature-revert witness
  `encoder_gate_speed1_rect_and_4way_25` (in `encoder_gate_e2e_byte_match.rs`) re-diverges if either
  fix is reverted. (Earlier "need test cells to validate" note was stale.)
- **#10 cpu-used 0..9 speed-feature sweep** (Gate 2) — **DONE ✅ (all speeds 0-9, ZERO pinned
  cells since 2026-08-02)**: speeds 0-7 (KB-8/KB-9/KB-10/KB-11; 6/7 = 64/64 canon each), speed 9
  = 64/64 canon + noise, speed 8 = **64/64** canon + noise (was 60/64 — the 4 diag
  "estimate-arm near-tie" cells closed with KB-12's `aom_hadamard_lp_8x8` transpose) — the nonrd
  PICKMODE (`use_nonrd_pick_mode`, `av1_nonrd_use_partition` single-pass walk,
  `av1_nonrd_pick_intra_mode` + `hybrid_intra_mode_search`). See KB-12. The speed sweep above is
  SYNTHETIC content; **REAL content at speed>=1 is a SEPARATE residual (KB-13, task #39)** — the
  synthetic gates are 64/64 but decoded-conformance content diverges at 36/60 speed-1..4 cells (all
  interior BLOCK_16X16/8X8 partition RD near-ties, port over-picks AB/SPLIT), pinned self-promoting
  in `encoder_gate_real_content_speed1to4_e2e`. Remaining Gate-2 byte-exactness is the
  KB-10/KB-11 speed-6/7 noise-cq63 near-tie + the KB-13 real-content set (the 4 speed-8 diag
  cells CLOSED 2026-08-02 — and they were never a tie: see KB-12's transpose root, which is a
  standing warning against reading this shape as a tie in the two that remain). (#8 qindex-from-cq and #21 decoder q62/q63 also DONE + CI-green.)

**Confirmed NON-divergences (ruled out — do not re-chase):**
- **#27 `model_based_prune_tx_search_level`.** `av1_set_speed_features_qindex_dependent` sets it
  to 0 for `{<720p, base_qindex ≤ thresh}` while the port keeps 1, but the field is **inter-only**:
  the C consumer gate lives in `av1_pick_recursive_tx_size_type_yrd` behind `is_inter_block`, so it
  is inert on the all-intra KEY path and the port never reads it. `prune_tx_size_level` is inter-only
  the same way. Coordinator independently confirmed both. Empirical guard: the new asserted
  `encoder_gate_e2e_low_qindex_speed0` (cq8–30 → qindex 32–120, 12 cells) byte-matches end-to-end
  with the field left at 1 — the previously-untested aggressive-web low-q regime is now covered.

**NOT blocking single-frame-primary (non-default single-frame knobs — these ARE single-frame work
to be done before "the rest"=inter-frame, but lower priority than the primary default config):**
- **#23 QM-on encode — DONE ✅ (2026-07-16)**: `--enable-qm=1` allintra KEY byte-matches real
  aomenc — `encoder_gate_qm_on_e2e` (40 cells, bd8+bd10, qm ranges (5,9)+(4,10), mono+420) +
  anti-vacuous witness. QM selection runs inside the RD search (`resolve_qm` per tx in
  `xform_quant`), levels via `aom_get_qmlevel_allintra`. KEY subtlety (root-caused via
  sibling-libaom dump): C's trellis weights its DISTORTION by the forward matrix ONLY under
  `dist_metric == QM_PSNR` (tune=IQ) — with default PSNR the trellis runs `qmatrix = NULL`
  while dequant still folds `iqmatrix` (`optimize_txb_qm` now takes `Option` for the dist qm).
  tune=IQ / tune=SSIMULACRA2 (QM_PSNR dist, 444-chroma level formula, chroma deltaq,
  sharpness=7) remain out of envelope. See STATUS.md 2026-07-16.
- **#7 CDEF-strength RD search — DONE ✅ (2026-07-17), BIT-IDENTICAL**: full `av1_cdef_search`
  port (`aom-encode/src/pickcdef.rs`) + the two-pass encode→LF→search→pack architecture
  (`pack_tile_from_trees`, pack.rs) — 14/14 cells byte-match real aomenc `--enable-cdef=1`
  (real content 196²/64² cq5..63 with cdef_bits=2 per-unit literals; mono/444/420/bd10
  synthetic axes). Gate: `encoder_gate_cdef_{real_content,synthetic_axes}_rd_close`
  (aom-bench, via the rd_close harness + full byte-identity asserts). CDEF stays off by
  default — the default envelope is untouched. FAST search levels 1..5 ported (table-level
  unit tests); only FULL (speed 0) is e2e-gated so far. See STATUS.md 2026-07-17.
- **Loop-restoration (Wiener/SGR) search — DONE ✅ (2026-07-18), BIT-IDENTICAL + DEFAULT-WIRED**:
  loop-restoration is **ON by default** in allintra (speeds 0-4; NOT a non-default knob — the
  prior "off by default" note here was WRONG, see the corrected primary-envelope note above). The
  byte-exact `av1_pick_filter_restoration` search (`crates/aom-dsp/src/restore/pick.rs`; PARITY C2) is now
  wired into the port's DEFAULT path (`aom-bench::EncodeCell::port_encode` derives the LR stage
  from the frame's `enable_restoration` = C's `is_restoration_used`). Gates: `lr_restoration_gate`
  (the search, 8/8 real content + 3/3 mono/444/bd12 format axis) + **`lr_default_parity`** (the
  port's default encode byte-matches a genuinely no-flags `aomenc --allintra` — the reference is
  the new `shim_encode_av1_kf_defaults`, tools at allintra defaults). Restoration-off remains a
  valid non-default config tested by the `encoder_gate_e2e_*` gates. Speeds 1-4 LR search arms +
  GOOD-mode arms are source-verified but PINNED (real-content base encode not yet byte-exact at
  speed >= 1, KB-13); speed >= 5 is structurally LR-off in C. See STATUS.md 2026-07-18.

## Coordination (parallel tracks)

- Max clean parallelism = **2** (one decoder agent + one encoder agent); cargo's shared
  target-dir lock serializes builds, which keeps the box safe.
- Strict crate ownership; commit with **explicit per-file staging** (`git add <paths>`, never
  `-A`/`-u`/`.`); shared `STATUS.md` via `git add -p`. Push `git push origin HEAD:main`; verify
  `git merge-base --is-ancestor HEAD origin/main`.
- Coordinator independently verifies every landing (on origin, boundary-clean, no `#[ignore]`
  / weakened asserts, gate is a real byte-identity assertion, CI green). Never trust a claim.

## Zen codec cross-cutting compliance (decode backend) — SPEC (2026-07-20)

zenav1-aom is a **decode backend** consumed by zenavif (feature `aom-backend`,
`decode_av1_obu_yuv_aomrs` → `aom_decode::frame::decode_frame_obus`). Its input
is an **untrusted AV1 bitstream**, so it carries the *high* bar: a hostile or
truncated stream must never panic, never abort on OOM, and must fail with a
*categorizable, located* error. This section specs the six zen cross-cutting
contracts against the current state (audited 2026-07-20). The reference codec
is zenavif; the contract types live in `zencodec` 0.1.26 (`src/error.rs`,
`src/limits.rs`, `src/estimate.rs`) and `whereat`/`enough`.

**Design rule: stay codec-only.** zenav1-aom must NOT take a hard dependency on
`zencodec`. It stays a pure codec crate; the *integration* crate (zenavif) owns
the `CategorizedError`/`ResourceLimits`/`estimate` trait impls. zenav1-aom's job
is to expose a **structured, located, category-bearing error enum**, accept a
**limits struct** and a **stop token**, and make its allocations **fallible on
demand** — so zenavif can map cleanly without losing information. Optional
`zencodec`/`whereat` integration may live behind default-off features.

### 1. Limits enforcement — PARTIAL (hardcoded ceiling landed; configurable still open)
- **Landed (task #60):** a hardcoded `MAX_DECODE_PIXELS = 1 << 28` (~268 Mpx)
  DoS bound rejects over-large frames **before** the recon allocation
  (`frame.rs:233`, commit `1b65d61`), plus film-grain scaling-point and
  segment-id bounds (`5922c47`, `606813d`). So the crafted-header OOM-abort is
  closed at a fixed ceiling.
- **Still open:** the ceiling is not yet *configurable* and there is no way for
  the caller (zenavif) to pass its own `ResourceLimits`. Add a `DecodeLimits
  { max_pixels, max_width, max_height, max_memory_bytes }` (all `Option`,
  `None` = the current hardcoded default) and a config-carrying entry
  `decode_frame_obus_with(data, &DecodeConfig)`; check header dims against the
  passed limits (still after header parse, before first alloc), returning the
  limit-exceeded variant (§4). Keep the bare `decode_frame_obus(data)` applying
  the `1<<28` default. This lets zenavif thread its `frame_size_limit` /
  `parser_*` caps through instead of relying on a fixed 268 Mpx.
- **Acceptance:** a caller passing `max_pixels = 1_000_000` gets `Err(…Limit…)`
  on a 2 Mpx header before allocating; the default path still stops at 268 Mpx.

### 2. Resource estimation — MISSING
- **Bar:** a caller must be able to pre-flight peak memory/time from the header
  without decoding. Currently there is no header-only probe (even
  `decode_frame_obus_prefilter` fully decodes a tile) and no estimate API.
- **Add:** (a) `probe_header(data) -> Result<FrameInfo, DecodeError>` that
  parses only the sequence+frame headers and returns dims/bit_depth/subsampling/
  monochrome — cheap, allocation-light, the input to both limit checks and
  estimation. (b) `estimate_decode(info) -> DecodeEstimate { peak_memory_bytes,
  time_ms }` keyed on pixels × bit_depth. zenavif's `heuristics::estimate_decode`
  is the shape to mirror; zenavif can call `probe_header` + its own calibrated
  model, so a minimal honest peak-memory bound here is enough.

### 3. whereat traces / structured errors — MISSING (`String` today)
- **Bar:** every fallible entry returns a structured error carrying a source
  location. Today both public entries return `Result<_, String>`
  (`frame.rs:679,723,1058`) — 21 distinct flat string reasons, no location, no
  categories; the zenavif seam discards even the string
  (`decode_av1.rs:625` → generic `code: -1`).
- **Add:** a `#[non_exhaustive] pub enum DecodeError` (thiserror) replacing
  `String`, with the variants in §4. Behind a default-off `whereat` feature,
  `define_at_crate_info!()` at the crate root and return `At<DecodeError>` so
  traces link to repo+commit; without the feature, the bare enum still carries
  category + a message. `DecodeError: core::error::Error` with a correct
  `source()` chain.

### 4. Category granularity (feeds zencodec `CategorizedError`) — MISSING
- **Bar:** the error enum must let a consumer distinguish, at minimum:
  corrupt-bitstream vs truncated-input vs unsupported-*type* vs
  unsupported-*feature* vs limit-exceeded vs internal-bug. zenavif main already
  implements `zencodec::CategorizedError for Error` (error.rs:161, two-level
  `ErrorCategory`: `Image(Malformed|UnexpectedEof|Unsupported{Type,Feature})`,
  `Request(...)`, `Resource(Limits(kind)|OutOfMemory)`, `Stopped`, `Internal`).
  The aom seam currently collapses all 21 reasons to one `Error::Decode
  { code:-1 }`, so every failure lands in the coarsest bucket.
- **Add:** `DecodeError` variants that map 1:1 onto those categories, so the
  zenavif seam translates variant→zenavif `Error`→existing category instead of
  flattening. Suggested set (each carries a `&'static str` / small context):
  - `Truncated` (short OBU/tile/leb128 — `"OBU size past end"`,
    `"truncated tile payload"`, `"truncated tile-size prefix"`) → maps to
    **Image::UnexpectedEof**.
  - `Malformed(reason)` (`"bad OBU header"`, header-before-seq-header, tile
    group without frame header, `"no frame in stream"`, corrupt block-size
    index at `lib.rs:5169`, invalid partition at `:5233`, invalid intrabc DV
    at `:3850`) → **Image::Malformed**. THESE MUST BECOME `Err`, NOT
    `panic!`/`expect`/`unreachable!` (§5).
  - `UnsupportedType(what)` (subsampling `"unsupported subsampling"`,
    `frame_type` unsupported) → **Image::Unsupported(Type)**.
  - `UnsupportedFeature(what)` (KEY/intra scope only: `"second frame"`,
    `show_existing_frame`, inter-before-ref, `frame_size_override`, mixed
    lossless segments, multi-tile superres, forced screen-content) →
    **Image::Unsupported(Feature)**. (These are honest "codec doesn't
    implement it yet", distinct from corruption.)
  - `LimitExceeded { kind, actual, max }` (§1) → **Resource::Limits(kind)**.
  - `AllocFailed` (§5 fallible path) → **Resource::OutOfMemory**.
  - `Cancelled(StopReason)` (§6) → **Stopped**.
  - `Internal(reason)` (broken invariant that is genuinely a code bug, not
    input-driven) → **Internal::Bug**.
- **Acceptance:** the zenavif seam (`decode_av1.rs`) maps each `DecodeError`
  variant to the matching zenavif `Error` variant — no more blanket `code:-1`
  — and a test asserts the category survives to `error_category()`.

### 5. Panic-freedom (LANDED, keep clean) + configurable fallible alloc (open)
- **Landed (task #60):** a cargo-fuzz harness now exists
  (`crates/aom-decode/fuzz`, commit `bbd7bc4`) over the OBU decode entry points,
  and it drove the elimination of 5 escaping panics (`88b4de3`) plus conversion
  of corrupt / out-of-envelope panics and bit-reader errors to `Err`
  (`bbd7bc4`, `5922c47`), with a stable-toolchain regression harness
  (`606813d`). Panic-freedom on the *current* untrusted surface is
  substantially met and mechanically guarded.
- **Fuzz-status (2026-07-23/24 sustained campaign, ~5.3 CPU-hours):** the two
  targets were built and driven on nightly `cargo-fuzz` 0.13.2, seeded from the
  553-file `decode_frame_obus` + 39-file `decode_frames` conformance corpora.
  A crash-finding round (short) plus a clean round of **4 workers × 2400 s per
  target** (≈2.67 CPU-h each, ≈5.3 CPU-h total) accumulated corpora of 9,474
  (`decode_obus`) / 7,189 (`decode_frames`) inputs. Two issues found and fixed:
  - **`9069a95` — spurious OOM (harness):** the targets called the bare
    `decode_frame_obus` / `decode_frames` (default `1<<28` ≈268 Mpx ceiling); a
    234-byte OBU can declare a ~268 Mpx frame whose in-bounds recon/mi alloc is
    ~3.2 GiB and trips libFuzzer's 2 GiB malloc limit. Not a decoder bug (the
    alloc IS bounded by the ceiling). Per §5 the targets + the stable
    `fuzz_regression.rs`/`fuzz_sweep.rs` harnesses now decode via `*_with` under
    a low `max_pixels = 1<<22` (4 Mpx). Seed:
    `fuzz/regression/decode_obus_oom_268mpx_declared_frame.obu`.
  - **`d7aa3c8` — real panic, arithmetic overflow:** `read_timing_info_header`
    computed `read_uvlc() + 1` for `NumTicksPerPicture`; `read_uvlc()` returns
    `u32::MAX` for ≥32 leading zeros (attacker-reachable in the seq-header
    timing_info), so `+1` panicked under overflow-checks — a DoS on untrusted
    input. Both entry points found it (12 crash inputs, one root cause). Fixed
    with `saturating_add(1)` (timing info is pixel-decode-irrelevant; matches
    spec intent, strictly better than libaom's C wrap-to-zero). Seed:
    `fuzz/regression/decode_frames_timing_num_ticks_uvlc_overflow.obu`.
  - **No remaining crashes/panics/OOMs** over the clean round on either target;
    the full `-p zenav1-aom-decode` suite (byte-identity tests included) stays
    green. One `slow-unit` (277 B → ~834 ms optimized / ~8–13 s instrumented)
    was triaged as **legitimate O(pixels) work**: it declares a 4.06 Mpx frame
    just under the fuzz cap; any smaller `max_pixels` rejects it in ~50 µs
    (`LimitExceeded`). Not a loop/leak — the pixel ceiling governs decode cost
    exactly as designed; a latency-sensitive caller uses a tighter `max_pixels`
    and the (speced §6) stop token. Fixed crashes archived to
    `/root/fuzz-corpus/zenav1-aom/`.
  - **Remaining fuzz risks / not-yet-done:** (a) the campaign fuzzed the default
    runtime-SIMD path — the `AOM_FORCE_SCALAR=1` scalar path was only
    *replayed* over the accumulated corpus (deterministic, clean), not fuzzed
    for new coverage; a future round should fuzz under `AOM_FORCE_SCALAR=1`.
    (b) Coverage is bounded by the `1<<22` fuzz cap (the reject path above that
    is not exercised for deep decode). (c) The inter-frame decode surface grows;
    per the bar below, each new feature must land with its own fuzz coverage.
- **Bar (keep it clean as features grow):** every NEW decode feature (inter,
  intraBC coeff arm, extended tools) must land with its fuzz coverage and must
  convert any new bitstream-derived `panic!`/`expect`/`unreachable!` into an
  `Err`, not a crash — do not regress the property task #60 established. Any
  remaining low-level infallible tile driver (`decode_tile_kf`, `TileKf`) that
  can be reached with attacker geometry either returns `Result` or carries a
  comment naming the guard that makes its indexing safe.
- **Bar (alloc — the perf trade is a SETTING, per user directive) — STILL
  OPEN:** fallible vs infallible allocation must be a **configurable knob**, not
  hardcoded. Buffers are still infallible `vec![v; n]` (one `calloc`, faster,
  but aborts on OOM); fallible `try_reserve_exact` returns
  `DecodeError::AllocFailed` gracefully. Add an `AllocMode { Fallible,
  Infallible }` (or a plumbed `zencodec::AllocPreference` at the seam) on
  `DecodeConfig`; route every header-sized buffer (`recon`, `mi`, `seg_map`,
  film-grain, superres) through a helper honoring it. Default: **Fallible** for
  this untrusted decoder (a decoder favours safety, and the `1<<28` ceiling
  already caps the size); a trusted/bench caller opts into Infallible for the
  single-`calloc` speed. Mirror zenavif `alloc_util.rs` (`AllocPref` +
  `alloc_filled`/`vec_with_capacity`).
- **Keep:** `#![forbid(unsafe_code)]` (already present, all three crates).
- **Bar (alloc — and the perf trade is a SETTING, per user directive):**
  fallible vs infallible allocation must be a **configurable knob**, not
  hardcoded. Infallible `vec![v; n]` is one `calloc` (faster) but aborts on
  OOM; fallible `try_reserve_exact` returns `DecodeError::AllocFailed`
  gracefully. Add an `AllocMode { Fallible, Infallible }` (or reuse a plumbed
  `zencodec::AllocPreference` at the seam) on `DecodeConfig`; route every
  header-sized buffer (`recon`, `mi`, `seg_map`, film-grain, superres) through
  a helper that honors it. Default: **Fallible** for this untrusted decoder
  (a decoder favours safety); a trusted/bench caller opts into Infallible for
  the single-`calloc` speed. Mirror zenavif `alloc_util.rs` (`AllocPref` +
  `alloc_filled`/`vec_with_capacity`).
- **Bar (fuzz — the enforcement):** add `fuzz/` with a `cargo-fuzz` target
  feeding arbitrary bytes to `decode_frame_obus_with` under a low
  `max_pixels`; any panic/abort/OOM is a bug. There is **no fuzzing today**
  (only conformance/diff corpora). This is the mechanical gate for the two
  bars above. Corpus/artifacts to block storage per the global rule, never
  committed.
- **Keep:** `#![forbid(unsafe_code)]` (already present, all three crates).

### 6. Stop-token cancellation — MISSING
- **Bar:** a long decode must be cancellable at coarse boundaries. No entry
  point takes a token today; a decode runs to completion or panic.
- **Add:** thread an `&impl enough::Stop` (default `enough::Unstoppable`,
  zero-cost) through `decode_frame_obus_with` and poll `stop.check()?` at tile
  boundaries (the tile loop in `decode_frame_tiles_kf`) and, for `decode_frames`,
  per frame; map `StopReason` → `DecodeError::Cancelled`. `enough` is zero-dep
  `no_std` — acceptable for a codec-only crate. Cadence: at least once per tile
  / superblock-row so cancellation is observed within bounded work.

### Priority order for this backend
0. **DONE (task #60):** panic-freedom on the current surface + `1<<28` DoS
   ceiling + cargo-fuzz harness (§1/§5 landed portions). Keep this property as
   features grow.
1. §3/§4 structured `DecodeError` (replace `String`) with category-bearing
   variants — the highest-value remaining item, so the zenavif seam stops
   collapsing 21 reasons to `code:-1`.
3. §5 configurable `AllocMode` (the perf/safety trade as a setting).
4. §2 `probe_header` + estimate, §6 stop token.

When any item lands, update the zenavif seam (`src/decode_av1.rs`
`decode_av1_obu_yuv_aomrs`) in the SAME change to consume it (pass limits/token,
map the new error variants) — a backend capability the integration ignores is
not "done".
