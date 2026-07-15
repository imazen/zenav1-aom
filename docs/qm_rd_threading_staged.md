# QM-on forward-quant: RD-search threading (STAGED design)

**Task #23 — QM-on allintra KEY-frame encode.** This document stages the *one
remaining* piece of #23: threading the per-block quantization-matrix (QM) context
through the encoder's RD search. It is written as a **design to apply**, not a
literal patch, because the target files (`tx_search.rs`, `intra_rd.rs`,
`partition_pick.rs`) are the encoder track's active primary domain (#25/#27/#10)
and will move under them — apply this additively when that work quiesces.

QM is **OFF by default** in allintra (`enable_qm=0`; `qm_min/max=4/10` are inert
unless `--enable-qm` / `tune=IQ` / `tune=SSIMULACRA2`). So this is Gate-4 knob
coverage, and the **overriding invariant is: when QM is off, every byte is
identical to today** (see §6).

## 1. What is already landed (byte-exact vs real C) — do NOT redo

| Piece | Where | Validation | Commit |
|---|---|---|---|
| Forward `wt_matrix_ref` table | `aom-quant/src/qm_fwd_tables.rs` | 9 source anchors + 912-cell diff vs `av1_qm_init` | `624e91d` |
| Forward selector `aom_quant::qmatrix(level,plane,tx,type)` | `aom-quant/src/qm.rs` | `qm_fwd_select_diff` (912 cells) | `624e91d` |
| qmlevel derivation `aom_get_qmlevel[_allintra]` | `aom-quant/src/quant_common.rs` | `qm_level_diff` (qindex 0..=255 × 6 ranges) | `a066cf8` |
| Inverse selector reuse `aom_decode::qm::iqmatrix` (doc-hidden `pub`) | `aom-decode/src/qm.rs` | `inverse_qmatrix_reuse_matches_c` (912 cells) | `abb68d9` |
| Block-level compose (qindex→level→select fwd+inv→`xform_quant`) vs C | `aom-encode/tests/qm_forward_block_diff.rs` | `forward_qm_block_realistic_matches_c` | `abb68d9` |
| QM quantizer kernels (`av1_quantize_*_qm`) + `xform_quant` qm/iqm dispatch | `aom-quant/src/lib.rs`, `aom-encode/src/lib.rs` | `xform_quant_diff`, `quantize_qm_diff` | (pre-existing) |
| C QM-encode shim `ref_encode_av1_kf_qm` (`AV1E_SET_ENABLE_QM/QM_MIN/QM_MAX`) | `aom-sys-ref` + `dec_shim.c` | `qm_encode_witness` (QM-on ≠ QM-off) | (pre-existing) + `be758f9` |

**The forward+inverse selection, the level derivation, the kernels, and the C
reference all byte-match libaom.** The only gap is *wiring the selection into the
production RD/encode path* so the port's own encode applies QM.

## 2. Why the RD search must carry QM (not just the final pack)

C applies QM **inside** `av1_xform_quant`, which runs during the mode / tx-type /
partition RD search — so QM changes *which* mode/tx/partition wins, not merely the
final coefficients. If only the final pack applied QM, the port's RD decisions
would diverge from C and the bitstream would not byte-match. Therefore the qm
context must reach every `xform_quant` / `xform_quant_optimize` call in the search.

## 3. C reference call chain (libaom v3.14.1)

- **Per frame** — `av1_set_quantizer` (`av1/encoder/av1_quantize.c:878`) sets
  `quant_params->qmatrix_level_y/u/v`. For **allintra** (usage=2) it uses
  `get_luma_qmlevel = aom_get_qmlevel_allintra`, `get_chroma_qmlevel =
  aom_get_qmlevel_allintra` (or `_444_chroma` when 4:4:4). Args:
  `level_y = get_luma(base_qindex, qm_min, qm_max)`,
  `level_u = get_chroma(base_qindex + u_ac_delta_q, …)`,
  `level_v = get_chroma(base_qindex + v_ac_delta_q, …)`.
  (Ported: `aom_quant::aom_get_qmlevel_allintra`.)
- **Per block/segment** — `set_qmatrix` (`av1_quantize.c:775`) fills
  `xd->plane[p].seg_qmatrix[seg_id]` from `qmatrix_level_*` when
  `av1_use_qmatrix(quant_params, xd, seg_id)` = `using_qmatrix && !lossless[seg]`,
  else the flat top level `NUM_QM_LEVELS-1`.
- **Per (plane, tx_size, tx_type)** — `av1_setup_qmatrix` (`av1_quantize.c:370`)
  sets `qparam->qmatrix = av1_get_qmatrix(quant_params, xd, plane, tx_size,
  tx_type)` and `qparam->iqmatrix = av1_get_iqmatrix(...)`. **This is the
  per-transform selection** — 1-D / identity transforms (`tx_type >= IDTX`) and
  the flat level select `NULL` (flat). (Ported: `aom_quant::qmatrix` +
  `aom_decode::qm::iqmatrix`.)
- `av1_xform_quant` (`av1/encoder/encodemb.c:284`) then calls the quant func with
  `qparam->qmatrix/iqmatrix`. (Ported: `aom_encode::xform_quant`'s qm/iqm arms.)

## 4. Port design (apply additively)

### 4a. Carry the qm context on `QuantParams` (in `aom-encode/src/lib.rs` — owned by the QM track)

`QuantParams` currently exposes pre-selected `qm: Option<&[u8]>` / `iqm:
Option<&[u8]>` slices (kept for the kernel diff tests). The RD loop reuses one
`QuantParams` across all `tx_type`s, but QM selection depends on `tx_type` — so the
selection must move **inside** `xform_quant`, driven by a frame-level context:

```rust
#[derive(Clone, Copy)]
pub struct QmCtx { pub qm_level: usize, pub plane: usize } // qm_level = frame qmatrix_level_[plane-group]

// add to QuantParams:
pub qm_ctx: Option<QmCtx>,   // None => QM off (flat), byte-identical to today
```

`from_plane_rows` sets `qm_ctx: None`. Add a QM constructor/builder used only on
the QM path, e.g. `QuantParams::from_plane_rows(...).with_qm(qm_level, plane)`.

### 4b. Select per (tx_size, tx_type) inside `xform_quant` / `xform_quant_optimize`

At the top of `xform_quant` (which already has `tx_size`, `tx_type`), resolve the
effective slices — mirroring `av1_setup_qmatrix`:

```rust
let (qm_sel, iqm_sel) = match qp.qm_ctx {
    Some(cx) => (
        aom_quant::qmatrix(cx.qm_level, cx.plane, tx_size, tx_type),   // fwd (None for 1-D/identity/flat)
        aom_decode::qm::iqmatrix(cx.qm_level, cx.plane, tx_size, tx_type),
    ),
    None => (qp.qm, qp.iqm),   // explicit-slice / flat path — unchanged
};
```

Then dispatch on `(kind, qm_sel, iqm_sel, hbd)` exactly as today. `qm_sel` and
`iqm_sel` are always both-`Some` or both-`None` (the selectors agree on the
`tx_type` gating — asserted in `qm_forward_block_diff`). The `xform_quant_optimize`
trellis path already branches on `(qp.qm, qp.iqm)` → route it through the same
resolved slices (its `optimize_txb_qm` is already ported+tested).

### 4c. Frame QM state source

The frame config that drives the encode must carry `enable_qm`, `qm_min`,
`qm_max` (from the shim / API controls), and derive, once per frame:

```
qmatrix_level[0] = if enable_qm { aom_get_qmlevel_allintra(base_qindex,               qm_min, qm_max) } else { 15 }
qmatrix_level[1] = if enable_qm { aom_get_qmlevel_allintra(base_qindex + u_ac_delta_q, qm_min, qm_max) } else { 15 }
qmatrix_level[2] = if enable_qm { aom_get_qmlevel_allintra(base_qindex + v_ac_delta_q, qm_min, qm_max) } else { 15 }
```

For the **primary allintra KEY 4:2:0** case (deltas 0, not lossless) all three
equal `aom_get_qmlevel_allintra(base_qindex, qm_min, qm_max)`. Level `15`
(=`NUM_QM_LEVELS-1`) makes every selector return `None` → the flat path. Mirror
the decode-side `frame_qm_levels` gating (`aom-decode/src/lib.rs:557`:
`using_qmatrix && !seg_lossless`).

### 4d. Call sites that must pass the qm context

Each site builds/uses `QuantParams`; on the QM path it must attach
`QmCtx { qm_level: qmatrix_level[plane_group], plane }`. **Line numbers are
approximate — these files churn under #10; match by function/role.**

FORBIDDEN for the QM track — the encoder track applies these (or coordinates):
- `tx_search.rs` ~`search_tx_type_intra` (builds `qp = from_plane_rows(inp.rows,
  kind, inp.bd)`; doc says "flat quant (no qmatrix), plane 0"). Add a qm field to
  `TxTypeSearchInputs` (the frame `qmatrix_level` + `plane`) and `.with_qm(...)`.
- `intra_rd.rs` — threads `qp: &QuantParams`; ensure its inputs/env carry the qm
  context so the `qp` it passes to `xform_quant_optimize` is QM-aware.
- `partition_pick.rs` — the frame driver (`PickFrameCfg`): carry `enable_qm/qm_min/
  qm_max`, derive `qmatrix_level[3]` once, and thread it into the leaf search
  inputs above.

Non-forbidden — the QM track can do these once the frame context exists:
- `intra_uv_rd.rs` — builds `inp` for `search_tx_type_intra` (chroma): pass
  `plane = 1` (+ the chroma qm level).
- `encode_intra.rs` — the reconstruct/encode pass (`from_plane_rows` at ~398/659):
  attach `QmCtx` for the winner block's plane.
- `pack.rs` — the final coeff pack re-runs `xform_quant_optimize` on the winner;
  attach the same `QmCtx` so the packed coefficients match the RD-search ones.

## 5. Reconstruction / dequant

The block reconstruct path (`aom_encode::reconstruct` → `aom_txb::dequant_txb`,
`lib.rs:~828`) already takes `iqmatrix: Option<&[u8]>`. On the QM path pass
`aom_decode::qm::iqmatrix(qm_level, plane, tx_size, tx_type)` so recon uses the
same inverse weights as decode (`dqcoeff = (qcoeff·dequant·iqm + 16) >> 5`).

## 6. Non-negotiable invariant: QM-off is byte-identical

When `enable_qm == 0` (the default), `qm_ctx` is `None` at **every** site →
`xform_quant` takes the exact flat path it takes today → all existing
`encoder_gate_e2e_*` gates stay byte-identical. This is the property that lets the
threading land without disturbing the primary #25/#27/#10 work. Verify by running
the full `aom-encode` suite before/after: **zero** diff on the QM-off gates.

## 7. Part C — the QM-on e2e gate (add once threading lands)

`aom-encode/tests/encoder_gate_qm_on_diff.rs`:
- Encode a textured KEY frame with the port at `enable_qm=1, qm_min=4, qm_max=10`;
  assert **port bytes == `ref_encode_av1_kf_qm(...)` bytes**, at **bd8 and bd10**.
- **Anti-vacuous** (already proven for the C reference in `qm_encode_witness`):
  assert the port's QM-on bytes **differ** from its own QM-off bytes for the same
  content — so the gate can't pass with QM silently doing nothing.
- Start with a single small cell (e.g. 64×64 textured, cq 32), then widen. If the
  first cell diverges, root-cause with the sibling-libaom RD-dump method (per KB-2/
  KB-3) — the divergence will be a per-block QM selection or a
  qmatrix_level-derivation mismatch, both now individually byte-validated, so the
  bug will be in the *threading* (wrong plane, wrong level, missed call site).

## 8. Suggested landing order

1. (QM track) frame QM state + `QmCtx` + `xform_quant`/`xform_quant_optimize`
   selection + `from_plane_rows().with_qm()` — all in `aom-encode/src/lib.rs`
   (owned), no-op when off. Add a unit test that `qm_ctx=None` reproduces today's
   `xform_quant` output exactly.
2. (encoder track / coordinated) the three forbidden RD-search sites (§4d).
3. (QM track) `encode_intra.rs` / `pack.rs` / `intra_uv_rd.rs` + reconstruct iqm.
4. (QM track) `encoder_gate_qm_on_diff` (§7), bd8+bd10, anti-vacuous.
