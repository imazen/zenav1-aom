# C-oracle instrumentation — versioned, never committed to the submodule

The divergence hunts that closed KB-55..68 all worked the same way: an env-gated
`fprintf` in the real libaom (`upstream/`, pinned at `03087864`, v3.14.1) paired with
an identically named print on the port side. The port half is committed, byte-inert,
and since 2026-09-24 behind a RUNTIME API rather than the environment:
`aom_dsp::trace` (`crates/aom-dsp/src/trace.rs`) — `install(TraceConfig)` /
`clear()`, a `Trace` kind per print family, a `Focus` slot per `<mi_row>,<mi_col>`
node selector, an optional output sink, and `trace_on!` / `trace_focus!` /
`trace_out!` at the sites. The published crates never read the environment on a
trace path; `TraceConfig::from_env()` / `install_from_env()` map the `AOM_*` names
below onto the API for hosts that want them, and the `trace-env` cargo feature
(default OFF, on for every in-tree harness build through `__internals`) seeds that
lazily so `AOM_TX_DBG=.. cargo test` keeps working. The C half cannot be committed —
the submodule gitlink IS the oracle pin, and the differential gates must link the
pristine tree — so it lives here as patches against that commit.

`.gitmodules` sets `ignore = dirty`, so a root `git status` does not show an
instrumented oracle. `just upstream-check` (first step of `just gate-landing`) does.

## Recipes

```
just upstream-instrument                # apply the newest set + relink the oracle
just upstream-instrument docs/upstream-instrumentation/2026-09-12-kb55-58-traces.patch
just upstream-pristine                  # revert + relink; landing gate requires this
just upstream-check                     # fails if upstream/ is dirty
git -C upstream diff > docs/upstream-instrumentation/<date>-<what>-traces.patch   # save a new set
```

Every patch applies to the pristine pinned tree. The two sets below are NOT a
superset chain: the 09-12 set has 149 added lines (`[ceob]`, `[cl]`, cdef-control
prints) that the 09-24 set dropped, and the 09-24 set adds 13 gates the 09-12 set
never had. Apply one at a time; `git apply --3way` of one over the other does not
resolve.

## Sets

| file | captured | gates | lines | hunts it served |
|---|---|---|---|---|
| `2026-09-12-kb55-58-traces.patch` | 2026-09-12 | `AOM_TX_DBG` `AOM_PART_DBG` `AOM_CDEF_DBG` `AOM_DQ_DBG` | 589 | KB-55 (phase-2 rdmult fold), KB-56/57 (CDEF pick / adaptive), KB-58 (delta-q plumbing) |
| `2026-09-24-kb59-68-traces.patch` | 2026-09-24 | the 17 below | 965 | KB-59..68: tune rdmult scopes, intra-dct-only chroma, CNN u16, AB-reuse stale map, winner map order, lossless IntraBC, SCM trial, `screen_512` |

## Gate inventory (2026-09-24 set)

Port sites are `crates/...`; every site is one relaxed atomic load of the installed
mask (an uncached `getenv` there measured 1.4 % flat at the shipping preset,
`7099a04`). `NAME=0` and `NAME=` are OFF in the env mapping. The `Trace`/`Focus`
column is the runtime-API name (`aom_dsp::trace`).

| env gate | `Trace` / `Focus` | C files (this set) | port site(s) |
|---|---|---|---|
| `AOM_TX_DBG=<r>,<c>` | `Focus::Tx` | reconintra.c, encodemb.c, intra_mode_search.c/.h, palette.c, partition_search.c, tx_search.c | `aom-encode/src/tx_search.rs` `tx_dbg_target` |
| `AOM_PART_DBG=<r>,<c>` | `Focus::Part` | partition_search.c | `aom-encode/src/partition_pick.rs` `part_dbg_target` |
| `AOM_UV_DBG=<r>,<c>` | `Focus::Uv` | reconintra.c, intra_mode_search.c, tx_search.c | `aom-encode/src/intra_uv_rd.rs` `uv_dbg_target` |
| `AOM_P4_NOBUDGET` | `Trace::P4NoBudget` | — (port-only: unlimited 4-way strip budget) | `aom-encode/src/partition_pick.rs` |
| `IBC_TRACE` | `Trace::Ibc` | rdopt.c, tx_search.c | `aom-encode/src/intrabc_search.rs` (`[r-mv]`, `[r-arms]`) |
| `AOM_IBC_WIN`, `AOM_IBC_COEFF_TRACE` | `Trace::IbcWin / Trace::IbcCoeff` | — (port-only) | `intrabc_search.rs`, `pack.rs` |
| `AOM_SCT_DBG` | `Trace::Sct` | encoder.c | `aom-encode/src/key_frame.rs` (`[sct-port]`, `[lf-port]`) |
| `AOM_SCT_TRIAL_DBG` | `Trace::SctTrial` | encoder.c, encoder_utils.c | `key_frame.rs` (`[sct-trial]`) |
| `AOM_SSM_DBG` | `Trace::Ssm` | partition_search.c | `aom-encode/src/encode_sb.rs` (`[ssm-port]`) |
| `AOM_DQ_DBG` | `Trace::Dq` | encodeframe.c | `key_frame.rs` (`[dq-port]`) |
| `AOM_DQ_DEC` | `Trace::DqDec` | — (port decoder) | `aom-dsp/src/entropy/partition.rs` (`[dq-dec]`) |
| `AOM_CDEF_DBG` | `Trace::Cdef` | bitstream.c, pickcdef.c | `aom-encode/src/pickcdef.rs`, `aom-dsp/src/entropy/header.rs` |
| `AOM_HDR_TRACE`, `AOM_HDR_DUMP=<path>` | `Trace::Hdr` / `TraceConfig::with_hdr_dump` | — (port-only header field boundaries / `{p:#?}` dump) | `aom-dsp/src/entropy/header.rs`, `key_frame.rs` |
| `AOM_SYM_TRACE` / `AOM_WSYM_TRACE` | `Trace::Sym / Trace::WSym` | entdec.c | `aom-dsp/src/entropy/dec.rs` / `enc.rs` |
| `AOM_PAL_TRACE` | `Trace::Pal` | decodemv.c | `aom-dsp/src/entropy/partition.rs` (`[pal]`, `[palc]`) |
| `AOM_PART_TRACE` | `Trace::Part` | decodeframe.c | `aom-decode/src/lib.rs` (`[dp]`) |
| `AOM_DV_TRACE` | `Trace::Dv` | — (port decoder) | `aom-decode/src/lib.rs` (`[dv]`) |
| `AOM_CTXB_TRACE` | `Trace::Ctxb` | decodetxb.c | `aom-encode/src/pack.rs` (`[wtxb]`) |
| `AOM_PACK_TRACE` | `Trace::Pack` | — (port-only) | `pack.rs` (`[pk]`) |
| `AOM_DBG_BLOCKS` | `Trace::DbgBlocks` | (use `/root/aom-inspect`, CLAUDE.md) | `aom-decode/src/lib.rs` |
| `AOM_TRELLIS_CALLS` | `Trace::TrellisCalls` | — (port-only call counter) | `aom-encode/src/lib.rs`, `tx_search.rs` |
| `AOM_TIME_PHASES` | `Trace::TimePhases` | — (port-only phase timing) | `key_frame.rs` |
| `AOM_LEAF_TRACE`, `AOM_CMODE_TRACE`, `AOM_CDV_TRACE`, `AOM_CDP_TRACE` | — (C only) | decodeframe.c, decodemv.c | **no port twin** — C-only decoder dumps used against `aomdec` output directly |

`AOM_FORCE_SCALAR` is not a trace; it is the dispatch pin (`aom-dsp/src/dispatch/mod.rs`)
and keeps its own set-once cached read — its tests assert the pin holds for the life of
the process, so it is deliberately outside `aom_dsp::trace`.
