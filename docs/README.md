# docs/ — what to read, in order

**Fresh eyes: read these three, in this order.**

| file | what it is |
|---|---|
| [`../CLAUDE.md`](../CLAUDE.md) | The standing goal, the six clauses with their **current** measured state, the gates, the landing rules. Kept small on purpose (loaded every turn). |
| [`ITERATION_PLAYBOOK.md`](ITERATION_PLAYBOOK.md) | The fast loops: land a perf lever, port a C feature, close a divergence, compare against other encoders, check CI. One command per step. |
| [`CYCLE_LEDGER_2026-09-08_11.md`](CYCLE_LEDGER_2026-09-08_11.md) | What the last 2.5-day cycle did, in what order, what it rejected and why, what was in flight when it ended, and the decisions that are the user's. |

**Live references** (append to these; do not duplicate their content elsewhere):

| file | holds |
|---|---|
| [`KNOWN_BUGS.md`](KNOWN_BUGS.md) | The durable bug log — every KB-* entry with root, fix, gate and bite proof. Indexed from `CLAUDE.md`. |
| [`CLAUSE_STATUS_LOG.md`](CLAUSE_STATUS_LOG.md) | The complete measured history of the six clauses; row (4) is the whole clause-(4) perf record. |
| [`COVERAGE_QUEUE.md`](COVERAGE_QUEUE.md) | Named-but-unmeasured axes, ranked by reachability. |
| [`DIFFERENTIAL_PLAYBOOK.md`](DIFFERENTIAL_PLAYBOOK.md) | The 15 rules for differential testing and benchmarking (§1 bite proofs, §6 control bands, §14 re-profile first). |
| [`../PARITY.md`](../PARITY.md), [`../STATUS.md`](../STATUS.md), [`../CHANGELOG.md`](../CHANGELOG.md) | Feature-by-feature parity ledger; the per-landing narrative log (newest first, 360 KB — grep it, do not read it); the changelog. |
| [`../PORTING.md`](../PORTING.md), [`ARCHITECTURE.md`](ARCHITECTURE.md), [`LIBAOM_UPSTREAM_NOTES.md`](LIBAOM_UPSTREAM_NOTES.md), [`MAGETYPES_VOCABULARY.md`](MAGETYPES_VOCABULARY.md) | How to port a C function; crate layout; libaom quirks/ISA-conditional kernels; what the SIMD crate can and cannot express. |
| [`docs/HANDOFF-TOGGLES.md`](docs/HANDOFF-TOGGLES.md) | The instrumented-sibling-C ("ar-swap") method for dumping C's per-block decisions — the tool that localised most encoder divergences. |
| [`ENCODER_PRIMARY_ENVELOPE.md`](ENCODER_PRIMARY_ENVELOPE.md), [`ZEN_COMPLIANCE_SPEC.md`](ZEN_COMPLIANCE_SPEC.md) | libaom's allintra defaults (verified); the six zen contracts spec (all landed). |
| `public-api/` | Generated public-API snapshots, enforced by `just api-doc-check` (pinned nightly). |
| `CONFIG_*_2026-07-30.md`, `DECODER_CONFIG_COVERAGE_2026-07-30.md`, `SIMD_REACH_AUDIT_2026-07-28.md`, `RDOPT_C_COVERAGE_2026-09-01.md` | Dated measurement records still cited from tests; correct as of their date. |

**Historical — do not plan from these:**

* `inter/` — inter-frame (video) roadmaps and handoffs. Inter is a **non-goal until the still-image ship**; the decoder side is gated, the encoder side is a zero-MV skeleton (see the README video table).
* `../handoff/2026-09-08/` — the imazen-26 datagen fleet's research handoff (`AOM_ADOPTION.md`, cell tables). Input to the standing goal, not a plan.
* `archive/` — superseded design notes and handoffs whose content has landed or been overtaken (`kb5_completion_spec`, `winner_mode_port_design`, `qm_rd_threading_staged`, `cpu_used_allintra_sweep_plan`, `inter-vartx-coeff-arm-notes`, `CONTEXT-HANDOFF`). Kept for `git blame` context only.
