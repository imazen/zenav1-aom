> Moved VERBATIM out of `CLAUDE.md` on 2026-09-11 so the per-turn instruction file stays small (it had reached 744 KB / ~186k tokens). Nothing below was edited in the move; "above"/"below" refer to positions in the original file. Keep appending here, not in `CLAUDE.md`.

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
