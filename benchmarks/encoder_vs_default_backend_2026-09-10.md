# The deployment question, MEASURED: the port is ~10x FASTER than the backend it would replace

Clause (1)'s two zenavif-side "by default" flips were recorded as **"blocked on
clause (4) alone"**. That conflates two different tests, and answering the right
one reverses the conclusion.

* **Clause (4) is a RATIO against libaom** — the user's stated bar, ≤ 1.5x, and it
  is genuinely unmet at **1.94x**. Nothing here changes that.
* **"Can zenavif default to it" is a comparison against ZENRAV1E**, the rav1e fork
  that is zenavif's `#[default]` backend (`zenavif/src/encoder.rs:262-265`,
  "production-proven"). That is the backend the flip would REPLACE, and it had
  never been measured at HEAD.

## The result

Identical content, identical zenavif config, single-threaded both, one box:

| arm | setting | time | bytes |
|---|---|---:|---:|
| **zenav1-aom** (`encode_key_frame`, the path zenavif calls) | cq27, **cpu-used 3** | **2,994.6 ms** | 40,237 |
| **zenrav1e** (zenavif's current default) | q127, **speed 4** | **30,538 ms** | 40,436 |

**10.20x, at a rate matched to 0.49 % — and matched in rav1e's favour** (its
output is 0.5 % LARGER, so it is not buying the time with a better rate point).

Port median over 3 runs 2,993.4 / 2,994.6 / 2,998.1 ms (0.16 % spread), bytes
40,237 every time — **the pinned clause-(4) value**, so this is the same cell
every band in this repo uses.

## Why speed 4 against cpu-used 3 is the matched comparison, not a handicap

Both come from ONE zenavif setting. `EncoderConfig::speed` defaults to **4**
(`zenavif/src/encoder.rs:506`); `backend_router.rs:754` passes
`speed_effective()` **straight through** to the rav1e backend, while
`encoder_aom.rs:248`'s `speed_to_cpu_used(s) = s - 1` maps the same 4 to
**cpu-used 3**. So a user who changes nothing gets rav1e speed 4 today and would
get aom cpu-used 3 after the flip. Measuring rav1e at speed 7 instead — which an
earlier table did — compares a setting zenavif never asks for.

## Two reasons the existing cross-encoder table cannot answer this

`benchmarks/xbench_2026-08-01.md` is the only prior cross-encoder data, and its
zenav1-aom column is unrepresentative of the product **twice over**:

1. **It predates the entire perf programme.** Taken 2026-08-01, the day BEFORE
   KB-PERF-1 (10.66x -> 3.36x) and ~two cycles before the shipping preset reached
   1.9443x.
2. **`drv-aom` does not time the shipping path.** Its timed region is
   `cell.port_encode_with(..)` — the aom-bench DIFFERENTIAL HARNESS — not
   `encode_key_frame`, which is what zenavif calls. Measured here on the identical
   cell, both emitting the identical 40,237 bytes: **drv-aom 5,687 ms vs
   `encode_key_frame` 2,994.6 ms, a 1.90x inflation.** So xbench's zenav1-aom
   MP/s figures understate the product by ~1.9x on top of being stale.

Anyone re-running xbench for a product question must fix (2) or read the wrong
number.

## What this does NOT establish — read before quoting the 10x

* **Rate-matched, not QUALITY-matched.** No SSIMULACRA2 or PSNR was computed.
  Equal bytes is a first-order proxy. What is known independently: the port is
  **byte-identical to libaom on 427/427 cells**, so its RD *is* libaom's RD, and
  the coding half of this comparison reduces to libaom-vs-rav1e at these
  settings — which this measurement does not settle.
* **Single-threaded both.** `drv-rav1e` pins `Config::with_threads(1)` to match
  this repo's whole measurement protocol. **rav1e threads in production**, so a
  threaded zenrav1e would narrow this gap by whatever its scaling is — unmeasured
  here, and the honest ceiling on the claim. (The port spawns no threads at all;
  `encoder_tile_threading_ceiling_2026-09-10.md` measures its own ~3.2x-at-4-threads
  headroom.)
* **One content, one size, one rate point, one box.** Photographic (the
  mirror-tiled `av1-1-b8-01-size-196x196` decode), 1024x1024, ~40 KB, x86-64 Linux.
  Screen content is deliberately excluded: KB-41 measures the port's IntraBC DV
  search at ~80 s per 1 MP there, which would answer a different question, and the
  committed corpus (`codec-corpus/gb82-sc`) holds only screen content — which is
  why `crates/aom-bench/examples/dump_cell_yuv.rs` exists to emit the photographic
  cell as I420.

## What it means

The two readings come apart cleanly, and both should be quoted:

* **against the BAR** — 1.94x vs 1.5x, clause (4) NOT met, unchanged;
* **against the INCUMBENT** — ~10x faster at matched rate and matched zenavif
  config, with libaom's RD rather than rav1e's.

So "blocked on clause (4) alone" is true of the bar and **false as a statement
about whether the flip would make zenavif worse**. Whether to flip the default on
that basis is the user's call, not this record's — but it should be made knowing
the incumbent is ~10x slower single-threaded, not knowing only that a libaom
ratio is unmet.
