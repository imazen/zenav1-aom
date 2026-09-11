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

Identical content, identical zenavif config, single-threaded both, one box
(the threaded-rav1e arms are measured further down and move this by 3 %):

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

## BOTH RESERVATIONS ARE NOW MEASURED, and neither rescues the incumbent

The first version of this record said the 10x should not be quoted without a
quality-matched run and a threaded-rav1e number. Both were run.

### 1. Quality at matched rate — the port wins this too

Both streams decoded with the PINNED upstream `aomdec` (v3.14.1) and scored by
`xtool score` (SSIMULACRA2 + butteraugli) against the source:

| arm | bytes | SSIMULACRA2 (higher better) | butteraugli max (lower better) | ba_3n |
|---|---:|---:|---:|---:|
| **zenav1-aom** cq27 cpu-used 3 | 40,237 | **79.22** | **3.435** | **1.110** |
| zenrav1e q127 speed 4 | 40,436 | 76.21 | 3.822 | 1.237 |

**+3.01 SSIMULACRA2 at 0.5 % FEWER bytes**, and better on both butteraugli
metrics. So this is not a speed-for-quality trade: the port is ahead on rate,
quality AND time simultaneously. At *matched quality* the gap would be wider
still, since the port would be run at a lower quality point.

### 2. Threaded rav1e — worth ~3 %, and it costs rate

`drv-rav1e` gained an optional 9th `threads` argument (tiles track threads, which
is how rav1e parallelises inside one frame):

| zenrav1e s4 q127 | time | bytes | vs the port |
|---|---:|---:|---:|
| threads=1 | 30,777 ms | 40,436 | 10.28x |
| threads=4 | 30,682 ms | 41,459 | 10.25x |
| threads=8 | 30,110 ms | 42,049 | 10.05x |
| threads=16 | 29,836 ms | **42,853** | **9.96x** |

**16 threads buys 3.1 % and costs 6.0 % more bytes.** The mechanism is not
mysterious: rav1e's main parallelism is FRAME-level, which a single-frame still
image cannot use at all, and intra-frame tiling both scales poorly at 1 MP and
charges rate for the tile boundaries. **So the "rav1e threads in production"
caveat is measured to be ~nil for the still-image case** — the gap goes 10.28x ->
9.96x while rav1e's output gets 6 % larger, i.e. the rate match degrades faster
than the time does.

Worth stating the asymmetry: the PORT spawns no threads at all and has its own
measured ~3.2x-at-4-threads headroom
(`encoder_tile_threading_ceiling_2026-09-10.md`), untouched here.

## What this still does NOT establish — read before quoting the 10x

* ~~Rate-matched, not quality-matched.~~ **MEASURED above: +3.01 SSIMULACRA2 to the
  port at 0.5 % fewer bytes.** The coding half is settled for this cell, and it
  goes the port's way — consistent with its RD *being* libaom's by 427/427
  byte-identity.
* ~~Single-threaded both / rav1e threads in production.~~ **MEASURED above: worth
  3.1 % to rav1e at 16 threads, at a 6 % rate cost.** Not a ceiling on the claim.
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
about whether the flip would make zenavif worse**. On the evidence now in hand,
for this cell, the flip would make zenavif **~10x faster AND ~3 SSIMULACRA2 points
better at the same rate**, and threading the incumbent does not change that.

Whether to flip remains the user's call — the bar is theirs to interpret, and one
content at one size is not a product decision. But the two named reservations that
stood against the 10x are now measured and both fell the same way, so the decision
should not be deferred on the grounds that the comparison is unknown. **The honest
summary is: the port loses to libaom on time by 1.94x, and beats the backend it
would actually replace on time, rate and quality simultaneously.**
