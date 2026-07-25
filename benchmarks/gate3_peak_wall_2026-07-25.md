# Gate-3 decode WALL baseline — post bd8-rows + CDEF-find_dir (2026-07-25)

The "how fast can the 8-bit path go" wall measurement after the full bd8
peak-perf series landed: Phase A/B/C (u8 planes → u8 kernels → i16 columns) +
`9f49ebc3` (i16 transform rows) + `ea61406f` (CDEF `cdef_find_dir`). Paired
port-vs-C via zenbench interleaved rounds (`cargo bench -p zenav1-aom-bench
--bench gate3 -- --group=dec`); every cell byte-verifies port==C output before
timing.

## Headline

**4K stills decode: port ≈1.19–1.24× C wall** (cq40 ≈1.16–1.24, cq20
≈1.20–1.35 across both runs' 95% CIs; run-1 cq20 CI was tight at 1.218–1.235).
Down from the Phase-B baseline 1.286× (cq20) / 1.250× (cq40) — the i16-rows +
find_dir landings are worth ≈4–6% of 4K wall. **The user-set ≤1.5× acceptance
bar is met at the 4K headline cells.** 2K and small-frame cells still exceed it
(1.66–1.9× at 2K, up to ~2.4× on tiny/entropy-dominated conformance cells) —
consistent with the Ir profile; ranked remaining levers in
`gate3_filters_2026-07-22.md` (deblock i16 repack, dispatch-prologue hoist,
wiener/CDEF madd — the latter two blocked on magetypes integer primitives).

## Results (port "95% CI vs base" = the wall ratio − 1; both runs)

| cell | run 1 | run 2 | reading |
|---|---|---|---|
| dec_mosaic_4k_cq20 | +21.8–23.5% | +20.1–34.5% | ≈1.22× (run-1 CI tight) |
| dec_mosaic_4k_cq40 | +18.2–23.9% | +16.3–21.2% | ≈1.19× |
| dec_mosaic_2k_cq20 | +59.5–93.4% | +66.6–69.0% | ≈1.67× |
| dec_mosaic_2k_cq40 | +78.9–88.5% | +69.1–82.5% | ≈1.75× |
| dec_64x64 | +41.0–62.6% | +42.9–58.3% | ≈1.5× (per-call overhead regime) |
| dec_196x196 | +118.3–130.6% | +112.1–118.2% | ≈2.15× |
| dec_352x288_q00 (lossless/WHT) | +14.4–46.6% | +17.8–22.3% | ≈1.2× |
| dec_352x288_q32 | +72.0–98.0% | +71.3–97.1% | ≈1.85× |
| dec_352x288_q63 | +115.0–145.3% | +127.6–144.8% | ≈2.35× |

Absolute 4K throughput (run 2): cq20 C 220.4±1.4ms vs port 279.4±8.5ms
(37.6 vs 29.7 Mpx/s); cq40 C 142.6±3.3ms vs port 169.4±6.3ms (58.2 vs 49.0
Mpx/s).

## Measurement-quality caveats (read before quoting)

- **zenbench's resource gate rejected most rounds on this VM in BOTH runs**
  (run 1: 1080 noisy of ~1100, ambient load ~3.8; run 2: 1069 noisy, ambient
  load ~1.6) — every group fell back to its 4-round minimum with a ⚠ flag.
  This is structural on this cloud VM (timer/steal-time jitter trips the gate
  even near-idle), not a transient. The numbers above survive it two ways:
  the 4K cells' per-call times are long (140–280 ms) with small mad, giving
  tight CIs even at N=4; and **two independent runs ~40 minutes apart agree
  on every cell** (all CIs overlap).
- Occasional per-cell drift flags (`drift r=0.80`) on the µs-scale cells —
  treat the small-cell ratios as ±10% indicative, not precise.
- For committed-baseline purposes the 4K cells are the quotable numbers; the
  Ir (callgrind) series in `bd8_*`/`gate3_*` remains the precise
  per-lever attribution tool on this box.
