# Mac ARM DSP tier check — 2026-10-16

Source: `93783de612062707501932d0a4d9d1c22630fb32` on Apple M4 Pro.

Commands from the repository root:

```sh
TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 12G --jobs 4 -- cargo bench -p zenav1-aom-dsp-bench --bench dsp_kernels -- --group=fwd_txfm/16x16_dct --format=llm
TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 12G --jobs 4 -- cargo bench -p zenav1-aom-dsp-bench --bench dsp_kernels -- --group=intra_dispatch --format=llm
```

The 16×16 forward DCT batch averaged **119.80 µs native SIMD** versus **465.15 µs forced scalar** (100 paired rounds; scalar +288.26%, 95% CI +260.42% to +316.38%). Variability was high: CV 58.1% / 43.4%. This confirms tier direction, not a small optimization delta.

For 15 intra predictor cells, forced scalar was significantly slower in six: Paeth, smooth and smooth-V at 16×16 and 32×32. The other nine had no significant tier difference by the harness test. In particular, the earlier 2026-09-06 small-shape scalar advantages were not reproduced here: Paeth 4×4 scalar +0.73% (CI −1.15% to +2.60%), smooth 4×4 +2.09% (CI −0.48% to +4.67%), smooth-V 4×4 −1.14% (CI −5.38% to +3.19%), and smooth-V 32×32 +692.55% (CI +635.94% to +748.25%). No dispatch change follows from this run.

`run-heavy` forward DCT: 26 s, peak RSS **0.25 GiB**, minimum available **5233 MiB**, peak load **3.90**. Intra: 307 s, peak RSS **0.05 GiB**, minimum available **5331 MiB**, peak load **3.58**. Full logs: `~/tmp/zenav1-aom-mac-kernels.log` and `~/tmp/zenav1-aom-mac-intra.log` on the Mac.
