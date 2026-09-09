# The preset zenavif actually ships is `--cpu-used 3` — and at 1 MP it is the port's BEST cell, 2.258x

**2026-09-09. No code changed.** Three results, one of which corrects a
conclusion published earlier the same day.

## 1. The open question was answerable from the code, not from the user

`encoder_ratio_vs_speed_2026-09-09.md` closed with *"Which speed zenavif selects
now matters to the goal ... That is a question for the integration, and it is
worth answering before more speed-0 optimisation is done."* It is answered by
reading the integration:

* `zenavif/src/encoder.rs:506` — the encode-config default is **`speed: 4`**;
* `zenavif/src/encoder_aom.rs:248` — `speed_to_cpu_used(s) = clamp(s,1,10) - 1`,
  one-to-one, no remap or clamp hiding a distinction;
* `zenavif/src/encoder_aom.rs:460` — that value reaches
  `KeyFrameConfig::cpu_used` directly.

**The shipping preset is `--cpu-used 3`.** Every ranked profile in this repo is
`--cpu-used 0`; the profile taken earlier today is `--cpu-used 6`. **Neither is
the configuration clause (1) reduces to.**

## 2. The 1 MP speed ladder at HEAD, one box, back to back

1024x1024 cq27, x86-64, `eprof_x86`, reps scaled so each row totals ~30 s:

| `--cpu-used` | reps | port | libaom C | **ratio** | bytes |
|---:|---:|---:|---:|---:|---:|
| 0 | 3 | 9650.29 ms | 4128.70 ms | **2.337x** | 39,694 both |
| **3 — the shipping preset** | 8 | **3498.73** | **1549.38** | **2.258x** | 40,237 both |
| 6 | 25 | 457.73 | 169.76 | **2.696x** | 43,012 both |

**Port and C emit byte-identical streams at all three speeds** — a free parity
check across the ladder, not merely a timing.

### This CORRECTS the ratio-vs-speed record's headline

That record measured **512x512** and read 2.189 / 2.223 / 2.652 at s0 / s3 / s6
— monotone — and titled itself *"the ratio gets WORSE as `--cpu-used` rises"*.

**At 1 MP it is not monotone: it DIPS at s3.** 2.337 -> **2.258** -> 2.696. The
shipping preset is the best of the three and the best cell measured anywhere at
HEAD. The 512x512 record is not wrong about 512x512; it is wrong as a general
claim, and the frame size that matters for a still-image backend is the larger
one.

**Do not read the dip as a mechanism.** Two effects run in opposite directions
here — libaom sheds search work faster than the port does as speed rises, and
the port's fixed per-call overhead amortises better over a larger frame — and a
three-point ladder on one cell cannot separate them. What is measured is the
ORDERING, which is what the goal needs.

## 3. The class table at the shipping preset — gap 1949.35 ms

Whole-symbol classification of both flat profiles (`.port.tsv`, `.c.tsv`, 394
and 480 symbols, 99.7 % / 100.5 % of samples accounted):

| class | port | C | **gap** | ratio | % of gap |
|---|---:|---:|---:|---:|---:|
| **transform** | 1072.4 ms | 283.4 ms | **+789.0** | 3.78x | **40.5 %** |
| intra-pred | 408.3 | 170.3 | +238.0 | 2.40x | 12.2 % |
| (unclassified) | 547.9 | 361.2 | +186.7 | 1.52x | 9.6 % |
| txb/trellis | 677.0 | 498.3 | +178.7 | **1.36x** | 9.2 % |
| memory | 174.2 | 27.0 | +147.2 | 6.45x | 7.6 % |
| loop-restoration | 167.9 | 29.1 | +138.8 | 5.77x | 7.1 % |
| distortion | 217.6 | 93.0 | +124.6 | 2.34x | 6.4 % |
| quantize | 114.1 | 56.7 | +57.4 | 2.01x | 2.9 % |
| loopfilter/cdef | 65.4 | 12.7 | +52.7 | 5.15x | 2.7 % |
| **CNN** | 40.9 | 9.3 | **+31.6** | 4.40x | **1.6 %** |

**The speed-0 ranking TRANSFERS to the shipping preset.** Compare the 1 MP s0
table (transform 38.3 %, loop-restoration 15.3 %, intra-pred, txb at 1.31x,
memory at 7.43x, quantize, distortion): the ORDER of the top item is identical
and transform is 40.5 % here against 38.3 % there. **The speed-0 optimisation
this cycle spent was well aimed after all** — which is the opposite of what a
reader would conclude from the speed-6 profile alone.

The one class that moves materially is **loop-restoration, 15.3 % of the s0 gap
and 7.1 % here**: at s3 libaom begins shedding restoration work the port still
does.

`optimize_txb_core` is **388.0 ms against `av1_optimize_txb`'s 415.2 — the
port's trellis is FASTER**, for the third consecutive profile (s0 1211 vs 1258,
s6 1.07x, s3 0.93x). It is 26.8 % of libaom's entire speed-3 encode. **Do not
spend a landing on it**, and note that the whole txb class runs at 1.36x here
precisely because that member offsets `txb_init_levels`' 4.5x.

## 4. THE CORRECTION: the CNN is a speed-6 lever, not a shipping-preset lever

`encoder_speed6_profile_2026-09-09.md`, written earlier today, measured the
intra-mode CNN at **10.4x, +32.05 ms, 11.1 % of the speed-6 gap** and concluded
that one piece of work — porting `cnn_avx2.c`'s 5x5/skip-4 specialisation —
*"serves BOTH clause (4) and KB-41 root #27"*. **The clause-(4) half of that does
not survive at the preset zenavif ships.**

The CNN's ABSOLUTE cost is essentially speed-invariant, because it is computed
once per 64x64 node (KB-PERF-1's cache) and the frame has the same number of
them at every speed:

| | port CNN | C CNN | gap | gap as % of that speed's total gap |
|---|---:|---:|---:|---:|
| `--cpu-used 6` | 35.47 ms | 3.42 ms | +32.05 | **11.1 %** |
| `--cpu-used 3` | 40.9 ms | 9.3 ms | +31.6 | **1.6 %** |

Same ~32 ms; a gap that is 290 ms at s6 and 1949 ms at s3. **At the shipping
preset the CNN is 1.6 % of the gap — a rounding error against transform's
40.5 %.** It is still live at s3 (`intra_cnn_based_part_prune_level` is 0 only
at speed 0, KB-23), so the finding is not vacuous; it is simply not a ranked
lever there.

**So the `cnn_avx2.c` port should be justified on PARITY grounds — KB-41 root
#27, the last carrier of the datagen wave's 13 cells — and not on clause (4).**
That is the harder half anyway (bit-identity to the dispatched C, not merely
speed), and it should be scheduled as parity work rather than as perf work.

**The transferable lesson: a lever's SHARE is a ratio, and profiling at a speed
the product does not ship inflates the numerator's importance without changing
the numerator.** Rank levers at the shipping configuration, or state the speed
next to the share.

## 5. What clause (4) now needs

Bar <= 1.5x. At the configuration that actually gates clause (1):

* today **3498.73 ms** against a bar of **2324.07 ms** (1.5 x 1549.38);
* gap **1949.35 ms**, of which **1174.66 ms — 60 % — must go.**

Several landings, not one. But the starting point is better than either
published number implied: the speed-0 headline understates the shipping ratio's
quality (2.337 vs 2.258) and the speed-6 profile overstates its difficulty
(2.696 vs 2.258). **Quote clause (4) as `2.258x at 1024x1024 --cpu-used 3`** —
with its cell AND its speed, the speed being the one the backend ships.

The ranked target is unchanged from the speed-0 table and is now confirmed at
the shipping preset: **transform, +789 ms, 40.5 % of the gap, 3.78x.**

## Limits

One cell, one content class, one box, bd8 4:2:0, cq27. Single timing pass per
row (reps inside each; no interleaved band, no rotation), so these are
ranking-grade ratios of the same status as every other table here — read the
ordering and the ~0.1x scale, not the third digit. Shares are each arm's own
wall. The 512x512 rows quoted in §2 come from the earlier record and were taken
separately, so that comparison crosses two sessions. `intra_model_rd_y`
(122.8 ms) and `txfm_rd_in_plane_intra` (130.1 ms) remain **inlining sinks**
(KB-PERF-10) and levers must not be costed off them.
