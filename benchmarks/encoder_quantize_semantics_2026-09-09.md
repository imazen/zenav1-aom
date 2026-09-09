# quantize_fp is NOT blocked on `mul_high` — it is blocked on a SEMANTICS DECISION

**2026-09-09. No code changed.** The last of the three levers the intrinsics
probe unblocked, analysed the same way wiener was — from the disassembly, before
writing anything. The conclusion is different from the other two: **the
instruction was never the obstacle, and the real one is a decision that is not a
kernel author's to make.**

## The measurement

`aom_dsp::quant::simd::quantize_fp_impl_v3` **79.1 ms** against
`av1_quantize_fp_avx2` **24.5 ms = 3.23x, +54.6 ms** at the shipping preset —
the second-largest item in the lane-width audit after directional intra.

Instruction mix, both from the shipped binary:

| | port `quantize_fp_impl_v3` | C `av1_quantize_fp_avx2` |
|---|---|---|
| total instructions | **365** (incl. cold panic pads) | **122** |
| arithmetic | `vpsrad` 14, `vpxor` 10, `vpsubd` 8, **`vpmulld` 8**, `vpmaxsd` 7, `vpminsd` 4, `vpcmpeqd` 4, `vpaddd` 4 | `vpsraw` 5, `vpxor` 4, `vpcmpgtw` 4, `vpmaxsw` 3, `vpsubw` 2, **`vpmullw` 2**, `vpsignw` 2 |
| lane width | **every op `*d` — 32-bit** | **every op `*w` — 16-bit** |

**The port is 32-bit end to end where libaom is 16-bit end to end** — the
lane-width thesis in its purest form, 2x on every operation, plus `vpmulld`
(2 uops) against `vpmullw`/`vpmulhw` (1), plus `vpsignw` doing in one
instruction what the port spends `vpxor` + `vpsubd` on.

## Where the recorded blocker was wrong

`CLAUDE.md` and the audit record this as *"blocked on `mul_high`"*, i.e. on
`_mm256_mulhi_epi16` for `(abs_r * quant) >> 16`. That instruction is now
available (KB-PERF-21/22 are the precedent) **and it is the safe part**: the
port already clamps `abs_r` into `[i16::MIN, i16::MAX]` before the multiply and
`quant` is `i16`, so `mulhi` reproduces that step exactly, at every bit depth,
with nothing to prove.

**The unsafe part is the OTHER multiply, and nobody had named it.** libaom's two
`vpmullw` compute `dqcoeff = tmp * dequant` in **16-bit wrapping** arithmetic.
The port computes it in `i32`:

    let absdq = tmp * dqv_v;    // i32x8, no wrap

`tmp` reaches 32767 and `dequant` reaches ~1800, so the product reaches ~59 M —
**far outside `i16`**. The two agree only where the product happens to stay in
range.

## Why that is a decision and not a bug

* The port is **byte-identical to real aomenc on 427/427 cells**
  (`self_contained_key_frame`, cq 0..63, bd 8/10/12), and aomenc dispatches
  `av1_quantize_fp_avx2`. So on that whole envelope the i32 and the wrapping i16
  results **coincide** — no reachable overflow there.
* But **KB-20 measured that they do NOT always coincide**: `av1_quantize_fp`'s
  tiers disagree with `_c` and with each other outside `int16` — NEON truncates
  (`vmovn_s32`), AVX2 saturates (`_mm_packs_epi32`) — and the port already had to
  model the DISPATCHED variant for the nonrd arm
  (`nonrd_pickmode::quantize_fp_dispatched`) to be byte-exact.

So converting this kernel to 16-bit is not "make it faster". It is choosing
which libaom to be bit-identical to:

| | today | after an i16 rewrite |
|---|---|---|
| matches `av1_quantize_fp_c` | **yes** | no, outside i16 |
| matches x86-64 aomenc (`_avx2`) | on the 427-cell envelope | **yes, by construction** |
| matches aarch64 aomenc (`_neon`) | on that envelope | **no** — NEON truncates where AVX2 saturates |

**That is the same fork `KB-ARM-FLOAT` root #1 already decided once**, for the
oracle: *"the port matches the x86-64 build, which is the platform every gate in
this repo was established on."* Applying that precedent here would say "go
16-bit and match `_avx2`" — but it would also make the port's own output
ISA-conditional in a kernel that feeds the bitstream, which is a far bigger
claim than a benchmark, and it is contradicted by nothing less than the
`AOM_FORCE_SCALAR` gate leg, which requires the scalar and vector tiers to agree
byte-for-byte.

**That last point is decisive and is why this stops here:** `just gate-landing`
runs the suite twice, and a 16-bit vector tier that wraps where the scalar
transcription does not **cannot** pass both legs on any cell that overflows. So
an i16 rewrite is only admissible together with an argument that overflow is
UNREACHABLE — which is a domain proof over `(coeff, quant, dequant, log_scale)`,
not a lane-width change.

## What is safely available, and what it is worth

The `mulhi` half alone — `(abs_r * quant) >> 16` at 16 lanes, keeping `dqcoeff`
in i32 — is exact and needs no proof. But it forces a widen back to i32 for the
dequant step (two `vpunpck` + `vperm2i128` per 16 coefficients), so it buys one
`vpmulld` -> one `vpmulhw` and costs the lane fix: on the wiener evidence
(where the same setup cost made the 8-column shape a wash) **this is very
likely null**, and it should be costed in uops before anyone writes it.

## Handoff

1. **Do not implement "quantize with mul_high" as a perf task.** The instruction
   is available and is not what is in the way.
2. The real question is a **project decision**: should the port's quantizer be
   bit-identical to `_c` (today, ISA-neutral) or to the dispatched `_avx2`
   (faster, ISA-conditional)? KB-ARM-FLOAT root #1 is the precedent for how this
   project has answered a question of that shape before.
3. If the answer is "stay `_c`", then the honest ceiling for this kernel is the
   16-lane `mulhi` step alone, and the +54.6 ms is largely **structural**, not
   addressable — which should be written into the audit so it stops being
   ranked as an available lever.
4. Either way, an overflow-reachability sweep over `(coeff, quant, dequant,
   log_scale)` is the prerequisite, and `xtask/audit_i16_fwd.py`'s exhaustive
   method is the model.
