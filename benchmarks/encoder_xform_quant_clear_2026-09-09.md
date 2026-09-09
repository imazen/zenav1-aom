# KB-PERF-19 — the quantize scratch re-zeroed itself on every call; KB-PERF-2's null did not survive the platform change

**Landed 2026-09-09.** Byte-identical; **−0.72 % at 512x512**, 24/24 rounds
faster, p < 0.0001, against a same-binary null of +0.02 %.

## The change is three deleted words

`xform_quant_into` opened with, three times over:

    coeff.clear();
    coeff.resize(full, 0);

`resize` **already** leaves the length exactly right on its own — it truncates
when the buffer is long enough and pads with zeros only when it genuinely
grows. The `clear()` in front of it is what forced all `n` elements to be
re-zeroed on **every** call. Deleting the three `clear()`s is the whole diff.

It is sound because every element these buffers hand on is written before it is
read, and **KB-PERF-2 established that rather than assuming it**:
`av1_fwd_txfm2d` writes every `coeff[..full]`, and all twelve quantizer variants
open by filling `qcoeff[..n]` / `dqcoeff[..n]`. The byte gates are the check
that matters here — stale data surviving into a coded block would change the
stream, and the output is byte-identical (10,912 B at 512x512).

## The finding: the same change is NULL on one platform and a win on another

**KB-PERF-2 built exactly this, measured it, and reverted it**, recording:

> A skip-the-re-zero variant was built ... and **measured inside the control
> band** (16 rounds, −3.34 % vs −2.40 %, each inside the other's spread), so it
> was reverted.

That measurement was **aarch64-apple-darwin at `--cpu-used 6`**. This one is
**x86-64 at `--cpu-used 0`**:

| | platform | speed | result |
|---|---|---|---|
| KB-PERF-2 | aarch64-apple-darwin | cpu-used 6 | **null** (inside the control band) |
| KB-PERF-19 | x86-64 linux | cpu-used 0 | **−0.72 %**, 24/24, p<0.0001 |

**This does not contradict KB-PERF-2 — it is the platform-dependence KB-PERF-2
itself predicted**, in the entry that opens with a boxed warning that every
millisecond in it is "an APPLE M4 PRO millisecond against APPLE'S allocator"
and whose central lesson is that the `memset`-vs-allocator split *is priced
differently by different heaps* (its own allocation lever moved from 21 % of
the win on Darwin to 86-99 % on Windows). A null on one (platform, speed) is
not a null on another.

Speed matters as much as platform here: at `--cpu-used 0` the transform search
runs **31 forward transforms per pixel**
(`encoder_txfm_size_census_2026-09-09.md`), so a per-call cost is paid an order
of magnitude more often than at cpu-used 6.

**The transferable rule:** a reverted-as-null perf change is refuted for its
measured cell, not for the codebase. Before re-running one, check what
(platform, speed, content) its null was taken at — and say so, as this record
does, rather than quietly relanding it.

## Measured

| cell | base | new | paired median | rounds faster | p | null |
|---|---:|---:|---:|---:|---:|---:|
| 512x512 cq27 s0 | 3436.0 ms | 3413.3 ms | **−0.72 %** | **24/24** | <0.0001 | +0.02 %, 11/24, p=0.84 |

## Not covered

* The 1 MP cell was not banded; the effect is per-call and the 512 cell resolves
  it more sharply, but that is a choice, not a measurement.
* Other `clear()` + `resize()` pairs in the tree were not swept. Each needs its
  own overwrite proof — this one inherits KB-PERF-2's, and a site without such a
  proof must not copy this change.
* One box, one content class, `--cpu-used 0`, x86-64.
