# The unported SCM trial: measured to be a DIVERGENCE, not a refusal

**2026-09-09.** The standing goal lists two classes that must CLOSE rather than
ship under the "measured, attributed, bounded and documented" cap, and names
only two examples of the refusal class:

> (a) anything that REFUSES a configuration a caller can reach
> (`--deltaq-mode` 2/3 at `--cpu-used` >= 8 asserts; **the unported SCM trial**)

The first closed as KB-46. This measures the second, and the answer is that it
was never a refusal on the path a caller uses.

## Why this needed a measurement rather than a reading

The SCM gap touches two things in **different crates**, and only one of them is
reachable by a caller:

| site | what it is | reachable by a caller? |
|---|---|---|
| `aom_bench`'s `port_encode_impl` | a hard `assert_eq!` that the ported screen-content decision matches the ORACLE HEADER, whose message names `av1_determine_sc_tools_with_encoding` as the remaining C arm | **No** — it fires while comparing against libaom |
| `aom_encode::key_frame::encode_key_frame` | the path zenavif calls; runs its OWN detector and lists the trial under "Not yet wired" | **Yes** |

`encode_key_frame`'s entire refusal surface is `PlaneSize`, `Unsupported`,
`Cancelled`, `SampleRange`, `AllocFailed`, `LimitExceeded` — **none of them
reachable from a screen-content decision**. That is an argument from reading the
enum, so it was checked against behaviour instead.

## The measurement

The class KB-41 names as the trial's own reproducer neighbourhood (tiny cells at
`cq >= ~40`, `--cpu-used` 4 and 6), swept through the PUBLIC entry point:

* sizes 59x128, 85x128, 128x80, 128x128
* cq {40, 44, 50, 57, 62} x `--cpu-used` {4, 6, 8}
* **60 cells: 60 encode, 0 refused, 0 panicked**, 1.46 s.

**Non-vacuity is asserted on the FIXTURE, and it is the part that matters.** The
census's existing `planes()` is a textured gradient, which is detector-NEGATIVE
by construction — and detector-negative content is exactly the class the trial
governs, since C's `av1_determine_sc_tools_with_encoding` returns early whenever
the detector has already said yes. A refusal census run only on gradients cannot
reach this at all. So the sweep uses flat few-colour panels with a repeating
glyph alphabet, and measures the fixture with libaom's own statistic — the
fraction of full 16x16 luma blocks holding 2..=4 distinct `pix >> (bd-8)` codes
(KB-17): **100.0 %**, against libaom's 10 % detector threshold.

## What this changes

**Class (a) has no known open member on the public API.** The SCM trial is a
byte DIVERGENCE, which the cap covers, and the byte gate already holds it
accountable per cell (427/427 byte-identical; two adversarial probes designed to
find a counterexample found none in 105 cells).

**What is NOT claimed:** the datagen fleet's 35 refused tiny cells are real, and
they are `aom-bench`'s. A differential harness that cannot model C's decision
SHOULD refuse rather than silently report a mismatch as a port defect — that is
the harness working. It is not the backend zenavif selects.

## The gate

`refusal_census::screen_shaped_tiny_cells_encode_rather_than_refuse`. It fails
BY NAME if a refusal ever appears in that class, and its message says what that
would mean: the item moves back onto the standing goal's must-close list,
because it would then be a refusal on a configuration a caller can produce
rather than a divergence under the cap.

It also fails if the fixture stops being screen-shaped, so it cannot decay into
the gradient census under a new name.
