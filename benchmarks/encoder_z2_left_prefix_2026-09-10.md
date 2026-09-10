# z2's left-gather prefix paid three bounds checks per pixel — **−0.274 %**

**2026-09-10.** Byte-identical, **−0.274 % at 1024x1024 cq27 `--cpu-used 3`,
20 of 24 rounds faster, p = 0.0015**, against a same-binary null of −0.063 %
(13/24, p = 0.84).

## How it was picked — by the rule, not by the ranking

The lever map ranks `directional intra z2` at **+47.3 ms, 2.28x**. But the rule
this session established is what selected the *change*:
**removal of work per iteration pays; reshaping how the same work is reached
does not** (three rejections, two of them measured slower).

So the question was not "can z2 be vectorised" — KB-PERF-4 already answered that,
and the left half is a true gather (`base_y` is not affine in `c`). The question
was **what is executed per pixel that need not be**. `perf annotate` on
`z2_high`:

```
6.08 %  cmpq    %r14, 0x10(%rsp)          <- a checked bound, reloaded
3.30 %  movabsq $0x7fffffffffffffff, %rcx <- the signed-index limit constant
2.83 %  movzwl  (%rbp,%r11,2), %r11d      <- the gather load
2.32 %  movw    %cx, (%rdi,%r14,2)        <- the per-pixel store
```

~9 % of the function in bounds-check machinery alone.

## The defect

`z2_high` dispatches the above-suffix to `two_tap_run` (vectorised) and walks the
left prefix scalar. That prefix was paying **three** bounds checks per pixel:

* `left.at(base_y)` and `left.at(base_y + 1)` — `EdgeRef16::at` is
  `self.data[(self.pad as i32 + i) as usize]`, a checked index through a signed
  cast, called **twice**;
* `dst[row + c]` — the store.

Now: one row slice for the destination (`drow`), so the store is unchecked inside
the loop and the vectorised tail becomes `&mut drow[c..]`; and **one two-element
window** `&ld[i0..i0 + 2]` for the tap pair instead of two independent checked
reads. Three checks per pixel become one.

## Exact, including the panics

Arithmetic, walk order and values are untouched — `w[0]`/`w[1]` are literally
`left.at(base_y)`/`left.at(base_y + 1)`. **Panic behaviour is identical too:**
`&ld[i0..i0 + 2]` panics exactly when the old `left.at(base_y + 1)` would have,
since both require `i0 + 2 <= len`.

Only `z2_high` is touched. `z1`/`z3` and every scalar core keep `at()`, which is
what keeps the bite proof asymmetric.

## Measurement

Two sha256-distinct binaries from one tree, arms rotated each round,
same-binary null, 24 rounds x 4 reps.

| | effect | rounds | p |
|---|---:|---:|---:|
| **z2 left prefix** | **−0.274 %** | **20/24** | **0.0015** |
| null | −0.063 % | 13/24 | 0.84 |

**Byte-identical on all four standard cells** — 40,237 / 39,694 / 10,912 /
11,961.

## The same transformation on z1 and z3: REJECTED, and it sharpens the rule

Applying the identical change to `z1_high_scalar` (contiguous inner loop, so
3 checks -> 1, exactly like z2) and `z3_high_scalar` (STRIDED inner loop, so only
the window applies, 3 -> 2) measures **+0.123 %, 10 of 24 rounds faster,
p = 1.0000** against a +0.040 % null. Byte-identical on four cells. **Reverted**;
band committed as `.z1z3_rejected.tsv`.

**Why it does not follow from z2's win:** `z2_high` carries **78 ms** of self
cost and lost **2 of 3** checks per pixel. `z3_high_scalar` carries **27 ms** and
loses **1 of 3** (its store is strided, so there is no row slice to take), and
z1's scalar path barely clears the profile floor at all. Same shape, roughly a
quarter of the work removed, and it disappears into the noise.

**This is a real qualification of the rule this session derived**, and it is
worth more than the change would have been. "Removes work and adds none" is
**necessary but not sufficient** — and it does not even guarantee the right sign:

| change | strictly removes work? | measured |
|---|---|---:|
| 4x4 variance (copy + walk, every unit) | yes | **−0.827 %** |
| z2 left prefix (2 of 3 checks, 78 ms symbol) | yes | **−0.274 %** |
| reconstruct round trip (2 copies, minority of txbs) | yes | −0.147 %, null |
| **z1/z3 (1 of 3 checks, 27 ms symbol)** | **yes** | **+0.123 %, null** |

Two of the four "strictly removes work" changes landed on the wrong side of zero.
So the KB-PERF-44 argument — *keep it anyway, it cannot be slower in principle* —
is **not** a reliable predictor of sign at this magnitude, and should be used only
to justify keeping a change that already measured negative, never to skip the
band or to keep one that did not.

## Not covered

The row is not closed: the gather itself remains scalar, and closing it properly
means libaom's transposing structure. One box, one content class, one quantizer,
`--cpu-used 3`, x86-64. `z1`/`z3`'s scalar prefixes carry the same `at()` shape
and are untouched.
