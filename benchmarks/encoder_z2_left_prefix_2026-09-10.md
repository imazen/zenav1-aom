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

## Not covered

The row is not closed: the gather itself remains scalar, and closing it properly
means libaom's transposing structure. One box, one content class, one quantizer,
`--cpu-used 3`, x86-64. `z1`/`z3`'s scalar prefixes carry the same `at()` shape
and are untouched.
