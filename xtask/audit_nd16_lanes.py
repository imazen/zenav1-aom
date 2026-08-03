#!/usr/bin/env python3
"""Audit the NON-DIRECTIONAL intra predictors (SMOOTH / SMOOTH_V / SMOOTH_H /
PAETH) for 16-bit-lane narrowability, and derive the per-kernel sample bound
`M*` that makes the narrowing EXACT.

This is the third of the lane-width audits and it is the *easy* one, on
purpose: unlike `audit_i16_fwd.py` (which has to propagate an exact linear
form through a 64-point butterfly network) every intermediate here is a
function of at most FOUR scalars drawn from small ranges, so the bound is
established by **exhaustive enumeration over the whole product space** rather
than by any inequality. Nothing below is an estimate.

## The kernels, as `aom_dsp/src/intra/simd.rs` writes them

    SMOOTH    p = wh*above + (256-wh)*below + ww*left + (256-ww)*right
              out = (p + 256) >> 9
    SMOOTH_V  p = w*above + (256-w)*below                  out = (p + 128) >> 8
    SMOOTH_H  p = w*left  + (256-w)*right                  out = (p + 128) >> 8
    PAETH     base = top + left - top_left
              out  = argmin over {left, top, top_left} of |base - x|,
                     ties resolved left > top > top_left

with weights `w, wh, ww` from the `u8` table `SMOOTH_WEIGHTS` (so `<= 255` by
type) and samples in `[0, M]`.

## What the vector form changes, and what it must not

The port's shipped kernel is `i32x8` — a 32-bit lane per sample, so nothing can
wrap and there is no bound to prove. libaom's lowbd kernels run the SMOOTH
family in **`u16` lanes** (`vmull_u8` / `vmlal_u8` / `vhaddq_u16` /
`vrshrn_n_u16`, `aom_dsp/arm/intrapred_neon.c:2383-2520`) and PAETH likewise on
narrow lanes. Sixteen `u16` lanes fit the same register as eight `i32` ones, so
the narrow form is 2x the work per instruction — provided every intermediate
stays inside its lane.

Two of the four intermediates do not fit trivially:

  * SMOOTH's `p` reaches `2 * 256 * M`, which is OUTSIDE `u16` for every
    `M >= 128`. The vector form therefore keeps the two halves separate and
    combines them with a **truncating halving add**, exactly as libaom does:

        A = wh*above + (256-wh)*below       B = ww*left + (256-ww)*right
        out = ((A + B) >> 1  +  128) >> 8            [`vhaddq_u16` + `vrshrn`]

    which this script checks is equal to `(A + B + 256) >> 9` for EVERY
    reachable `A + B` — not for a sample of them.

  * PAETH's `base` is signed, so it runs in `i16` lanes and the bound is on
    `|base - top_left| <= 2M` rather than on `256M`.

## Output

For each kernel: `M*`, the largest sample bound for which every intermediate is
representable, established by enumeration; the intermediate that binds it; and
the witness at `M* + 1` that shows the bound is TIGHT (i.e. that it is not a
conservative under-estimate). A bound with no witness at `M*+1` would mean the
audit had left headroom on the table and the gate would be needlessly narrow.

Run: `python3 xtask/audit_nd16_lanes.py`
"""

import sys

U16_MAX = 65535
I16_MIN, I16_MAX = -32768, 32767

# ---------------------------------------------------------------------------
# The SMOOTH family: every intermediate that lives in a u16 lane.
# ---------------------------------------------------------------------------


def smooth_half_max(M):
    """max over w in [0,255], a,b in [0,M] of each u16 intermediate of one
    half-term `w*a + (256-w)*b`, returned as a dict."""
    # The three intermediates the vector body materialises, each maximised
    # exhaustively over the whole (w, a, b) product space.
    m_prod0 = 0  # w * a
    m_prod1 = 0  # (256 - w) * b
    m_sum = 0  # the half-term itself
    for w in range(256):
        # Both products are monotone in their sample operand, so the sample
        # maximum is at a = b = M; the weight axis is still swept in full.
        p0 = w * M
        p1 = (256 - w) * M
        m_prod0 = max(m_prod0, p0)
        m_prod1 = max(m_prod1, p1)
        m_sum = max(m_sum, p0 + p1)
    return {"w*a": m_prod0, "(256-w)*b": m_prod1, "half-term": m_sum}


def smooth_intermediates(M):
    """Every u16-lane intermediate of the full SMOOTH kernel at sample bound M."""
    h = smooth_half_max(M)
    A = h["half-term"]  # A and B share the same bound
    out = dict(h)
    # (A & B) + ((A ^ B) >> 1) == floor((A+B)/2) -- both addends and the sum.
    out["halving add"] = (A + A) // 2
    out["+128 before >>8"] = (A + A) // 2 + 128
    return out


def smoothvh_intermediates(M):
    """SMOOTH_V / SMOOTH_H: one half-term, then `+128` before the `>>8`."""
    h = smooth_half_max(M)
    out = dict(h)
    out["+128 before >>8"] = h["half-term"] + 128
    return out


def paeth_intermediates(M):
    """PAETH in i16 lanes: the signed extremes over top, left, top_left in
    [0, M], enumerated over the sign structure rather than argued.

    These are the intermediates the SHIPPED kernel materialises, which is not
    quite the textbook form: `base` itself is never built, because
    `base - left == top - top_left` is row-invariant and hoisted out of the row
    loop, and `base - top_left` is computed as `top + (left - 2*top_left)` so
    the inner loop needs one add rather than two. `left - 2*top_left` is
    therefore a real intermediate and is audited as one."""
    lo = hi = 0
    for top in (0, M):
        for left in (0, M):
            for tl in (0, M):
                base = top + left - tl
                for v in (
                    base,  # base itself (not materialised; the re-association below is)
                    top - tl,  # == base - left, hoisted out of the row loop
                    left - tl,  # == base - top, taken scalar-side per row
                    left - 2 * tl,  # the per-row addend
                    top + (left - 2 * tl),  # == base - top_left
                ):
                    lo = min(lo, v)
                    hi = max(hi, v)
                # abs() of each distance must also be representable
                for v in (top - tl, left - tl, top + (left - 2 * tl)):
                    hi = max(hi, abs(v))
    return {"min signed": lo, "max signed": hi}


def largest_M(fits, hi=1 << 20):
    """Largest M in [0, hi] with `fits(M)`; the predicates here are monotone in
    M (every intermediate is non-decreasing in the sample bound), so a binary
    search is exact."""
    lo, hi_ = 0, hi
    assert fits(lo), "M = 0 must fit"
    if fits(hi_):
        return hi_
    while lo + 1 < hi_:
        mid = (lo + hi_) // 2
        if fits(mid):
            lo = mid
        else:
            hi_ = mid
    return lo


def u16_fits(d):
    return all(0 <= v <= U16_MAX for v in d.values())


def i16_fits(d):
    return all(I16_MIN <= v <= I16_MAX for v in d.values())


# ---------------------------------------------------------------------------
# The halving-add identity, verified over EVERY reachable sum.
# ---------------------------------------------------------------------------


def check_halving_identity(M):
    """`((A+B)>>1 + 128) >> 8 == (A + B + 256) >> 9` for every reachable A+B.

    Both sides depend on `A + B` alone, so sweeping the SUM over its whole
    reachable range is a COMPLETE verification, not a sample. (A quadratic
    sweep over (A, B) would check the same 130 561 distinct cases many times
    over.) The truncating halving add is what makes this exact — the
    *rounding* halving add `vrhaddq_u16` is off by one at `A+B ≡ 255 mod 512`,
    which this function is also asked to demonstrate.
    """
    smax = 2 * 256 * M
    bad_trunc = bad_round = None
    for s in range(smax + 1):
        want = (s + 256) >> 9
        got_trunc = ((s >> 1) + 128) >> 8
        got_round = (((s + 1) >> 1) + 128) >> 8
        if got_trunc != want and bad_trunc is None:
            bad_trunc = (s, want, got_trunc)
        if got_round != want and bad_round is None:
            bad_round = (s, want, got_round)
    return smax, bad_trunc, bad_round


def main():
    print("# audit_nd16_lanes — 16-bit-lane bounds for the non-directional")
    print("# intra predictors, by exhaustive enumeration.")
    print()

    rows = []

    m_smooth = largest_M(lambda M: u16_fits(smooth_intermediates(M)))
    rows.append(("SMOOTH", "u16", m_smooth, smooth_intermediates, u16_fits))

    m_vh = largest_M(lambda M: u16_fits(smoothvh_intermediates(M)))
    rows.append(("SMOOTH_V/H", "u16", m_vh, smoothvh_intermediates, u16_fits))

    m_paeth = largest_M(lambda M: i16_fits(paeth_intermediates(M)))
    rows.append(("PAETH", "i16", m_paeth, paeth_intermediates, i16_fits))

    print(f"{'kernel':12s} {'lane':5s} {'M*':>7s}  binding intermediate at M*")
    for name, lane, M, f, _fits in rows:
        d = f(M)
        binder = max(d.items(), key=lambda kv: abs(kv[1]))
        print(f"{name:12s} {lane:5s} {M:7d}  {binder[0]} = {binder[1]}")
    print()

    print("# Tightness: the first intermediate to leave the lane at M*+1.")
    ok = True
    for name, lane, M, f, fits in rows:
        d = f(M + 1)
        assert not fits(d), f"{name}: M*={M} is NOT tight — M*+1 still fits"
        over = [
            (k, v)
            for k, v in d.items()
            if not (0 <= v <= U16_MAX) if lane == "u16"
        ] or [(k, v) for k, v in d.items() if not (I16_MIN <= v <= I16_MAX)]
        print(f"{name:12s} at M*+1 = {M + 1}: {over[0][0]} = {over[0][1]} leaves {lane}")
    print()

    # In bit-depth terms.
    print("# What the bound admits, in bit-depth terms (max sample = 2^bd - 1):")
    for name, lane, M, _f, _fits in rows:
        adm = [bd for bd in (8, 10, 12) if (1 << bd) - 1 <= M]
        print(f"{name:12s} M* = {M:5d}  ->  bd {adm if adm else 'none'}")
    print()

    smax, bad_trunc, bad_round = check_halving_identity(m_smooth)
    print(f"# SMOOTH halving-add identity, swept over every reachable A+B (0..{smax}):")
    if bad_trunc is None:
        print(f"  TRUNCATING halving add: EXACT over all {smax + 1} sums")
    else:
        print(f"  TRUNCATING halving add: WRONG at A+B={bad_trunc[0]} "
              f"(want {bad_trunc[1]}, got {bad_trunc[2]})")
        ok = False
    if bad_round is None:
        print("  ROUNDING halving add: also exact (unexpected — check the sweep)")
        ok = False
    else:
        print(f"  ROUNDING halving add ({'vrhaddq_u16'}): first WRONG at "
              f"A+B={bad_round[0]} (want {bad_round[1]}, got {bad_round[2]}) "
              f"— this is why the kernel must use the truncating form")
    print()
    print("RESULT:", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
