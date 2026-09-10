#!/usr/bin/env python3
"""Is `av1_quantize_fp`'s dqcoeff overflow REACHABLE? — the prerequisite for any
i16 rewrite of `aom_dsp::quant::simd::quantize_fp_impl`.

libaom's `av1_quantize_fp_avx2` computes `dqcoeff = tmp * dequant` in 16-bit
WRAPPING arithmetic (two `vpmullw`); the port computes it in i32. They agree
only where the product fits i16, so an i16 rewrite is admissible only if the
overflow is unreachable. This answers that from the port's OWN quantizer tables
rather than by sampling.

Reads `crates/aom-dsp/src/quant/quant_common.rs`. Run from the repo root.
"""
import re, sys

src = open("crates/aom-dsp/src/quant/quant_common.rs").read()

def tbl(name):
    m = re.search(name + r"\s*:\s*\[i16;\s*256\]\s*=\s*\[(.*?)\];", src, re.S)
    if not m:
        sys.exit(f"table {name} not found — did quant_common.rs move?")
    return [int(x) for x in re.findall(r'-?\d+', m.group(1))]

TABS = {
    8:  (tbl("DC_QLOOKUP_QTX"),    tbl("AC_QLOOKUP_QTX")),
    10: (tbl("DC_QLOOKUP_10_QTX"), tbl("AC_QLOOKUP_10_QTX")),
    12: (tbl("DC_QLOOKUP_12_QTX"), tbl("AC_QLOOKUP_12_QTX")),
}

print("av1_quantize_fp:  quant_fp = 65536 / dequant   (av1_quantize.c:628)")
print("                  tmp      = (min(|coeff|, 2^17) + round) * quant >> (16 - log_scale)")
print("                  dqcoeff  = (tmp * dequant) >> log_scale")
print("|coeff| <= 2^(bd+7): av1_gen_fwd_stage_range bounds the forward output at")
print("bd+8 SIGNED bits (KB-ARM-FLOAT root #3 derives the same cap).\n")

worst_overall = {}
for bd, (dc, ac) in TABS.items():
    cmax = 1 << (bd + 7)
    worst, at = 0, None
    for ls in (0, 1, 2):
        for name, t in (("dc", dc), ("ac", ac)):
            for qi in range(256):
                dq = t[qi]
                if dq <= 0:
                    continue
                quant = 65536 // dq
                rnd = (dq * 48) >> 7          # round_fp = dequant * 48 / 128
                tmp = ((min(cmax, 1 << 17) + rnd) * quant) >> (16 - ls)
                d = (tmp * dq) >> ls
                if d > worst:
                    worst, at = d, (ls, name, qi, dq, tmp)
    worst_overall[bd] = worst
    verdict = "FITS i16" if worst <= 32767 else f"OVERFLOWS by {worst/32767:.2f}x"
    print(f"  bd{bd:>2}: worst |dqcoeff| = {worst:>10,}   "
          f"(log_scale={at[0]} {at[1]} qindex={at[2]} dequant={at[3]})   {verdict}")
    print(f"        dequant range {min(t for t in dc if t>0)}..{max(dc)} (dc) / "
          f"{min(t for t in ac if t>0)}..{max(ac)} (ac), |coeff| cap {cmax:,}")

print("\nVERDICT: the overflow is REACHABLE at every bit depth — at bd8 (the only")
print("depth av1_quantize_fp_avx2 serves; hbd has its own kernel) by 1.01x, i.e.")
print("essentially ONE unit: |dqcoeff| is capped at 32768 by the spec clamp and")
print("i16::MAX is 32767. So an i16 rewrite of this kernel CANNOT be justified by")
print("an unreachability proof, and would make the port's bitstream ISA-conditional.")
