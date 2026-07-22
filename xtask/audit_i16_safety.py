#!/usr/bin/env python3
"""Audit the generated scalar inverse 1-D kernels for i16-lane narrowability at bd8.

Domain rules (bd8: every stage_range value == 16, driver input pre-clamped to i16):
  - input[..] loads and copies of i16 values                      -> I16
  - clamp_value(_, stage_range[..]) outputs (== saturate to i16)  -> I16
  - half_btf(w, a, w, b) with I16 inputs: |w|<=2^12, |in|<=2^15
      -> products <= 2^27 (no i32 wrap), pair sum + rnd <= 2^28
      -> exact in i32; output <= ~2^16.5                          -> T17 (17-bit transient)
  - half_btf with a T17 input: products <= 2^29 (no wrap), sum <= 2^30, exact i32,
      output <= ~2^17.5                                           -> T18 (would need proof)
  - anything else (identity multiplies, unclamped adds/negs)      -> flagged

The i16 column-pass design is exact iff:
  1. every half_btf input is I16 (so madd-based butterflies are exact), and
  2. every clamp_value operand is I16 or T17 (i32 add + saturating pack is exact), and
  3. every terminal (returned) value is I16 (so the i16 round-shift + packus recon is exact).
Kernels violating any rule are left on the i32 path (reported).
"""
import re, sys, collections

SRC = "crates/aom-dsp/src/transform/inv_txfm1d_gen.rs"
text = open(SRC).read()

fn_re = re.compile(r"pub fn (av1_i\w+)\(input[^{]+\{(.*?)\n\}", re.S)
asn_re = re.compile(r"(\w+)\[(\d+)\]\s*=\s*(.+?);")

def classify(fn_name, body):
    # domain map: (arr, idx) -> 'I16' | 'T17' | 'BAD:<why>'
    dom = {}
    problems = []
    stmts = []
    for m in asn_re.finditer(body):
        stmts.append((m.group(1), int(m.group(2)), m.group(3)))
    def rd(expr_arr, expr_idx):
        return dom.get((expr_arr, expr_idx), "I16?unset")
    ref_re = re.compile(r"(\w+)\[(\d+)\]")
    for arr, idx, rhs in stmts:
        refs = [(a, int(i)) for a, i in ref_re.findall(rhs)]
        refs = [(a, i) for (a, i) in refs if a in ("out", "step", "input")]
        srcs = []
        for (a, i) in refs:
            srcs.append("I16" if a == "input" else rd(a, i))
        if rhs.startswith("clamp_value"):
            # operands may be I16 or T17; result I16
            bad = [s for s in srcs if s not in ("I16", "T17")]
            if bad:
                problems.append(f"{arr}[{idx}] clamp over {bad} <- {rhs}")
            dom[(arr, idx)] = "I16"
        elif rhs.startswith("half_btf"):
            bad = [s for s in srcs if s != "I16"]
            if bad:
                problems.append(f"{arr}[{idx}] half_btf over {bad} <- {rhs}")
                dom[(arr, idx)] = "T18"
            else:
                dom[(arr, idx)] = "T17"
        elif re.fullmatch(r"(input|out|step)\[\d+\]", rhs):
            dom[(arr, idx)] = srcs[0] if srcs else "I16"
        else:
            # anything else: round_shift, muls, unclamped adds/negs...
            allsrc = ",".join(sorted(set(srcs))) or "-"
            problems.append(f"{arr}[{idx}] OTHER({allsrc}) <- {rhs}")
            dom[(arr, idx)] = f"OTHER"
    # terminal domain: the final values in `out`
    term = collections.Counter()
    for (a, i), d in dom.items():
        pass
    # replay to get FINAL out[] domains (dom holds last write already)
    outdoms = collections.Counter(d for (a, i), d in dom.items() if a == "out")
    # But dom[(out,i)] is the LAST write to out[i] — what we want.
    term_bad = [i for (a, i), d in dom.items() if a == "out" and d != "I16"]
    return problems, outdoms, sorted(term_bad)

for m in fn_re.finditer(text):
    name, body = m.group(1), m.group(2)
    problems, outdoms, term_bad = classify(name, body)
    status = "OK-i16" if not problems and not term_bad else "NOT-i16"
    print(f"== {name}: {status}  terminal-out domains: {dict(outdoms)}")
    if term_bad:
        print(f"   terminal non-I16 out indices: {term_bad}")
    for p in problems:
        print(f"   PROBLEM: {p}")
