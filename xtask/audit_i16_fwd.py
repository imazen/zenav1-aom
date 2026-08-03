#!/usr/bin/env python3
"""Audit the FORWARD 1-D kernels for i16-lane narrowability, and derive the
per-kernel input bound `M*` that makes the narrowing EXACT.

This is the forward twin of `audit_i16_safety.py`, and the two audits are
shaped differently for a structural reason:

  * the INVERSE kernels `clamp_value(_, stage_range[i])` at every stage, so
    their audit is a DOMAIN question ("is every value either an i16 clamp
    output or a bounded transient?");
  * the FORWARD kernels contain NO clamp at all (`av1_fwd_txfm1d.c` has no
    range check in the production config), so every value's magnitude is a
    pure function of the INPUT magnitude. The audit is therefore a BOUND
    question: how large may |input| be before some intermediate leaves i16?

## The bound is exact, not a triangle-inequality estimate

Each kernel is a fixed sequence of adds, negations and
`half_btf(w0,a,w1,b,bit) = (w0*a + w1*b + 2^(bit-1)) >> bit`. Every value is
therefore an exact integer linear form in the inputs plus an accumulated
rounding error. This script propagates BOTH:

    value  =  sum_i c_i * input_i   +   e,      |e| <= E

    input[i]                 c = unit vector e_i,          E = 0
    a +/- b                  c = c_a +/- c_b,              E = E_a + E_b
    -a                       c = -c_a,                     E = E_a
    half_btf(w0,a,w1,b,bit)  c = (w0 c_a + w1 c_b)/2^bit,  E = (|w0|E_a+|w1|E_b)/2^bit + 1/2
    round_shift(a*C, k)      c = C c_a / 2^k,              E = |C| E_a / 2^k + 1/2
    a * k                    c = k c_a,                    E = |k| E_a

Coefficients are exact `Fraction`s (all denominators are powers of two), so

    |value|  <=  M * sum_i |c_i|  +  E                                    (*)

is a SOUND upper bound, and it is also TIGHT: the sign vertex
`input_i = M * sign(c_i)` attains `M * sum|c_i|` in the linear part. The
triangle-inequality bound that treats each butterfly operand as independently
maximal is 1.5-2x looser and would reject kernels that are provably safe
(fdct32's column pass at bd8 is exactly such a case).

`M*` is then the largest `M` for which (*) stays <= i16::MAX at EVERY value
of the kernel, and that single condition proves the two things the vector code
needs:

  * no `wrapping_*` in the scalar kernel actually wraps at that bound, so i16
    lane arithmetic (which would wrap at 2^15) computes the same integer;
  * every `half_btf` product |w * v| <= 2^13 * 2^15 = 2^28 and each pair sum
    <= 2^29 + 2^12, i.e. exact in the i32 accumulator a widening multiply
    gives — the same argument the inverse audit makes for its T17 transients.

Usage:
    python3 xtask/audit_i16_fwd.py             # M* table + bd8 reach
    python3 xtask/audit_i16_fwd.py --cells     # + the per-(tx_size,tx_type) grid
"""

import re
import sys
from fractions import Fraction

ROOT = "crates/aom-dsp/src/transform/"
COS_BIT_MIN = 10
I16_MAX = 32767

COSPI = [
    [1024, 1024, 1023, 1021, 1019, 1016, 1013, 1009, 1004, 999, 993, 987, 980,
     972, 964, 955, 946, 936, 926, 915, 903, 891, 878, 865, 851, 837,
     822, 807, 792, 775, 759, 742, 724, 706, 688, 669, 650, 630, 610,
     590, 569, 548, 526, 505, 483, 460, 438, 415, 392, 369, 345, 321,
     297, 273, 249, 224, 200, 175, 150, 125, 100, 75, 50, 25],
    [2048, 2047, 2046, 2042, 2038, 2033, 2026, 2018, 2009, 1998, 1987,
     1974, 1960, 1945, 1928, 1911, 1892, 1872, 1851, 1829, 1806, 1782,
     1757, 1730, 1703, 1674, 1645, 1615, 1583, 1551, 1517, 1483, 1448,
     1412, 1375, 1338, 1299, 1260, 1220, 1179, 1138, 1096, 1053, 1009,
     965, 921, 876, 830, 784, 737, 690, 642, 595, 546, 498,
     449, 400, 350, 301, 251, 201, 151, 100, 50],
    [4096, 4095, 4091, 4085, 4076, 4065, 4052, 4036, 4017, 3996, 3973,
     3948, 3920, 3889, 3857, 3822, 3784, 3745, 3703, 3659, 3612, 3564,
     3513, 3461, 3406, 3349, 3290, 3229, 3166, 3102, 3035, 2967, 2896,
     2824, 2751, 2675, 2598, 2520, 2440, 2359, 2276, 2191, 2106, 2019,
     1931, 1842, 1751, 1660, 1567, 1474, 1380, 1285, 1189, 1092, 995,
     897, 799, 700, 601, 501, 401, 301, 201, 101],
    [8192, 8190, 8182, 8170, 8153, 8130, 8103, 8071, 8035, 7993, 7946,
     7895, 7839, 7779, 7713, 7643, 7568, 7489, 7405, 7317, 7225, 7128,
     7027, 6921, 6811, 6698, 6580, 6458, 6333, 6203, 6070, 5933, 5793,
     5649, 5501, 5351, 5197, 5040, 4880, 4717, 4551, 4383, 4212, 4038,
     3862, 3683, 3503, 3320, 3135, 2948, 2760, 2570, 2378, 2185, 1990,
     1795, 1598, 1401, 1202, 1003, 803, 603, 402, 201],
]
SINPI = [
    [0, 330, 621, 836, 951],
    [0, 660, 1241, 1672, 1901],
    [0, 1321, 2482, 3344, 3803],
    [0, 2642, 4964, 6689, 7606],
]
NEW_SQRT2 = 5793
NEW_SQRT2_BITS = 12
HALF = Fraction(1, 2)


# ---------------------------------------------------------------------------
# Exact linear forms


class Lin:
    """`sum_i c[i]*input_i + e` with `|e| <= E`. All arithmetic exact."""

    __slots__ = ("c", "E")

    def __init__(self, c, E=Fraction(0)):
        self.c = c
        self.E = E

    @staticmethod
    def unit(n, i):
        c = [Fraction(0)] * n
        c[i] = Fraction(1)
        return Lin(c)

    def __add__(self, o):
        return Lin([a + b for a, b in zip(self.c, o.c)], self.E + o.E)

    def __sub__(self, o):
        return Lin([a - b for a, b in zip(self.c, o.c)], self.E + o.E)

    def __neg__(self):
        return Lin([-a for a in self.c], self.E)

    def scale_exact(self, k):
        """Multiply by an exact integer (no rounding)."""
        return Lin([a * k for a in self.c], self.E * abs(k))

    def l1(self):
        return sum(abs(a) for a in self.c)

    def bound(self, M):
        """Sound and tight upper bound on |value| for |input_i| <= M."""
        return self.l1() * M + self.E


def half_btf(w0, a, w1, b, bit):
    d = Fraction(1, 1 << bit)
    c = [(w0 * x + w1 * y) * d for x, y in zip(a.c, b.c)]
    return Lin(c, (abs(w0) * a.E + abs(w1) * b.E) * d + HALF)


def round_shift_mul(a, C, bit):
    d = Fraction(1, 1 << bit)
    return Lin([x * C * d for x in a.c], abs(C) * a.E * d + HALF)


# ---------------------------------------------------------------------------
# Parsing the generated kernels


def read(path):
    with open(ROOT + path) as f:
        return f.read()


def extract(name, text):
    m = re.search(r"pub fn %s\((.*?)\n\}" % re.escape(name), text, re.S)
    if not m:
        raise SystemExit("kernel %s not found" % name)
    return m.group(1)


HBTF = re.compile(
    r"^half_btf\((-?)cospi\[(\d+)\], (\w+)\[(\d+)\], (-?)cospi\[(\d+)\], "
    r"(\w+)\[(\d+)\], cos_bit\)$")
ASSIGN = re.compile(r"^\s*(\w+)\[(\d+)\]\s*=\s*(.+?);\s*$", re.M)
COPY = re.compile(r"^(\w+)\[(\d+)\]$")
NEGCOPY = re.compile(r"^(\w+)\[(\d+)\]\.wrapping_neg\(\)$")
ADD = re.compile(r"^(\w+)\[(\d+)\]\.wrapping_add\((\w+)\[(\d+)\]\)$")
SUB = re.compile(r"^(\w+)\[(\d+)\]\.wrapping_sub\((\w+)\[(\d+)\]\)$")
NEGADD = re.compile(
    r"^(\w+)\[(\d+)\]\.wrapping_neg\(\)\.wrapping_add\((\w+)\[(\d+)\]\)$")


def eval_kernel(body, n, cos_bit):
    """Every assigned value of one generated kernel, as exact linear forms."""
    cospi = COSPI[cos_bit - COS_BIT_MIN]
    env = {}

    def get(arr, idx):
        idx = int(idx)
        if arr == "input":
            return Lin.unit(n, idx)
        v = env.get((arr, idx))
        if v is None:
            raise SystemExit("read before write: %s[%d]" % (arr, idx))
        return v

    allvals = []
    for arr, idx, rhs in ASSIGN.findall(body):
        if arr not in ("out", "output", "step", "bf1"):
            continue
        rhs = rhs.strip()
        m = HBTF.match(rhs)
        if m:
            w0 = cospi[int(m.group(2))] * (-1 if m.group(1) else 1)
            w1 = cospi[int(m.group(6))] * (-1 if m.group(5) else 1)
            v = half_btf(w0, get(m.group(3), m.group(4)),
                         w1, get(m.group(7), m.group(8)), cos_bit)
        elif NEGADD.match(rhs):
            g = NEGADD.match(rhs)
            v = get(g.group(3), g.group(4)) - get(g.group(1), g.group(2))
        elif NEGCOPY.match(rhs):
            g = NEGCOPY.match(rhs)
            v = -get(g.group(1), g.group(2))
        elif COPY.match(rhs):
            g = COPY.match(rhs)
            v = get(g.group(1), g.group(2))
        elif ADD.match(rhs):
            g = ADD.match(rhs)
            v = get(g.group(1), g.group(2)) + get(g.group(3), g.group(4))
        elif SUB.match(rhs):
            g = SUB.match(rhs)
            v = get(g.group(1), g.group(2)) - get(g.group(3), g.group(4))
        else:
            raise SystemExit("UNPARSED forward statement: %s" % rhs)
        env[(arr, int(idx))] = v
        allvals.append(v)
    outs = [v for (a, i), v in env.items() if a in ("out", "output")]
    return allvals, outs


def eval_fadst4(cos_bit):
    """fadst4 in exact linear forms. It has no array statements: it works in
    scalar `let` bindings over a PRE-SHIFT domain, where every stage-1 value is
    `sinpi[j] * x` held UNSHIFTED, i.e. ~2^13 times the input."""
    n = 4
    sinpi = SINPI[cos_bit - COS_BIT_MIN]
    x = [Lin.unit(n, i) for i in range(4)]
    s0 = x[0].scale_exact(sinpi[1])
    s1 = x[0].scale_exact(sinpi[4])
    s2 = x[1].scale_exact(sinpi[2])
    s3 = x[1].scale_exact(sinpi[1])
    s4 = x[2].scale_exact(sinpi[3])
    s5 = x[3].scale_exact(sinpi[4])
    s6 = x[3].scale_exact(sinpi[2])
    s7 = (x[0] + x[1]) - x[3]
    y0 = (s0 + s2) + s5
    y1 = s7.scale_exact(sinpi[3])
    y2 = (s1 - s3) + s6
    y3 = s4
    t0 = y0 + y3
    t1 = y1
    t2 = y2 - y3
    t3 = (y2 - y0) + y3
    allv = [s0, s1, s2, s3, s4, s5, s6, s7, y0, y1, y2, y3, t0, t1, t2, t3]
    outs = [round_shift_mul(t, 1, cos_bit) for t in (t0, t1, t2, t3)]
    return allv + outs, outs


def eval_fidentity(name, cos_bit):
    n = {"fidentity4": 4, "fidentity8": 8, "fidentity16": 16,
         "fidentity32": 32}[name]
    x = [Lin.unit(n, i) for i in range(n)]
    if name == "fidentity4":
        outs = [round_shift_mul(v, NEW_SQRT2, NEW_SQRT2_BITS) for v in x]
    elif name == "fidentity8":
        outs = [v.scale_exact(2) for v in x]
    elif name == "fidentity16":
        outs = [round_shift_mul(v, 2 * NEW_SQRT2, NEW_SQRT2_BITS) for v in x]
    else:
        outs = [v.scale_exact(4) for v in x]
    return outs, outs


# ---------------------------------------------------------------------------

GEN = read("txfm1d_gen.rs")
FDCT = read("fdct.rs")
SPECIAL = read("special.rs")

KERNELS = {
    "fdct4": ("av1_fdct4", FDCT, 4),
    "fdct8": ("av1_fdct8", GEN, 8),
    "fdct16": ("av1_fdct16", GEN, 16),
    "fdct32": ("av1_fdct32", GEN, 32),
    "fdct64": ("av1_fdct64", GEN, 64),
    "fadst8": ("av1_fadst8", GEN, 8),
    "fadst16": ("av1_fadst16", GEN, 16),
}
IDENTITY = ["fidentity4", "fidentity8", "fidentity16", "fidentity32"]
ALL = list(KERNELS) + IDENTITY + ["fadst4"]

_cache = {}


def analyse(name, cos_bit):
    key = (name, cos_bit)
    if key in _cache:
        return _cache[key]
    if name == "fadst4":
        allv, outs = eval_fadst4(cos_bit)
    elif name in IDENTITY:
        allv, outs = eval_fidentity(name, cos_bit)
    else:
        fn, text, n = KERNELS[name]
        allv, outs = eval_kernel(extract(fn, text), n, cos_bit)
    peak_l1 = max(v.l1() for v in allv)
    peak_E = max(v.E for v in allv)
    mstar = None
    for v in allv:
        if v.l1() == 0:
            continue
        m = int((Fraction(I16_MAX) - v.E) / v.l1())
        mstar = m if mstar is None else min(mstar, m)
    out_l1 = max(v.l1() for v in outs)
    out_E = max(v.E for v in outs)
    res = (mstar, peak_l1, peak_E, out_l1, out_E)
    _cache[key] = res
    return res


def out_bound(name, cos_bit, m_in):
    _m, _pl, _pe, ol1, oe = analyse(name, cos_bit)
    return int(ol1 * m_in + oe)


# ---------------------------------------------------------------------------
# Driver tables (mirrors of crates/aom-dsp/src/transform/txfm2d.rs)

TX_W = [4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64]
TX_H = [4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16]
SHIFT = [
    [2, 0, 0], [2, -1, 0], [2, -2, 0], [2, -4, 0], [0, -2, -2],
    [2, -1, 0], [2, -1, 0], [2, -2, 0], [2, -2, 0], [2, -4, 0],
    [2, -4, 0], [0, -2, -2], [2, -4, -2], [2, -1, 0], [2, -1, 0],
    [2, -2, 0], [2, -2, 0], [0, -2, 0], [2, -4, 0],
]
COS_BIT_COL = [
    [13, 13, 13, 0, 0], [13, 13, 13, 12, 0], [13, 13, 13, 12, 13],
    [0, 13, 13, 12, 13], [0, 0, 13, 12, 13],
]
COS_BIT_ROW = [
    [13, 13, 12, 0, 0], [13, 13, 13, 12, 0], [13, 13, 12, 13, 12],
    [0, 12, 13, 12, 11], [0, 0, 12, 11, 10],
]
VTX = [0, 1, 0, 1, 2, 0, 2, 1, 2, 3, 0, 3, 1, 3, 2, 3]
HTX = [0, 0, 1, 1, 0, 2, 2, 2, 1, 3, 3, 0, 3, 1, 3, 2]
TXFM_TYPE_LS = [
    [0, 5, 5, 8], [1, 6, 6, 9], [2, 7, 7, 10], [3, -1, -1, 11], [4, -1, -1, -1],
]
TYPE_NAME = {0: "fdct4", 1: "fdct8", 2: "fdct16", 3: "fdct32", 4: "fdct64",
             5: "fadst4", 6: "fadst8", 7: "fadst16",
             8: "fidentity4", 9: "fidentity8", 10: "fidentity16",
             11: "fidentity32"}
LOG2 = {4: 0, 8: 1, 16: 2, 32: 3, 64: 4}


def rshift_bound(m, bit):
    return (m + (1 << (bit - 1))) >> bit if bit > 0 else m


def main():
    print("=== M*: the i16-safe INPUT BOUND per (kernel, cos_bit) ===")
    print("the largest |input| for which EVERY value inside that kernel stays")
    print("inside i16::MAX, hence for which i16 lane arithmetic is bit-identical")
    print("to the scalar i32 kernel. `L1` is the peak coefficient sum (the gain")
    print("that binds) and `E` the peak accumulated rounding error.")
    print()
    print("%-13s %8s %8s %8s %8s   %9s %6s" %
          ("kernel", "cb=10", "cb=11", "cb=12", "cb=13", "peak L1", "E"))
    for name in ALL:
        row = [analyse(name, cb)[0] for cb in (10, 11, 12, 13)]
        _m, pl1, pE, _o, _oe = analyse(name, 13)
        print("%-13s %8d %8d %8d %8d   %9.4f %6.2f" %
              (name, row[0], row[1], row[2], row[3], float(pl1), float(pE)))
    print()

    print("=== bd8 reach: |residual| <= 255 pushed through the driver ===")
    print("col_in = 255 << shift[0];  row_in = round_shift(col_out_bound, -shift[1]).")
    print("A pass is REACHABLE when its entry bound is <= that kernel's M*.")
    print()
    cells = []
    for ts in range(19):
        w, h = TX_W[ts], TX_H[ts]
        for tt in range(16):
            cti = TXFM_TYPE_LS[LOG2[h]][VTX[tt]]
            rti = TXFM_TYPE_LS[LOG2[w]][HTX[tt]]
            if cti < 0 or rti < 0:
                continue
            cn, rn = TYPE_NAME[cti], TYPE_NAME[rti]
            cb_c = COS_BIT_COL[LOG2[w]][LOG2[h]]
            cb_r = COS_BIT_ROW[LOG2[w]][LOG2[h]]
            sh = SHIFT[ts]
            cin = 255 << sh[0]
            cms = analyse(cn, cb_c)[0]
            rin = rshift_bound(out_bound(cn, cb_c, cin), -sh[1])
            rms = analyse(rn, cb_r)[0]
            cells.append((ts, tt, cn, cin, cms, cin <= cms,
                          rn, rin, rms, rin <= rms))
    if "--cells" in sys.argv:
        print("%-8s %-8s %-12s %8s %8s %-4s  %-12s %8s %8s %-4s" % (
            "tx_size", "tx_type", "col kernel", "col_in", "M*", "col",
            "row kernel", "row_in", "M*", "row"))
        for c in cells:
            print("%-8d %-8d %-12s %8d %8d %-4s  %-12s %8d %8d %-4s" % (
                c[0], c[1], c[2], c[3], c[4], "OK" if c[5] else "NO",
                c[6], c[7], c[8], "OK" if c[9] else "NO"))
        print()

    def summarise(label, kidx, okidx):
        ok, no = {}, {}
        for c in cells:
            d = ok if c[okidx] else no
            d[c[kidx]] = d.get(c[kidx], 0) + 1
        print("%s pass: %d of %d cells reachable at bd8." %
              (label, sum(ok.values()), len(cells)))
        print("   reachable kernels: %s" % dict(sorted(ok.items())))
        print("   blocked   kernels: %s" % dict(sorted(no.items())))
    summarise("COLUMN", 2, 5)
    print()
    summarise("ROW   ", 6, 9)


main()
