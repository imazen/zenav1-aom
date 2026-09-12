#!/usr/bin/env python3
"""Extract per-function CALL COUNTS and self-Ir from a callgrind out file, and
optionally diff two arms (port vs C oracle).

`perf record` attributes TIME; it cannot distinguish "called 4x too often" from
"4x too slow per call". Callgrind records `calls=N` on every call arc, so this
script answers the excess-work question directly: which functions does the port
invoke more often than libaom on the same cell?

Usage:
  callgrind_counts.py FILE.out                 # per-function table, one arm
  callgrind_counts.py PORT.out C.out           # diff: functions by call-count ratio
  callgrind_counts.py PORT.out C.out --top 40  # limit the diff table

Caveats (read before trusting a row):
  * Counts are of calls that EXIST IN THE BINARY. An LLVM-inlined callee leaves
    no call edge on the Rust side where C's RTCD dispatch keeps one — so a
    missing port row can mean "inlined", not "absent". Per-line Ir still carries
    the inlined work; a feature-gated count!{} macro is the complement for a
    named site.
  * SIMD kernels count once per call regardless of lane width. Calls answer
    "how often"; Ir answers "how much work". An excess-work candidate wants the
    count ratio AND the Ir ratio to disagree.
"""
import re
import sys
from collections import defaultdict


def parse(path):
    """Return (calls, self_ir, self_dw, arcs).

    arcs: callee -> {caller -> [n_calls, inclusive_Ir]} — the "who calls
    memcpy and how often" table."""
    ids = {}          # (kind, num) -> name ; kind in fl/fn/ob
    cur_fn = None     # current fn= context
    cur_cfn = None    # current cfn= context (call arc target)
    cur_calls = 0
    calls = defaultdict(int)
    self_ir = defaultdict(int)
    self_dw = defaultdict(int)
    arcs = defaultdict(lambda: defaultdict(lambda: [0, 0]))

    def resolve(kind, tok):
        # tok is either 'name' or '(num) name' or '(num)'
        m = re.match(r"^\((\d+)\)\s*(.*)$", tok)
        if m:
            num, name = int(m.group(1)), m.group(2).strip()
            if name:
                ids[(kind, num)] = name
                return name
            return ids.get((kind, num), f"<{kind}:{num}>")
        return tok

    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith("fn="):
                cur_fn = resolve("fn", line[3:].strip())
                cur_cfn = None
            elif line.startswith("cfn="):
                cur_cfn = resolve("fn", line[4:].strip())
            elif line.startswith(("cfl=", "cob=", "ob=", "fl=")):
                pass
            elif line.startswith("calls="):
                n = int(line.split()[0][6:])
                if cur_cfn is not None:
                    calls[cur_cfn] += n
                    cur_calls = n
            elif line and line[0].isdigit() or line[:1] in "+-*":
                parts = line.split()
                if len(parts) >= 2 and parts[0][0] in "+-*0123456789":
                    try:
                        ir = int(parts[1])
                    except ValueError:
                        continue
                    dw = 0
                    if len(parts) >= 4:
                        try:
                            dw = int(parts[3])
                        except ValueError:
                            pass
                    if cur_cfn is None:
                        if cur_fn is not None:
                            self_ir[cur_fn] += ir
                            self_dw[cur_fn] += dw
                    else:
                        # first cost line under a cfn= is the arc's inclusive cost
                        a = arcs[cur_cfn][cur_fn or "?"]
                        a[0] += cur_calls
                        a[1] += ir
                        cur_calls = 0
    return calls, self_ir, self_dw, arcs


def short(name):
    # strip hash suffixes and paths for readability
    name = re.sub(r"::h[0-9a-f]{16}$", "", name)
    return name


def main():
    top = 40
    skip = set()
    for flag in ("--top", "--callers-of"):
        if flag in sys.argv:
            i = sys.argv.index(flag)
            skip.update((i, i + 1))
            if flag == "--top":
                top = int(sys.argv[i + 1])
    args = [a for i, a in enumerate(sys.argv[1:], 1)
            if not a.startswith("--") and i not in skip]

    if "--callers-of" in sys.argv:
        target = sys.argv[sys.argv.index("--callers-of") + 1]
        _, _, _, arcs = parse(args[0])
        for name, callers in arcs.items():
            if target in name:
                rows = sorted(callers.items(), key=lambda kv: -kv[1][0])
                print(f"== callers of {short(name)} (total {sum(v[0] for v in callers.values()):,} calls) ==")
                for caller, (n, ir) in rows[:top]:
                    print(f"{n:>12,} calls {ir:>14,} incl-Ir  {short(caller)[:90]}")
                print()
        return

    if len(args) == 1:
        calls, ir, dw, _ = parse(args[0])
        rows = sorted(ir.items(), key=lambda kv: -kv[1])[:top]
        print(f"{'self_Ir':>14} {'calls':>12} {'Ir/call':>9} {'Dw':>12}  function")
        for name, i in rows:
            c = calls.get(name, 0)
            ipc = i / c if c else float("nan")
            print(f"{i:>14,} {c:>12,} {ipc:>9.0f} {dw.get(name,0):>12,}  {short(name)[:90]}")
        return

    pc, pir, pdw, _ = parse(args[0])
    cc, cir, cdw, _ = parse(args[1])

    names = set(pir) | set(cir) | set(pc) | set(cc)
    rows = []
    for n in names:
        p_calls, c_calls = pc.get(n, 0), cc.get(n, 0)
        p_ir, c_ir = pir.get(n, 0), cir.get(n, 0)
        rows.append((n, p_calls, c_calls, p_ir, c_ir))

    print("== Largest CALL-COUNT divergences (port calls vs C calls) ==")
    print(f"{'port_calls':>12} {'c_calls':>12} {'ratio':>7} {'port_Ir':>14} {'c_Ir':>14}  function")
    # rows where BOTH sides have a nonzero count and counts diverge most
    both = [r for r in rows if r[1] > 0 and r[2] > 0]
    both.sort(key=lambda r: -abs(r[1] / r[2] - 1) * min(r[3], r[4] or 1))
    for n, p, c, pi, ci in both[:top]:
        print(f"{p:>12,} {c:>12,} {p/c:>6.2f}x {pi:>14,} {ci:>14,}  {short(n)[:80]}")

    print("\n== Port-only functions by self-Ir (no C call edge; may be inlined in C) ==")
    ponly = [r for r in rows if r[1] > 0 and r[2] == 0]
    ponly.sort(key=lambda r: -r[3])
    for n, p, c, pi, ci in ponly[:top // 2]:
        print(f"{p:>12,} {'—':>12} {'':>7} {pi:>14,} {'—':>14}  {short(n)[:80]}")

    print("\n== C-only functions by self-Ir (dispatched in C, inlined in port) ==")
    conly = [r for r in rows if r[1] == 0 and r[2] > 0]
    conly.sort(key=lambda r: -r[4])
    for n, p, c, pi, ci in conly[:top // 2]:
        print(f"{'—':>12} {c:>12,} {'':>7} {'—':>14} {ci:>14,}  {short(n)[:80]}")


if __name__ == "__main__":
    main()
