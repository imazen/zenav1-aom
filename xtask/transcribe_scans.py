#!/usr/bin/env python3
"""Transcribe av1_scan_orders (scan + iscan tables) from libaom scan.c into Rust.

Mechanical transcription; fidelity is enforced by the equality differential
test in aom-txb/tests/txb_diff.rs (every entry of every (tx_size, tx_type)
combo vs the C data through the oracle shim).
"""
import re

src = open("reference/libaom/av1/common/scan.c").read()
# Strip comments.
src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
body_nocomment = re.sub(r"//[^\n]*", "", src)

# All int16_t arrays (scan + iscan), both DECLARE_ALIGNED forms.
arrays = {}
for m in re.finditer(
    r"DECLARE_ALIGNED\(\s*\d+\s*,\s*(?:static\s+)?const\s+int16_t\s*,\s*([A-Za-z0-9_]+)\[(\d+)\]\)\s*=\s*\{(.*?)\};",
    body_nocomment, re.S,
):
    name, n, body = m.group(1), int(m.group(2)), m.group(3)
    vals = [int(v) for v in re.findall(r"-?\d+", body)]
    assert len(vals) == n, (name, len(vals), n)
    arrays[name] = vals
assert len(arrays) == 84, len(arrays)

# The 19x16 initializer of {scan, iscan} name pairs.
init = re.search(r"const SCAN_ORDER av1_scan_orders\[TX_SIZES_ALL\]\[TX_TYPES\] = \{(.*?)\n\};",
                 body_nocomment, re.S).group(1)
pairs = re.findall(r"\{\s*([A-Za-z0-9_]+)\s*,\s*([A-Za-z0-9_]+)\s*\}", init)
assert len(pairs) == 19 * 16, len(pairs)
for s, i in pairs:
    assert s in arrays and i in arrays, (s, i)

def rname(n):
    return n.upper()

out = []
out.append("//! `av1_scan_orders` scan + iscan tables, transcribed from libaom v3.14.1")
out.append("//! `av1/common/scan.c` by `xtask/transcribe_scans.py`. Verified")
out.append("//! entry-for-entry against the C data by `tests/txb_diff.rs` — do not")
out.append("//! hand-edit.")
out.append("")
for name, vals in arrays.items():
    rows = [", ".join(str(v) for v in vals[i:i+20]) for i in range(0, len(vals), 20)]
    out.append("#[rustfmt::skip]")
    out.append(f"static {rname(name)}: [i16; {len(vals)}] = [")
    for r in rows:
        out.append(f"    {r},")
    out.append("];")
out.append("")
out.append("/// `av1_scan_orders[tx_size][tx_type]` -> (scan, iscan).")
out.append("#[rustfmt::skip]")
out.append("pub static SCAN_ORDERS: [[(&[i16], &[i16]); 16]; 19] = [")
for t in range(19):
    row = pairs[t*16:(t+1)*16]
    entries = ", ".join(f"(&{rname(s)}, &{rname(i)})" for s, i in row)
    out.append(f"    [{entries}],")
out.append("];")
out.append("")
out.append("/// The scan order for `(tx_size, tx_type)`: coefficient positions in")
out.append("/// coding order (transposed raster indices).")
out.append("pub fn scan(tx_size: usize, tx_type: usize) -> &'static [i16] {")
out.append("    SCAN_ORDERS[tx_size][tx_type].0")
out.append("}")
out.append("")
out.append("/// The inverse scan for `(tx_size, tx_type)`: position -> scan index.")
out.append("pub fn iscan(tx_size: usize, tx_type: usize) -> &'static [i16] {")
out.append("    SCAN_ORDERS[tx_size][tx_type].1")
out.append("}")
out.append("")
open("crates/aom-txb/src/scan.rs", "w").write("\n".join(out))
total = sum(len(v) for v in arrays.values())
print(f"wrote {len(arrays)} arrays, {total} i16 entries, 19x16 order table")
