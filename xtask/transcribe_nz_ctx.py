#!/usr/bin/env python3
"""Transcribe av1_nz_map_ctx_offset tables from libaom txb_common.c into Rust.

Faithful mechanical transcription; correctness is enforced by the
table-equality differential test in aom-txb/tests/txb_diff.rs.
"""
import re, sys

src = open("reference/libaom/av1/common/txb_common.c").read()

# Per-size arrays.
arrays = {}
for m in re.finditer(
    r"static const int8_t (av1_nz_map_ctx_offset_(\d+x\d+))\[(\d+)\] = \{(.*?)\};",
    src, re.S,
):
    name, shape, n, body = m.group(1), m.group(2), int(m.group(3)), m.group(4)
    vals = [int(v) for v in re.findall(r"-?\d+", body)]
    assert len(vals) == n, (name, len(vals), n)
    arrays[shape] = vals

# Pointer table order (19 TX sizes).
ptr_body = re.search(r"const int8_t \*av1_nz_map_ctx_offset\[19\] = \{(.*?)\};", src, re.S).group(1)
ptrs = re.findall(r"av1_nz_map_ctx_offset_(\d+x\d+)", ptr_body)
assert len(ptrs) == 19, ptrs

out = []
out.append("//! `av1_nz_map_ctx_offset` tables, transcribed from libaom v3.14.1")
out.append("//! `av1/common/txb_common.c` by `xtask/transcribe_nz_ctx.py`.")
out.append("//! Verified entry-for-entry against the C data by the table-equality")
out.append("//! differential test in `tests/txb_diff.rs` — do not hand-edit.")
out.append("")
for shape, vals in arrays.items():
    rows = [", ".join(str(v) for v in vals[i:i+19]) for i in range(0, len(vals), 19)]
    out.append(f"#[rustfmt::skip]")
    out.append(f"static NZ_MAP_CTX_OFFSET_{shape.upper().replace('X','X')}: [i8; {len(vals)}] = [")
    for r in rows:
        out.append(f"    {r},")
    out.append("];")
    out.append("")
out.append("/// `av1_nz_map_ctx_offset[TX_SIZES_ALL]`: per-tx-size 2D context-offset")
out.append("/// table (indexed by transposed raster `coeff_idx`), with libaom's exact")
out.append("/// alias mapping (e.g. TX_8X4 -> the 16x4 table, TX_64X64 -> 32x32).")
out.append("pub fn nz_map_ctx_offset(tx_size: usize) -> &'static [i8] {")
out.append("    match tx_size {")
for i, shape in enumerate(ptrs):
    out.append(f"        {i} => &NZ_MAP_CTX_OFFSET_{shape.upper()},")
out.append("        _ => &NZ_MAP_CTX_OFFSET_4X4,")
out.append("    }")
out.append("}")
out.append("")
open("crates/aom-txb/src/tables.rs", "w").write("\n".join(out))
print(f"wrote {len(arrays)} arrays, {sum(len(v) for v in arrays.values())} entries, 19 aliases")
