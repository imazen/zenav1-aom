#!/usr/bin/env python3
"""eprof_rollup.py — turn a `/usr/bin/sample` call-graph report into a ranked
SELF-cost table and a per-stage rollup, for the encoder hotspot profile.

Method (the sampling analogue of the callgrind rollup in
`benchmarks/gate3_decode_profile_2026-07-19.md`):

* `sample` prints an indented call tree whose per-node number is the
  **inclusive** sample count of that node *at that call site*.
* `self(node) = count(node) - sum(count(child))`, summed over every call site
  of a symbol, is that symbol's self cost.  This is exact (it is just the tree
  arithmetic) and it is correct under recursion, which matters here because
  `rd_pick_partition_real` is recursive.
* Inclusive cost per symbol is reported as the sum over the node's *outermost*
  occurrences only (a node whose symbol already appears among its ancestors is
  not counted again), so a recursive symbol's inclusive number is the honest
  "time under the top-level entry", not a multiple of it.
* Symbols are classified into stages by name.  Rust symbols carry their crate
  and module (`aom_encode::partition_pick::...`); libaom C symbols carry
  libaom's own naming.  Anything unmatched lands in `other` and is printed, so
  the residual is visible rather than hidden.

Usage:
    eprof_rollup.py <sample.txt> [--tsv out.tsv] [--top N] [--arm LABEL]
"""

import argparse
import re
import sys
from collections import defaultdict

# `<prefix chars> <count> <symbol>  (in <image>) + <off>  [addr]  [file:line]`
LINE = re.compile(r"^(?P<pre>[ +!:|]*)(?P<count>\d+) (?P<rest>\S.*)$")
SYM = re.compile(r"^(?P<sym>.*?)\s+\(in (?P<image>[^)]*)\)")

# --------------------------------------------------------------- demangling --
# Rust v0 mangling: _RNvNtCs<hash>_10aom_encode4pack9pack_tile
# We do not need a full demangler — we need the crate + module + fn path, which
# is exactly the sequence of <len><name> components after the crate-id.
V0 = re.compile(r"^_R[A-Za-z]*")


def demangle_v0(s):
    """Extract the `crate::mod::fn` path out of a Rust v0 symbol, best effort."""
    if not s.startswith("_R"):
        return s
    # Strip the crate-disambiguator tokens `Cs<base62>_` FIRST: they sit between
    # the namespace tags and the length-prefixed crate name, and their base62
    # payload contains digits that would otherwise be read as a length.
    s = re.sub(r"Cs[0-9A-Za-z]+_", "", s)
    parts = []
    i = 0
    n = len(s)
    while i < n:
        if s[i].isdigit():
            j = i
            while j < n and s[j].isdigit():
                j += 1
            ln = int(s[i:j])
            name = s[j:j + ln]
            i = j + ln
            # `Cs<hash>_` disambiguators are attached to the crate root; strip
            # the leading hash form `Cs...._` that precedes a crate name.
            parts.append(name)
        else:
            i += 1
    # drop trailing hash-ish components and generic-instantiation noise
    parts = [p for p in parts if p and not re.fullmatch(r"[0-9a-zA-Z]{16,}", p)]
    return "::".join(parts) if parts else s


def pretty(sym):
    if sym.startswith("_R"):
        return demangle_v0(sym)
    if sym.startswith("_ZN"):  # legacy mangling
        return re.sub(r"\d+", lambda m: "::", sym)
    return sym.lstrip("_")


# ------------------------------------------------------------------- stages --
# Ordered; first match wins.  The two arms use different vocabularies, so both
# sets live here and each is anchored on identifiers that only occur in one.
RUST_STAGES = [
    ("cnn-partition-prune", r"aom_encode::cnn_partition"),
    ("hog-intra-prune", r"aom_encode::hog"),
    ("partition-search", r"aom_encode::(partition_pick|partition|part4_prune|var_part)"),
    ("intra-mode-rd", r"aom_encode::(intra_rd|rd_pick|intra_uv_rd|encode_intra)"),
    ("tx-search", r"aom_encode::(tx_search|var_tx|prune_tx_2d|interp_rd)"),
    ("trellis(optimize_txb)", r"aom_dsp::txb::optimize"),
    ("pack/entropy-write", r"aom_encode::(pack|obu_assemble|inter_pack)"),
    ("nonrd", r"aom_encode::nonrd_pickmode"),
    ("screen-tools", r"aom_encode::(palette_search|intrabc_search)"),
    ("postfilter-search", r"aom_encode::(lf_search|pickcdef)|aom_dsp::(restore|loopfilter|cdef)"),
    ("entropy/rate-model", r"aom_dsp::(txb|entropy)|aom_encode::(rd|real_costs|mode_costs|inter_costs|rc|speed_features)"),
    ("allintra-vis", r"aom_encode::allintra_vis"),
    ("dsp:transform", r"aom_dsp::transform"),
    ("dsp:intra-pred", r"aom_dsp::intra"),
    ("dsp:quant", r"aom_dsp::quant"),
    ("dsp:dist(sad/var/satd)", r"aom_dsp::dist"),
    ("dsp:convolve", r"aom_dsp::convolve"),
    ("dsp:recon", r"aom_dsp::recon"),
    ("dsp:other", r"aom_dsp::"),
    ("encode-driver", r"aom_encode::|aom_bench::"),
    ("aom_decode", r"aom_decode::"),
    ("alloc/libc", r"(malloc|free|realloc|calloc|bzero|memset|memcpy|memmove|xzm_|_platform_|nanov2|szone|tiny_|DYLD-STUB)"),
    ("os/setjmp", r"(sigprocmask|sigaltstack|setjmp|longjmp|pthread|mach_)"),
    ("rust-core", r"^(core::|alloc::|std::|__rust)"),
]

C_STAGES = [
    ("cnn-partition-prune", r"(av1_cnn|cnn_|av1_intra_mode_cnn|av1_nn_predict|nn_predict|ml_predict)"),
    ("hog-intra-prune", r"(gradient_info|prune_intra_mode_with_hog|av1_get_gradient_hist|generate_hog)"),
    ("screen-content-detect", r"(av1_set_screen_content_options|count_colors|av1_calc_normalized_variance|is_screen_content)"),
    ("trellis(optimize_txb)", r"(av1_optimize_txb|av1_optimize_b|optimize_txb|av1_get_txb_entropy_context)"),
    ("partition-search", r"(rd_pick_partition|av1_rd_use_partition|encode_sb|pick_sb_modes|none_partition_search|rectangular_partition|ab_partition|prune_part|split_partition|first_partition|set_offsets|init_partition|av1_setup_src_planes|av1_source_variance|av1_get_perpixel_variance|av1_get_max_min_partition|encode_rd_sb_row)"),
    ("intra-mode-rd", r"(intra_mode|rd_pick_intra|handle_intra|av1_rd_pick_intra|search_intra_modes|intra_block_yrd|rd_pick_intra_sbuv|cfl_|av1_predict_intra_block|av1_init_intra_predictors|palette_rd|rd_pick_palette|filter_intra|intra_rd_variance|encode_block_intra|av1_encode_intra)"),
    ("tx-search", r"(tx_type|txfm_search|av1_txfm_rd|tx_size_rd|search_tx|pick_tx|prune_tx|block_rd_txfm|av1_estimate_txfm|select_tx_size|super_block_yrd|av1_xform|av1_txb_|dist_block|txfm_yrd|foreach_transformed_block|av1_encode_block|encode_block)"),
    ("pack/entropy-write", r"(pack_|write_modes|av1_pack_bitstream|write_mb|av1_write_|od_ec_enc|aom_write|encode_superblock|write_tx|write_intra|write_partition|write_coeffs|av1_update_and_record|update_cdf)"),
    ("entropy/rate-model", r"(av1_cost_|cost_coeffs|get_rate|av1_rd_|rd_cost|av1_fill_|av1_set_rd|av1_init_rd|estimate_rd|av1_get_entropy_contexts|av1_get_syntax|get_txb_ctx|txb_common)"),
    ("postfilter-search", r"(pick_filter|av1_pick|cdef_|search_cdef|restoration|lr_|wiener|sgrproj|loop_restor|av1_loop_filter|filter_level|lpf_|loop_filter_)"),
    ("screen-tools", r"(intrabc|dv_|hash_table|av1_get_palette|palette)"),
    ("dsp:transform", r"(fwd_txfm|inv_txfm|av1_fdct|av1_fadst|av1_fidentity|av1_fwht|highbd_fwd|av1_round_shift|txfm_param|av1_get_fwd_txfm|_txfm2d_|fdct|fadst|idct|iadst)"),
    ("dsp:intra-pred", r"(dc_predictor|v_predictor|h_predictor|_predictor_|smooth_|paeth_|d45|d63|d113|d135|d157|d203|directional_intra|av1_dr_prediction|av1_filter_intra_edge|av1_upsample_intra_edge|build_intra_predictors|highbd_.*pred)"),
    ("dsp:quant", r"(quantize|av1_quant|dequant|aom_qm)"),
    ("dsp:dist(sad/var/satd)", r"(aom_sad|aom_variance|aom_get[0-9]|aom_mse|aom_sse|aom_highbd_sad|aom_highbd_var|_sub_pixel_var|aom_satd|aom_hadamard|aom_int_pro|aom_vector_var|aom_sum|aom_avg|aom_minmax|aom_subtract_block|av1_block_error)"),
    ("dsp:convolve", r"(convolve|aom_scaled)"),
    ("dsp:recon", r"(av1_inverse_transform|av1_reconstruct|aom_convolve_copy|copy_rect|extend_)"),
    ("alloc/libc", r"(malloc|free|realloc|calloc|bzero|memset|memcpy|memmove|xzm_|_platform_|nanov2|szone|tiny_|aom_memalign|aom_calloc|aom_free|aom_alloc|DYLD-STUB)"),
    ("os/setjmp", r"(sigprocmask|sigaltstack|setjmp|longjmp|pthread|mach_)"),
]


def classify(name, rules):
    for stage, pat in rules:
        if re.search(pat, name):
            return stage
    return "other"


# --------------------------------------------------------------- the parser --
def parse(path):
    """Returns (nodes, total) where nodes is a list of
    (depth, count, symbol, image, srcline, self_count, is_outermost)."""
    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    try:
        start = next(i for i, l in enumerate(lines) if l.startswith("Call graph:"))
    except StopIteration:
        sys.exit(f"{path}: no 'Call graph:' section")
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].startswith(("Total number in stack", "Binary Images:",
                                "Sort by top of stack")):
            end = i
            break

    raw = []
    for l in lines[start + 1:end]:
        m = LINE.match(l)
        if not m:
            continue
        rest = m.group("rest")
        sm = SYM.match(rest)
        sym = sm.group("sym").strip() if sm else rest.split("  ")[0].strip()
        image = sm.group("image") if sm else "?"
        src = ""
        ms = re.search(r"\]\s+(\S+:\d+)\s*$", rest)
        if ms:
            src = ms.group(1)
        raw.append((len(m.group("pre")), int(m.group("count")), sym, image, src))

    # children sums + recursion detection via an explicit ancestor stack
    child_sum = [0] * len(raw)
    outermost = [True] * len(raw)
    stack = []           # (depth, index)
    ancestors = []       # symbols currently on the stack
    for i, (d, c, sym, _img, _src) in enumerate(raw):
        while stack and stack[-1][0] >= d:
            stack.pop()
            ancestors.pop()
        if stack:
            child_sum[stack[-1][1]] += c
        outermost[i] = sym not in ancestors
        stack.append((d, i))
        ancestors.append(sym)

    nodes = []
    for i, (d, c, sym, img, src) in enumerate(raw):
        nodes.append((d, c, sym, img, src, c - child_sum[i], outermost[i]))
    total = sum(c for d, c, *_ in raw if d == min(x[0] for x in raw))
    return nodes, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sample")
    ap.add_argument("--tsv")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--arm", default="")
    ap.add_argument("--rules", choices=["rust", "c"], default="rust")
    a = ap.parse_args()

    nodes, total = parse(a.sample)
    rules = RUST_STAGES if a.rules == "rust" else C_STAGES

    self_by = defaultdict(int)
    incl_by = defaultdict(int)
    src_by = {}
    for _d, c, sym, _img, src, slf, outer in nodes:
        name = pretty(sym)
        self_by[name] += slf
        if outer:
            incl_by[name] += c
        if src and name not in src_by:
            src_by[name] = src

    stage_self = defaultdict(int)
    for name, s in self_by.items():
        stage_self[classify(name, rules)] += s

    print(f"# {a.arm or a.sample}: {total} samples in the sampled window "
          f"({a.rules} stage rules)")
    print()
    print("## stage rollup (SELF samples)")
    print(f"{'stage':<24}{'self':>10}{'%':>8}")
    for st, s in sorted(stage_self.items(), key=lambda kv: -kv[1]):
        print(f"{st:<24}{s:>10}{100*s/total:>7.2f}%")
    print()
    print(f"## top {a.top} symbols by SELF samples")
    print(f"{'self':>9}{'%':>7}{'incl':>10}{'incl%':>7}  {'stage':<22} symbol")
    top = sorted(self_by.items(), key=lambda kv: -kv[1])[:a.top]
    for name, s in top:
        print(f"{s:>9}{100*s/total:>6.2f}%{incl_by.get(name,0):>10}"
              f"{100*incl_by.get(name,0)/total:>6.2f}%  "
              f"{classify(name, rules):<22} {name}  {src_by.get(name,'')}")

    if a.tsv:
        with open(a.tsv, "w") as f:
            f.write("arm\tkind\tstage\tsymbol\tself_samples\tself_pct\t"
                    "incl_samples\tincl_pct\tsrc\n")
            for st, s in sorted(stage_self.items(), key=lambda kv: -kv[1]):
                f.write(f"{a.arm}\tstage\t{st}\t\t{s}\t{100*s/total:.4f}\t\t\t\n")
            for name, s in sorted(self_by.items(), key=lambda kv: -kv[1]):
                if s == 0:
                    continue
                f.write(f"{a.arm}\tsymbol\t{classify(name, rules)}\t{name}\t{s}\t"
                        f"{100*s/total:.4f}\t{incl_by.get(name,0)}\t"
                        f"{100*incl_by.get(name,0)/total:.4f}\t{src_by.get(name,'')}\n")
        print(f"\n# wrote {a.tsv}")


if __name__ == "__main__":
    main()
