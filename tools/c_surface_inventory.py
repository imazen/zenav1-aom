#!/usr/bin/env python3
"""Inventory of the libaom C **inter-encode** surface vs what this port has translated.

Answers ONE question: which C functions have no Rust counterpart yet. It does
that by NAME, which is a heuristic and is stated as one — the port deliberately
names its functions after C's, so a name hit is good evidence of a port and a
name miss is good evidence of a gap, but neither is proof. Treat the output as
a WORK QUEUE, not a coverage claim.

    tools/c_surface_inventory.py [--tsv out.tsv] [--all] [--scope NAME]

Adapted 2026-08-31 from the sibling zenav1-svt tool of the same name.

WHAT IT COUNTS
  Every function DEFINITION at column 0 in the scoped `.c` files under
  `upstream/av1/{encoder,common}`.  The regex is deliberately conservative: it
  only sees definitions whose ENTIRE signature is on one line and which start at
  column 0.  libaom wraps many long signatures across lines and hides others
  behind macros, so **every total printed here is a LOWER BOUND on the real
  surface.**  A file reported "0 gap" is not proven complete.

WHAT `ported` MEANS
  A Rust `fn` of that name (or of the name with a leading `av1_` / `aom_`
  stripped) is DEFINED in the port's own source — excluding `crates/aom-sys-ref`
  (the oracle: its wrappers and `extern` blocks name every C function they
  drive), `tests/`, `benches/` and `examples/` (a differential names the C
  function it compares against), and every `.c` / `.h` file (the shims are the
  oracle too).

  This still MISSES a port that renamed the function — several here did, e.g.
  `backup_stats` is `CompTypeCosts::backup` — so a MISSING row is a work item
  to triage, not a proven absence.  It no longer counts a DOC COMMENT: before
  2026-08-31 this matched the C name anywhere in the concatenated tree, so a
  module doc that listed a function as *not* ported was scored as a port, and
  a file with 9 known gaps reported 34/34.

WHAT `sym` MEANS  (the column the SVT tool did not have)
  Whether an exported, linkable symbol of that name exists in the built oracle
  archive `upstream/build/libaom.a`.  `Y` means a **tier-1 differential against
  the real C function is possible** through `crates/aom-sys-ref`; `n` means the
  function is `static` (or inlined away) and any oracle for it must be a shim
  that calls its caller, or hand-derived vectors labelled tier 4.  This column
  is measured with `nm -g`, not guessed from headers.

SCOPES
  `inter-encode` (default) — the C files the inter-frame encoder walks:
  motion search, inter RD, inter predictor build, reference/GOP management,
  rate control, and the inter arms of the shared partition/tx/bitstream code.
  `--all` inventories every `.c` under av1/encoder + av1/common instead.
"""
import os, re, subprocess, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSRC = os.path.join(REPO, "upstream", "av1")
RSRC = [os.path.join(REPO, "crates")]
LIBAOM_A = os.path.join(REPO, "upstream", "build", "libaom.a")

# The inter-ENCODE scope.  Each entry is <subdir>/<file.c> under upstream/av1/.
# Chosen from INTER-ENCODE-ROADMAP.md §2's C-path gap map plus the call trees it
# names.  Files that are purely intra/still are excluded even when the encoder
# links them (pickcdef.c, pickrst.c, palette.c, intra_mode_search.c, ...): the
# port already holds those byte-exact on the ALLINTRA track.
INTER_ENCODE = [
    # --- motion estimation -------------------------------------------------
    "encoder/mcomp.c",                 # full-pel + subpel search
    "encoder/motion_search_facade.c",  # single/joint/compound motion search
    "encoder/mv_prec.c",               # mv precision selection
    "encoder/encodemv.c",              # MV entropy write + cost tables
    "encoder/hash_motion.c",           # intrabc / hash-me source hash
    # --- inter mode decision ----------------------------------------------
    "encoder/rdopt.c",                 # av1_rd_pick_inter_mode_sb, handle_inter_mode
    "encoder/interp_search.c",         # interp-filter RD
    "encoder/compound_type.c",         # compound-type RD (wedge/diffwtd)
    "encoder/wedge_utils.c",           # wedge sse/sign kernels
    "encoder/nonrd_pickmode.c",        # speeds 8/9 inter pickmode
    "encoder/nonrd_opt.c",
    "encoder/var_based_part.c",        # variance-based partition (RT)
    # --- inter predictor build (encoder side) ------------------------------
    "encoder/reconinter_enc.c",
    "common/reconinter.c",
    "common/warped_motion.c",
    "common/convolve.c",
    "common/scale.c",
    # --- MV reference / prediction contexts --------------------------------
    "common/mvref_common.c",
    "common/pred_common.c",
    # --- global motion -----------------------------------------------------
    "encoder/global_motion.c",
    "encoder/global_motion_facade.c",
    # --- transform / coeff arms the inter path shares ----------------------
    "encoder/tx_search.c",             # var-tx recursion lives here
    "encoder/encodemb.c",
    "encoder/txb_rdopt.c",
    "encoder/encodetxb.c",
    "encoder/tokenize.c",
    # --- partition / frame walk -------------------------------------------
    "encoder/partition_search.c",
    "encoder/partition_strategy.c",
    "encoder/encodeframe.c",
    "encoder/encodeframe_utils.c",
    "encoder/context_tree.c",
    "encoder/rd.c",
    "encoder/segmentation.c",
    # --- frame/GOP/reference management + rate control ---------------------
    "encoder/encode_strategy.c",
    "encoder/gop_structure.c",
    "encoder/ratectrl.c",
    "encoder/firstpass.c",
    "encoder/pass2_strategy.c",
    "encoder/temporal_filter.c",
    "encoder/tpl_model.c",
    "encoder/lookahead.c",
    "encoder/encoder_utils.c",
    # --- bitstream ---------------------------------------------------------
    "encoder/bitstream.c",
]

# A C function definition at column 0: <type...> name(args) {  — deliberately
# conservative; it misses macro-generated and multi-line-signature functions,
# which is why every total is a LOWER BOUND on the surface.
DEF = re.compile(r'^(?:static\s+)?(?:const\s+)?[A-Za-z_][\w \*]*?\b([a-z_][a-z0-9_]*)\s*\([^;]*?\)\s*\{', re.M)
KEYWORDS = {"if", "for", "while", "switch", "return", "sizeof", "else", "do"}


def scope_files(all_files):
    if all_files:
        out = []
        for sub in ("encoder", "common"):
            d = os.path.join(CSRC, sub)
            for fn in sorted(os.listdir(d)):
                if fn.endswith(".c"):
                    out.append(f"{sub}/{fn}")
        return out
    return INTER_ENCODE


def c_functions(files):
    out = {}
    for rel in files:
        path = os.path.join(CSRC, rel)
        if not os.path.isfile(path):
            print(f"warning: scope file missing: {rel}", file=sys.stderr)
            continue
        text = open(path, errors="ignore").read()
        for m in DEF.finditer(text):
            name = m.group(1)
            if name in KEYWORDS:
                continue
            out.setdefault(rel, []).append(name)
    return out


# A Rust function DEFINITION.  Matching this rather than the bare name is the
# whole difference between "the port has this function" and "some file in the
# tree mentions it".
RUST_FN_RE = re.compile(r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)")


def rust_fn_names():
    """Every `fn <name>` defined in the port's own Rust source.

    THREE exclusions, each of which was over-counting:

    * **`.c` and `.h` files** — `crates/aom-sys-ref/shim/*.c` is the ORACLE.
      A shim that wraps `foo` so a differential can call it names `foo`; that
      is evidence the function is testABLE, not that it is ported.
    * **`crates/aom-sys-ref/`** — same reason for its Rust side: every
      `ref_*` wrapper and `extern "C"` block names the C function it drives.
    * **`tests/`, `benches/` and `examples/`** — a differential names the C
      function it compares against, in prose and in the test's own name.

    Substring matching over the concatenated tree (what this tool did before)
    also credited a DOC COMMENT.  Measured 2026-08-31: after a landing whose
    module docs listed every function in `compound_type.c` — 25 ported and 9
    explicitly named as NOT ported — the tool reported that file as 34/34
    matched, i.e. it read "MISSING: av1_compound_type_rd" as a port of
    `av1_compound_type_rd`.
    """
    names = set()
    for base in RSRC:
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [
                d
                for d in dirnames
                if d not in ("target", ".git", "vendor", "tests", "benches", "examples")
            ]
            if os.sep + "aom-sys-ref" in dirpath + os.sep:
                continue
            for f in filenames:
                if not f.endswith(".rs"):
                    continue
                text = open(os.path.join(dirpath, f), errors="ignore").read()
                names.update(RUST_FN_RE.findall(text))
    return names


def exported_symbols():
    """Names with a linkable T/D symbol in the built oracle archive.

    Measured, not guessed: `nm -g` on upstream/build/libaom.a.  Mach-O prefixes
    every C symbol with `_`, ELF does not; strip one leading underscore only
    when the un-stripped name is not itself a C identifier we expect.
    """
    if not os.path.isfile(LIBAOM_A):
        print(f"warning: {LIBAOM_A} absent — `sym` column will read '?' for every row.\n"
              f"         build it with `cargo build -p zenav1-aom-sys-ref`.", file=sys.stderr)
        return None
    try:
        raw = subprocess.run(["nm", "-g", LIBAOM_A], capture_output=True, text=True).stdout
    except FileNotFoundError:
        print("warning: `nm` not on PATH — `sym` column will read '?'.", file=sys.stderr)
        return None
    syms = set()
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        kind, name = parts[-2], parts[-1]
        if kind not in ("T", "t", "D", "S", "B"):
            continue
        syms.add(name)
        if name.startswith("_"):
            syms.add(name[1:])
    return syms


def main():
    all_files = "--all" in sys.argv
    files = scope_files(all_files)
    cfns = c_functions(files)
    fn_names = rust_fn_names()
    syms = exported_symbols()

    rows = []
    for path, names in cfns.items():
        for n in sorted(set(names)):
            # `av1_` / `aom_` prefixes and the `_c` reference-implementation
            # suffix are libaom's naming, not the port's: `aom_upsampled_pred_c`
            # is ported as `upsampled_pred`. Strip them in every combination.
            stems = {n}
            for p in ("av1_", "aom_"):
                if n.startswith(p):
                    stems.add(n[len(p):])
            for stem in list(stems):
                if stem.endswith("_c"):
                    stems.add(stem[:-2])
            hit = any(s in fn_names for s in stems)
            if syms is None:
                sym = "?"
            else:
                sym = "Y" if n in syms else "n"
            rows.append((path, n, "ported" if hit else "MISSING", sym))

    per_file = {}
    for path, n, st, sym in rows:
        d = per_file.setdefault(path, [0, 0, []])
        d[0] += 1
        if st == "ported":
            d[1] += 1
        else:
            d[2].append(n)

    total = len(rows)
    ported = sum(1 for r in rows if r[2] == "ported")
    linkable_gap = sum(1 for r in rows if r[2] == "MISSING" and r[3] == "Y")
    scope = "ALL av1/{encoder,common}" if all_files else "inter-encode"
    print(f"scope: {scope}   ({len(files)} C files)")
    print(f"C functions found: {total}   name-matched in the Rust tree: {ported}"
          f"   no match: {total - ported}   ({100.0 * ported / total:.1f}% matched)")
    print(f"of the {total - ported} unmatched, {linkable_gap} have an exported symbol in "
          f"libaom.a (tier-1 differential possible)")
    print("NAME-matched, and the regex only sees single-line column-0 definitions: "
          "these totals are a LOWER BOUND and a work queue, not a coverage claim.")
    print()
    print(f"{'file':40} {'total':>6} {'matched':>8} {'gap':>5}")
    for path in sorted(per_file, key=lambda p: -(per_file[p][0] - per_file[p][1])):
        t, p, miss = per_file[path]
        if t - p == 0:
            continue
        print(f"{path:40} {t:>6} {p:>8} {t - p:>5}")

    if "--tsv" in sys.argv:
        out = sys.argv[sys.argv.index("--tsv") + 1]
        with open(out, "w") as fh:
            fh.write("file\tfunction\tstatus\tsym\n")
            for path, n, st, sym in rows:
                fh.write(f"{path}\t{n}\t{st}\t{sym}\n")
        print(f"\nwrote {out}")


main()
