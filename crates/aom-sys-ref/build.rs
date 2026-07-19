//! Cargo-driven build of the pinned C libaom oracle that aom-rs measures
//! bit-exactness against ("fresh-box easy mode", repo-reorg #3 Phase 1).
//!
//! A plain `cargo test` builds everything with no manual step:
//!   1. verify the C toolchain (cmake / nasm / a C compiler) is present, else
//!      panic with the one-line install — not a cryptic linker error;
//!   2. auto-init the `upstream/` git submodule (pinned libaom v3.14.1) if empty;
//!   3. build libaom ONCE via cmake in the deterministic single-thread oracle
//!      config (see reference/BUILD_CONFIG.md), cached by the submodule SHA so it
//!      never rebuilds on an unchanged tree;
//!   4. compile the aom-sys-ref shims against the libaom source + generated config;
//!   5. link the static archives.
//!
//! aom-sys-ref is a dev-dependency only, so this build.rs never runs for a
//! downstream consumer of a published crate (invariant A: dependency usage is
//! zero-C).

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Pinned libaom oracle commit — must match `.gitmodules` / reference/BUILD_CONFIG.md.
/// Used only as a fallback stamp key if `git` can't report the checked-out SHA.
const PINNED_SHA: &str = "03087864cf4bea6abb0d28f95cf7843511413d8f";

/// The oracle shims (aom-sys-ref/shim/*.c), compiled into libaom_shim.a.
const SHIMS: &[&str] = &[
    "entropy_shim",
    "intra_shim",
    "sadvar_shim",
    "convolve_shim",
    "cdef_shim",
    "highbd_intra_shim",
    "hbd_lpf_shim",
    "hbd_sadvar_shim",
    "txb_shim",
    "intra_edge_shim",
    "quant_fp_shim",
    "qm_shim",
    "wb_shim",
    "modeinfo_shim",
    "avail_shim",
    "rd_shim",
    "hog_shim",
    "dec_shim",
    "pickrst_shim",
    "superres_shim",
    "prune_tx_shim",
    "inter_shim",
    "warp_shim",
    "obmc_shim",
    "me_shim",
    "interintra_shim",
];

fn main() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // Workspace root = two levels up from crates/aom-sys-ref.
    let workspace_root = manifest
        .join("../..")
        .canonicalize()
        .expect("cannot resolve workspace root from CARGO_MANIFEST_DIR");
    let upstream = workspace_root.join("upstream");

    // 1. Toolchain first — fail fast with a friendly message BEFORE any slow work.
    check_toolchain();

    // 2. Ensure the pinned libaom source is present (auto-init the submodule).
    ensure_submodule(&workspace_root, &upstream);

    // 3. Build libaom once (SHA-keyed cache), producing upstream/build/{libaom.a,...}.
    let build_dir = build_libaom(&upstream);

    // 4. Compile the shims against the libaom source + its generated config headers.
    let shim_lib_dir = compile_shims(&manifest, &upstream, &build_dir);

    // 5. Link: shims first, then libaom, then the C++/math/pthread runtimes.
    println!("cargo:rustc-link-search=native={}", shim_lib_dir.display());
    println!("cargo:rustc-link-lib=static=aom_shim");
    println!("cargo:rustc-link-search=native={}", build_dir.display());
    println!("cargo:rustc-link-lib=static=aom");
    // libaom is C, but the archive is linked by CXX; pull in libstdc++ + libm
    // + pthread in case any TU needs them.
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=m");
    println!("cargo:rustc-link-lib=dylib=pthread");

    // Re-run when the pinned submodule commit changes (best-effort).
    if let Some(head) = submodule_head_file(&upstream) {
        println!("cargo:rerun-if-changed={}", head.display());
    }
    println!(
        "cargo:rerun-if-changed={}",
        build_dir.join("libaom.a").display()
    );
}

/// Verify the C-oracle build tools exist. Panic with the one-line install if not,
/// rather than letting cmake/the linker fail cryptically later.
fn check_toolchain() {
    let cmake = tool_ok("cmake", "--version");
    let nasm = tool_ok("nasm", "--version");
    let cc = pick_c_compiler().is_some();

    if cmake && nasm && cc {
        return;
    }
    let mut missing = Vec::new();
    if !cmake {
        missing.push("cmake");
    }
    if !nasm {
        missing.push("nasm");
    }
    if !cc {
        missing.push("a C compiler (build-essential / clang)");
    }
    panic!(
        "aom-sys-ref: missing C-oracle build tool(s): {}.\n\
         The pinned libaom oracle is built from source (cmake + nasm + a C \
         compiler). Install with:\n\
         \n    sudo apt-get install cmake nasm build-essential\n\n\
         (see reference/BUILD_CONFIG.md for the oracle build config).",
        missing.join(", ")
    );
}

/// True if `<cmd> <arg>` runs and exits 0 (output suppressed).
fn tool_ok(cmd: &str, arg: &str) -> bool {
    Command::new(cmd)
        .arg(arg)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Pick a C compiler for the shims: prefer clang (what the shims were developed
/// against), else the platform `cc`, else gcc.
fn pick_c_compiler() -> Option<&'static str> {
    for cc in ["clang", "cc", "gcc"] {
        if tool_ok(cc, "--version") {
            return Some(cc);
        }
    }
    None
}

/// Ensure the `upstream/` libaom submodule is checked out. Auto-init if empty;
/// otherwise fail with the exact command to run.
fn ensure_submodule(workspace_root: &Path, upstream: &Path) {
    // libaom's source root has a CMakeLists.txt — use it as the "present" sentinel.
    let sentinel = upstream.join("CMakeLists.txt");
    if sentinel.exists() {
        return;
    }

    eprintln!(
        "aom-sys-ref: `upstream/` submodule is empty — running \
         `git submodule update --init upstream` ..."
    );
    let ran = Command::new("git")
        .args(["submodule", "update", "--init", "upstream"])
        .current_dir(workspace_root)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);

    if !ran || !sentinel.exists() {
        panic!(
            "aom-sys-ref: the pinned libaom C oracle submodule at `upstream/` is \
             not checked out, and auto-init failed. Run:\n\
             \n    git submodule update --init upstream\n\n\
             (or clone the repo with `git clone --recurse-submodules ...`)."
        );
    }
}

/// Build libaom once, cached on the checked-out submodule SHA. Returns the build
/// directory (upstream/build) that holds libaom.a + the generated config headers.
fn build_libaom(upstream: &Path) -> PathBuf {
    let build_dir = upstream.join("build");
    let lib = build_dir.join("libaom.a");
    let stamp = build_dir.join(".aom-oracle-sha");
    let sha = current_sha(upstream);

    // Cache (invariant C, the dominant cost): skip the minutes-long cmake build
    // when libaom.a already exists for the current submodule SHA.
    if lib.exists() {
        if let Ok(prev) = std::fs::read_to_string(&stamp) {
            if prev.trim() == sha {
                return build_dir;
            }
        }
    }

    std::fs::create_dir_all(&build_dir).expect("cannot create the oracle build dir");

    // Configure — the deterministic single-thread oracle config (BUILD_CONFIG.md).
    // CONFIG_MULTITHREAD=0 => deterministic encoder output target; this is the
    // definition against which aom-rs bit-exactness is measured. DO NOT change
    // these flags without updating BUILD_CONFIG.md and the CI cache salt.
    let configure = Command::new("cmake")
        .arg("-S")
        .arg(upstream)
        .arg("-B")
        .arg(&build_dir)
        .args([
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCONFIG_MULTITHREAD=0",
            "-DENABLE_TESTS=1",
            "-DENABLE_EXAMPLES=1",
            "-DENABLE_TOOLS=1",
            "-DCONFIG_AV1_DECODER=1",
            "-DCONFIG_AV1_ENCODER=1",
        ])
        .status()
        .expect("failed to spawn cmake (configure) for the libaom oracle");
    assert!(
        configure.success(),
        "cmake configure of the libaom oracle failed"
    );

    // Build the library + the aomenc/aomdec tools (used by the conformance +
    // coverage xtasks). One build, reused by every test binary.
    let jobs = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .to_string();
    let build = Command::new("cmake")
        .arg("--build")
        .arg(&build_dir)
        .arg("--target")
        .arg("aom")
        .arg("aomenc")
        .arg("aomdec")
        .arg("-j")
        .arg(&jobs)
        .status()
        .expect("failed to spawn cmake (build) for the libaom oracle");
    assert!(build.success(), "cmake build of the libaom oracle failed");

    assert!(
        lib.exists(),
        "libaom.a missing after the oracle build: {}",
        lib.display()
    );
    // Stamp the build so unchanged trees skip the rebuild next time.
    std::fs::write(&stamp, &sha).ok();
    build_dir
}

/// The checked-out submodule commit SHA, or the pinned constant if git can't tell.
fn current_sha(upstream: &Path) -> String {
    Command::new("git")
        .arg("-C")
        .arg(upstream)
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| PINNED_SHA.to_string())
}

/// Absolute path to the submodule's `HEAD` file (for rerun tracking), if resolvable.
fn submodule_head_file(upstream: &Path) -> Option<PathBuf> {
    let out = Command::new("git")
        .arg("-C")
        .arg(upstream)
        .args(["rev-parse", "--absolute-git-dir"])
        .output()
        .ok()
        .filter(|o| o.status.success())?;
    let git_dir = String::from_utf8(out.stdout).ok()?;
    let head = PathBuf::from(git_dir.trim()).join("HEAD");
    head.exists().then_some(head)
}

/// Compile the oracle shims into libaom_shim.a against the libaom source and its
/// cmake-generated config headers. Returns the directory holding the archive.
fn compile_shims(manifest: &Path, upstream: &Path, build_dir: &Path) -> PathBuf {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let shim_dir = manifest.join("shim");
    let cc = pick_c_compiler().expect("no C compiler for the shims (checked earlier)");

    let mut objs = Vec::new();
    for name in SHIMS {
        let shim_c = shim_dir.join(format!("{name}.c"));
        let obj = out_dir.join(format!("{name}.o"));
        let status = Command::new(cc)
            .args(["-O2", "-c"])
            .arg(&shim_c)
            .arg("-o")
            .arg(&obj)
            .arg(format!("-I{}", upstream.display()))
            .arg(format!("-I{}", build_dir.display()))
            .status()
            .unwrap_or_else(|e| panic!("failed to spawn {cc} for shim {name}: {e}"));
        assert!(status.success(), "{name} shim compile failed");
        println!("cargo:rerun-if-changed={}", shim_c.display());
        objs.push(obj);
    }

    let lib = out_dir.join("libaom_shim.a");
    let _ = std::fs::remove_file(&lib); // ar `crus` appends; start clean.
    let mut ar = Command::new("ar");
    ar.arg("crus").arg(&lib);
    for o in &objs {
        ar.arg(o);
    }
    assert!(
        ar.status().expect("failed to spawn ar").success(),
        "ar failed to build libaom_shim.a"
    );
    out_dir
}
