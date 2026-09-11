//! Every crate this workspace would publish must be publishable, and the set of
//! them must be a DELIBERATE list.
//!
//! **A crates.io name is permanent — a published crate cannot be deleted.** So
//! the expensive mistake is not a broken manifest (that fails loudly at
//! `cargo publish` and costs a retry); it is publishing a name nobody meant to
//! own, or publishing a surface nobody reviewed. Both are unrecoverable. This
//! scan therefore pins the publishable SET by name alongside the mechanical
//! manifest checks, so adding a crate to crates.io takes an edit here.
//!
//! Why a metadata scan and not `cargo publish --dry-run`: a first publish is
//! bottom-up, and cargo resolves a downstream crate's version requirements
//! against the real index, so `--dry-run` on `zenav1-aom-encode` cannot succeed
//! until `zenav1-aom-dsp` is actually on crates.io ("no matching package named
//! `zenav1-aom-dsp` found"). That failure is inherent, not a defect, so it
//! cannot be a gate. What CAN be gated offline is everything cargo checks
//! BEFORE it touches the index — which is where every real defect this scan was
//! written for lived.
//!
//! Measured when it was written (2026-09-10): three of the four publishable
//! crates could not have been published at all — `zenav1-aom`,
//! `zenav1-aom-encode` and `zenav1-aom-decode` carried intra-workspace path
//! dependencies with no version requirement, which cargo rejects with "all
//! dependencies must have a version requirement specified when publishing", and
//! `zenav1-aom-encode` had no `repository` field.

use std::collections::BTreeSet;
use std::process::Command;

/// The crates this workspace publishes. **Pinned by name on purpose.** A name on
/// crates.io cannot be given back, so a crate joining this list is a decision,
/// not a side effect of dropping `publish = false`.
///
/// **MEASURED 2026-09-10, and the window is still open: NONE of these four are
/// on crates.io yet** (queried directly; all four return 404 while `zenavif`
/// returns 200). So the published set is still fully revisable — and it stops
/// being revisable at the first `cargo publish`.
///
/// **The one open question, stated here because this is where a publisher will
/// look: `zenav1-aom` (the facade) has NO CONSUMER.** zenavif — the only
/// external consumer — depends on `zenav1-aom-decode` and `zenav1-aom-encode`
/// DIRECTLY by git rev, and `zenav1_aom::` appears nowhere in its source. The
/// argument each way is real and this list deliberately does not settle it:
///   * DON'T publish it — publishing is irreversible and a crate with no
///     consumer is the exact mistake this pin exists to prevent; not publishing
///     is reversible, publishing is not, so the asymmetry favours waiting.
///   * DO publish it — `zenav1-aom` is the PRIMARY name. Publishing the three
///     `-dsp`/`-encode`/`-decode` crates while leaving the unsuffixed name free
///     lets someone else take the brand, which is also unrecoverable.
/// It is left in the list (the name-defence reading) because that is the status
/// quo, NOT because the question was resolved. Resolve it before publishing.
const PUBLISHED: &[&str] = &[
    "zenav1-aom",
    "zenav1-aom-decode",
    "zenav1-aom-dsp",
    "zenav1-aom-encode",
];

/// crates.io refuses an upload without these, and each is load-bearing for a
/// consumer: `description` and `license` are shown on the crate page and are
/// what a licence audit reads, `repository` is the only link back to the source
/// a published tarball has.
const REQUIRED_FIELDS: &[&str] = &["description", "license", "repository"];

fn metadata() -> serde_json::Value {
    let out = Command::new(env!("CARGO"))
        .args([
            "metadata",
            "--no-deps",
            "--format-version",
            "1",
            "--manifest-path",
            "../Cargo.toml",
        ])
        .output()
        .expect("run cargo metadata on the parent workspace");
    assert!(
        out.status.success(),
        "cargo metadata failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).expect("parse cargo metadata")
}

/// `publish = false` in a manifest arrives as `"publish": []`; anything else
/// (including absent) means the crate would go to crates.io.
fn is_published(pkg: &serde_json::Value) -> bool {
    match pkg.get("publish") {
        Some(serde_json::Value::Array(a)) => !a.is_empty(),
        _ => true,
    }
}

#[test]
fn the_published_set_is_exactly_the_pinned_list() {
    let md = metadata();
    let actual: BTreeSet<String> = md["packages"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|p| is_published(p))
        .map(|p| p["name"].as_str().unwrap().to_string())
        .collect();
    let pinned: BTreeSet<String> = PUBLISHED.iter().map(|s| s.to_string()).collect();

    let added: Vec<_> = actual.difference(&pinned).collect();
    let removed: Vec<_> = pinned.difference(&actual).collect();
    assert!(
        added.is_empty(),
        "these crates would be published to crates.io but are not in PUBLISHED: {added:?}\n\
         A crates.io name is PERMANENT and cannot be deleted. If publishing them is intended, \
         add them to PUBLISHED in apidoc/tests/crates_io_publishable.rs; otherwise set \
         `publish = false` in the manifest."
    );
    assert!(
        removed.is_empty(),
        "these crates are pinned as published but their manifests now say `publish = false`: \
         {removed:?}\n\
         If a crate is already ON crates.io, marking it unpublishable here does not remove it — \
         the name stays taken. Update PUBLISHED deliberately."
    );
}

#[test]
fn every_published_crate_carries_the_metadata_crates_io_requires() {
    let md = metadata();
    let mut problems = Vec::new();
    for pkg in md["packages"].as_array().unwrap() {
        if !is_published(pkg) {
            continue;
        }
        let name = pkg["name"].as_str().unwrap();
        for field in REQUIRED_FIELDS {
            let ok = pkg
                .get(*field)
                .and_then(|v| v.as_str())
                .is_some_and(|s| !s.trim().is_empty());
            if !ok {
                problems.push(format!("{name}: missing `{field}`"));
            }
        }
    }
    assert!(
        problems.is_empty(),
        "crates.io requires these and the upload is rejected without them:\n  {}",
        problems.join("\n  ")
    );
}

/// The defect this file was written for. cargo strips `path` from a published
/// manifest, so a path dependency with no `version` leaves the uploaded crate
/// pointing at nothing — cargo refuses it up front with "all dependencies must
/// have a version requirement specified when publishing".
///
/// Dev-dependencies are deliberately EXEMPT: cargo drops a dev-dependency that
/// carries no version from the published manifest, which is what lets the
/// publishable crates dev-depend on `zenav1-aom-sys-ref` (a `publish = false`
/// crate whose build.rs needs the libaom submodule and cmake). Giving one of
/// those a version would be the bug — see the test below.
#[test]
fn published_crates_pin_a_version_on_every_intra_workspace_dependency() {
    let md = metadata();
    let versions: std::collections::BTreeMap<&str, &str> = md["packages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| {
            (
                p["name"].as_str().unwrap(),
                p["version"].as_str().unwrap(),
            )
        })
        .collect();

    let mut problems = Vec::new();
    for pkg in md["packages"].as_array().unwrap() {
        if !is_published(pkg) {
            continue;
        }
        let name = pkg["name"].as_str().unwrap();
        for dep in pkg["dependencies"].as_array().unwrap() {
            if dep.get("path").is_none() {
                continue;
            }
            let dname = dep["name"].as_str().unwrap();
            // "dev" | "build" | null(normal)
            let kind = dep.get("kind").and_then(|k| k.as_str());
            let req = dep["req"].as_str().unwrap();
            if kind == Some("dev") {
                continue;
            }
            if req == "*" {
                problems.push(format!(
                    "{name}: path dependency `{dname}` ({}) has no version requirement — \
                     cargo rejects the upload",
                    kind.unwrap_or("normal")
                ));
                continue;
            }
            // A hardcoded requirement drifts silently when the member's own
            // version bumps, and the mismatch only surfaces at publish time.
            if let Some(actual) = versions.get(dname) {
                let bare = req.trim_start_matches('^');
                if bare != *actual {
                    problems.push(format!(
                        "{name}: requires `{dname}` {req} but that crate is at {actual} in this \
                         workspace — bump the requirement with the version"
                    ));
                }
            }
        }
    }
    assert!(
        problems.is_empty(),
        "these manifests cannot be published as they stand:\n  {}",
        problems.join("\n  ")
    );
}

/// The mirror of the exemption above, and the reason it is safe. A
/// dev-dependency on a `publish = false` crate must carry NO version: with one,
/// cargo keeps the dependency in the published manifest and then tries to
/// resolve a crate that is not on the index and never will be.
#[test]
fn dev_dependencies_on_unpublished_crates_carry_no_version() {
    let md = metadata();
    let unpublished: BTreeSet<&str> = md["packages"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|p| !is_published(p))
        .map(|p| p["name"].as_str().unwrap())
        .collect();

    let mut problems = Vec::new();
    let mut reached = 0usize;
    for pkg in md["packages"].as_array().unwrap() {
        if !is_published(pkg) {
            continue;
        }
        let name = pkg["name"].as_str().unwrap();
        for dep in pkg["dependencies"].as_array().unwrap() {
            let dname = dep["name"].as_str().unwrap();
            if dep.get("kind").and_then(|k| k.as_str()) != Some("dev")
                || !unpublished.contains(dname)
            {
                continue;
            }
            reached += 1;
            if dep["req"].as_str().unwrap() != "*" {
                problems.push(format!(
                    "{name}: dev-dependency `{dname}` is `publish = false` but carries a version \
                     requirement — the published manifest would reference a crate that is not on \
                     crates.io"
                ));
            }
        }
    }
    assert!(problems.is_empty(), "{}", problems.join("\n  "));
    // Non-vacuity: this test asserts a property of a set, so an empty set makes
    // it pass while checking nothing. `zenav1-aom-sys-ref` is dev-depended on by
    // three of the four published crates.
    assert!(
        reached >= 1,
        "no published crate dev-depends on an unpublished one, so this test checked nothing — \
         if that is now genuinely true, delete it rather than leaving it vacuous"
    );
}

/// Ties the two halves of `just api-doc-check` together: a crate we would put on
/// crates.io permanently must have its public surface committed, so the review
/// that precedes an irreversible publish has something to read.
#[test]
fn every_published_crate_has_a_committed_public_api_snapshot() {
    let md = metadata();
    let mut problems = Vec::new();
    for pkg in md["packages"].as_array().unwrap() {
        if !is_published(pkg) {
            continue;
        }
        let name = pkg["name"].as_str().unwrap();
        let path = format!("../docs/public-api/{name}.txt");
        match std::fs::read_to_string(&path) {
            Ok(s) if !s.trim().is_empty() => {}
            Ok(_) => problems.push(format!("{name}: {path} is empty")),
            Err(e) => problems.push(format!("{name}: {path} is missing ({e})")),
        }
    }
    assert!(
        problems.is_empty(),
        "run `just api-doc` to generate the missing snapshots:\n  {}",
        problems.join("\n  ")
    );
}
