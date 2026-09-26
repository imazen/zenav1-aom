//! Runtime diagnostic traces — the divergence-hunt prints, behind a runtime API.
//!
//! Every `[pk]`, `[dp]`, `[txt]`, `[ssm-port]` ... line the encoder and decoder
//! can print is a byte-inert aid that pairs with an identically named
//! `getenv("AOM_*")` in the instrumented C oracle
//! (`docs/upstream-instrumentation/README.md` is the inventory). Until
//! 2026-09-24 each site read its own environment variable, which meant a
//! LIBRARY read the host's environment and wrote to the host's stderr on a
//! switch the host never declared, read once per process, discoverable only by
//! grepping. This module is the replacement:
//!
//! * **One process-wide switch set: [`install`] / [`clear`].** A bitmask of
//!   [`Trace`] kinds, three optional [`Focus`] nodes (`(mi_row, mi_col)` for
//!   the per-node RD dumps), an optional header-dump path, and an optional
//!   output [`Sink`]. Atomics throughout; a site pays one relaxed load, which is
//!   what the previous `OnceLock` deref cost.
//! * **The library never reads the environment.** Hosts call [`install`].
//!   The harness front door is [`install_from_env`], which maps the documented
//!   `AOM_*` names onto this API so every script, KB body and C pairing keeps
//!   working; the `trace-env` cargo feature (default OFF, on for the in-tree
//!   harness) makes that seeding lazy and automatic so `AOM_TX_DBG=.. cargo
//!   test` still behaves. A published crate built with default features has no
//!   `getenv` on any trace path.
//! * **Output goes to the installed sink**, default stderr, so a host can
//!   capture or silence it. `NAME=0` and `NAME=` mean OFF in the env mapping,
//!   the convention `AOM_FORCE_SCALAR` already uses.
//!
//! The dispatch pin (`AOM_FORCE_SCALAR`, `crate::dispatch::scalar_forced`) is
//! deliberately NOT here: its tests assert it holds for the life of the process,
//! so it keeps set-once semantics.

use std::fmt;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::Relaxed};
use std::sync::{OnceLock, RwLock};

/// A trace kind. Each maps to one documented `AOM_*` environment name
/// ([`Trace::env_name`]) and to the identically gated print in the C oracle.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Trace {
    /// `AOM_SYM_TRACE` — decoder: sequence, value, alphabet per read symbol.
    Sym = 0,
    /// `AOM_WSYM_TRACE` — encoder twin of [`Trace::Sym`].
    WSym,
    /// `AOM_HDR_TRACE` — header field boundaries (`[hdr] name @bit`).
    Hdr,
    /// `AOM_CDEF_DBG` — CDEF strength search + header fields.
    Cdef,
    /// `AOM_PAL_TRACE` — decoder palette cache / colours.
    Pal,
    /// `AOM_DQ_DEC` — decoder delta-q reads.
    DqDec,
    /// `AOM_DBG_BLOCKS` — decoder per-block position / mode / `tell_frac`.
    DbgBlocks,
    /// `AOM_DV_TRACE` — decoder IntraBC DV validation.
    Dv,
    /// `AOM_PART_TRACE` — decoder partition symbols.
    Part,
    /// `AOM_SSM_DBG` — per-node rdmult fold (`[ssm-port]`).
    Ssm,
    /// `AOM_TRELLIS_CALLS` — trellis call/eval counters.
    TrellisCalls,
    /// `AOM_P4_NOBUDGET` — run 4-way strip searches under an unlimited budget.
    P4NoBudget,
    /// `AOM_PACK_TRACE` — pack-side per-leaf mode info (`[pk]`).
    Pack,
    /// `AOM_IBC_COEFF_TRACE` — IntraBC coefficient arm (`[ibc-coeff]`).
    IbcCoeff,
    /// `AOM_CTXB_TRACE` — written txb contexts (`[wtxb]`).
    Ctxb,
    /// `IBC_TRACE` — IntraBC candidate / mode-rate prints (`[r-mv]`, `[r-arms]`).
    Ibc,
    /// `AOM_IBC_WIN` — IntraBC winner updates (`[ibc-win]`).
    IbcWin,
    /// `AOM_SCT_TRIAL_DBG` — screen-content trial decision (`[sct-trial]`).
    SctTrial,
    /// `AOM_SCT_DBG` — screen-content decision + derived loop-filter (`[sct-port]`).
    Sct,
    /// `AOM_DQ_DBG` — encoder per-SB delta-q (`[dq-port]`).
    Dq,
    /// `AOM_TIME_PHASES` — per-phase wall timing of `encode_key_frame`.
    TimePhases,
}

impl Trace {
    /// Every kind, in bit order.
    pub const ALL: [Trace; 21] = [
        Trace::Sym,
        Trace::WSym,
        Trace::Hdr,
        Trace::Cdef,
        Trace::Pal,
        Trace::DqDec,
        Trace::DbgBlocks,
        Trace::Dv,
        Trace::Part,
        Trace::Ssm,
        Trace::TrellisCalls,
        Trace::P4NoBudget,
        Trace::Pack,
        Trace::IbcCoeff,
        Trace::Ctxb,
        Trace::Ibc,
        Trace::IbcWin,
        Trace::SctTrial,
        Trace::Sct,
        Trace::Dq,
        Trace::TimePhases,
    ];

    /// The documented environment name this kind pairs with in the C oracle.
    pub const fn env_name(self) -> &'static str {
        match self {
            Trace::Sym => "AOM_SYM_TRACE",
            Trace::WSym => "AOM_WSYM_TRACE",
            Trace::Hdr => "AOM_HDR_TRACE",
            Trace::Cdef => "AOM_CDEF_DBG",
            Trace::Pal => "AOM_PAL_TRACE",
            Trace::DqDec => "AOM_DQ_DEC",
            Trace::DbgBlocks => "AOM_DBG_BLOCKS",
            Trace::Dv => "AOM_DV_TRACE",
            Trace::Part => "AOM_PART_TRACE",
            Trace::Ssm => "AOM_SSM_DBG",
            Trace::TrellisCalls => "AOM_TRELLIS_CALLS",
            Trace::P4NoBudget => "AOM_P4_NOBUDGET",
            Trace::Pack => "AOM_PACK_TRACE",
            Trace::IbcCoeff => "AOM_IBC_COEFF_TRACE",
            Trace::Ctxb => "AOM_CTXB_TRACE",
            Trace::Ibc => "IBC_TRACE",
            Trace::IbcWin => "AOM_IBC_WIN",
            Trace::SctTrial => "AOM_SCT_TRIAL_DBG",
            Trace::Sct => "AOM_SCT_DBG",
            Trace::Dq => "AOM_DQ_DBG",
            Trace::TimePhases => "AOM_TIME_PHASES",
        }
    }

    const fn bit(self) -> u32 {
        1 << (self as u32)
    }
}

/// A per-node focus for the RD dumps: the one `(mi_row, mi_col)` node whose
/// stage-by-stage costs are printed.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Focus {
    /// `AOM_TX_DBG=<r>,<c>` — per-tx-type / pixel-domain-distortion dumps.
    Tx = 0,
    /// `AOM_PART_DBG=<r>,<c>` — partition stage / strip dumps.
    Part,
    /// `AOM_UV_DBG=<r>,<c>` — chroma mode loop dumps.
    Uv,
}

impl Focus {
    /// Every focus slot.
    pub const ALL: [Focus; 3] = [Focus::Tx, Focus::Part, Focus::Uv];

    /// The documented environment name (value `<mi_row>,<mi_col>`).
    pub const fn env_name(self) -> &'static str {
        match self {
            Focus::Tx => "AOM_TX_DBG",
            Focus::Part => "AOM_PART_DBG",
            Focus::Uv => "AOM_UV_DBG",
        }
    }
}

/// Where trace lines go. Called with one complete line, no trailing newline.
pub type Sink = Box<dyn Fn(&str) + Send + Sync>;

/// What to install. Build with [`TraceConfig::new`] and the `with_*` methods.
#[non_exhaustive]
#[derive(Default)]
pub struct TraceConfig {
    flags: u32,
    focus: [Option<(i32, i32)>; 3],
    hdr_dump: Option<std::path::PathBuf>,
    sink: Option<Sink>,
}

impl fmt::Debug for TraceConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TraceConfig")
            .field("flags", &format_args!("{:#x}", self.flags))
            .field("focus", &self.focus)
            .field("hdr_dump", &self.hdr_dump)
            .field("sink", &self.sink.as_ref().map(|_| "custom"))
            .finish()
    }
}

impl TraceConfig {
    /// Nothing enabled, stderr sink.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable one trace kind.
    pub fn with(mut self, t: Trace) -> Self {
        self.flags |= t.bit();
        self
    }

    /// Set one focus node.
    pub fn with_focus(mut self, f: Focus, mi_row: i32, mi_col: i32) -> Self {
        self.focus[f as usize] = Some((mi_row, mi_col));
        self
    }

    /// Write the assembled frame-header parameter block (`{p:#?}`) to this path
    /// after each encode (`AOM_HDR_DUMP`).
    pub fn with_hdr_dump(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.hdr_dump = Some(path.into());
        self
    }

    /// Route every trace line here instead of stderr.
    pub fn with_sink(mut self, sink: Sink) -> Self {
        self.sink = Some(sink);
        self
    }

    /// `true` when at least one kind or focus is set.
    pub fn is_empty(&self) -> bool {
        self.flags == 0 && self.focus.iter().all(Option::is_none) && self.hdr_dump.is_none()
    }

    /// Read the documented `AOM_*` names from the process environment.
    /// `NAME=0` / `NAME=` are OFF. Never called by the library on its own.
    pub fn from_env() -> Self {
        let on = |name: &str| std::env::var_os(name).is_some_and(|v| !v.is_empty() && v != "0");
        let mut cfg = Self::new();
        for t in Trace::ALL {
            if on(t.env_name()) {
                cfg = cfg.with(t);
            }
        }
        for f in Focus::ALL {
            if let Some((r, c)) = std::env::var(f.env_name())
                .ok()
                .and_then(|v| parse_node(&v))
            {
                cfg = cfg.with_focus(f, r, c);
            }
        }
        if let Some(p) = std::env::var_os("AOM_HDR_DUMP").filter(|v| !v.is_empty()) {
            cfg = cfg.with_hdr_dump(p);
        }
        cfg
    }
}

fn parse_node(v: &str) -> Option<(i32, i32)> {
    let (r, c) = v.split_once(',')?;
    Some((r.trim().parse().ok()?, c.trim().parse().ok()?))
}

const FOCUS_NONE: u64 = u64::MAX;

fn pack_node(n: Option<(i32, i32)>) -> u64 {
    match n {
        // (r,c) both fit in 31 bits in practice (mi units); pack as two u32
        // lanes and reserve all-ones for None.
        Some((r, c)) => ((r as u32 as u64) << 32) | (c as u32 as u64),
        None => FOCUS_NONE,
    }
}

fn unpack_node(v: u64) -> Option<(i32, i32)> {
    (v != FOCUS_NONE).then(|| ((v >> 32) as u32 as i32, v as u32 as i32))
}

static FLAGS: AtomicU32 = AtomicU32::new(0);
static FOCUS: [AtomicU64; 3] = [
    AtomicU64::new(FOCUS_NONE),
    AtomicU64::new(FOCUS_NONE),
    AtomicU64::new(FOCUS_NONE),
];
static HDR_DUMP: RwLock<Option<std::path::PathBuf>> = RwLock::new(None);
static SINK: RwLock<Option<Sink>> = RwLock::new(None);
/// Set by the first [`install`] / [`clear`] (or the lazy env seed) so a later
/// query never re-seeds over an explicit configuration.
static INSTALLED: OnceLock<()> = OnceLock::new();

/// Install `cfg` for the whole process, replacing whatever was there.
pub fn install(cfg: TraceConfig) {
    let _ = INSTALLED.set(());
    for (slot, n) in FOCUS.iter().zip(cfg.focus) {
        slot.store(pack_node(n), Relaxed);
    }
    *HDR_DUMP.write().unwrap_or_else(|e| e.into_inner()) = cfg.hdr_dump;
    *SINK.write().unwrap_or_else(|e| e.into_inner()) = cfg.sink;
    // Flags last: a site that sees the bit also sees the focus/sink it needs.
    FLAGS.store(cfg.flags, Relaxed);
}

/// Disable everything and restore the stderr sink.
pub fn clear() {
    install(TraceConfig::new());
}

/// Read the documented `AOM_*` names and [`install`] them. The harness front
/// door; a host that wants env control calls this once at startup.
pub fn install_from_env() {
    install(TraceConfig::from_env());
}

#[inline]
fn seed_if_needed() {
    #[cfg(feature = "trace-env")]
    {
        // Lazy, once, only when nothing was installed explicitly: preserves the
        // pre-2026-09-24 behaviour for the in-tree harness without making the
        // published crates read the environment.
        if INSTALLED.get().is_none() {
            INSTALLED.get_or_init(|| {
                let cfg = TraceConfig::from_env();
                for (slot, n) in FOCUS.iter().zip(cfg.focus) {
                    slot.store(pack_node(n), Relaxed);
                }
                *HDR_DUMP.write().unwrap_or_else(|e| e.into_inner()) = cfg.hdr_dump;
                *SINK.write().unwrap_or_else(|e| e.into_inner()) = cfg.sink;
                FLAGS.store(cfg.flags, Relaxed);
            });
        }
    }
}

/// Is `t` enabled? One relaxed load on the hot path.
#[inline]
pub fn enabled(t: Trace) -> bool {
    seed_if_needed();
    FLAGS.load(Relaxed) & t.bit() != 0
}

/// The focus node for `f`, if one is installed.
#[inline]
pub fn focus(f: Focus) -> Option<(i32, i32)> {
    seed_if_needed();
    unpack_node(FOCUS[f as usize].load(Relaxed))
}

/// The header-dump path, if one is installed.
pub fn hdr_dump_path() -> Option<std::path::PathBuf> {
    seed_if_needed();
    HDR_DUMP.read().unwrap_or_else(|e| e.into_inner()).clone()
}

/// Deliver one line to the installed sink (stderr when none). Sites reach this
/// through [`trace_out!`](crate::trace_out); it is only ever called on an enabled path.
pub fn emit(args: fmt::Arguments<'_>) {
    let guard = SINK.read().unwrap_or_else(|e| e.into_inner());
    match guard.as_ref() {
        Some(sink) => sink(&args.to_string()),
        None => eprintln!("{args}"),
    }
}

/// `true` when [`Trace`] `$t` is enabled. Cached-atomic, hot-path safe.
#[macro_export]
macro_rules! trace_on {
    ($t:expr) => {
        $crate::trace::enabled($t)
    };
}

/// The installed [`Focus`] node for `$f`, `Option<(i32, i32)>`.
#[macro_export]
macro_rules! trace_focus {
    ($f:expr) => {
        $crate::trace::focus($f)
    };
}

/// `crate::trace_out!`-shaped: format and deliver one trace line to the installed sink.
#[macro_export]
macro_rules! trace_out {
    ($($arg:tt)*) => {
        $crate::trace::emit(::core::format_args!($($arg)*))
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    // The statics are process-global; nextest gives each test its own process,
    // and under plain `cargo test` these run serially through `install`, so
    // each one installs what it needs and clears at the end.

    #[test]
    fn install_sets_flags_focus_and_sink_and_clear_resets() {
        let lines = Arc::new(Mutex::new(Vec::new()));
        let sink_lines = lines.clone();
        install(
            TraceConfig::new()
                .with(Trace::Pack)
                .with_focus(Focus::Tx, 12, 34)
                .with_sink(Box::new(move |l| {
                    sink_lines.lock().unwrap().push(l.to_string())
                })),
        );
        assert!(enabled(Trace::Pack));
        assert!(!enabled(Trace::Sym));
        assert_eq!(focus(Focus::Tx), Some((12, 34)));
        assert_eq!(focus(Focus::Part), None);
        if trace_on!(Trace::Pack) {
            trace_out!("[pk] mi({},{}) test", 1, 2);
        }
        assert_eq!(lines.lock().unwrap().as_slice(), ["[pk] mi(1,2) test"]);
        clear();
        assert!(!enabled(Trace::Pack));
        assert_eq!(focus(Focus::Tx), None);
    }

    #[test]
    fn env_mapping_honours_zero_and_empty_and_parses_nodes() {
        std::env::set_var("AOM_PACK_TRACE", "1");
        std::env::set_var("AOM_SYM_TRACE", "0");
        std::env::set_var("AOM_CDEF_DBG", "");
        std::env::set_var("AOM_TX_DBG", " 7 , 9 ");
        std::env::set_var("AOM_UV_DBG", "junk");
        let cfg = TraceConfig::from_env();
        assert_eq!(cfg.flags & Trace::Pack.bit(), Trace::Pack.bit());
        assert_eq!(cfg.flags & Trace::Sym.bit(), 0);
        assert_eq!(cfg.flags & Trace::Cdef.bit(), 0);
        assert_eq!(cfg.focus[Focus::Tx as usize], Some((7, 9)));
        assert_eq!(cfg.focus[Focus::Uv as usize], None);
        for n in [
            "AOM_PACK_TRACE",
            "AOM_SYM_TRACE",
            "AOM_CDEF_DBG",
            "AOM_TX_DBG",
            "AOM_UV_DBG",
        ] {
            std::env::remove_var(n);
        }
    }

    #[test]
    fn every_kind_has_a_distinct_bit_and_env_name() {
        let mut bits = 0u32;
        let mut names = std::collections::HashSet::new();
        for t in Trace::ALL {
            assert_eq!(bits & t.bit(), 0, "{t:?} bit collides");
            bits |= t.bit();
            assert!(names.insert(t.env_name()), "{t:?} env name collides");
        }
        for f in Focus::ALL {
            assert!(names.insert(f.env_name()), "{f:?} env name collides");
        }
    }

    #[test]
    fn node_packing_round_trips_negatives() {
        for n in [
            Some((0, 0)),
            Some((-1, 5)),
            Some((i32::MAX, i32::MIN)),
            None,
        ] {
            assert_eq!(unpack_node(pack_node(n)), n);
        }
    }
}
