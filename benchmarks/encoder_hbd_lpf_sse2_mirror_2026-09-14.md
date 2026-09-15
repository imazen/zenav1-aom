# highbd loop-filter v3 re-mirrored onto the real dispatched SSE2 kernels — 2026-09-14

Cell: 1024x1024 cq27 `--cpu-used 3` (the preset zenavif ships), eprof_x86 `port`,
reps=1 per round, `nice -n 19`, rotated base/new/baseB. Band TSV:
`benchmarks/encoder_hbd_lpf_sse2_mirror_2026-09-14.band1024s3.tsv`.
base = `d719fd1`; new = working tree.

## Result

```
base     median   2461.14 ms  min   2451.29  n=24  bytes=['40237']
baseB    median   2466.04 ms  min   2452.55  n=24  bytes=['40237']
new      median   2447.22 ms  min   2439.37  n=24  bytes=['40237']
new vs base   : paired median -0.589 %   23/24 rounds faster   p=2.98e-06
null (baseB vs base): paired median +0.112 %   9/24 rounds faster   p=0.3075
```

Clears the same-binary null and p < 0.05 — a real landing. Byte-identical
(40,237 B every round, both arms).

## Mechanism

Paired callgrind (196x196 cq27 s0, 20 reps) put `lpf_impl_v3` at ~100M Ir
against C's entire dispatched highbd lpf family at ~20M — ~5x excess, and the
largest remaining single-symbol Ir gap after the `quantize_fp` v3 mirror
(`d719fd1`). The previous v3 body was a `magetypes` i32x4 kernel whose every
tap load was a per-lane scalar gather (~144 Ir/edge-position vs C's ~18).

libaom's real x86 kernels (`aom_dsp/x86/highbd_loopfilter_sse2.c` +
`lpf_common_sse2.h`) do contiguous `loadl`/`loadu` row loads, unpack/transpose
into packed `[p | q]` u16 lanes (one op covers both sides of the edge), use
SSE2 saturating arithmetic for the filter terms, and blend per-lane by mask.
The new `#[archmage::arcane]` `lpf_impl_v3` in
`crates/aom-dsp/src/loopfilter/simd.rs` mirrors that structure
instruction-for-instruction for widths 4/6/8/14, both directions, using the
`archmage::intrinsics::x86_64` safe load/store wrappers (no raw pointers).

Fallback stays panic-free: unsupported widths and any `&mut [u16]` window too
short for the C-style wider loads route to `lpf_scalar`, which is untouched.
`AOM_FORCE_SCALAR` still pins the scalar leg at dispatch entry.

Kernel cost: ~577 -> ~330 Ir/call (-43 %), measured on the same callgrind
config; the residual vs C's ~113-340 Ir/call per-width is the safe-Rust
bounds bookkeeping on the wider C-style spans, which the preflight span check
converts to a single rejection branch rather than a panic path.

## Verification

- `hbd_lpf_sse2_diff` (new): v3 vs the REAL exported
  `aom_highbd_lpf_{horizontal,vertical}_{4,6,8,14}_sse2` symbols — the kernels
  libaom actually dispatches on x86-64 — over ~288k adversarial cases:
  bd {8,10,12} x both directions x all four widths x per-position flat /
  soft-edge / hard-edge / hev-heavy lane mixtures (divergent per-lane masks
  inside one call). Shim: `aom-sys-ref/shim/hbd_lpf_sse2_shim.c`.
- `hbd_lpf_diff` (v3 vs C `_c` scalar) and `lpf_simd_diff` (tier parity)
  stay green.
- `aom-dsp` `all`: 403/403 under nextest (the gate runner).
- `self_contained_key_frame` 10/10, `encoder_gate_e2e_byte_match` 32/32 —
  every shipping-cell byte identical.

## Deliberately not done

- C's `_dual`/`_quad` batching variants (one call filters two/four
  same-parameter edges): the port's frame traversal issues one kernel call
  per edge (1.29M calls/20 reps at 196^2 s0 vs C's ~84k). That is a traversal
  restructure, not a kernel change — the per-call cost is now near C's, so
  the remaining gap is call-count, a separate lever.
- The u8 (`lpf_lowbd`) path was not touched — same magetypes-gather shape,
  smaller share; the mirror pattern transfers if it ever tops a profile.

## Note: upstream archmage harness race surfaced (not a port bug)

The new differential initially flaked under multi-threaded `cargo test`:
`X64V3Token::summon()` reads a process-wide detection cache that
`dangerously_disable_token_process_wide` forces to 1 and re-enable resets to
0; a concurrent `summon()` in the re-enable->disable window enters
`x64_v3_detect()`, whose unconditional `CACHE.store(2)` can land INSIDE the
next (scalar) permutation — a kernel then legally dispatches v3 while the
harness believes the tier is off. This is the residual `dispatch_serial.rs`
already documents ("a sweep also races any concurrent test that merely reads
dispatch state"); the band runner (`nextest`, process-per-test) cannot hit
it. The new test takes `archmage::testing::lock_token_testing()` so it can
neither observe a transiently-disabled v3 nor poison a permutation during
its own 288k-summon run; a sibling `cdef_find_dir_simd_diff` red under
multi-threaded `cargo test` is that documented residual, green under
nextest.
