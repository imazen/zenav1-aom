# get_txb_ctx — square-tx const-generic specialization, mirroring C's SPECIALIZE_GET_TXB_CTX (2026-09-15)

## Mechanism

`crates/aom-dsp/src/txb/entropy_ctx.rs` `get_txb_ctx` ran the general body with
runtime `w_unit`/`h_unit` loop bounds on every call (~102k/rep at the 196²
attribution cell). C's wrapper (txb_common.h:446) dispatches the four square tx
sizes through `SPECIALIZE_GET_TXB_CTX` macros "so the compiler can compile away
the while loops".

The port now does the same via monomorphization: `tx_size` 0..3
(TX_4X4/8X8/16X16/32X32) route to `get_txb_ctx_units::<W, H>` with `W`/`H`
literal unit dims, and every other shape keeps the general path. Both arms
share `get_txb_ctx_body` over pre-sliced `&a[..W]`/`&l[..H]` context arrays, so
the per-element slice indexing C's macro bodies avoid is gone in the
specialized legs. `debug_assert`s pin `W`/`H` to `TX_WIDE/HIGH_UNIT[tx_size]`.

## Measurement

Callgrind 196x196 cq27 `--cpu-used 3`, 5 reps, identical call counts:

- `get_txb_ctx` self Ir: **94.2M → 87.5M (−7 %)**

Below band resolution — landed on the strict-Ir precedent.

## Correctness

- `entropy_ctx_diff::get_txb_ctx_matches_c` — real-C oracle — green.
- Shipping cell byte-identical: 40,237 B.
