//! Dump the port decoder's leaf sequence (coding order) for a key-frame
//! stream — the port-side counterpart of the `AOM_LEAF_TRACE` instrumentation
//! in upstream aomdec, for locating decoder-mirror desyncs.
//!
//! ```text
//! dump_port_leaves <stream.obu>
//! ```

use aom_decode::frame::decode_frame_obus_prefilter;

fn main() {
    let path = std::env::args().nth(1).expect("usage: dump_port_leaves <obu>");
    let bytes = std::fs::read(&path).expect("read stream");
    let (t, _c, _h) = decode_frame_obus_prefilter(&bytes).expect("decode");
    for b in &t.blocks {
        let i = &b.info;
        println!(
            "[pleaf] mi({},{}) bs={} part={} mode={} ibc={} dv=({},{}) skip={} tx={} pal={:?} txbs={:?}",
            b.mi_row, b.mi_col, b.bsize, b.partition, i.y_mode, i.use_intrabc,
            i.dv_row, i.dv_col, i.skip, b.tx_size, i.palette_size, b.txbs
        );
    }
    eprintln!("leaves={} tree={}", t.blocks.len(), t.tree.len());
}
