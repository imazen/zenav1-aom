//! Decode-both localizer for two key-frame streams (port vs real C): decodes
//! each with the port decoder and finds the first divergent DECISION, not the
//! first divergent byte — partition tree node-for-node, then per-leaf mode/tx
//! fields, then the reconstruction pixels.
//!
//! Same method as `decode_diff_multisb.rs`, but driven by two stream files so
//! it works on any cell (the `eprof_yuv`/`dump_kf_stream` dumps).
//!
//! ```text
//! localize_stream_diff <port.obu> <c.obu>
//! ```

use aom_decode::frame::decode_frame_obus_prefilter;
use aom_decode::KfTileDecode;
use aom_dsp::entropy::partition::get_partition_subsize;

/// MI width of each bsize enum index (decode_diff_multisb.rs's table).
const MI_WIDE: [i32; 22] = [
    1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 1, 4, 2, 8, 4, 16,
];
const PARTITION_NAMES: [&str; 10] = [
    "NONE", "HORZ", "VERT", "SPLIT", "HORZ_A", "HORZ_B", "VERT_A", "VERT_B", "HORZ_4", "VERT_4",
];

fn replay(
    tree: &[i8],
    cur: &mut usize,
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
    mi_rows: i32,
    mi_cols: i32,
    out: &mut Vec<(i32, i32, usize, i8)>,
) {
    if mi_row >= mi_rows || mi_col >= mi_cols {
        return;
    }
    let p = tree[*cur];
    out.push((mi_row, mi_col, bsize, p));
    *cur += 1;
    if p == 3 {
        let hbs = MI_WIDE[bsize] / 2;
        let sub = get_partition_subsize(bsize, p as i32) as usize;
        replay(tree, cur, mi_row, mi_col, sub, mi_rows, mi_cols, out);
        replay(tree, cur, mi_row, mi_col + hbs, sub, mi_rows, mi_cols, out);
        replay(tree, cur, mi_row + hbs, mi_col, sub, mi_rows, mi_cols, out);
        replay(tree, cur, mi_row + hbs, mi_col + hbs, sub, mi_rows, mi_cols, out);
    }
}

/// Whole-tile node sequence: the recorded tree walks the SB grid in raster
/// order; replay each SB root against the shared cursor.
fn expand(t: &KfTileDecode, mi_rows: i32, mi_cols: i32, sb128: bool) -> Vec<(i32, i32, usize, i8)> {
    let root = if sb128 { 15usize } else { 12 }; // BLOCK_128X128 / BLOCK_64X64
    let mi = if sb128 { 32 } else { 16 };
    let mut out = Vec::new();
    let mut cur = 0usize;
    let mut r = 0;
    while r < mi_rows {
        let mut c = 0;
        while c < mi_cols {
            replay(&t.tree, &mut cur, r, c, root, mi_rows, mi_cols, &mut out);
            c += mi;
        }
        r += mi;
    }
    out
}

fn compare(a: &KfTileDecode, b: &KfTileDecode, mi_rows: i32, mi_cols: i32, sb128: bool) {
    let sa = expand(a, mi_rows, mi_cols, sb128);
    let sb = expand(b, mi_rows, mi_cols, sb128);
    println!("tree lens: a={} b={}", sa.len(), sb.len());
    let mut part_diverged = false;
    for (x, y) in sa.iter().zip(sb.iter()) {
        if x.3 != y.3 {
            println!(
                ">>> FIRST PARTITION DIVERGENCE (mi_row={}, mi_col={}, bsize={}): a={} b={}",
                x.0, x.1, x.2, PARTITION_NAMES[x.3 as usize], PARTITION_NAMES[y.3 as usize]
            );
            part_diverged = true;
            break;
        }
        if (x.0, x.1, x.2) != (y.0, y.1, y.2) {
            println!("positions diverged before any value diff — tree structures differ in arity");
            part_diverged = true;
            break;
        }
    }
    if !part_diverged {
        println!("partition trees agree on shared prefix");
    }

    // Even when the partition tree diverges, still compare leaf fields on the
    // blocks that exist at the same position+bsize in both streams — an
    // earlier mode-only divergence (which drifts entropy contexts without
    // touching the tree) would otherwise be hidden behind the partition diff.
    // `b.blocks` is in coding order, so the first reported mismatch is the
    // earliest leaf-level divergence.
    let mut leaf_reported = false;
    for rb in &b.blocks {
        match a.blocks.iter().find(|x| {
            x.mi_row == rb.mi_row && x.mi_col == rb.mi_col && x.bsize == rb.bsize
        }) {
            Some(ob) => {
                let i = &ob.info;
                let j = &rb.info;
                if ob.bsize != rb.bsize
                    || ob.partition != rb.partition
                    || i.y_mode != j.y_mode
                    || i.angle_delta_y != j.angle_delta_y
                    || i.use_filter_intra != j.use_filter_intra
                    || i.filter_intra_mode != j.filter_intra_mode
                    || i.uv_mode != j.uv_mode
                    || i.angle_delta_uv != j.angle_delta_uv
                    || i.cfl_alpha_idx != j.cfl_alpha_idx
                    || i.cfl_joint_sign != j.cfl_joint_sign
                    || i.palette_size != j.palette_size
                    || i.use_intrabc != j.use_intrabc
                    || i.dv_row != j.dv_row
                    || i.dv_col != j.dv_col
                    || i.skip != j.skip
                    || ob.tx_size != rb.tx_size
                    || ob.txbs != rb.txbs
                    || ob.txbs_uv != rb.txbs_uv
                {
                    println!(
                        ">>> FIRST LEAF MISMATCH (mi_row={}, mi_col={}):\n  a: bsize={} part={} y_mode={} adl={} fi={}/{} tx={} uv={} cfl={}/{} pal={:?} ibc={} dv=({},{}) skip={} txbs={:?} txbs_uv={:?}\n  b: bsize={} part={} y_mode={} adl={} fi={}/{} tx={} uv={} cfl={}/{} pal={:?} ibc={} dv=({},{}) skip={} txbs={:?} txbs_uv={:?}",
                        rb.mi_row, rb.mi_col,
                        ob.bsize, ob.partition, i.y_mode, i.angle_delta_y, i.use_filter_intra, i.filter_intra_mode, ob.tx_size, i.uv_mode, i.cfl_alpha_idx, i.cfl_joint_sign, i.palette_size, i.use_intrabc, i.dv_row, i.dv_col, i.skip, ob.txbs, ob.txbs_uv,
                        rb.bsize, rb.partition, j.y_mode, j.angle_delta_y, j.use_filter_intra, j.filter_intra_mode, rb.tx_size, j.uv_mode, j.cfl_alpha_idx, j.cfl_joint_sign, j.palette_size, j.use_intrabc, j.dv_row, j.dv_col, j.skip, rb.txbs, rb.txbs_uv,
                    );
                    leaf_reported = true;
                    // Dump the leaf sequence ending at the mismatch so the
                    // immediately-preceding committed leaves (the ctx writers)
                    // are visible.
                    let idx = b.blocks.iter().position(|x| {
                        x.mi_row == rb.mi_row && x.mi_col == rb.mi_col
                    });
                    if let Some(i0) = idx {
                        let lo = i0.saturating_sub(8);
                        for x in &b.blocks[lo..=i0] {
                            println!(
                                "  leaf-seq mi({},{}) bs{} part{} txbs={:?}",
                                x.mi_row, x.mi_col, x.bsize, x.partition, x.txbs
                            );
                        }
                    }
                    break;
                }
            }
            None => {}
        }
    }
    if !leaf_reported && !part_diverged {
        println!("all leaf mode/tx fields agree — divergence is coefficient VALUES");
    }

    // Recon-pixel diff (prefilter planes — post-loop-filter pixels are the
    // FrameDecode surface; KfTileDecode.recon is pre-CDEF/LR, which is what
    // localizes the block itself).
    let (pa, pb) = (a.recon.to_u16(), b.recon.to_u16());
    for (k, (x, y)) in pa.iter().zip(pb.iter()).enumerate() {
        if x != y {
            let (r, c) = (k / a.stride, k % a.stride);
            println!("first luma recon diff at px ({c},{r}) = a:{x} b:{y}  (mi {},{})", c / 4, r / 4);
            return;
        }
    }
    println!("recon planes identical — residual syntax-only divergence");
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 {
        eprintln!("usage: localize_stream_diff <a.obu> <b.obu>");
        std::process::exit(2);
    }
    let fa = std::fs::read(&a[1]).unwrap();
    let fb = std::fs::read(&a[2]).unwrap();
    let (ta, ca, _ha) = decode_frame_obus_prefilter(&fa).expect("decode a");
    let (tb, cb, _hb) = decode_frame_obus_prefilter(&fb).expect("decode b");
    assert_eq!(ca.sb_size_128, cb.sb_size_128, "SB size differs — header-level divergence");
    assert_eq!((ca.mi_rows, ca.mi_cols), (cb.mi_rows, cb.mi_cols));
    println!("mi {}x{} sb128={}", ca.mi_cols, ca.mi_rows, ca.sb_size_128);
    compare(&ta, &tb, ca.mi_rows, ca.mi_cols, ca.sb_size_128);
}
