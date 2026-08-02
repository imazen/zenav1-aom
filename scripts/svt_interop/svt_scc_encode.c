/*
 * svt_scc_encode — SCREEN-CONTENT still-picture encode driver for the REAL C
 * SVT-AV1 (imazen/zenav1-svt-c, an upstream SVT-AV1 mirror), used to generate
 * the interop corpus for GitHub issue #5 (aom-decode rejecting conformant
 * non-libaom IntraBC streams).
 *
 *   svt_scc_encode <w> <h> <qp 0..63> <preset 0..13> <scm 0|1|2> <in.yuv> <out.obu>
 *   stdout: "BYTES=<n>"
 *
 * Optional env: SVT_TILE_COLS / SVT_TILE_ROWS (log2 tile counts, default 0).
 * Tiles matter here because `av1_is_dv_valid` reads `tile->mi_col_start/end`
 * directly (the tile-bound clamp, `total_sb64_per_row`, and the wavefront
 * gradient all derive from them), so a multi-tile stream exercises DV validity
 * inputs a single-tile stream structurally cannot reach.
 *
 * This is drv_svtc.c (benchmarks/xbench/csrc) minus the timing loop, plus the
 * one knob that matters here: `cfg.screen_content_mode`. SVT enables IntraBC
 * (and palette) only when its screen-content path is armed --
 *   0 = off, 1 = forced on, 2 = auto-detect (see Source/Lib/Codec/pd_process.c
 *   and pic_analysis_process.c, both switching on `screen_content_mode`).
 * Everything else is byte-for-byte the same config the xbench svt-c arm uses,
 * so the streams here are directly comparable to that measurement:
 *   still/AVIF CQP (rc 0, aq 0, --avif 1, --lp 1), 8-bit 4:2:0, single tile.
 *
 * NOTHING here modifies the SVT-AV1 C tree: it links the out-of-tree static lib
 * that scripts/xbench_build.sh already produces. Build with
 * scripts/svt_interop/build.sh.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "EbSvtAv1.h"
#include "EbSvtAv1Enc.h"

static void die(const char* msg, int32_t err) {
    fprintf(stderr, "svt_scc_encode: %s (err=0x%x)\n", msg, (unsigned)err);
    exit(1);
}

int main(int argc, char** argv) {
    if (argc != 8) {
        fprintf(stderr,
                "usage: %s <w> <h> <qp 0..63> <preset 0..13> <scm 0|1|2> <in.yuv> <out.obu>\n",
                argv[0]);
        return 2;
    }
    const uint32_t w      = (uint32_t)atoi(argv[1]);
    const uint32_t h      = (uint32_t)atoi(argv[2]);
    const uint32_t qp     = (uint32_t)atoi(argv[3]);
    const int8_t   preset = (int8_t)atoi(argv[4]);
    const uint32_t scm    = (uint32_t)atoi(argv[5]);
    const char*    in_yuv = argv[6];
    const char*    out    = argv[7];

    const size_t ysz         = (size_t)w * h;
    const size_t cw          = ((size_t)w + 1) / 2;
    const size_t chh         = ((size_t)h + 1) / 2;
    const size_t csz         = cw * chh;
    const size_t frame_bytes = ysz + 2 * csz;

    uint8_t* yuv = malloc(frame_bytes);
    if (!yuv) die("oom", 0);
    FILE* fi = fopen(in_yuv, "rb");
    if (!fi) die("cannot open input .yuv", 0);
    if (fread(yuv, 1, frame_bytes, fi) != frame_bytes) die("short read (need w*h*3/2 I420)", 0);
    fclose(fi);

    EbComponentType*         handle = NULL;
    EbSvtAv1EncConfiguration cfg;
    memset(&cfg, 0, sizeof(cfg));
    EbErrorType err = svt_av1_enc_init_handle(&handle, &cfg);
    if (err != EB_ErrorNone) die("svt_av1_enc_init_handle", err);

    cfg.source_width           = w;
    cfg.source_height          = h;
    cfg.enc_mode               = preset;
    cfg.rate_control_mode      = 0;    /* CQP/CRF */
    cfg.aq_mode                = 0;    /* rc 0 + aq 0 == CQP */
    cfg.qp                     = qp;   /* CLI domain 0..63 */
    cfg.avif                   = true; /* still_picture + reduced_still_picture_header */
    cfg.level_of_parallelism   = 1;    /* --lp 1 (single-threaded) */
    cfg.encoder_bit_depth      = 8;
    cfg.encoder_color_format   = EB_YUV420;
    cfg.frame_rate_numerator   = 30;
    cfg.frame_rate_denominator = 1;
    cfg.screen_content_mode    = scm; /* --scm: the main delta vs drv_svtc.c */
    {
        const char* tc = getenv("SVT_TILE_COLS");
        const char* tr = getenv("SVT_TILE_ROWS");
        if (tc) cfg.tile_columns = atoi(tc);
        if (tr) cfg.tile_rows = atoi(tr);
    }

    err = svt_av1_enc_set_parameter(handle, &cfg);
    if (err != EB_ErrorNone) die("svt_av1_enc_set_parameter", err);
    err = svt_av1_enc_init(handle);
    if (err != EB_ErrorNone) die("svt_av1_enc_init", err);

    EbSvtIOFormat io;
    memset(&io, 0, sizeof(io));
    io.luma      = yuv;
    io.cb        = yuv + ysz;
    io.cr        = yuv + (ysz + csz);
    io.y_stride  = w;
    io.cb_stride = (uint32_t)cw;
    io.cr_stride = (uint32_t)cw;

    EbBufferHeaderType in_hdr;
    memset(&in_hdr, 0, sizeof(in_hdr));
    in_hdr.size         = sizeof(EbBufferHeaderType);
    in_hdr.p_buffer     = (uint8_t*)&io;
    in_hdr.n_filled_len = (uint32_t)frame_bytes;
    in_hdr.pts          = 0;
    in_hdr.pic_type     = EB_AV1_INVALID_PICTURE;

    EbBufferHeaderType eos_hdr;
    memset(&eos_hdr, 0, sizeof(eos_hdr));
    eos_hdr.size     = sizeof(EbBufferHeaderType);
    eos_hdr.flags    = EB_BUFFERFLAG_EOS;
    eos_hdr.pic_type = EB_AV1_INVALID_PICTURE;

    err = svt_av1_enc_send_picture(handle, &in_hdr);
    if (err != EB_ErrorNone) die("svt_av1_enc_send_picture", err);
    err = svt_av1_enc_send_picture(handle, &eos_hdr);
    if (err != EB_ErrorNone) die("svt_av1_enc_send_picture(EOS)", err);

    FILE*    fo     = fopen(out, "wb");
    if (!fo) die("cannot open output", 0);
    uint32_t nbytes = 0;
    for (;;) {
        EbBufferHeaderType* pkt = NULL;
        err                     = svt_av1_enc_get_packet(handle, &pkt, 1);
        if (err == EB_ErrorMax) die("svt_av1_enc_get_packet", err);
        if (pkt == NULL) break;
        if (pkt->n_filled_len) {
            fwrite(pkt->p_buffer, 1, pkt->n_filled_len, fo);
            nbytes += pkt->n_filled_len;
        }
        const uint32_t last = (pkt->flags & EB_BUFFERFLAG_EOS) != 0;
        svt_av1_enc_release_out_buffer(&pkt);
        if (last) break;
    }
    fclose(fo);
    svt_av1_enc_deinit(handle);
    svt_av1_enc_deinit_handle(handle);

    printf("BYTES=%u\n", nbytes);
    free(yuv);
    return 0;
}
