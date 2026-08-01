/*
 * drv_svtc — timed still-picture encode driver for the REAL C SVT-AV1
 * (imazen/zenav1-svt-c, an upstream SVT-AV1 mirror).
 *
 * Uniform driver contract, identical to the three Rust drivers here:
 *   drv <w> <h> <qp 0..63> <preset 0..13> <in.yuv> <out.obu> <warmup> <reps>
 *   stdout: "NS=<n> NS=<n> ... BYTES=<m>"
 *
 * Timed region = svt_av1_enc_send_picture(frame) + send(EOS) + the get_packet
 * drain, i.e. the per-frame encode work only. The one-time setup
 * (init_handle / set_parameter / svt_av1_enc_init: table build, buffer alloc,
 * thread spawn) is done BEFORE the clock starts on every rep, symmetric with
 * the Rust drivers which exclude their own constructors.
 *
 * Config: still/AVIF CQP (--rc 0 --aq-mode 0 --qp Q --avif 1 --lp 1 -n 1),
 * 8-bit 4:2:0, single tile. This knob set is copied from the zenav1-svt port's
 * own byte-identity oracle harness (rust/tools/perf_c_encode/perf_c_encode.c),
 * so the C encoder here is driven exactly as the port's parity gates drive it.
 *
 * Built out-of-tree by scripts/xbench_build.sh — the SVT-AV1 C checkout is
 * never modified.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "EbSvtAv1.h"
#include "EbSvtAv1Enc.h"

static void die(const char* msg, int32_t err) {
    fprintf(stderr, "drv_svtc: %s (err=0x%x)\n", msg, (unsigned)err);
    exit(1);
}

static int64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + (int64_t)ts.tv_nsec;
}

static int64_t encode_once(uint32_t w, uint32_t h, uint32_t qp, int8_t preset, const uint8_t* yuv,
                           size_t frame_bytes, size_t ysz, size_t csz, size_t cw,
                           const char* out_path, uint32_t* out_bytes) {
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

    err = svt_av1_enc_set_parameter(handle, &cfg);
    if (err != EB_ErrorNone) die("svt_av1_enc_set_parameter", err);
    err = svt_av1_enc_init(handle);
    if (err != EB_ErrorNone) die("svt_av1_enc_init", err);

    EbSvtIOFormat io;
    memset(&io, 0, sizeof(io));
    io.luma      = (uint8_t*)yuv;
    io.cb        = (uint8_t*)yuv + ysz;
    io.cr        = (uint8_t*)yuv + (ysz + csz);
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

    /* ---- TIMED ---- */
    const int64_t t0 = now_ns();
    err              = svt_av1_enc_send_picture(handle, &in_hdr);
    if (err != EB_ErrorNone) die("svt_av1_enc_send_picture", err);
    err = svt_av1_enc_send_picture(handle, &eos_hdr);
    if (err != EB_ErrorNone) die("svt_av1_enc_send_picture(EOS)", err);

    FILE*    fo     = out_path ? fopen(out_path, "wb") : NULL;
    uint32_t nbytes = 0;
    for (;;) {
        EbBufferHeaderType* pkt = NULL;
        err                     = svt_av1_enc_get_packet(handle, &pkt, 1);
        if (err == EB_ErrorMax) die("svt_av1_enc_get_packet", err);
        if (pkt == NULL) break;
        if (pkt->n_filled_len) {
            if (fo) fwrite(pkt->p_buffer, 1, pkt->n_filled_len, fo);
            nbytes += pkt->n_filled_len;
        }
        const uint32_t last = (pkt->flags & EB_BUFFERFLAG_EOS) != 0;
        svt_av1_enc_release_out_buffer(&pkt);
        if (last) break;
    }
    const int64_t t1 = now_ns();
    /* ---- END TIMED ---- */

    if (fo) fclose(fo);
    svt_av1_enc_deinit(handle);
    svt_av1_enc_deinit_handle(handle);
    if (out_bytes) *out_bytes = nbytes;
    return t1 - t0;
}

int main(int argc, char** argv) {
    if (argc != 9) {
        fprintf(stderr,
                "usage: %s <w> <h> <qp 0..63> <preset 0..13> <in.yuv> <out.obu> <warmup> <reps>\n",
                argv[0]);
        return 2;
    }
    const uint32_t w      = (uint32_t)atoi(argv[1]);
    const uint32_t h      = (uint32_t)atoi(argv[2]);
    const uint32_t qp     = (uint32_t)atoi(argv[3]);
    const int8_t   preset = (int8_t)atoi(argv[4]);
    const char*    in_yuv = argv[5];
    const char*    out    = argv[6];
    const int      warmup = atoi(argv[7]);
    const int      reps   = atoi(argv[8]);

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

    for (int k = 0; k < warmup; k++)
        (void)encode_once(w, h, qp, preset, yuv, frame_bytes, ysz, csz, cw, NULL, NULL);

    uint32_t bytes = 0;
    for (int k = 0; k < reps; k++) {
        const int64_t ns =
            encode_once(w, h, qp, preset, yuv, frame_bytes, ysz, csz, cw,
                        (k == reps - 1) ? out : NULL, &bytes);
        printf("NS=%lld ", (long long)ns);
    }
    printf("BYTES=%u\n", bytes);
    free(yuv);
    return 0;
}
