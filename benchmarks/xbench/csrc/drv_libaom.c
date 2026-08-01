/*
 * drv_libaom — timed still-picture encode driver for the REAL C libaom, the
 * encoder zenav1-aom is a port of.
 *
 * Uniform driver contract, identical to the four drivers already here:
 *   drv <w> <h> <cq 0..63> <cpu-used 0..9> <in.yuv> <out.obu> <warmup> <reps>
 *   stdout: "NS=<n> NS=<n> ... BYTES=<m>"
 *
 * WHY THIS EXISTS: benchmarks/xbench_2026-08-01.md decomposed the Rust SVT
 * port's BD-rate deficit into "throughput" and "coding" by adding an `svt-c@p4`
 * arm — the same C encoder at the port's qualifying preset. It could not do the
 * same for `zenav1-aom` because libaom C was not one of the encoders under
 * test. This driver is that missing arm.
 *
 * CONFIG FIDELITY — the point of the whole exercise. The setup below is a
 * line-for-line transcription of `shim_encode_av1_kf_defaults`
 * (crates/aom-sys-ref/shim/dec_shim.c), which is the function `drv-aom` calls
 * (via `EncodeCell::c_encode_defaults`) to produce its sequence-header
 * bootstrap. Same `aom_codec_enc_config_default(AOM_USAGE_ALL_INTRA)`, same
 * five controls, same 8-bit 4:2:0 image, same single FORCE_KF frame + flush.
 * So `drv_libaom` and `drv-aom` at the same (w,h,q,speed) differ ONLY in which
 * encoder produced the frame OBU — which makes a whole-stream sha256 of the two
 * `.obu` files a direct byte-identity test of the port against its oracle. That
 * transcription is not taken on trust: scripts/xbench.py's `byteid` subcommand
 * runs exactly that comparison over the RD corpus.
 *
 * TIMED REGION: `aom_codec_encode(img)` + the packet drain + the flush call +
 * its drain — i.e. the per-frame encode work only. Excluded, on every rep:
 * config build, `aom_img_alloc` + the plane copy, `aom_codec_enc_init` (table
 * build, buffer alloc), the controls, and `aom_codec_destroy`. That is the same
 * boundary drv_svtc.c draws around SVT-AV1 (`svt_av1_enc_init` excluded), and
 * the same rule the harness states for all drivers: no process start, no file
 * I/O, no constructor/init.
 *
 * A fresh codec context is built per rep, exactly as drv_svtc.c does, so no rep
 * inherits another's warmed internal state. libaom allocates some per-frame
 * working buffers lazily on the first `aom_codec_encode`, so that allocation
 * IS inside the timed region here — symmetric with `drv-aom`, whose timed
 * `port_encode` also allocates its own working buffers on every call.
 *
 * Built out-of-tree by scripts/xbench_build.sh against the PINNED in-tree
 * oracle libaom: the `upstream/` submodule (v3.14.1, 03087864) built by
 * crates/aom-sys-ref/build.rs into upstream/build/libaom.a. NOT the Homebrew
 * aomenc, whose version and build flags are unpinned. The oracle build config
 * (Release, CONFIG_MULTITHREAD=0, -ffp-contract=off) is documented in
 * reference/BUILD_CONFIG.md; CONFIG_MULTITHREAD=0 is exactly the
 * single-threaded operating point this benchmark requires, and NEON is on
 * (aom_config.h CONFIG_RUNTIME_CPU_DETECT / HAVE_NEON — checked, reported in
 * the .meta).
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "aom/aom_encoder.h"
#include "aom/aom_image.h"
#include "aom/aomcx.h"

static void die(const char* msg, int err) {
    fprintf(stderr, "drv_libaom: %s (err=%d)\n", msg, err);
    exit(1);
}

static int64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + (int64_t)ts.tv_nsec;
}

/* One full encode. Returns the TIMED nanoseconds; writes the stream to
 * `out_path` (may be NULL) and its length to *out_bytes. */
static int64_t encode_once(int w, int h, int cq_level, int cpu_used, const uint8_t* yuv,
                           const char* out_path, uint32_t* out_bytes) {
    /* ---- UNTIMED setup (mirrors shim_encode_av1_kf_defaults) ---- */
    aom_codec_iface_t*  iface = aom_codec_av1_cx();
    aom_codec_enc_cfg_t cfg;
    if (aom_codec_enc_config_default(iface, &cfg, AOM_USAGE_ALL_INTRA))
        die("aom_codec_enc_config_default", 0);
    cfg.g_w               = (unsigned int)w;
    cfg.g_h               = (unsigned int)h;
    cfg.g_limit           = 1;
    cfg.g_lag_in_frames   = 0;
    cfg.g_threads         = 1;
    cfg.g_pass            = AOM_RC_ONE_PASS;
    cfg.rc_end_usage      = AOM_Q;
    cfg.monochrome        = 0;
    cfg.g_input_bit_depth = 8;
    cfg.g_bit_depth       = AOM_BITS_8;
    cfg.g_profile         = 0; /* 8-bit 4:2:0 */

    aom_image_t* img = aom_img_alloc(NULL, AOM_IMG_FMT_I420, w, h, 32);
    if (!img) die("aom_img_alloc", 0);
    img->monochrome = 0;
    img->bit_depth  = 8;
    const int cw = (w + 1) >> 1;
    const int ch = (h + 1) >> 1;
    for (int plane = 0; plane < 3; plane++) {
        const uint8_t* src = plane == 0 ? yuv
                             : plane == 1 ? yuv + (size_t)w * h
                                          : yuv + (size_t)w * h + (size_t)cw * ch;
        const int      pw  = plane == 0 ? w : cw;
        const int      ph  = plane == 0 ? h : ch;
        for (int r = 0; r < ph; r++)
            memcpy(img->planes[plane] + (size_t)r * img->stride[plane],
                   src + (size_t)r * pw, (size_t)pw);
    }

    aom_codec_ctx_t ctx;
    if (aom_codec_enc_init(&ctx, iface, &cfg, 0)) die("aom_codec_enc_init", 0);
#define DFLTCTRL(id, val)                                     \
    do {                                                      \
        if (aom_codec_control(&ctx, (id), (val)))             \
            die("aom_codec_control " #id, 0);                 \
    } while (0)
    /* ONLY the operating point + the SB64 envelope — no coding-tool controls,
     * so every tool sits at its ALLINTRA default (cdef OFF, loop-restoration
     * ON, qm OFF). Identical list, identical order, to the shim. */
    DFLTCTRL(AOME_SET_CPUUSED, cpu_used);
    DFLTCTRL(AOME_SET_CQ_LEVEL, cq_level);
    DFLTCTRL(AV1E_SET_SUPERBLOCK_SIZE, AOM_SUPERBLOCK_SIZE_64X64);
    DFLTCTRL(AV1E_SET_TILE_COLUMNS, 0);
    DFLTCTRL(AV1E_SET_TILE_ROWS, 0);
#undef DFLTCTRL

    static uint8_t* out     = NULL;
    static size_t   out_cap = 0;
    const size_t    want    = (size_t)w * h * 3 + (1u << 20);
    if (out_cap < want) {
        free(out);
        out     = (uint8_t*)malloc(want);
        out_cap = want;
        if (!out) die("oom (output buffer)", 0);
    }

    /* ---- TIMED ---- */
    const int64_t t0    = now_ns();
    long          total = 0;
    for (int pass = 0; pass < 2; pass++) {
        if (aom_codec_encode(&ctx, pass == 0 ? img : NULL, 0, 1,
                             pass == 0 ? AOM_EFLAG_FORCE_KF : 0))
            die("aom_codec_encode", pass);
        aom_codec_iter_t          iter = NULL;
        const aom_codec_cx_pkt_t* pkt;
        while ((pkt = aom_codec_get_cx_data(&ctx, &iter)) != NULL) {
            if (pkt->kind != AOM_CODEC_CX_FRAME_PKT) continue;
            if ((size_t)total + pkt->data.frame.sz > out_cap) die("output overflow", 0);
            memcpy(out + total, pkt->data.frame.buf, pkt->data.frame.sz);
            total += (long)pkt->data.frame.sz;
        }
    }
    const int64_t t1 = now_ns();
    /* ---- END TIMED ---- */

    aom_codec_destroy(&ctx);
    aom_img_free(img);

    if (out_path) {
        FILE* fo = fopen(out_path, "wb");
        if (!fo) die("cannot open output", 0);
        fwrite(out, 1, (size_t)total, fo);
        fclose(fo);
    }
    if (out_bytes) *out_bytes = (uint32_t)total;
    return t1 - t0;
}

int main(int argc, char** argv) {
    if (argc != 9) {
        fprintf(stderr,
                "usage: %s <w> <h> <cq 0..63> <cpu-used 0..9> <in.yuv> <out.obu> <warmup> "
                "<reps>\n",
                argv[0]);
        return 2;
    }
    const int   w        = atoi(argv[1]);
    const int   h        = atoi(argv[2]);
    const int   cq_level = atoi(argv[3]);
    const int   cpu_used = atoi(argv[4]);
    const char* in_yuv   = argv[5];
    const char* out      = argv[6];
    const int   warmup   = atoi(argv[7]);
    const int   reps     = atoi(argv[8]);
    if (w % 2 || h % 2) die("even dims only", 0);

    const size_t ysz         = (size_t)w * h;
    const size_t csz         = (size_t)((w + 1) / 2) * ((h + 1) / 2);
    const size_t frame_bytes = ysz + 2 * csz;

    uint8_t* yuv = (uint8_t*)malloc(frame_bytes);
    if (!yuv) die("oom", 0);
    FILE* fi = fopen(in_yuv, "rb");
    if (!fi) die("cannot open input .yuv", 0);
    if (fread(yuv, 1, frame_bytes, fi) != frame_bytes) die("short read (need w*h*3/2 I420)", 0);
    fclose(fi);

    for (int k = 0; k < warmup; k++)
        (void)encode_once(w, h, cq_level, cpu_used, yuv, NULL, NULL);

    uint32_t bytes = 0;
    for (int k = 0; k < reps; k++) {
        const int64_t ns =
            encode_once(w, h, cq_level, cpu_used, yuv, (k == reps - 1) ? out : NULL, &bytes);
        printf("NS=%lld ", (long long)ns);
    }
    printf("BYTES=%u\n", bytes);
    free(yuv);
    return 0;
}
