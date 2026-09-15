/* Shim over aom_highbd_lpf_{horizontal,vertical}_{4,6,8,14}_sse2 — the REAL
 * dispatched x86-64 kernels (what aomenc runs on v3 hardware), not the `_c`
 * references. Oracle only.
 *
 * The kernels read the blimit/limit/thresh rows with _mm_load_si128, so the
 * scalars are first broadcast into 16-byte-aligned rows — matching the
 * production `loop_filter_thresh` row layout where the caller passes
 * &[u8; 16].
 */
#include <stdint.h>
#include <string.h>
#define D(n) void aom_highbd_lpf_horizontal_##n##_sse2(uint16_t*,int,const uint8_t*,const uint8_t*,const uint8_t*,int); \
             void aom_highbd_lpf_vertical_##n##_sse2(uint16_t*,int,const uint8_t*,const uint8_t*,const uint8_t*,int);
D(4) D(6) D(8) D(14)
void shim_hbd_lpf_sse2(int dir, int width, uint16_t* s, int p, uint8_t bl,
                       uint8_t li, uint8_t th, int bd) {
  uint8_t blr[16] __attribute__((aligned(16)));
  uint8_t lir[16] __attribute__((aligned(16)));
  uint8_t thr[16] __attribute__((aligned(16)));
  memset(blr, bl, 16);
  memset(lir, li, 16);
  memset(thr, th, 16);
  if (dir==0) { switch(width){
    case 4: aom_highbd_lpf_horizontal_4_sse2(s,p,blr,lir,thr,bd); return;
    case 6: aom_highbd_lpf_horizontal_6_sse2(s,p,blr,lir,thr,bd); return;
    case 8: aom_highbd_lpf_horizontal_8_sse2(s,p,blr,lir,thr,bd); return;
    default: aom_highbd_lpf_horizontal_14_sse2(s,p,blr,lir,thr,bd); return; }
  } else { switch(width){
    case 4: aom_highbd_lpf_vertical_4_sse2(s,p,blr,lir,thr,bd); return;
    case 6: aom_highbd_lpf_vertical_6_sse2(s,p,blr,lir,thr,bd); return;
    case 8: aom_highbd_lpf_vertical_8_sse2(s,p,blr,lir,thr,bd); return;
    default: aom_highbd_lpf_vertical_14_sse2(s,p,blr,lir,thr,bd); return; }
  }
}
