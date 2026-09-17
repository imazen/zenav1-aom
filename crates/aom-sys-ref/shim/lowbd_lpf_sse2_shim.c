/* Shim over aom_lpf_{horizontal,vertical}_{4,6,8,14}_sse2 — the REAL dispatched
   x86-64 lowbd kernels, not the `_c` references. The kernels load the scalar
   thresholds from 16-byte broadcast rows; the shim broadcasts them.
   NOTE: these SSE2 kernels are NOT bit-identical to aom_lpf_*_c — their mask
   sums use saturating u8 arithmetic (adds_epu8 caps at 255), so with
   blimit==255 they filter edges whose unclamped sum exceeds 255. This shim
   exists precisely so the port's verbatim SSE2 mirror can be gated against
   the dispatched kernel rather than the scalar reference. */
#include <stdint.h>
#include <string.h>
#define D(n) void aom_lpf_horizontal_##n##_sse2(uint8_t*,int,const uint8_t*,const uint8_t*,const uint8_t*); \
             void aom_lpf_vertical_##n##_sse2(uint8_t*,int,const uint8_t*,const uint8_t*,const uint8_t*);
D(4) D(6) D(8) D(14)
void shim_lowbd_lpf_sse2(int dir, int width, uint8_t* s, int p, uint8_t bl,
                         uint8_t li, uint8_t th) {
  uint8_t blr[16] __attribute__((aligned(16)));
  uint8_t lir[16] __attribute__((aligned(16)));
  uint8_t thr[16] __attribute__((aligned(16)));
  memset(blr, bl, 16);
  memset(lir, li, 16);
  memset(thr, th, 16);
  if (dir==0) { switch(width){
    case 4: aom_lpf_horizontal_4_sse2(s,p,blr,lir,thr); return;
    case 6: aom_lpf_horizontal_6_sse2(s,p,blr,lir,thr); return;
    case 8: aom_lpf_horizontal_8_sse2(s,p,blr,lir,thr); return;
    default: aom_lpf_horizontal_14_sse2(s,p,blr,lir,thr); return; }
  } else { switch(width){
    case 4: aom_lpf_vertical_4_sse2(s,p,blr,lir,thr); return;
    case 6: aom_lpf_vertical_6_sse2(s,p,blr,lir,thr); return;
    case 8: aom_lpf_vertical_8_sse2(s,p,blr,lir,thr); return;
    default: aom_lpf_vertical_14_sse2(s,p,blr,lir,thr); return; }
  }
}
