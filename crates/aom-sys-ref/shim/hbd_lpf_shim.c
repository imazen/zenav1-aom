/* Shim over aom_highbd_lpf_{horizontal,vertical}_{4,6,8,14}_c. Oracle only. */
#include <stdint.h>
#define D(n) void aom_highbd_lpf_horizontal_##n##_c(uint16_t*,int,const uint8_t*,const uint8_t*,const uint8_t*,int); \
             void aom_highbd_lpf_vertical_##n##_c(uint16_t*,int,const uint8_t*,const uint8_t*,const uint8_t*,int);
D(4) D(6) D(8) D(14)
void shim_hbd_lpf(int dir, int width, uint16_t* s, int p, const uint8_t* bl,
                  const uint8_t* li, const uint8_t* th, int bd) {
  if (dir==0) { switch(width){
    case 4: aom_highbd_lpf_horizontal_4_c(s,p,bl,li,th,bd); return;
    case 6: aom_highbd_lpf_horizontal_6_c(s,p,bl,li,th,bd); return;
    case 8: aom_highbd_lpf_horizontal_8_c(s,p,bl,li,th,bd); return;
    default: aom_highbd_lpf_horizontal_14_c(s,p,bl,li,th,bd); return; }
  } else { switch(width){
    case 4: aom_highbd_lpf_vertical_4_c(s,p,bl,li,th,bd); return;
    case 6: aom_highbd_lpf_vertical_6_c(s,p,bl,li,th,bd); return;
    case 8: aom_highbd_lpf_vertical_8_c(s,p,bl,li,th,bd); return;
    default: aom_highbd_lpf_vertical_14_c(s,p,bl,li,th,bd); return; }
  }
}
