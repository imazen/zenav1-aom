// Reference oracle for the VarDCT-FP *quant-matrix* quantizer paths. The QM
// arithmetic lives in the two static helpers `quantize_fp_helper_c` /
// `highbd_quantize_fp_helper_c` (av1/encoder/av1_quantize.c), which are not
// exported. Rather than transcribe them, these wrappers reach them through the
// real exported facades (`av1_quantize_fp_facade` /
// `av1_highbd_quantize_fp_facade`), which route to the static helper whenever
// both qmatrix and iqmatrix are non-NULL. So this is the genuine C code path.
#include <stdint.h>
#include <string.h>
#include "av1/common/enums.h"
#include "av1/common/entropymode.h"  // SCAN_ORDER
#include "av1/encoder/block.h"       // MACROBLOCK_PLANE
#include "av1/encoder/av1_quantize.h"  // QUANT_PARAM + facade decls

// Fill the QTX tables the facade forwards. The FP path reads only
// round_fp_QTX / quant_fp_QTX / dequant_QTX (indices 0/1); zbin/quant_shift are
// (void)-cast inside the helper but must be non-NULL pointers.
static void fill_plane(MACROBLOCK_PLANE *p, const int16_t *round,
                       const int16_t *quant, const int16_t *dequant) {
  memset(p, 0, sizeof(*p));
  p->round_fp_QTX = round;
  p->quant_fp_QTX = quant;
  p->dequant_QTX = dequant;
  p->zbin_QTX = round;         // unused by the FP path
  p->quant_shift_QTX = quant;  // unused by the FP path
}

uint16_t shim_quantize_fp_qm(const tran_low_t *coeff, int n,
                             const int16_t *round, const int16_t *quant,
                             const int16_t *dequant, const int16_t *scan,
                             const int16_t *iscan, const qm_val_t *qm,
                             const qm_val_t *iqm, int log_scale,
                             tran_low_t *qcoeff, tran_low_t *dqcoeff) {
  MACROBLOCK_PLANE p;
  fill_plane(&p, round, quant, dequant);
  SCAN_ORDER sc = { scan, iscan };
  QUANT_PARAM qparam;
  memset(&qparam, 0, sizeof(qparam));
  qparam.log_scale = log_scale;
  qparam.qmatrix = qm;
  qparam.iqmatrix = iqm;
  uint16_t eob = 0;
  av1_quantize_fp_facade(coeff, (intptr_t)n, &p, qcoeff, dqcoeff, &eob, &sc,
                         &qparam);
  return eob;
}

uint16_t shim_highbd_quantize_fp_qm(const tran_low_t *coeff, int n,
                                    const int16_t *round, const int16_t *quant,
                                    const int16_t *dequant, const int16_t *scan,
                                    const int16_t *iscan, const qm_val_t *qm,
                                    const qm_val_t *iqm, int log_scale,
                                    tran_low_t *qcoeff, tran_low_t *dqcoeff) {
  MACROBLOCK_PLANE p;
  fill_plane(&p, round, quant, dequant);
  SCAN_ORDER sc = { scan, iscan };
  QUANT_PARAM qparam;
  memset(&qparam, 0, sizeof(qparam));
  qparam.log_scale = log_scale;
  qparam.qmatrix = qm;
  qparam.iqmatrix = iqm;
  uint16_t eob = 0;
  av1_highbd_quantize_fp_facade(coeff, (intptr_t)n, &p, qcoeff, dqcoeff, &eob,
                                &sc, &qparam);
  return eob;
}
