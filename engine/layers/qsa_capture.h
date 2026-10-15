// Opt-in reproducible QSA input snapshots. No capture on the normal fast path.
#pragma once
#include <fstream>
#include <string>
#include <vector>
#include "ops.h"
#include "qsa_capture_policy.h"
#include "../switches.h"

namespace crucible {

template<class T>
inline void QsaWriteCapture(const std::string& name, const T* device, size_t count) {
  std::vector<T> host(count);
  HIP_OK(hipMemcpy(host.data(),device,count*sizeof(T),hipMemcpyDeviceToHost));
  std::ofstream file(name,std::ios::binary);
  if (!file.write(reinterpret_cast<const char*>(host.data()),count*sizeof(T))) {
    std::fprintf(stderr,"cannot write QSA capture: %s\n",name.c_str());
    std::exit(1);
  }
}

// q_normed: a.q holds the normalized query (false in prefill).
inline void CaptureQsa(const QsaArgs& a, int layer, bool q_normed = true) {
  const char* dir = crucible::SwitchText("CRUCIBLE_QSA_CAPTURE_DIR");
  const char* at = crucible::SwitchText("CRUCIBLE_QSA_CAPTURE_AT");
  if (!dir || !at || a.pos0+a.R != std::atoi(at)) return;
  const int n = a.pos0+a.R;
  const std::string prefix = std::string(dir)+"/qsa_"+std::to_string(layer);
  const QsaCaptureKind mode = QsaCaptureMode(crucible::SwitchText("CRUCIBLE_QSA_CAPTURE_INDEX_ONLY"), q_normed);
  if (mode == QsaCaptureKind::kRefused) {
    std::fprintf(stderr,"QSA capture at %d: %s\n",n,kQsaFullCaptureRefusal);
    std::exit(2);
  }
  std::ofstream meta(prefix+".meta");
  if (!(meta << a.pos0 << ' ' << a.R << '\n')) {
    std::fprintf(stderr,"cannot write QSA metadata: %s\n",prefix.c_str());
    std::exit(1);
  }
  if (mode == QsaCaptureKind::kIndexOnly) {
    std::ofstream kind(prefix+".capture-kind");
    if (!(kind << "index-only\n")) {
      std::fprintf(stderr,"cannot label QSA capture: %s\n",prefix.c_str());
      std::exit(1);
    }
    QsaWriteCapture(prefix+".idx.f32",a.idx,(size_t)a.R*kQsaIdxWidth);
    QsaWriteCapture(prefix+".pool.f32",a.c.ik_pool,QsaPoolFloats(n));
    QsaWriteCapture(prefix+".sel.i32",a.sel,(size_t)a.R*512);
    QsaWriteCapture(prefix+".nsel.i32",a.nsel,a.R);
    return;
  }
  QsaWriteCapture(prefix+".q.f32",a.q,(size_t)a.R*kQsaQWidth);
  QsaWriteCapture(prefix+".idx.f32",a.idx,(size_t)a.R*kQsaIdxWidth);
  QsaWriteCapture(prefix+".pool.f32",a.c.ik_pool,QsaPoolFloats(n));
  QsaWriteCapture(prefix+".k.f16",a.c.k,(size_t)n*512);
  QsaWriteCapture(prefix+".v.f16",a.c.v,(size_t)n*512);
  QsaWriteCapture(prefix+".sel.i32",a.sel,(size_t)a.R*512);
  QsaWriteCapture(prefix+".nsel.i32",a.nsel,a.R);
  QsaWriteCapture(prefix+".suh.f16",a.suh_o,kQsaOutWidth);
}

}  // namespace crucible
