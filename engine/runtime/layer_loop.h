// The one trunk layer loop of every forward pass.
// Decode, batch, packed verification and prefill each give three callables: the pre-layer input (PLE
// before the layers where the model config says PleBefore), the layer body, and what runs after the
// layer (dumps, the prefill interleave hook). Do not add another loop over the trunk layers:
// tests/test_layer_loops.py allows exactly one, here.
// Host code only. The callables are inline lambdas, so the launches and their order do not change.
#pragma once

#include "../model/flashnext/geometry.h"

namespace crucible {

template <class Pre, class Body, class After>
inline void RunTrunk(Pre&& pre, Body&& body, After&& after) {
  for (int l = 0; l < kLayers; ++l) {
    if (PleBefore(l)) pre(l);
    body(l);
    after(l);
  }
}

}  // namespace crucible
