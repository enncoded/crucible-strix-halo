#include <algorithm>
#include <array>
#include <cassert>
#include <climits>
#include <cstdio>

#include "../engine/runtime/bounds.h"

using namespace crucible;

int main() {
  constexpr int vocab = 248320;
  const std::array good{0, vocab - 1};
  const std::array negative{0, -1};
  const std::array high{vocab};
  assert(ValidTokenIds(good, vocab));
  assert(!ValidTokenIds(negative, vocab));
  assert(!ValidTokenIds(high, vocab));
  assert(ContextRowsFit(0, 8, 8, 8));
  assert(ContextRowsFit(INT_MAX - 1, 1, INT_MAX, 8));
  assert(!ContextRowsFit(INT_MAX, 1, INT_MAX, 8));
  assert(!ContextRowsFit(-1, 1, 8, 8));
  assert(!ContextRowsFit(0, 0, 8, 8));
  assert(!ContextRowsFit(0, 9, 16, 8));
  assert(RequestFitsContext(24, 9, 32));
  assert(!RequestFitsContext(24, 10, 32));
  assert(!RequestFitsContext(32, 2, 32));
  assert(!RequestFitsContext(24, 0, 32));
  assert(!RequestFitsContext(0, 1, 32));
  assert(!RequestFitsContext(24, 1, -1));
  assert(BoundedDraftDepth(2, 32, 30, 29, 1) == 1);
  assert(BoundedDraftDepth(2, 32, 31, 30, 1) == 0);
  assert(BoundedDraftDepth(2, 16, 8, 15, 1) == 1);
  assert(BoundedDraftDepth(2, 16, 8, 15, 2) == 0);
  for (int ctx = 1; ctx <= 64; ++ctx) {
    for (int trunk = 0; trunk <= ctx; ++trunk) {
      for (int mtp = 0; mtp <= ctx; ++mtp) {
        for (int rows = 1; rows <= 8; ++rows) {
          const int depth = BoundedDraftDepth(7, ctx, trunk, mtp, rows);
          assert(depth >= 0 && depth <= 7);
          if (depth == 0) continue;
          assert(trunk + depth + 1 <= ctx);
          assert(mtp + rows <= ctx);
          assert(mtp + rows + depth - 1 <= ctx);
        }
      }
    }
  }
  // the MTP cache may hold fewer positions than the trunk context.
  assert(BoundedDraftDepth(7, 196608, 1000, 33270, 3, 33280) == 7);
  assert(BoundedDraftDepth(7, 196608, 1000, 33275, 3, 33280) == 3);
  assert(BoundedDraftDepth(7, 196608, 1000, 33278, 3, 33280) == 0);
  assert(BoundedDraftDepth(7, 196608, 1000, 33278, 3, 196608) == 7);
  assert(!MtpRestart(33270, 3, 7, 33280));
  assert(MtpRestart(33275, 3, 7, 33280));
  assert(!MtpRestart(0, 3, 7, 4));       // a restart at position 0 gains nothing
  assert(MtpRestart(33280 - 7, 8, 1, 33280));
  assert(!MtpRestart(33280 - 8, 8, 1, 33280));
  for (int ctx = 1; ctx <= 48; ++ctx) {
    for (int cap = 1; cap <= ctx; ++cap) {
      for (int trunk = 0; trunk < ctx; ++trunk) {
        for (int mtp = 0; mtp <= cap; ++mtp) {
          for (int rows = 1; rows <= 8; ++rows) {
            const int at = MtpRestart(mtp, rows, 7, cap) ? 0 : mtp;
            const int depth = BoundedDraftDepth(7, ctx, trunk, at, rows, cap);
            assert(depth <= BoundedDraftDepth(7, ctx, trunk, at, rows, ctx));
            if (depth == 0) continue;
            assert(trunk + depth + 1 <= ctx);
            assert(at + rows + depth - 1 <= cap);
          }
          // After a restart, a step is limited only by the trunk and by a catch-up that the cache cannot hold.
          if (MtpRestart(mtp, 1, 7, cap))
            assert(BoundedDraftDepth(7, ctx, trunk, 0, 1, cap) == std::min({7, ctx - trunk - 1, cap}));
        }
      }
    }
  }
  assert(BoundedDraftDepth(2, 32, 30, 29, 1) == BoundedDraftDepth(2, 32, 30, 29, 1, 32));
  std::puts("PASS: token IDs, request bounds, row bounds, draft-depth bounds, and MTP restarts.");
}
