// Prior for a draft's acceptance from the MTP top-2 logit margin. Batched serve and the CLI
// generation loop both size a draft chain with it. A proposal choice only: verification commits tokens.
#pragma once

// P(MTP draft accepted | its top-2 logit margin), pooled over 8 prompts at
// depth 7 with the 32K adaptive head (09:50).
inline float DraftAcceptProb(float margin) {
  static constexpr float edges[] = {1, 2, 3, 4, 6, 8, 12};
  static constexpr float p[] = {0.35f, 0.43f, 0.51f, 0.75f, 0.87f, 0.94f, 0.99f, 0.99f};
  int b = 0;
  while (b < 7 && margin >= edges[b]) ++b;
  return p[b];
}
constexpr float kDraftNextPrior = 0.75f;  // typical acceptance of a draft not yet computed
