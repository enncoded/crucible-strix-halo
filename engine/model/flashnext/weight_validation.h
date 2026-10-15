// CPU-only validation must finish before device allocation or tensor conversion.
#pragma once
#include "../../exl3/safetensors.h"
#include "../../exl3/decode_support.h"
#include "geometry.h"

namespace crucible {

inline int ValidateLinear(st::Checkpoint& ck, const std::string& p, int I, int O) {
  const std::string name = p + ".trellis";
  const auto& ti = ck.Info(name);
  if (ti.shape.size() != 3 || ti.shape[2] % 16 != 0 ||
      ti.shape[2] < 0 || ti.shape[2] > 16 * 8 || !SupportedDecodeK(int(ti.shape[2] / 16))) {
    std::string supported;
    for (const int k : kSupportedDecodeK) supported += (supported.empty() ? "" : ",") + std::to_string(k);
    ck.Error(name, "supported K values are " + supported + "; fractional widths are not supported");
  }
  ck.Require(name, {"I16"}, {I / 16, O / 16, ti.shape[2]});
  ck.Require(p + ".suh", {"F16"}, {I});
  ck.Require(p + ".svh", {"F16"}, {O});
  ck.Require(p + ".mul1", {"I32"}, {});
  if (ck.Has(p + ".mcg")) ck.Error(p + ".mcg", "only the mul1 codebook is supported");
  uint32_t marker;
  ck.ReadBytes(p + ".mul1", &marker, sizeof(marker));
  if (marker != 0x83DCD12D) ck.Error(p + ".mul1", "unsupported mul1 codebook constant");
  return int(ti.shape[2] / 16);
}

inline void ValidateHc(st::Checkpoint& ck, const std::string& p, bool inject) {
  ck.Require(p + "hc_norm.weight", {"F16", "BF16"}, {kWide});
  ck.Require(p + "input_mix_weight_down.weight", {"F16"}, {kRank, kWide});
  ck.Require(p + "input_mix_weight_up.weight", {"F16"}, {kWide, kRank});
  if (inject) ck.Require(p + "block_inject_weight.weight", {"F16"}, {kStreams, kWide});
}

inline void ValidateLayer(st::Checkpoint& ck, const std::string& p, bool attn) {
  ValidateHc(ck, p + "attn_hyper_connection.", true);
  ValidateHc(ck, p + "mlp_hyper_connection.", true);
  if (attn) {
    const std::string a = p + "self_attn.";
    ValidateLinear(ck, a + "q_proj", kHidden, kQsaQWidth);
    ValidateLinear(ck, a + "k_proj", kHidden, kQsaKvWidth);
    ValidateLinear(ck, a + "v_proj", kHidden, kQsaKvWidth);
    ValidateLinear(ck, a + "indexer.index_qk_proj", kHidden, kQsaIdxWidth);
    ValidateLinear(ck, a + "o_proj", kQsaOutWidth, kHidden);
    ck.Require(a + "q_norm.weight", {"F16", "BF16"}, {kHeadDim});
    ck.Require(a + "k_norm.weight", {"F16", "BF16"}, {kHeadDim});
    ck.Require(a + "indexer.q_layernorm.weight", {"F16", "BF16"}, {kIdxDim});
    ck.Require(a + "indexer.k_layernorm.weight", {"F16", "BF16"}, {kIdxDim});
  } else {
    const std::string g = p + "linear_attn.";
    ValidateLinear(ck, g + "in_proj_qkv", kHidden, kGdnQkvWidth);
    ValidateLinear(ck, g + "in_proj_z", kHidden, kGdnValueWidth);
    ValidateLinear(ck, g + "out_proj", kGdnValueWidth, kHidden);
    ck.Require(g + "in_proj_a.weight", {"F16"}, {kGdnHeads, kHidden});
    ck.Require(g + "in_proj_b.weight", {"F16"}, {kGdnHeads, kHidden});
    ck.Require(g + "conv1d.weight", {"F16", "BF16"}, {kGdnQkvWidth, 1, 4});
    ck.Require(g + "A_log", {"F16", "BF16"}, {kGdnHeads});
    ck.Require(g + "dt_bias", {"F16", "BF16"}, {kGdnHeads});
    ck.Require(g + "norm.weight", {"F16", "BF16"}, {kGdnDim});
  }
  const std::string m = p + "mlp.";
  ck.Require(m + "gate.weight", {"F16"}, {kExperts, kHidden});
  ck.Require(m + "shared_expert_gate.weight", {"F16"}, {1, kHidden});
  // MoeFusedKernel<R, KR> (moe_rows.h) runs every routed projection at one
  // width KR (3 or 4) and every shared projection at KR + 2.
  const std::string e0 = m + "experts.0.gate_proj";
  const int kr = ValidateLinear(ck, e0, kHidden, kMoeInter);
  if (kr != 3 && kr != 4) ck.Error(e0 + ".trellis", "routed experts require K=3 or 4 on this engine");
  for (const std::string proj : {"gate_proj", "up_proj", "down_proj"}) {
    const bool down = proj == "down_proj";
    const std::string sp = m + "shared_expert." + proj;
    if (ValidateLinear(ck, sp, down ? kMoeInter : kHidden, down ? kHidden : kMoeInter) != kr + 2)
      ck.Error(sp + ".trellis", "the shared expert requires K = routed K + 2 on this engine");
  }
  for (int e = 0; e < kExperts; ++e) {
    const std::string ep = m + "experts." + std::to_string(e) + ".";
    for (const std::string proj : {"gate_proj", "up_proj", "down_proj"}) {
      const bool down = proj == "down_proj";
      if (ValidateLinear(ck, ep + proj, down ? kMoeInter : kHidden, down ? kHidden : kMoeInter) != kr)
        ck.Error(ep + proj + ".trellis", "all routed expert projections require one K on this engine");
    }
  }
}

inline void ValidatePle(st::Checkpoint& ck, const std::string& p) {
  ck.Require(p + "key_proj.weight", {"F16"}, {kWide, kHidden});
  ck.Require(p + "value_proj.weight", {"F16"}, {kHidden, kHidden});
  for (const std::string norm : {"norm_key", "norm_query", "norm_conv"})
    ck.Require(p + norm + ".weight", {"F16", "BF16"}, {kWide});
  ck.Require(p + "conv1d.weight", {"F16", "BF16"}, {kWide, 1, 4});
  const std::string n = p + "ple_embedding.ngram_embedding.";
  ck.Require(n + "layer_multipliers", {"I64"}, {3});
  ck.Require(n + "head_vocab_sizes", {"I64"}, {16});
  ck.Require(n + "head_offsets", {"I64"}, {16});
  ck.Require(n + "head_bias", {"F16"}, {16, 160});
  int64_t rows = 0;
  // The 4.05 pack stores one table tensor; the 3.05 pack stores shards.
  std::vector<std::string> tables;
  if (ck.Has(n + "trellis")) tables.push_back(n + "trellis");
  else
    for (int shard = 0; ck.Has(n + "shard_" + std::to_string(shard) + ".trellis"); ++shard)
      tables.push_back(n + "shard_" + std::to_string(shard) + ".trellis");
  for (const std::string& name : tables) {
    const auto& ti = ck.Info(name);
    // PleGatherKernel<K> uses a K-bit row codec with 10K+1 words (K=5 or 6),
    // not the linear trellis codec.
    if (ti.shape.size() != 2 || ti.shape[0] <= 0 || (ti.shape[1] != 51 && ti.shape[1] != 61))
      ck.Error(name, "PLE gather requires nonempty 5- or 6-bit rows with 51 or 61 words on this engine");
    ck.Require(name, {"I16"}, ti.shape);
    if (rows > std::numeric_limits<int64_t>::max() - ti.shape[0]) ck.Error(name, "PLE row count overflow");
    rows += ti.shape[0];
    if (uint64_t(rows) > (std::numeric_limits<size_t>::max() - 255) / 122)
      ck.Error(name, "PLE table byte count exceeds the host address space");
  }
  if (!rows) ck.Error(n + "shard_0.trellis", "required PLE table is missing");
  int64_t vocab[16], offsets[16];
  ck.ReadBytes(n + "head_vocab_sizes", vocab, sizeof(vocab));
  ck.ReadBytes(n + "head_offsets", offsets, sizeof(offsets));
  for (int h = 0; h < 16; ++h) {
    if (vocab[h] <= 0) ck.Error(n + "head_vocab_sizes", "PLE moduli must be positive");
    if (offsets[h] < 0 || offsets[h] > rows || vocab[h] > rows - offsets[h])
      ck.Error(n + "head_offsets", "PLE head range exceeds the packed table");
  }
}

inline void ValidateCheckpoint(st::Checkpoint& ck) {
  // About 37,000 marker reads: through the mapping each one read up to 4 MB, and the
  // arena evicted those pages before the load read them again (1.55x the checkpoint).
  st::NoMappedReads small_reads_only(ck);
  // The markers are independent 4-byte reads; read them all at once on several threads.
  ck.Prefetch(ck.NamesEndingWith(".mul1"));
  const std::string p = "model.language_model.";
  ck.Require(p + "embed_tokens.weight", {"BF16"}, {kVocab, kHidden});
  for (int l = 0; l < kLayers; ++l) ValidateLayer(ck, p + "layers." + std::to_string(l) + ".", IsQsa(l));
  ValidateLinear(ck, "mtp.fc_embedding", kHidden, kHidden);
  ValidateLinear(ck, "mtp.fc_hidden", kHidden, kHidden);
  ck.Require("mtp.pre_fc_norm_embedding.weight", {"F16", "BF16"}, {kHidden});
  ck.Require("mtp.pre_fc_norm_hidden.weight", {"F16", "BF16"}, {kWide});
  ValidateLayer(ck, "mtp.layers.0.", IsQsa(kMtpLayer));
  ValidateHc(ck, "mtp.hyper_connection_mixer.", false);
  ValidateHc(ck, p + "hyper_connection_mixer.", false);
  ValidateLinear(ck, "lm_head", kHidden, kVocab);
  ValidatePle(ck, p + kPleTensor);
}
}  // namespace crucible
