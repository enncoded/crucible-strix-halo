// Real-checkpoint codec gate. This reads selected tensors, not the full model.
#include <filesystem>
#include "../engine/model/flashnext/model.h"
#include "kernels/exl3_gemv.h"

using namespace crucible;

static void Write(const std::string& file, const void* data, size_t bytes) {
  std::ofstream out(file, std::ios::binary);
  if (!out.write(static_cast<const char*>(data), bytes)) std::exit(1);
}

template <int K, int Mode>
static std::vector<__half> Decode(const uint32_t* tiles, int A, int B) {
  __half* dst;
  const size_t count = (size_t)A * B * 256;
  HIP_OK(hipMalloc(&dst, count * 2));
  Exl3DecodeKernel<K, Mode><<<(A * B + 7) / 8, 256>>>(tiles, A, B, dst);
  HIP_OK(hipGetLastError());
  std::vector<__half> out(count);
  HIP_OK(hipMemcpy(out.data(), dst, count * 2, hipMemcpyDeviceToHost));
  HIP_OK(hipFree(dst));
  return out;
}

template <int K>
static void Linear(st::Checkpoint& ck, const std::string& name, const std::string& output) {
  const auto& ti = ck.Info(name + ".trellis");
  const int A = ti.shape[0], B = ti.shape[1], T = K * 8;
  if (ti.shape[2] != K * 16) std::abort();
  const uint32_t* src = reinterpret_cast<const uint32_t*>(ck.Data(name + ".trellis"));
  std::vector<uint32_t> host((size_t)A * B * T);
  for (int b = 0; b < B; ++b)
    for (int a = 0; a < A; ++a)
      std::memcpy(host.data() + ((size_t)b * A + a) * T,
                  src + ((size_t)a * B + b) * T, T * 4);
  uint32_t* tiles;
  HIP_OK(hipMalloc(&tiles, host.size() * 4));
  HIP_OK(hipMemcpy(tiles, host.data(), host.size() * 4, hipMemcpyHostToDevice));
  const auto ref = Decode<K, kMulLo>(tiles, A, B);
  const auto fast = Decode<K, kMulTrue16>(tiles, A, B);
  const auto u24 = Decode<K, kMulU24>(tiles, A, B);
  if (std::memcmp(ref.data(), fast.data(), ref.size() * 2) ||
      std::memcmp(ref.data(), u24.data(), ref.size() * 2)) std::abort();
  Write(output, ref.data(), ref.size() * 2);
  HIP_OK(hipFree(tiles));
  std::printf("K=%d %s: %zu values, three multiply modes agree\n", K, name.c_str(), ref.size());
}

static void CheckPle(st::Checkpoint& ck, const std::string& output) {
  const std::string p = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.";
  ck.AddFile("ngram_embedding.safetensors");
  const bool single = ck.Has(p + "trellis");
  const auto& info = ck.Info(p + (single ? "trellis" : "shard_0.trellis"));
  const int words = info.shape[1], K = (words - 1) / 10;
  if (K != 5 && K != 6) std::abort();
  constexpr int64_t count = 320001536;
  std::vector<uint16_t> packed(16 * words);
  std::vector<int64_t> rows(16);
  for (int h = 0; h < 16; ++h) {
    const int64_t row = h < 8 ? h : count - 16 + h;
    const int64_t local = single ? row : row % info.shape[0];
    const std::string name = p + (single ? "trellis" : "shard_" + std::to_string(row / info.shape[0]) + ".trellis");
    std::memcpy(packed.data() + h * words, ck.Data(name) + local * words * 2, words * 2);
    rows[h] = h;
  }
  uint16_t* table;
  int64_t* indices;
  __half* bias;
  float* out;
  HIP_OK(hipMalloc(&table, packed.size() * 2));
  HIP_OK(hipMalloc(&indices, rows.size() * 8));
  HIP_OK(hipMalloc(&bias, 2560 * 2));
  HIP_OK(hipMalloc(&out, 2560 * 4));
  HIP_OK(hipMemcpy(table, packed.data(), packed.size() * 2, hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(indices, rows.data(), rows.size() * 8, hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(bias, ck.Data(p + "head_bias"), 2560 * 2, hipMemcpyHostToDevice));
  if (K == 5) PleGatherKernel<5><<<16, 160>>>(table, indices, bias, out);
  else PleGatherKernel<6><<<16, 160>>>(table, indices, bias, out);
  HIP_OK(hipGetLastError());
  std::vector<float> result(2560);
  HIP_OK(hipMemcpy(result.data(), out, 2560 * 4, hipMemcpyDeviceToHost));
  Write(output, result.data(), 2560 * 4);
  HIP_OK(hipFree(table)); HIP_OK(hipFree(indices));
  HIP_OK(hipFree(bias)); HIP_OK(hipFree(out));
  std::printf("PLE K=%d: 16 rows, including both table boundaries\n", K);
}

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  st::Checkpoint ck(argv[1]);
  std::filesystem::create_directories(argv[2]);
  const std::string prefix[] = {"model.language_model.layers.0.linear_attn.in_proj_qkv",
                              "model.language_model.layers.0.mlp.experts.0.down_proj", "mtp.fc_embedding"};
  for (int i = 0; i < 3; ++i) {
    const std::string out = std::string(argv[2]) + "/linear" + std::to_string(i) + ".f16";
    const int K = ck.Info(prefix[i] + ".trellis").shape[2] / 16;
    if (K == 3) Linear<3>(ck, prefix[i], out);
    else if (K == 4) Linear<4>(ck, prefix[i], out);
    else if (K == 5) Linear<5>(ck, prefix[i], out);
    else if (K == 6) Linear<6>(ck, prefix[i], out);
    else return 1;
  }
  CheckPle(ck, std::string(argv[2]) + "/ple.f32");
}
