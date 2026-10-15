#include "../engine/layers/ablation.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using crucible::kHidden;
using crucible::kStreams;

namespace {

struct Tensor {
  std::string name, dtype = "F32";
  std::vector<int64_t> shape{kHidden};
  std::vector<float> values;
};

std::vector<uint8_t> Safetensors(const std::vector<Tensor>& tensors, const std::string& extra = "") {
  std::string header = "{\"__metadata__\":{\"tool\":\"test\"}";
  std::vector<uint8_t> payload;
  for (const Tensor& t : tensors) {
    const size_t begin = payload.size();
    const auto* p = reinterpret_cast<const uint8_t*>(t.values.data());
    payload.insert(payload.end(), p, p + t.values.size() * sizeof(float));
    std::string shape;
    for (const int64_t d : t.shape) shape += (shape.empty() ? "" : ",") + std::to_string(d);
    header += ",\"" + t.name + "\":{\"dtype\":\"" + t.dtype + "\",\"shape\":[" + shape +
              "],\"data_offsets\":[" + std::to_string(begin) + "," + std::to_string(payload.size()) + "]}";
  }
  header += extra + "}";
  std::vector<uint8_t> out(8);
  for (int i = 0; i < 8; ++i) out[i] = uint8_t(uint64_t(header.size()) >> (8 * i));
  out.insert(out.end(), header.begin(), header.end());
  out.insert(out.end(), payload.begin(), payload.end());
  return out;
}

std::vector<float> Unit(int seed) {
  std::vector<float> v(kHidden);
  double n = 0;
  for (int i = 0; i < kHidden; ++i) {
    v[i] = std::sin(0.37 * (i + 1) * (seed + 1)) + 0.01f * seed;
    n += double(v[i]) * v[i];
  }
  for (float& x : v) x = float(x / std::sqrt(n));
  return v;
}

std::vector<float> Scaled(const std::vector<float>& v, float s) {
  std::vector<float> o(v);
  for (float& x : o) x *= s;
  return o;
}

std::string Key(int layer, const char* site, char ab) {
  return "model.language_model.layers." + std::to_string(layer) + ".abl_" + site + "." + ab;
}

std::vector<Tensor> Pair(int layer, const char* site, float lambda) {
  const std::vector<float> a = Unit(layer * 2 + (site[0] == 'm'));
  return {{Key(layer, site, 'a'), "F32", {kHidden}, a}, {Key(layer, site, 'b'), "F32", {kHidden}, Scaled(a, lambda)}};
}

void Rejects(const std::vector<uint8_t>& bytes, const char* needle) {
  try {
    crucible::ParseAblation(bytes);
  } catch (const std::invalid_argument& e) {
    if (!std::strstr(e.what(), needle)) {
      std::fprintf(stderr, "message '%s' lacks '%s'\n", e.what(), needle);
      assert(false);
    }
    return;
  }
  std::fprintf(stderr, "accepted a file that should fail with '%s'\n", needle);
  assert(false);
}

void TestSha256() {
  const std::string abc = "abc";
  assert(crucible::Sha256Hex(reinterpret_cast<const uint8_t*>(abc.data()), abc.size()) ==
         "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  assert(crucible::Sha256Hex(nullptr, 0) ==
         "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  // 1,000,000 times 'a' crosses many block boundaries and the length padding.
  const std::string million(1000000, 'a');
  assert(crucible::Sha256Hex(reinterpret_cast<const uint8_t*>(million.data()), million.size()) ==
         "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
  const std::string two_blocks = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
  assert(crucible::Sha256Hex(reinterpret_cast<const uint8_t*>(two_blocks.data()), two_blocks.size()) ==
         "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
}

void TestParse() {
  std::vector<Tensor> t = Pair(30, "attn", 2.9f);
  for (const Tensor& x : Pair(17, "mlp", 0.8f)) t.push_back(x);
  const crucible::AblationSet set = crucible::ParseAblation(Safetensors(t));
  assert(set.sites.size() == 2);
  // Sites are sorted by layer, then attention before MoE.
  assert(set.sites[0].layer == 17 && set.sites[0].mlp);
  assert(set.sites[1].layer == 30 && !set.sites[1].mlp);
  assert(set.sites[1].a == t[0].values && set.sites[1].b == t[1].values);

  Rejects(Safetensors({}), "no ablation vectors");
  Rejects(Safetensors({Pair(30, "attn", 2.f)[0]}), "needs both");
  Rejects(Safetensors({Pair(30, "attn", 2.f)[1]}), "needs both");
  auto bad_layer = Pair(48, "attn", 2.f);
  Rejects(Safetensors(bad_layer), "layer");
  auto mtp = Pair(3, "attn", 2.f);
  mtp[0].name = "mtp.layers.0.abl_attn.a";
  mtp[1].name = "mtp.layers.0.abl_attn.b";
  Rejects(Safetensors(mtp), "unexpected tensor");
  auto scale = Pair(30, "attn", 2.f);
  scale.push_back({Key(30, "attn", 's'), "F32", {kHidden}, std::vector<float>(kHidden, 1.f)});
  Rejects(Safetensors(scale), "unexpected tensor");
  auto f16 = Pair(30, "attn", 2.f);
  f16[0].dtype = "F16";
  f16[0].values.resize(kHidden / 2);
  Rejects(Safetensors(f16), "F32");
  auto shape = Pair(30, "attn", 2.f);
  shape[0].shape = {1, kHidden};
  Rejects(Safetensors(shape), "shape");
  auto nan = Pair(30, "attn", 2.f);
  nan[1].values[7] = std::nanf("");
  Rejects(Safetensors(nan), "finite");
  auto norm = Pair(30, "attn", 2.f);
  norm[0].values = Scaled(norm[0].values, 1.1f);
  Rejects(Safetensors(norm), "unit");
  auto dup = Pair(30, "attn", 2.f);
  Rejects(Safetensors(dup, ",\"" + Key(30, "attn", 'a') + "\":{\"dtype\":\"F32\",\"shape\":[2560],\"data_offsets\":[0,10240]}"),
          "duplicate");
  std::vector<uint8_t> truncated = Safetensors(Pair(30, "attn", 2.f));
  truncated.resize(truncated.size() - 4);
  Rejects(truncated, "payload");
  Rejects(std::vector<uint8_t>(4, 0), "short");
}

void TestPin() {
  assert(crucible::FindAblationPin(
             "05f77ae18082de58b31fb645f52ec5689dce22820add8bb81439e06f01998e5f") != nullptr);
  assert(crucible::FindAblationPin(std::string(64, '0')) == nullptr);
  const crucible::AblationPin& pin = crucible::kAblationPins[0];
  crucible::CheckAblationCheckpoint(pin, 3, 5);  // 3.05bpw_h5
  bool threw = false;
  try { crucible::CheckAblationCheckpoint(pin, 4, 6); } catch (const std::invalid_argument&) { threw = true; }
  assert(threw);
}

void TestReference() {
  const std::vector<float> a = Unit(3);
  std::vector<float> y(kHidden);
  for (int i = 0; i < kHidden; ++i) y[i] = std::cos(0.11f * i) + 0.5f * a[i];
  double exact = 0;
  for (int i = 0; i < kHidden; ++i) exact += double(y[i]) * a[i];
  const float c = crucible::AblationDotRef(y.data(), a.data());
  assert(std::fabs(c - exact) < 1e-5 * (1 + std::fabs(exact)));

  // lambda = 2 reflects the component along a; post gates scale the write.
  const std::vector<float> b = Scaled(a, 2.f);
  const float post[kStreams] = {1.f, 0.5f, 2.f, 0.f};
  std::vector<float> H(size_t(kStreams) * kHidden, 1.f);
  crucible::AblateWriteRef(y.data(), a.data(), b.data(), post, H.data());
  for (int s = 0; s < kStreams; ++s) {
    double along = 0, n = 0;
    for (int i = 0; i < kHidden; ++i) {
      const double d = (H[size_t(s) * kHidden + i] - 1.0) ;
      along += d * a[i];
      n += 1;
    }
    assert(std::fabs(along + post[s] * exact) < 1e-4 * (1 + std::fabs(exact)));
  }
  // A stream with gate 0 is unchanged bit for bit.
  for (int i = 0; i < kHidden; ++i) assert(H[3 * size_t(kHidden) + i] == 1.f);
}

}  // namespace

int main() {
  TestSha256();
  TestParse();
  TestPin();
  TestReference();
  std::puts("ablation_test: ok");
}
