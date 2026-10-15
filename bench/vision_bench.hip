// Vision encoder check without the trunk: loads only the tower, encodes raw RGB files, and writes
// bf16 embeddings for ref/vision_compare.py. Timing uses device events around each encoder pass.
//   vision_bench CKPT OUTDIR IMAGE.rgb:HEIGHT:WIDTH [...] [--repeat N] [--interleave]
// --interleave then encodes all images again in slices, round robin, each with its own residual buffer, and
// overwrites the shared GEMM workspaces between slices (as other sessions' forwards do in batch serve). Every
// embedding must equal its one-pass embedding bit for bit.
#define ENGINE_LIBRARY
#include "../engine/engine.hip"

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: vision_bench CKPT OUTDIR IMAGE.rgb:H:W [...] [--repeat N]\n");
    return 2;
  }
  int repeat = 3;
  bool interleave = false;
  std::vector<std::string> inputs;
  for (int i = 3; i < argc; ++i) {
    if (std::string(argv[i]) == "--repeat" && i + 1 < argc) repeat = std::atoi(argv[++i]);
    else if (std::string(argv[i]) == "--interleave") interleave = true;
    else inputs.push_back(argv[i]);
  }
  std::vector<ImageInput> images;
  std::vector<std::vector<uint8_t>> pixels;
  std::vector<std::vector<uint16_t>> results;
  // Workspaces sized like the 4096-row GEMM prefill engine.
  const size_t partial_cap = size_t(4096) * 16384, xh_cap = size_t(4096) * 10240;
  float* partial = Zeros<float>(partial_cap);
  __half* xh = Zeros<__half>(xh_cap);
  VisionEncoder encoder(argv[1], partial, partial_cap, xh, xh_cap);
  hipEvent_t a, b;
  HIP_OK(hipEventCreate(&a));
  HIP_OK(hipEventCreate(&b));
  for (const auto& spec : inputs) {
    const auto c1 = spec.rfind(':'), c0 = spec.rfind(':', c1 - 1);
    ImageInput image;
    const std::string path = spec.substr(0, c0);
    image.height = std::stoi(spec.substr(c0 + 1, c1 - c0 - 1));
    image.width = std::stoi(spec.substr(c1 + 1));
    std::ifstream f(path, std::ios::binary);
    std::vector<uint8_t> rgb(std::istreambuf_iterator<char>(f), {});
    if (rgb.size() != image.Bytes() || image.height % 32 || image.width % 32) {
      std::fprintf(stderr, "%s: bad size\n", path.c_str());
      return 2;
    }
    uint16_t* out = nullptr;
    const size_t rows = size_t(image.Tokens());
    HIP_OK(hipMalloc(&out, rows * kHidden * sizeof(uint16_t)));
    std::vector<float> ms;
    for (int r = 0; r < repeat; ++r) {
      HIP_OK(hipEventRecord(a, 0));
      encoder.Encode(image, rgb.data(), out);
      HIP_OK(hipEventRecord(b, 0));
      HIP_OK(hipEventSynchronize(b));
      float t = 0;
      HIP_OK(hipEventElapsedTime(&t, a, b));
      ms.push_back(t);
    }
    std::vector<uint16_t> host(rows * kHidden);
    HIP_OK(hipMemcpy(host.data(), out, host.size() * 2, hipMemcpyDeviceToHost));
    HIP_OK(hipFree(out));
    images.push_back(image);
    pixels.push_back(rgb);
    results.push_back(host);
    const std::string name = path.substr(path.rfind('/') + 1);
    std::ofstream o(std::string(argv[2]) + "/" + name + ".bf16", std::ios::binary);
    o.write(reinterpret_cast<const char*>(host.data()), host.size() * 2);
    const float first = ms.front();
    std::sort(ms.begin(), ms.end());
    std::printf("{\"image\":\"%s\",\"height\":%d,\"width\":%d,\"patches\":%d,\"tokens\":%zu,"
                "\"ms_first\":%.3f,\"ms_median\":%.3f,\"ms_min\":%.3f}\n",
                name.c_str(), image.height, image.width, image.GridH() * image.GridW(), rows,
                first, ms[ms.size() / 2], ms.front());
    std::fflush(stdout);
  }
  if (interleave) {
    std::vector<float*> x(images.size());
    std::vector<uint16_t*> out(images.size());
    std::vector<int> next(images.size(), 0);
    std::vector<float> longest(images.size(), 0);
    for (size_t i = 0; i < images.size(); ++i) {
      HIP_OK(hipMalloc(&x[i], VisionEncoder::ResidualFloats(images[i]) * sizeof(float)));
      HIP_OK(hipMalloc(&out[i], size_t(images[i].Tokens()) * kHidden * sizeof(uint16_t)));
    }
    int slices = 0;
    for (bool left = true; left;) {
      left = false;
      for (size_t i = 0; i < images.size(); ++i) {
        if (next[i] == VisionEncoder::Slices(images[i])) continue;
        HIP_OK(hipEventRecord(a, 0));
        encoder.EncodeSlice(images[i], next[i] == 0 ? pixels[i].data() : nullptr, x[i], next[i], out[i]);
        HIP_OK(hipEventRecord(b, 0));
        HIP_OK(hipEventSynchronize(b));
        float t = 0;
        HIP_OK(hipEventElapsedTime(&t, a, b));
        longest[i] = std::max(longest[i], t);
        ++next[i];
        ++slices;
        left = true;
        HIP_OK(hipMemsetAsync(partial, 0x5a, partial_cap * sizeof(float), 0));
        HIP_OK(hipMemsetAsync(xh, 0x3c, xh_cap * sizeof(__half), 0));
      }
    }
    bool equal = true;
    for (size_t i = 0; i < images.size(); ++i) {
      std::vector<uint16_t> host(results[i].size());
      HIP_OK(hipMemcpy(host.data(), out[i], host.size() * 2, hipMemcpyDeviceToHost));
      const bool same = host == results[i];
      equal = equal && same;
      std::printf("{\"interleave_image\":%zu,\"slices\":%d,\"longest_slice_ms\":%.3f,\"bitwise_equal\":%s}\n", i,
                  VisionEncoder::Slices(images[i]), longest[i], same ? "true" : "false");
      HIP_OK(hipFree(x[i]));
      HIP_OK(hipFree(out[i]));
    }
    std::printf("{\"interleave_slices\":%d,\"interleave_equal\":%s}\n", slices, equal ? "true" : "false");
    if (!equal) return 1;
  }
  std::fprintf(stderr, "vision scratch %.1f MiB, weights %.1f MiB, %zu plans kept\n",
               encoder.ScratchBytes() / 1048576.0, encoder.WeightBytes() / 1048576.0, encoder.PlanCount());
  return 0;
}
