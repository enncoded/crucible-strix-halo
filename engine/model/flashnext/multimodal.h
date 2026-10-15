// Host-side image request rules: protocol parsing, multimodal RoPE positions, and cache keys.
// No HIP here, so CPU tests cover every rule. Positions follow HF qwen4_exp get_rope_index
// (ref/vision_ref.py mrope_positions is the checked reference).
#pragma once
#include "../../exl3/safetensors.h"
#include "../../runtime/sha256.h"
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <set>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace crucible {
constexpr int kImagePad = 248056;
constexpr int kVisionUnit = 32;            // patch 16 x spatial merge 2
// The encoder merges at most 4096 tokens (VisionEncoder::kChunk), so 4096 * 32 * 32 pixels.
// preprocessor_config.json allows 16777216; another local engine and the server default stop at 2560 x 1440.
constexpr int kVisionMaxPixels = 4194304;
constexpr int kMaxImages = 256;  // the server's --vision-max-images limit
constexpr size_t kMaxImagePath = 4096;
// Cache keys of image tokens start here. Token IDs stay below 2^18, so text keys never collide.
constexpr int kImageKeyBase = 1 << 30;

struct ImageInput {
  int offset = 0, height = 0, width = 0;
  std::string digest;          // 64 hex characters: sha256 of shape and pixels (server/vision.py)
  // A private file of height * width * 3 RGB bytes. The server owns it and deletes it when the request ends;
  // the engine reads it only when the image needs an encoder pass.
  std::string path;
  int GridH() const { return height / 16; }
  int GridW() const { return width / 16; }
  int Tokens() const { return (height / kVisionUnit) * (width / kVisionUnit); }
  int End() const { return offset + Tokens(); }
  size_t Bytes() const { return size_t(height) * width * 3; }
};

// server/vision.py image_digest: sha256 of the little-endian uint32 pair (height, width), then the pixels.
inline std::string ImageDigest(int height, int width, const uint8_t* rgb, size_t size) {
  uint8_t shape[8];
  for (int i = 0; i < 4; ++i) {
    shape[i] = uint8_t(uint32_t(height) >> (8 * i));
    shape[4 + i] = uint8_t(uint32_t(width) >> (8 * i));
  }
  Sha256 hash;
  hash.Update(shape, sizeof(shape));
  hash.Update(rgb, size);
  return hash.HexDigest();
}

// Error texts never name the file: paths stay out of logs and client errors.
inline int OpenImageFile(const ImageInput& image) {
  const int fd = ::open(image.path.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
  if (fd < 0) throw std::runtime_error("an image file could not be opened");
  struct stat info {};
  if (::fstat(fd, &info) != 0 || !S_ISREG(info.st_mode) || size_t(info.st_size) != image.Bytes()) {
    ::close(fd);
    throw std::runtime_error("an image file is not a regular file of height * width * 3 bytes");
  }
  return fd;
}

// The text of a request that ends because its host pixel buffer could not be allocated (batch serve).
inline const char* ImageBufferError() { return "an image buffer could not be allocated"; }

// Before any state change: the file exists with the right size. The digest is checked when the pixels are read.
inline void CheckImageFile(const ImageInput& image) { ::close(OpenImageFile(image)); }

// Closes the file on every path, also when the pixel buffer cannot be allocated.
struct ImageFileGuard {
  int fd;
  ~ImageFileGuard() { ::close(fd); }
};

inline std::vector<uint8_t> ReadImagePixels(const ImageInput& image) {
  std::vector<uint8_t> rgb;
  size_t at = 0;
  {
    const ImageFileGuard file{OpenImageFile(image)};
    rgb.resize(image.Bytes());
    while (at < rgb.size()) {
      const ssize_t got = ::read(file.fd, rgb.data() + at, rgb.size() - at);
      if (got < 0 && errno == EINTR) continue;
      if (got <= 0) break;
      at += size_t(got);
    }
  }
  if (at != rgb.size()) throw std::runtime_error("an image file is shorter than height * width * 3 bytes");
  if (ImageDigest(image.height, image.width, rgb.data(), rgb.size()) != image.digest)
    throw std::runtime_error("an image file does not match its digest");
  return rgb;
}

inline ImageInput ParseImage(st::Scanner& scanner) {
  ImageInput image;
  std::set<std::string> fields;
  scanner.Expect('{');
  while (!scanner.Peek('}')) {
    const std::string key = scanner.Str();
    if (!fields.insert(key).second) throw std::runtime_error("duplicate image field: " + key);
    scanner.Expect(':');
    auto number = [&] {
      const int64_t value = scanner.Int();
      if (value < 0 || value > (int64_t(1) << 30)) throw std::runtime_error("image integer out of range");
      return int(value);
    };
    if (key == "offset") image.offset = number();
    else if (key == "height") image.height = number();
    else if (key == "width") image.width = number();
    else if (key == "digest") image.digest = scanner.Str();
    else if (key == "path") image.path = scanner.Str();
    else throw std::runtime_error("unknown image field");
    scanner.Separator('}');
  }
  scanner.Expect('}');
  if (fields.size() != 5) throw std::runtime_error("an image needs offset, height, width, digest, and path");
  return image;
}

inline std::vector<ImageInput> ParseImages(st::Scanner& scanner) {
  std::vector<ImageInput> images;
  scanner.Expect('[');
  while (!scanner.Peek(']')) {
    if (images.size() >= size_t(kMaxImages)) throw std::runtime_error("a request can contain at most 256 images");
    images.push_back(ParseImage(scanner));
    scanner.Separator(']');
  }
  scanner.Expect(']');
  return images;
}

// Every rule the device path relies on, checked before any state changes.
inline void CheckImages(std::span<const ImageInput> images, std::span<const int> ids) {
  int end = 0;
  for (const auto& image : images) {
    if (image.height < kVisionUnit || image.width < kVisionUnit || image.height % kVisionUnit ||
        image.width % kVisionUnit || int64_t(image.height) * image.width > kVisionMaxPixels)
      throw std::runtime_error("image sides must be multiples of 32 within 4194304 pixels");
    if (image.path.empty() || image.path.front() != '/' || image.path.size() > kMaxImagePath ||
        image.path.find('\0') != std::string::npos)
      throw std::runtime_error("image path must be absolute and at most 4096 bytes");
    if (image.digest.size() != 64 ||
        !std::all_of(image.digest.begin(), image.digest.end(), [](char c) { return std::isxdigit((unsigned char)c); }))
      throw std::runtime_error("image digest must be 64 hex characters");
    if (image.offset < end || size_t(image.End()) > ids.size())
      throw std::runtime_error("images must be ordered, disjoint, and inside the prompt");
    for (int i = image.offset; i < image.End(); ++i)
      if (ids[i] != kImagePad) throw std::runtime_error("every image token must be the image pad ID");
    end = image.End();
  }
}

// [capacity][4] = (T, H, W, 0). An image at s gives merged cell (r, c) the triple (s, s+r, s+c);
// the next token continues at s + max(rows, cols). Text and generated tokens get three equal values.
inline std::vector<int> MropeTable(std::span<const ImageInput> images, size_t capacity) {
  std::vector<int> table(capacity * 4, 0);
  int position = 0;
  size_t index = 0;
  auto text_until = [&](size_t end) {
    for (; index < end && index < capacity; ++index, ++position)
      for (int a = 0; a < 3; ++a) table[index * 4 + a] = position;
  };
  for (const auto& image : images) {
    text_until(size_t(image.offset));
    const int rows = image.height / kVisionUnit, cols = image.width / kVisionUnit;
    for (int k = 0; k < rows * cols && index < capacity; ++k, ++index) {
      table[index * 4] = position;
      table[index * 4 + 1] = position + k / cols;
      table[index * 4 + 2] = position + k % cols;
    }
    position += std::max(rows, cols);
  }
  text_until(capacity);
  return table;
}

// FNV-1a over the digest and the token index: 31 bits per image token.
inline int ImageTokenKey(const std::string& digest, int k) {
  uint64_t h = 1469598103934665603ull;
  auto mix = [&](uint8_t byte) { h = (h ^ byte) * 1099511628211ull; };
  for (const char c : digest) mix(uint8_t(c));
  for (int s = 0; s < 32; s += 8) mix(uint8_t(uint32_t(k) >> s));
  h ^= h >> 29;
  h *= 0xbf58476d1ce4e5b9ull;
  h ^= h >> 32;
  return kImageKeyBase + int(h & ((1u << 30) - 1));
}

// Conversation-cache key: text keys equal token IDs; image tokens carry their image identity.
// `ids` may be a prefix of the prompt (a snapshot inside an image) or extend past it (a generated tail).
inline std::vector<int> CacheKey(std::span<const int> ids, std::span<const ImageInput> images) {
  std::vector<int> key(ids.begin(), ids.end());
  for (const auto& image : images)
    for (int k = 0; k < image.Tokens() && size_t(image.offset + k) < key.size(); ++k)
      key[size_t(image.offset + k)] = ImageTokenKey(image.digest, k);
  return key;
}

// Device bytes that one batch slot can hold for images: the encoder residual of the largest image (patches x 1152
// floats) and the bf16 embeddings of the images that overlap one prompt chunk of `chunk` rows. Images are freed when
// the cursor passes them, so one image can reach back across the chunk start and one past its end.
inline uint64_t VisionSlotBytes(int slot_context, int chunk, int hidden) {
  constexpr uint64_t kMaxTokens = kVisionMaxPixels / (kVisionUnit * kVisionUnit);
  constexpr uint64_t kResidual = kMaxTokens * 4 * 1152 * sizeof(float);
  const uint64_t rows = std::min<uint64_t>(uint64_t(std::max(slot_context, 0)), uint64_t(chunk) + 2 * (kMaxTokens - 1));
  return kResidual + rows * uint64_t(hidden) * sizeof(uint16_t);
}

// The rows of the longest bulk launch, which bounds the images that one prefill command keeps: a merged launch
// (CRUCIBLE_PF_LAUNCH_ROWS above the grid) reads the embeddings of all its rows. 8 rows (kMaxRows) without bulk prefill.
inline int VisionReserveRows(int prefill_rows, int launch_rows) {
  return prefill_rows > 0 ? std::max(prefill_rows, launch_rows) : 8;
}

// Images whose tokens reach past `cached` prompt tokens, so their embeddings are needed.
inline std::vector<size_t> ImagesToEncode(std::span<const ImageInput> images, size_t cached) {
  std::vector<size_t> out;
  for (size_t i = 0; i < images.size(); ++i)
    if (size_t(images[i].End()) > cached) out.push_back(i);
  return out;
}
}  // namespace crucible
