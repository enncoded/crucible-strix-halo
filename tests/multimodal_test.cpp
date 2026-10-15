#include "../engine/model/flashnext/multimodal.h"
#include <fstream>
#include <unistd.h>
#include <cassert>
#include <cstdlib>
#include <dirent.h>
#include <functional>
#include <new>
#include <string>
using namespace crucible;

// A test switch: the next allocation of at least this many bytes throws std::bad_alloc (0 = off).
static size_t fail_allocation_from = 0;
void* operator new(size_t size) {
  if (fail_allocation_from && size >= fail_allocation_from) {
    fail_allocation_from = 0;
    throw std::bad_alloc();
  }
  if (void* p = std::malloc(size ? size : 1)) return p;
  throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, size_t) noexcept { std::free(p); }

static int OpenFiles() {
  int count = 0;
  DIR* dir = opendir("/proc/self/fd");
  assert(dir);
  while (readdir(dir)) ++count;
  closedir(dir);
  return count;
}

static ImageInput Image(int offset, int height, int width, char fill = 'a') {
  ImageInput image;
  image.offset = offset; image.height = height; image.width = width;
  image.digest = std::string(64, fill);
  image.path = "/dev/shm/unused";
  return image;
}

static void Sha256Vectors() {
  auto hex = [](const std::string& text) {
    Sha256 h;
    h.Update(reinterpret_cast<const uint8_t*>(text.data()), text.size());
    return h.HexDigest();
  };
  assert(hex("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  assert(hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  assert(hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
         "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
  // Split updates across block edges give the same digest.
  const std::string million(1000000, 'a');
  Sha256 split;
  for (size_t at = 0, step = 1; at < million.size(); at += step, step = step * 3 % 97 + 1)
    split.Update(reinterpret_cast<const uint8_t*>(million.data()) + at, std::min(step, million.size() - at));
  assert(split.HexDigest() == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

// server/vision.py image_digest: sha256 of "<II" (height, width) then the pixels.
static void DigestMatchesServer() {
  const std::vector<uint8_t> rgb{1, 2, 3, 4, 5, 6};
  // hashlib.sha256(struct.pack("<II", 1, 2) + bytes([1, 2, 3, 4, 5, 6])).hexdigest()
  assert(ImageDigest(1, 2, rgb.data(), rgb.size()) ==
         "4cb90cb51e66702075ecb4cf1fd9fa4fd15ce37dc64dc656143415789fcebafd");
}

static std::string TempDir() {
  char pattern[] = "/tmp/crucible-mm-XXXXXX";
  const char* dir = mkdtemp(pattern);
  assert(dir);
  return dir;
}

static void ImageFiles() {
  const std::string dir = TempDir();
  std::vector<uint8_t> rgb(32 * 64 * 3);
  for (size_t i = 0; i < rgb.size(); ++i) rgb[i] = uint8_t(i * 7 + 3);
  const std::string path = dir + "/a.rgb";
  { std::ofstream out(path, std::ios::binary); out.write(reinterpret_cast<const char*>(rgb.data()), rgb.size()); }
  ImageInput image = Image(0, 32, 64);
  image.path = path;
  image.digest = ImageDigest(32, 64, rgb.data(), rgb.size());
  CheckImageFile(image);
  assert(ReadImagePixels(image) == rgb);
  auto fails = [](const ImageInput& bad, bool stat_too) {
    bool read = false, stat = false;
    try { ReadImagePixels(bad); } catch (const std::runtime_error& error) {
      read = true;
      // The error text never names the file.
      assert(std::string(error.what()).find('/') == std::string::npos);
    }
    try { CheckImageFile(bad); } catch (const std::runtime_error&) { stat = true; }
    return read && stat == stat_too;
  };
  ImageInput wrong_digest = image;
  wrong_digest.digest[0] = wrong_digest.digest[0] == 'a' ? 'b' : 'a';
  assert(fails(wrong_digest, false));  // the digest is checked at read time only
  ImageInput missing = image;
  missing.path = dir + "/missing.rgb";
  assert(fails(missing, true));
  ImageInput shape = image;
  shape.height = 64;
  assert(fails(shape, true));
  const std::string link = dir + "/link.rgb";
  assert(symlink(path.c_str(), link.c_str()) == 0);
  ImageInput linked = image;
  linked.path = link;
  assert(fails(linked, true));
  ImageInput directory = image;
  directory.path = dir;
  assert(fails(directory, true));
  for (const auto& p : {link, path}) std::remove(p.c_str());
  rmdir(dir.c_str());
}

static void ParseAndCheck() {
  const std::string line = R"([{"offset":2,"height":32,"width":64,"digest":")" + std::string(64, 'f') +
                           R"(","path":"/dev/shm/crucible-vision-1-x/0a"}])";
  st::Scanner scanner(line, "images", true);
  const auto images = ParseImages(scanner);
  assert(images.size() == 1 && images[0].Tokens() == 2 && images[0].path == "/dev/shm/crucible-vision-1-x/0a");
  const std::vector<int> ids{1, 2, kImagePad, kImagePad, 5};
  CheckImages(images, ids);
  // Inline pixels are gone: an rgb field is an unknown field.
  const std::string inline_rgb = R"([{"offset":2,"height":32,"width":64,"digest":")" + std::string(64, 'f') +
                                 R"(","rgb":"AAAA"}])";
  bool threw = false;
  try { st::Scanner old(inline_rgb, "images", true); ParseImages(old); } catch (const std::runtime_error&) { threw = true; }
  assert(threw);
  auto rejects = [&](std::function<void(std::vector<ImageInput>&, std::vector<int>&)> edit) {
    auto copy = images;
    auto prompt = ids;
    edit(copy, prompt);
    try { CheckImages(copy, prompt); } catch (const std::runtime_error&) { return true; }
    return false;
  };
  assert(rejects([](auto& im, auto&) { im[0].height = 48; }));
  assert(rejects([](auto& im, auto&) { im[0].path = "relative/a.rgb"; }));
  assert(rejects([](auto& im, auto&) { im[0].path = ""; }));
  assert(rejects([](auto& im, auto&) { im[0].path = "/a/" + std::string(4096, 'x'); }));
  assert(rejects([](auto& im, auto&) { im[0].digest[3] = 'x'; }));
  assert(rejects([](auto& im, auto&) { im[0].offset = 4; }));
  assert(rejects([](auto&, auto& p) { p[3] = 7; }));
  assert(rejects([](auto& im, auto&) { im.push_back(im[0]); }));
  // The encoder limit: 4096 merged tokens. 2048x2048 passes; one more row of cells fails.
  auto sized = [&](int height, int width) {
    auto copy = images;
    copy[0].height = height;
    copy[0].width = width;
    std::vector<int> prompt(2 + copy[0].Tokens(), kImagePad);
    try { CheckImages(copy, prompt); } catch (const std::runtime_error&) { return false; }
    return true;
  };
  assert(sized(2048, 2048));
  assert(!sized(2048, 2080));
  // The server flag allows 256 images per request, so the engine accepts 256.
  std::string many = "[";
  for (int i = 0; i < 257; ++i)
    many += std::string(i ? "," : "") + R"({"offset":)" + std::to_string(i) + R"(,"height":32,"width":32,"digest":")" +
            std::string(64, 'a') + R"(","path":"/p"})";
  // Scanner keeps a string_view, so each line lives in a named string.
  const std::string at_limit = many.substr(0, many.rfind(",{")) + "]", over_limit = many + "]";
  st::Scanner most(at_limit, "images", true);
  assert(ParseImages(most).size() == 256);
  threw = false;
  try { st::Scanner over(over_limit, "images", true); ParseImages(over); } catch (const std::runtime_error&) { threw = true; }
  assert(threw);
}

static void MropeMatchesReference() {
  // ref/test_vision_ref.py test_mrope_one_image: grid (1, 6, 10) is 3x5 merged cells at s = 3.
  std::vector<ImageInput> images{Image(3, 96, 160)};
  const auto table = MropeTable(images, 24);
  for (int i = 0; i < 3; ++i) assert(table[i * 4] == i && table[i * 4 + 1] == i && table[i * 4 + 2] == i);
  for (int k = 0; k < 15; ++k) {
    const int* p = &table[(3 + k) * 4];
    assert(p[0] == 3 && p[1] == 3 + k / 5 && p[2] == 3 + k % 5 && p[3] == 0);
  }
  for (int i = 18; i < 24; ++i)
    for (int a = 0; a < 3; ++a) assert(table[i * 4 + a] == 8 + (i - 18));
  // Text only: every axis equals the token index, so RoPE matches the 1D path.
  const auto plain = MropeTable({}, 10);
  for (int i = 0; i < 10; ++i) assert(plain[i * 4] == i && plain[i * 4 + 2] == i);
  // Two adjacent images: (1,4,4) -> 2x2 cells, then (1,2,4) -> 1x2 cells, as the Python test.
  std::vector<ImageInput> two{Image(0, 64, 64), Image(4, 32, 64)};
  const auto t2 = MropeTable(two, 7);
  assert(t2[4 * 4] == 2 && t2[4 * 4 + 1] == 2 && t2[4 * 4 + 2] == 2);
  assert(t2[6 * 4] == 4 && t2[6 * 4 + 1] == 4);
}

static void CacheKeysSeparateImages() {
  const std::vector<int> ids{1, kImagePad, kImagePad, 2};
  const std::vector<ImageInput> a{Image(1, 32, 64, 'a')}, b{Image(1, 32, 64, 'b')};
  const auto ka = CacheKey(ids, a), kb = CacheKey(ids, b), ka2 = CacheKey(ids, a);
  assert(ka == ka2 && ka != kb);
  assert(ka[0] == 1 && ka[3] == 2 && ka[1] >= kImageKeyBase && ka[1] != ka[2]);
  assert(CacheKey(ids, {}) == ids);
  assert(ImagesToEncode(a, 1) == std::vector<size_t>{0});
  assert(ImagesToEncode(a, 3).empty());
}

static void VisionSlotBytesBound() {
  constexpr uint64_t residual = uint64_t(16384) * 1152 * 4, row = 2560 * 2;
  // A 4096-row chunk can hold embeddings of 4096 rows plus one 4096-token image on each side, less one row each.
  assert(VisionSlotBytes(102400, 4096, 2560) == residual + (4096 + 2 * 4095) * row);
  // A short slot never holds more rows than its context.
  assert(VisionSlotBytes(8192, 4096, 2560) == residual + 8192 * row);
}

// a pixel buffer that cannot be allocated must not leak the open file.
static void PixelBufferFailureClosesTheFile() {
  char pattern[] = "/tmp/crucible-mm-fd-XXXXXX";
  const std::string dir = mkdtemp(pattern);
  std::vector<uint8_t> rgb(64 * 64 * 3, 5);
  const std::string path = dir + "/a.rgb";
  { std::ofstream out(path, std::ios::binary); out.write(reinterpret_cast<const char*>(rgb.data()), rgb.size()); }
  ImageInput image = Image(0, 64, 64);
  image.path = path;
  image.digest = ImageDigest(64, 64, rgb.data(), rgb.size());
  const int before = OpenFiles();
  for (int i = 0; i < 3; ++i) {
    fail_allocation_from = rgb.size();
    bool threw = false;
    try { ReadImagePixels(image); } catch (const std::bad_alloc&) { threw = true; }
    assert(threw && !fail_allocation_from);
  }
  assert(OpenFiles() == before);
  assert(ReadImagePixels(image) == rgb && OpenFiles() == before);
  std::remove(path.c_str());
  rmdir(dir.c_str());
}

// a merged bulk launch (CRUCIBLE_PF_LAUNCH_ROWS above the grid) reads the embeddings of every
// image in the launch, so the admission reserve counts the launch rows, not the grid.
static void ReserveRowsFollowTheLaunch() {
  assert(VisionReserveRows(4096, 4096) == 4096);
  assert(VisionReserveRows(4096, 16384) == 16384);
  assert(VisionReserveRows(0, 0) == 8);  // small-row prefill: kMaxRows rows a chunk
  assert(VisionSlotBytes(102400, VisionReserveRows(4096, 16384), 2560) -
         VisionSlotBytes(102400, VisionReserveRows(4096, 4096), 2560) == uint64_t(12288) * 2560 * 2);
}

int main() {
  PixelBufferFailureClosesTheFile();
  ReserveRowsFollowTheLaunch();
  VisionSlotBytesBound();
  Sha256Vectors();
  DigestMatchesServer();
  ImageFiles();
  ParseAndCheck();
  MropeMatchesReference();
  CacheKeysSeparateImages();
  return 0;
}
