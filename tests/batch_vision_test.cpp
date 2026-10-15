// Batch serve runs the image encoder in slices inside prefill commands, so other sessions decode between them
// The mock counts slices and forwards; the GPU check of bitwise-equal embeddings is
// bench/vision_bench.hip --interleave.
#include "../engine/serve/batch_session.h"
#include <cassert>
#include <fstream>
#include <map>
#include <unistd.h>

struct MockSequence {
  std::vector<crucible::ImageInput> pending;
  int slice = 0;
  std::vector<uint8_t> rgb;  // verified pixels of pending.front(), read before its first slice
};

struct VisionEngine {
  int selected = 0, prefill_rows = 0;
  std::map<int, std::vector<int>> histories;
  std::map<int, MockSequence> sequences;
  std::vector<std::string> log;  // "bind <slot> <deferred>", "slice <slot>", "forward <slot>", "batch"
  static constexpr int kSlicesPerImage = 3;
  int Context() const { return 4096; }
  int PrefillRows() const { return prefill_rows; }
  void ConfigureServeStreams(int, uint64_t = 0, int = 0, int = 0) {}
  int StreamCapacity(int) const { return Context(); }
  void SelectStream(int slot) { selected = slot; }
  void Reset() { histories[selected].clear(); }
  bool HasVision() const { return true; }
  MockSequence& ActiveSequence() { return sequences[selected]; }
  double BindImages(MockSequence& state, const std::vector<crucible::ImageInput>& images, size_t cached,
                    bool deferred = false) {
    state = {};
    for (const size_t i : crucible::ImagesToEncode(images, cached)) state.pending.push_back(images[i]);
    log.push_back("bind " + std::to_string(selected) + " " + std::to_string(int(deferred)));
    if (!deferred) while (VisionSlicesLeft(state)) EncodeVisionSlice(state);
    return 0;
  }
  int VisionSlicesLeft(const MockSequence& state) const { return VisionSlicesBefore(state, size_t(-1)); }
  int VisionSlicesBefore(const MockSequence& state, size_t end) const {
    int left = 0;
    for (const auto& image : state.pending) if (size_t(image.offset) < end) left += kSlicesPerImage;
    return std::max(left - state.slice, 0);
  }
  void ReleaseImageRows(MockSequence&, size_t cursor) { log.push_back("release " + std::to_string(cursor)); }
  // As the engine: reads and checks the next image's file on the host, before any device work of the command.
  bool alloc_fails = false;  // the host pixel buffer cannot be allocated
  void PrepareVisionPixels(MockSequence& state) {
    if (state.pending.empty() || state.slice || !state.rgb.empty()) return;
    if (alloc_fails) throw std::bad_alloc();
    state.rgb = crucible::ReadImagePixels(state.pending.front());
    log.push_back("read " + std::to_string(selected));
  }
  double EncodeVisionSlice(MockSequence& state) {
    assert(!state.pending.empty());
    if (state.slice == 0) {
      if (state.rgb.empty()) PrepareVisionPixels(state);  // the serial path reads here
      state.rgb.clear();
    }
    log.push_back("slice " + std::to_string(selected));
    if (++state.slice == kSlicesPerImage) { state.pending.erase(state.pending.begin()); state.slice = 0; }
    return 0.001;
  }
  std::vector<int> Forward(const int* ids, int rows, int, int) {
    log.push_back("forward " + std::to_string(selected));
    std::vector<int> out;
    for (int row = 0; row < rows; ++row) { histories[selected].push_back(ids[row]); out.push_back(row % 16); }
    return out;
  }
  std::vector<int> Prefill(const int* ids, int rows, const std::string&, bool predict, crucible::PrefillBetween* = nullptr, int = 0) {
    const auto out = Forward(ids, rows, 0, rows);
    return predict ? std::vector<int>{out.back()} : std::vector<int>{};
  }
  std::vector<int> ForwardBatch(const std::vector<int>& ids, const std::vector<int>& slots) {
    log.push_back("batch");
    for (size_t row = 0; row < ids.size(); ++row) histories[slots[row]].push_back(ids[row]);
    return std::vector<int>(ids.size(), 1);
  }
  std::vector<int> ForwardVerifySpans(const std::vector<crucible::VerifySpan>&) { return {}; }
  std::vector<float> CopyLogits(int, int = 0) const { return std::vector<float>(16); }
};

static std::string WriteImage(const std::string& dir, const std::string& name, int height, int width) {
  std::vector<uint8_t> rgb(size_t(height) * width * 3, uint8_t(name.size()));
  const std::string path = dir + "/" + name;
  std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(rgb.data()), std::streamsize(rgb.size()));
  return path;
}

static crucible::ServeRequest ImageRequest(const std::string& dir, int images) {
  crucible::ServeRequest r;
  r.ids = {1, 2};
  for (int i = 0; i < images; ++i) {
    crucible::ImageInput image;
    image.offset = int(r.ids.size());
    image.height = image.width = 32;
    image.path = WriteImage(dir, "image" + std::to_string(i), 32, 32);
    const std::vector<uint8_t> rgb(32 * 32 * 3, uint8_t(("image" + std::to_string(i)).size()));
    image.digest = crucible::ImageDigest(32, 32, rgb.data(), rgb.size());
    r.ids.push_back(crucible::kImagePad);
    r.images.push_back(image);
  }
  r.ids.push_back(3);
  r.max_tokens = 3; r.spec = 0; r.stop_ids.clear();
  return r;
}

static crucible::ServeRequest TextRequest() {
  crucible::ServeRequest r;
  r.ids = {4, 5}; r.max_tokens = 20; r.spec = 0; r.stop_ids.clear();
  return r;
}

int main() {
  char pattern[] = "/tmp/crucible-bv-XXXXXX";
  const std::string dir = mkdtemp(pattern);

  VisionEngine engine;
  crucible::BatchSession<VisionEngine> session(engine, 2);
  // A text request decodes while the image request opens.
  const auto text = session.Open(1, TextRequest(), true);
  assert(!text.front().vision_slices);
  while (session.Prefill(1).empty()) {}
  engine.log.clear();

  // The open binds the images without an encoder pass and reports the slice count.
  const auto opened = session.Open(2, ImageRequest(dir, 2), true);
  assert(opened.size() == 1 && opened.front().vision_slices == 2 * VisionEngine::kSlicesPerImage);
  assert(engine.log == std::vector<std::string>{"bind 1 1"});

  // Each prefill command runs one slice and publishes nothing; a decode step of the peer fits between them.
  // The first slice of each image reads its file first.
  for (int slice = 0; slice < 2 * VisionEngine::kSlicesPerImage; ++slice) {
    engine.log.clear();
    assert(session.Prefill(2).empty());
    if (slice % VisionEngine::kSlicesPerImage == 0) assert((engine.log == std::vector<std::string>{"read 1", "slice 1"}));
    else assert(engine.log == std::vector<std::string>{"slice 1"});
    // A slice moves no prompt row, so the command names no prompt cursor (the server counts no rows).
    assert(!session.PrefillCursor());
    assert(session.Step({1}).size() == 1);
  }
  // Then the prompt chunk runs and publishes the first token.
  engine.log.clear();
  const auto first = session.Prefill(2);
  assert(first.size() == 1 && first.front().token);
  assert((engine.log == std::vector<std::string>{"forward 1", "release 5"}));
  assert(session.PrefillCursor() == 5);

  // Close in the middle of the encoder work: the next open of the slot rebinds, so no slice is left over.
  session.Close(2);
  session.Open(3, ImageRequest(dir, 1), true);
  assert(session.Prefill(3).empty());
  session.Close(3);
  engine.log.clear();
  const auto plain = session.Open(4, TextRequest(), true);
  assert(!plain.front().vision_slices);
  assert(!session.Prefill(4).empty());
  assert((engine.log == std::vector<std::string>{"bind 1 1", "forward 1", "release 2"}));
  session.Close(4);

  // A missing or wrong-size image file fails the open alone, before any state change.
  for (const bool missing : {true, false}) {
    auto bad = ImageRequest(dir, 2);
    if (missing) bad.images[1].path = dir + "/none";
    else bad.images[1].path = WriteImage(dir, "short", 32, 16);
    engine.log.clear();
    try { session.Open(5, bad, true); assert(false); }
    catch (const crucible::RecoverableOpenError& error) {
      assert(std::string(error.what()).find('/') == std::string::npos);
    }
    assert(engine.log.empty() && !session.Poisoned() && session.Active() == 1);
  }

  // The direct open (Open runs its own prefill loop) also runs every slice before the prompt.
  engine.log.clear();
  const auto direct = session.Open(6, ImageRequest(dir, 1));
  assert(!direct.empty() && direct.front().token);
  assert((engine.log == std::vector<std::string>{"bind 1 1", "read 1", "slice 1", "slice 1", "slice 1", "forward 1", "release 4"}));

  session.Close(6);

  // A chunk encodes only the images that it reads: with 8-row chunks, the image at row 2 is encoded before the
  // first chunk and the image at row 12 only before the second.
  auto spread = ImageRequest(dir, 2);
  spread.ids = {1, 2, crucible::kImagePad, 4, 5, 6, 7, 8, 9, 10, 11, 12, crucible::kImagePad, 14, 15, 16, 17, 18};
  spread.images[1].offset = 12;
  engine.log.clear();
  session.Open(7, spread, true);
  std::vector<crucible::BatchEvent> events;
  while (events.empty()) events = session.Prefill(7);
  assert((engine.log == std::vector<std::string>{"bind 1 1", "read 1", "slice 1", "slice 1", "slice 1", "forward 1",
                                                 "release 8", "read 1", "slice 1", "slice 1", "slice 1", "forward 1",
                                                 "release 16", "forward 1", "release 18"}));
  session.Close(7);

  // A file that changes after the open (other pixels, cut short, or removed) ends only its own request: one error
  // event, no device work, no poisoned session, and the peer keeps decoding (train14 review F1).
  for (int change = 0; change < 3; ++change) {
    auto later = ImageRequest(dir, 1);
    session.Open(8 + change, later, true);
    const std::string& path = later.images[0].path;
    if (change == 0) WriteImage(dir, "image0x", 32, 32), std::rename((dir + "/image0x").c_str(), path.c_str());
    if (change == 1) WriteImage(dir, "short", 32, 16), std::rename((dir + "/short").c_str(), path.c_str());
    if (change == 2) std::remove(path.c_str());
    engine.log.clear();
    const auto failed = session.Prefill(8 + change);
    assert(failed.size() == 1 && failed.front().request_id == uint64_t(8 + change) && failed.front().error &&
           failed.front().error->find('/') == std::string::npos && !failed.front().token && !failed.front().done);
    assert(engine.log.empty() && !session.Poisoned() && session.Active() == 1);
    // The command names no prompt cursor (the last chunk's cursor must not reach its batch_end).
    assert(!session.PrefillCursor());
    try { session.Prefill(8 + change); assert(false); } catch (const std::invalid_argument&) {}
    assert(session.Step({1}).size() == 1);
  }
  // A host buffer that cannot be allocated ends only its request, as a failed file check does.
  engine.alloc_fails = true;
  session.Open(20, ImageRequest(dir, 1), true);
  const auto starved = session.Prefill(20);
  assert(starved.size() == 1 && starved.front().error && !session.Poisoned() && session.Active() == 1);
  engine.alloc_fails = false;
  for (const char* name : {"image0", "image1", "short", "image0x"}) std::remove((dir + "/" + name).c_str());
  rmdir(dir.c_str());
  return 0;
}
