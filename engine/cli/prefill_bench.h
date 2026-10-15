// Included by engine.hip after the Engine class (needs Engine, ParseIds, Now).
#pragma once

// Loads the checkpoint once and runs prefill configurations back to back:
// crucible pfbench CKPT --prefill MAX RUN... with
// (rows up to MAX, or up to CRUCIBLE_PF_LAUNCH_ROWS when that is set)
// RUN = tag:ids_file:rows:moe_rm:moe_packed:in_rm:in_packed:out_rm:out_packed.
// Each run starts from Reset(); the first run of a row count also builds plans,
// so every run executes twice and reports the second time. Chunks before the last
// read no trunk stacks and the last chunk reads its final kDefaultMtpPrefix rows,
// as in serve with MTP; a tag that starts with "noprune" keeps every row (an
// in-process A/B of the last-layer row pruning).
int PrefillBench(int argc, char** argv) {
  const std::string ckpt = argv[2];
  if (argc < 6 || std::string(argv[3]) != "--prefill") return 2;
  const int maxr = std::stoi(argv[4]);
  if (maxr < 9 || maxr > kMaxPrefillRows) return 2;
  Engine eng(ckpt, 32768, maxr);
  const bool prof = crucible::SwitchText("CRUCIBLE_PROFILE") != nullptr;
  const char* out = crucible::SwitchText("CRUCIBLE_PF_OUT");
  // CRUCIBLE_PFBENCH_PASSES=1 times the first pass (ABAB pairs in one lease); the default 2 reports the second.
  const char* passes_env = crucible::SwitchText("CRUCIBLE_PFBENCH_PASSES");
  const int passes = passes_env && std::string(passes_env) == "1" ? 1 : 2;
  for (int a = 5; a < argc; ++a) {
    std::stringstream spec(argv[a]);
    std::string tag, file, f;
    std::vector<int> v;
    std::getline(spec, tag, ':'); std::getline(spec, file, ':');
    while (std::getline(spec, f, ':')) v.push_back(std::stoi(f));
    if (v.size() != 7 || v[0] < 9 || v[0] > eng.PrefillLaunchRows()) return 2;
    std::ifstream in(file);
    std::stringstream ss; ss << in.rdbuf();
    const std::vector<int> ids = ParseIds(ss.str());
    if (ids.empty() || !ValidTokenIds(ids, kVocab) || ids.size() > 32768) return 2;
    // moe_rm 0 selects the default shapes, which follow the chunk rows.
    const PrefillTiles tiles = v[1] == 0 ? PrefillTiles{} : PrefillTiles{true,
        {std::min(8, ParseRows(v[1])), v[2] != 0}, {ParseRows(v[3]), v[4] != 0},
        {ParseRows(v[5]), v[6] != 0}};
    eng.SetPrefillTiles(tiles);
    const bool keep_all = tag.rfind("noprune", 0) == 0;
    double tp = 0, start = 0; int top = -1;
    for (int pass = 0; pass < passes; ++pass) {
      eng.Reset();
      if (prof && pass == passes - 1) eng.EnableProfile();
      else eng.DisableProfile();
      HIP_OK(hipDeviceSynchronize());
      const double t0 = Now();
      start = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
      for (size_t i = 0; i < ids.size(); i += v[0]) {
        const int R = (int)std::min<size_t>(v[0], ids.size() - i);
        const bool final = i + R == ids.size();
        const int keep = keep_all ? 0 : final ? std::max(0, R - kDefaultMtpPrefix) : R;
        const auto last = eng.Prefill(ids.data() + i, R, "", final, nullptr, keep);
        if (!last.empty()) top = last[0];
      }
      HIP_OK(hipDeviceSynchronize());
      tp = Now() - t0;
    }
    if (out) QsaWriteCapture(std::string(out) + "/" + tag + ".logits.f32", eng.logits(), kVocab);
    std::printf("%s: %zu tok rows=%d moe=%d/%d in=%d/%d out=%d/%d top1=%d %.3f s %.1f tok/s start=%.3f\n",
                tag.c_str(), ids.size(), v[0], tiles.Moe(v[0]).rm, tiles.Moe(v[0]).packed,
                tiles.In(v[0]).rm, tiles.In(v[0]).packed, tiles.Out(v[0]).rm, tiles.Out(v[0]).packed,
                top, tp, ids.size() / tp, start);
    std::fflush(stdout);
    if (prof) { eng.Report(1); eng.ResetProfile(); }
  }
  return 0;
}
