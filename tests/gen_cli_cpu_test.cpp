// The Python runner includes the real driver, without GPU CLI entry points.
#include "gen_cli_cpu_fixture.h"
#include "gen_cli_cpu.h"

// The prefill 16 and 64 cases must run the bulk path of RunPrompt, not the Forward fallback.
static_assert(HasGemmPrefill<Engine>);

int main() {
  for (double temperature : {0., 0.7}) for (int prefill : {0, 16, 64})
    for (int prompt : {1, 24, 65, 257, 3000}) for (int cap : {0, 8, 256}) {
      GenOptions options = CliGenOptions();
      options.n = 1; options.ctx = 4096; options.ids.assign(prompt, 1);
      options.sampling.temperature = temperature; options.mtp_prefix = cap;
      std::vector<int> emitted;
      options.emit = [&](int token) { emitted.push_back(token); };
      Engine engine(4096, prefill);
      const auto first = RunGen(engine, options, {});
      assert(first.gen.size() == 1 && engine.mtp_calls == 0 && first.gen == emitted);
      emitted.clear();
      const auto repeated = RunGen(engine, options, {});
      assert(first.gen == repeated.gen && engine.mtp_calls == 0 && repeated.gen == emitted);
      if (prompt > 1) {
        options.n = 2;
        const auto longer = RunGen(engine, options, {});
        assert(longer.gen.size() == 2 && engine.mtp_calls > 0);
      }
    }
  std::puts("PASS: actual greedy and sampled RunGen skip all MTP work for one-token requests");
}
