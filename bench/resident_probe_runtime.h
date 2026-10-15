#pragma once

namespace crucible::residentprobe {
// Reset request state, not weights or plans. A failed request aborts the role.
template<class Engine, class Jobs, class Native, class Clipped>
void Run(Engine& engine, const Jobs& jobs, Native native, Clipped clipped) {
  for (const auto& job : jobs) {
    engine.Reset();
    native(engine,job);
    engine.Reset();
    clipped(engine,job);
  }
}
}  // namespace crucible::residentprobe
