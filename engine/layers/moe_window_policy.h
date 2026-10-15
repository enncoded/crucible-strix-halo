#pragma once

// Keep the original compile switch for A/B builds. Zero selects the old body.
#ifndef CRUCIBLE_MOE_WINDOW_PROBE
#define CRUCIBLE_MOE_WINDOW_PROBE 1
#endif
#if CRUCIBLE_MOE_WINDOW_PROBE != 0 && CRUCIBLE_MOE_WINDOW_PROBE != 1
#error "CRUCIBLE_MOE_WINDOW_PROBE must be zero or one"
#endif
