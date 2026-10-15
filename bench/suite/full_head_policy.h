#pragma once

// Archived engines predate draft_head/adapt_set. New CLI defaults must not change suite policy.
template <class Options>
void SetFullHeadPolicy(Options& o) {
    o.draft_blocks.clear();
    o.draft_dense = false;
    o.adapt = 0;
    o.draft_q = 0.f;
    o.draft_margin = 0.f;
    o.lookup = 0;
    o.selection_options = 0;
    o.last_head = 1;
    o.overlap = 0;
    if constexpr (requires { o.draft_head; }) o.draft_head = "full";
    if constexpr (requires { o.adapt_set; }) o.adapt_set = true;
    if constexpr (requires { o.head_auto; }) o.head_auto = 0;
}
