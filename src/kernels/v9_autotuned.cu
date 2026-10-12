// v9_autotuned.cu — not a new kernel: a dispatcher that runs the v7/v8 configuration measured
// fastest for the problem size.
//
// WHAT CHANGED vs v7 / v8
//   No new device code. The tuning study (bench --tune) measured 18 tile configurations of the
//   v7 and v8 templates at every size. scripts/gen_dispatch.py turns that CSV into
//   v9_dispatch_table.inc: one {size, config} entry per tuned size, choosing the config with
//   the best % of paired cuBLAS among those that passed the cuBLAS check at every size. At run
//   time v9 picks the entry whose size is closest, in log scale, to cbrt(M*N*K).
//
// WHY IT SHOULD BE FASTER
//   No single tile is best everywhere. A 128x128 tile leaves most of the 40 SMs idle at
//   N = 256 (a fraction of one wave), while small tiles waste reuse at N = 4096. This is the
//   same idea as cuBLAS's internal heuristics, with the table built from our own measurements.
//
// HONEST MEASUREMENT
//   The tuning run's winners are biased upward: the best of 18 noisy measurements is likely to
//   be a lucky one ("winner's curse"). So bench measures v9 again, in its own run, and only that
//   number is reported for v9.
//
// WHAT THE LIMIT IS NOW
//   The table only knows square sizes from the tuning run; other shapes get the nearest entry.
//   The kernels themselves are v7/v8, with their limits.

#include "kernels.h"

#include <cmath>
#include <cstring>

namespace {

struct Entry {
    int size;
    const char* config;
};

const Entry kTable[] = {
#include "v9_dispatch_table.inc"
};

const TuneConfig& find_config(const char* name) {
    for (const auto* list : {&v7_tune_configs(), &v8_tune_configs()})
        for (const TuneConfig& c : *list)
            if (std::strcmp(c.name, name) == 0) return c;
    std::fprintf(stderr, "v9: config %s from v9_dispatch_table.inc is not compiled in\n", name);
    std::exit(EXIT_FAILURE);
}

const TuneConfig& pick(int M, int N, int K) {
    const double s = std::cbrt((double)M * N * K);
    const Entry* best = &kTable[0];
    for (const Entry& e : kTable)
        if (std::fabs(std::log(e.size / s)) < std::fabs(std::log(best->size / s))) best = &e;
    return find_config(best->config);
}

}  // namespace

void launch_v9_autotuned(int M, int N, int K, const float* A, const float* B, float* C) {
    pick(M, N, K).launch(M, N, K, A, B, C);
}

// Resources of the configuration used at the largest tuned size.
KernelResources resources_v9_autotuned() {
    const Entry& last = kTable[sizeof(kTable) / sizeof(kTable[0]) - 1];
    return find_config(last.config).resources();
}
