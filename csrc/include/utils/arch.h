#pragma once

#include <cstddef>

namespace astrai {

struct CudaTarget {
    int cc;
    char feature; // 'a': exact architecture, 'f': family, 0: generic PTX
};

constexpr bool supports(CudaTarget target, int cc) {
    if (target.cc <= 0 || cc < target.cc)
        return false;
    if (target.feature == 'a')
        return cc == target.cc;
    if (target.feature == 'f')
        return cc / 10 == target.cc / 10;
    return true;
}

template <std::size_t N> constexpr bool supports(const CudaTarget (&targets)[N], int cc) {
    for (CudaTarget target : targets)
        if (supports(target, cc))
            return true;
    return false;
}

} // namespace astrai
