// Copyright (c) Dietmar Wolz. MIT license; see the repository LICENSE.
// Exact inverted generational distance for minimization fronts.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace fcmaes::front_distance {
namespace detail {
using Triple = std::array<double, 3>;
using Iter = std::vector<Triple>::iterator;

inline void build(Iter first, Iter last, unsigned axis) {
    if (last - first <= 20)
        return;
    const auto mid = first + (last - first) / 2;
    std::nth_element(first, mid, last,
                     [axis](const auto& a, const auto& b) { return a[axis] < b[axis]; });
    build(first, mid, (axis + 1) % 3);
    build(mid + 1, last, (axis + 1) % 3);
}
inline double squared3(const Triple& p, const double* r) {
    const double x = r[0] - p[0], y = r[1] - p[1], z = r[2] - p[2];
    return x * x + y * y + z * z;
}
inline void nearest(Iter first, Iter last, unsigned axis, const double* r, double& best) {
    if (last - first <= 20) {
        for (; first != last; ++first)
            best = std::min(best, squared3(*first, r));
        return;
    }
    const auto mid = first + (last - first) / 2;
    best = std::min(best, squared3(*mid, r));
    const auto dx = r[axis] - (*mid)[axis];
    const auto next = (axis + 1) % 3;
    if (dx < 0) {
        nearest(first, mid, next, r, best);
        if (dx * dx < best)
            nearest(mid + 1, last, next, r, best);
    } else {
        nearest(mid + 1, last, next, r, best);
        if (dx * dx < best)
            nearest(first, mid, next, r, best);
    }
}

// Scaling is needed only if even the nearest distance exceeds DBL_MAX. Its
// contribution to the mean can still be representable when nref > 1.
inline double extreme_mean(const double* reference, std::size_t nref, const double* front,
                           std::size_t nfront, std::size_t dim, int shift) {
    double mean = 0;
    for (std::size_t i = 0; i < nref; ++i) {
        double best = std::numeric_limits<double>::infinity();
        for (std::size_t j = 0; j < nfront; ++j) {
            double norm = 0;
            for (std::size_t k = 0; k < dim; ++k) {
                const auto a = reference[i * dim + k], b = front[j * dim + k];
                double delta = a - b;
                if (shift)
                    delta = std::isfinite(delta) ? std::scalbn(delta, -shift)
                                                 : std::scalbn(a, -shift) - std::scalbn(b, -shift);
                norm = std::hypot(norm, delta);
            }
            best = std::min(best, norm);
        }
        if (!std::isfinite(best))
            return best;
        mean += (best - mean) / double(i + 1);
    }
    return std::scalbn(mean, shift);
}
} // namespace detail

// Mean Euclidean distance from each reference row to its nearest front row.
// The approximation front comes first, as in fcmaes-rust's igd(front, reference).
inline double igd(const double* front, std::size_t nfront, const double* reference, std::size_t nref,
                  std::size_t dim) {
    if (!dim || !nref || !reference || (nfront && !front) || nref > SIZE_MAX / dim ||
        nfront > SIZE_MAX / dim)
        throw std::invalid_argument("IGD requires a nonempty reference and matching positive dimension");
    bool scaled = false;
    auto inspect = [&](const double* data, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) {
            const auto a = std::abs(data[i]);
            if (!std::isfinite(a))
                throw std::invalid_argument("IGD coordinates must be finite");
            scaled |= a > 1e120 || (a != 0. && a < 1e-120);
        }
    };
    inspect(reference, nref * dim);
    inspect(front, nfront * dim);
    const auto infinity = std::numeric_limits<double>::infinity();
    if (!nfront)
        return infinity;
    if (scaled) {
        const auto mean = detail::extreme_mean(reference, nref, front, nfront, dim, 0);
        if (std::isfinite(mean) || nref == 1)
            return mean;
        constexpr int shift = std::numeric_limits<std::size_t>::digits / 2 + 2;
        return detail::extreme_mean(reference, nref, front, nfront, dim, shift);
    }
    double sum = 0;
    if (dim == 2 && nfront >= 40) {
        using Pair = std::array<double, 2>;
        std::vector<Pair> points(nfront);
        for (std::size_t j = 0; j < nfront; ++j)
            points[j] = {front[2 * j], front[2 * j + 1]};
        std::sort(points.begin(), points.end());
        const bool monotone = std::is_sorted(points.begin(), points.end(),
                                             [](const Pair& a, const Pair& b) { return a[1] > b[1]; });
        for (std::size_t i = 0; i < nref; ++i) {
            const auto* r = reference + 2 * i;
            auto pivot = std::lower_bound(points.begin(), points.end(), r[0],
                                          [](const Pair& a, double x) { return a[0] < x; });
            double best = infinity;
            auto visit = [&](const Pair& p, bool left) {
                const auto x = r[0] - p[0], dx = x * x;
                if (dx >= best)
                    return false;
                const auto y = r[1] - p[1], dist = dx + y * y;
                if (monotone && dist >= best && (left ? y <= 0 : y >= 0))
                    return false;
                best = std::min(best, dist);
                return true;
            };
            for (auto it = pivot; it != points.begin();)
                if (!visit(*--it, true))
                    break;
            for (auto it = pivot; it != points.end(); ++it)
                if (!visit(*it, false))
                    break;
            sum += std::sqrt(best);
        }
    } else if (dim == 3 && nfront >= 128) {
        std::vector<detail::Triple> tree(nfront);
        for (std::size_t i = 0; i < nfront; ++i)
            std::copy_n(front + 3 * i, 3, tree[i].begin());
        detail::build(tree.begin(), tree.end(), 0);
        for (std::size_t i = 0; i < nref; ++i) {
            double best = infinity;
            detail::nearest(tree.begin(), tree.end(), 0, reference + 3 * i, best);
            sum += std::sqrt(best);
        }
    } else {
        for (std::size_t i = 0; i < nref; ++i) {
            double best = infinity;
            for (std::size_t j = 0; j < nfront; ++j) {
                double dist = 0;
                for (std::size_t k = 0; k < dim; ++k) {
                    double v = reference[i * dim + k] - front[j * dim + k];
                    dist += v * v;
                }
                best = std::min(best, dist);
            }
            sum += std::sqrt(best);
        }
    }
    return sum / double(nref);
}
} // namespace fcmaes::front_distance
