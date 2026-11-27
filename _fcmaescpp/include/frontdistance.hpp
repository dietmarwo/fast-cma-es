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
template<std::size_t D>
double squared_distance(const double* a, const double* b) {
    double distance = 0.;
    for (std::size_t j = 0; j < D; ++j) {
        const double delta = a[j] - b[j];
        distance += delta * delta;
    }
    return distance;
}

template<std::size_t D>
struct SearchTree {
    static constexpr std::size_t leaf_size = D == 3 ? 64 : 24;
    std::vector<std::array<double, D>> points;

    void build(std::size_t lo, std::size_t hi, unsigned axis) {
        if (hi - lo <= leaf_size)
            return;
        const auto mid = lo + (hi - lo) / 2;
        std::nth_element(points.begin() + lo, points.begin() + mid, points.begin() + hi,
                         [axis](const auto& a, const auto& b) { return a[axis] < b[axis]; });
        const auto next = (axis + 1) % D;
        build(lo, mid, next);
        build(mid + 1, hi, next);
    }

    template<bool Count = false>
    void search(std::size_t lo, std::size_t hi, unsigned axis, const double* r,
                double& best, std::size_t& tested) const {
        if (hi - lo <= leaf_size) {
            if constexpr (Count)
                tested += hi - lo;
            for (auto i = lo; i < hi; ++i)
                best = std::min(best, squared_distance<D>(r, points[i].data()));
            return;
        }
        if constexpr (Count)
            ++tested;
        const auto mid = lo + (hi - lo) / 2;
        best = std::min(best, squared_distance<D>(r, points[mid].data()));
        const auto delta = r[axis] - points[mid][axis];
        const auto next = (axis + 1) % D;
        if (delta < 0.) {
            search<Count>(lo, mid, next, r, best, tested);
            if (delta * delta < best)
                search<Count>(mid + 1, hi, next, r, best, tested);
        } else {
            search<Count>(mid + 1, hi, next, r, best, tested);
            if (delta * delta < best)
                search<Count>(lo, mid, next, r, best, tested);
        }
    }
};

template<std::size_t D>
double indexed_distance(const double* front, std::size_t nfront,
                        const double* reference, std::size_t nref) {
    SearchTree<D> tree;
    tree.points.resize(nfront);
    for (std::size_t i = 0; i < nfront; ++i)
        std::copy_n(front + D * i, D, tree.points[i].begin());
    tree.build(0, nfront, 0);
    const double infinity = std::numeric_limits<double>::infinity();
    bool use_tree = true;
    std::size_t tested = 0;
    if constexpr (D >= 4) {
        // A small sample detects data for which split-plane pruning visits
        // most of the tree, such as a distant reference or flat coordinates.
        constexpr std::size_t samples = 8;
        for (std::size_t i = 0; i < samples; ++i) {
            double nearest = infinity;
            tree.template search<true>(0, nfront, 0, reference + D * (i * (nref / samples)),
                                       nearest, tested);
        }
        use_tree = tested / samples < nfront / 2;
    }
    double sum = 0.;
    for (std::size_t i = 0; i < nref; ++i) {
        const auto* r = reference + D * i;
        double best = infinity;
        if (use_tree) {
            tree.search(0, nfront, 0, r, best, tested);
        } else {
            for (std::size_t p = 0; p < nfront; ++p)
                best = std::min(best, squared_distance<D>(r, front + D * p));
        }
        sum += std::sqrt(best);
    }
    return sum / double(nref);
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
        double high = 0., low = std::numeric_limits<double>::infinity();
        auto bounds = [&](const double* data, std::size_t count) {
            for (std::size_t i = 0; i < count; ++i) {
                double v=std::abs(data[i]);
                if (v != 0.)
                    low = std::min(low, v);
                high = std::max(high, v);
            }
        };
        bounds(front,nfront*dim);
        bounds(reference,nref*dim);
        // Scaling is exact and keeps nonzero coordinates in a safe range
        // when their exponents differ by at most 64.
        if (low >= std::scalbn(high, -64)) {
            int scale;
            std::frexp(high, &scale);
            std::vector<double> a(nfront * dim), b(nref * dim);
            for (std::size_t j = 0; j < a.size(); ++j)
                a[j] = std::scalbn(front[j], -scale);
            for (std::size_t j = 0; j < b.size(); ++j)
                b[j] = std::scalbn(reference[j], -scale);
            const double scaled_mean = igd(a.data(), nfront, b.data(), nref, dim);
            const double mean = std::scalbn(scaled_mean, scale);
            if (mean < std::numeric_limits<double>::max() * (1. - 1e-12))
                return mean;
        }
        const auto mean = detail::extreme_mean(reference, nref, front, nfront, dim, 0);
        if (std::isfinite(mean) || nref == 1)
            return mean;
        constexpr int shift = std::numeric_limits<std::size_t>::digits / 2 + 2;
        return detail::extreme_mean(reference, nref, front, nfront, dim, shift);
    }
    if (dim == 4 && nfront >= 256 && nref >= 128)
        return detail::indexed_distance<4>(front, nfront, reference, nref);
    if (dim == 8 && nfront >= 2048 && nref >= 256)
        return detail::indexed_distance<8>(front, nfront, reference, nref);
    double sum = 0;
    if (dim == 1 && nfront >= 32) {
        std::vector<double> axis(front, front+nfront);
        std::sort(axis.begin(),axis.end());
        for (std::size_t i=0; i<nref; ++i) {
            const auto pos=std::lower_bound(axis.begin(),axis.end(),reference[i]);
            double distance=pos==axis.end()?infinity:*pos-reference[i];
            if (pos!=axis.begin()) distance=std::min(distance,reference[i]-*std::prev(pos));
            sum += distance;
        }
    } else if (dim == 2 && nfront >= 40) {
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
        return detail::indexed_distance<3>(front, nfront, reference, nref);
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
