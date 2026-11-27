/**
 * @file rankbox.hpp
 * Exact closed-box queries over a fixed point set in component-wise rank space.
 *
 * Derived from two C++ programs that are not part of the published BiteOpt
 * repository (https://github.com/avaneev/biteopt): grid_point_index.cpp, the
 * sparse uniform grid in rank space (Copyright (c) 2026 Aleksey Vaneev), and
 * borda_box_query.cpp, the Borda rank-band selector (no author notice). The
 * fcmaes adaptations by Dietmar Wolz add input validation, a shared rank
 * construction with a deterministic tie order, and the library interface.
 *
 * Original code: Copyright (c) 2026 Aleksey Vaneev.
 * fcmaes adaptations: Copyright (c) 2026 Dietmar Wolz.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

namespace fcmaes::rankbox {

using Id = std::uint32_t;

namespace detail {

struct Ranks {
    using Order = std::vector<Id>;
    std::vector<std::vector<double>> columns;
    std::vector<Order> orders, by_id;

    // Monotone integer key of a finite coordinate; both zeros share one key.
    static std::uint64_t key(double v) {
        static_assert(sizeof(double) == sizeof(std::uint64_t) &&
                      std::numeric_limits<double>::is_iec559);
        std::uint64_t bits = 0;
        if (v != 0.)
            std::memcpy(&bits, &v, sizeof(bits));
        constexpr std::uint64_t sign = UINT64_C(1) << 63;
        return (bits & sign) ? ~bits : (bits ^ sign);
    }

    Ranks(const double* points, std::size_t n, std::size_t d) {
        if (!d || n > std::numeric_limits<Id>::max() || n > SIZE_MAX / d || (n && !points))
            throw std::invalid_argument("invalid point matrix dimensions");
        for (std::size_t i = 0; i < n * d; ++i)
            if (!std::isfinite(points[i]))
                throw std::invalid_argument("stored coordinates must be finite");
        columns.resize(d);
        orders.resize(d);
        by_id.resize(d);
        // Each axis is sorted by coordinate; equal coordinates, including -0.0
        // and 0.0, keep ascending row order. The order is therefore the same on
        // every platform and in the Rust indexes of fcmaes-rust.
        struct Item {
            std::uint64_t key;
            Id id;
        };
        std::vector<Item> entries(n), scratch(n);
        for (std::size_t j = 0; j < d; ++j) {
            auto& ids = orders[j];
            ids.resize(n);
            auto value = [&](Id id) { return points[std::size_t(id) * d + j]; };
            for (std::size_t i = 0; i < n; ++i)
                entries[i] = {key(value(static_cast<Id>(i))), static_cast<Id>(i)};
            if (n < 256) {
                std::stable_sort(entries.begin(), entries.end(),
                                 [](const Item& a, const Item& b) { return a.key < b.key; });
            } else {
                // Stable LSD radix sort on 11-bit digits.
                for (unsigned shift = 0; shift < 64; shift += 11) {
                    std::array<std::size_t, 2048> cursor{};
                    for (const auto& e : entries)
                        ++cursor[(e.key >> shift) & 2047];
                    if (std::find(cursor.begin(), cursor.end(), n) != cursor.end())
                        continue;
                    std::size_t off = 0;
                    for (auto& c : cursor) {
                        auto count = c;
                        c = off;
                        off += count;
                    }
                    for (const auto& e : entries)
                        scratch[cursor[(e.key >> shift) & 2047]++] = e;
                    entries.swap(scratch);
                }
            }
            for (std::size_t i = 0; i < n; ++i)
                ids[i] = entries[i].id;
            auto& ranks = by_id[j];
            auto& column = columns[j];
            ranks.resize(n);
            column.resize(n);
            for (std::size_t k = 0; k < n; ++k) {
                column[k] = value(ids[k]);
                ranks[ids[k]] = static_cast<Id>(k);
            }
        }
    }
};

struct Span {
    Id lo, width;
};
inline bool span(const std::vector<double>& col, double lo, double hi, Span& result) {
    if (!(lo <= hi))
        return false;
    auto a = std::lower_bound(col.begin(), col.end(), lo);
    auto b = std::upper_bound(a, col.end(), hi);
    if (a == b)
        return false;
    result = {static_cast<Id>(a - col.begin()), static_cast<Id>(b - a - 1)};
    return true;
}

} // namespace detail

struct BordaResult {
    std::vector<Id> ids;
    std::uint64_t axis_candidates = 0, score_candidates = 0;
};

class BordaBoxIndex {
    template <class Rank> using Layout = std::vector<std::vector<std::vector<Rank>>>;
    std::vector<std::vector<double>> columns_;
    std::vector<std::vector<Id>> ids_;
    std::vector<std::vector<std::uint64_t>> scores_;
    std::variant<Layout<std::uint16_t>, Layout<Id>> cross_;

    template <class Rank> void build_layout(const detail::Ranks& ranks) {
        auto& layout = cross_.emplace<Layout<Rank>>(dim());
        for (std::size_t a = 0; a < dim(); ++a) {
            layout[a].resize(dim());
            for (std::size_t j = 0; j < dim(); ++j) {
                if (j == a)
                    continue;
                auto& col = layout[a][j];
                col.reserve(ids_[a].size());
                for (Id id : ids_[a])
                    col.push_back(static_cast<Rank>(ranks.by_id[j][id]));
            }
        }
    }

public:
    BordaBoxIndex(const double* points, std::size_t n, std::size_t d) {
        detail::Ranks ranks(points, n, d);
        columns_ = std::move(ranks.columns);
        ids_ = std::move(ranks.orders);
        std::vector<std::uint64_t> sums(n);
        for (const auto& col : ranks.by_id)
            for (std::size_t i = 0; i < n; ++i)
                sums[i] += col[i];
        scores_.resize(d);
        for (std::size_t j = 0; j < d; ++j) {
            scores_[j].reserve(n);
            for (Id id : ids_[j])
                scores_[j].push_back(sums[id]);
        }
        if (n <= 65536)
            build_layout<std::uint16_t>(ranks);
        else
            build_layout<Id>(ranks);
    }

    std::size_t dim() const { return columns_.size(); }

    BordaResult query(const double* lo, const double* hi, std::vector<Id>* sink = nullptr) const {
        if (!lo || !hi)
            throw std::invalid_argument("box bounds must be non-null");
        BordaResult out;
        auto& found = sink ? *sink : out.ids;
        std::vector<detail::Span> spans(dim());
        std::size_t axis = 0;
        std::uint64_t low = 0, high = 0;
        for (std::size_t j = 0; j < dim(); ++j) {
            auto& s = spans[j];
            if (!detail::span(columns_[j], lo[j], hi[j], s))
                return out;
            if (s.width < spans[axis].width)
                axis = j;
            low += s.lo;
            high += std::uint64_t(s.lo) + s.width;
        }
        const auto& ids = ids_[axis];
        const std::size_t begin = spans[axis].lo, end = begin + spans[axis].width + 1;
        out.axis_candidates = end - begin;
        for (std::size_t i = begin; i < end; ++i)
            out.score_candidates += (scores_[axis][i] - low <= high - low);
        std::vector<std::size_t> active;
        for (std::size_t j = 0; j < dim(); ++j)
            if (j != axis && std::size_t(spans[j].width) + 1 != ids.size())
                active.push_back(j);
        if (active.empty()) {
            found.insert(found.end(), ids.begin() + begin, ids.begin() + end);
            return out;
        }
        std::visit(
            [&](const auto& layout) {
                for (std::size_t p = begin; p < end; p += 64) {
                    const auto count = std::min<std::size_t>(64, end - p);
                    std::array<unsigned char, 64> hit;
                    std::fill_n(hit.begin(), count, 1);
                    for (auto j : active) {
                        const auto* col = layout[axis][j].data() + p;
                        for (std::size_t i = 0; i < count; ++i)
                            hit[i] &= (Id(col[i]) - spans[j].lo <= spans[j].width);
                        if (std::none_of(hit.begin(), hit.begin() + count,
                                         [](unsigned char flag) { return flag != 0; }))
                            break;
                    }
                    for (std::size_t i = 0; i < count; ++i)
                        if (hit[i])
                            found.push_back(ids[p + i]);
                }
            },
            cross_);
        return out;
    }
};

struct SparseRankGrid {
    struct CellRange {
        std::uint32_t begin;
        std::uint32_t end;
    };

    std::uint32_t resolution = 0;
    std::vector<std::uint64_t> keys;
    std::vector<CellRange> ranges;
    std::vector<std::uint32_t> ids;

    SparseRankGrid() = default;

    template <typename RankColumns>
    SparseRankGrid(const RankColumns& ranks, std::size_t n, std::size_t d, std::uint32_t g) {
        resolution = g;
        if (!n)
            return;
        if (n > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) || g < 2 || !d)
            throw std::invalid_argument("invalid rank grid dimensions");

        std::uint64_t maxKey = 0;
        for (std::size_t j = 0; j < d; ++j) {
            if (maxKey > (std::numeric_limits<std::uint64_t>::max() - (g - 1)) / g)
                throw std::length_error("grid cell key exceeds 64 bits");
            maxKey = maxKey * g + (g - 1);
        }
        std::vector<std::uint32_t> rankCell(n);
        for (std::size_t r = 0; r < n; ++r)
            rankCell[r] = static_cast<std::uint32_t>((std::uint64_t(r) * g) / n);
        std::vector<std::uint64_t> pointKey(n, 0);
        for (std::size_t j = 0; j < d; ++j)
            for (std::size_t i = 0; i < n; ++i)
                pointKey[i] = pointKey[i] * g + rankCell[ranks[j][i]];

        ids.resize(n);
        // At most 1 MiB of persistent direct lookup, and proportional to n.
        if (maxKey < 262144 && maxKey < 8 * std::uint64_t(n) + 64) {
            const auto cells = static_cast<std::size_t>(maxKey + 1);
            std::vector<std::uint32_t> cursor(cells, 0);
            for (auto key : pointKey)
                ++cursor[key];
            firstOccupied.resize(cells);
            std::uint32_t offset = 0;
            for (std::size_t key = 0; key < cells; ++key) {
                const auto count = cursor[key];
                firstOccupied[key] = static_cast<std::uint32_t>(keys.size());
                cursor[key] = offset;
                if (count) {
                    keys.push_back(key);
                    ranges.push_back({offset, offset + count});
                    offset += count;
                }
            }
            for (std::size_t id = 0; id < n; ++id)
                ids[cursor[pointKey[id]]++] = static_cast<std::uint32_t>(id);
        } else {
            std::iota(ids.begin(), ids.end(), 0u);
            if (n < 256) {
                std::stable_sort(ids.begin(), ids.end(),
                    [&](Id a, Id b) { return pointKey[a] < pointKey[b]; });
            } else {
                // Stable key sort retains ascending input IDs inside a cell.
                std::vector<Id> scratch(n);
                for (unsigned shift = 0; shift < 64 && (maxKey >> shift); shift += 8) {
                    std::array<std::size_t, 256> cursor{};
                    for (auto id : ids)
                        ++cursor[(pointKey[id] >> shift) & 255];
                    std::size_t offset = 0;
                    for (auto& count : cursor) {
                        const auto size = count;
                        count = offset;
                        offset += size;
                    }
                    for (auto id : ids)
                        scratch[cursor[(pointKey[id] >> shift) & 255]++] = id;
                    ids.swap(scratch);
                }
            }
            std::size_t begin = 0;
            while (begin < n) {
                const auto key = pointKey[ids[begin]];
                std::size_t end = begin + 1;
                while (end < n && pointKey[ids[end]] == key)
                    ++end;
                keys.push_back(key);
                ranges.push_back({static_cast<std::uint32_t>(begin), static_cast<std::uint32_t>(end)});
                begin = end;
            }
            // Bound the binary search with a directory on high key bits.
            if (keys.size() >= 256) {
                while ((maxKey >> bucket_shift_) >= 4096)
                    ++bucket_shift_;
                const std::size_t count = (maxKey >> bucket_shift_) + 2;
                directory_.resize(count);
                std::size_t pos = 0;
                for (std::size_t i = 0; i < count; ++i) {
                    while (pos < keys.size() && (keys[pos] >> bucket_shift_) < i)
                        ++pos;
                    directory_[i] = static_cast<Id>(pos);
                }
            }
        }
    }

    std::size_t lowerCell(std::uint64_t key) const {
        if (!firstOccupied.empty())
            return firstOccupied[key];
        if (!directory_.empty()) {
            const auto slot = key >> bucket_shift_;
            if (slot >= directory_.size() - 1)
                return keys.size();
            const auto begin = keys.begin() + directory_[slot];
            const auto end = keys.begin() + directory_[slot + 1];
            return std::lower_bound(begin, end, key) - keys.begin();
        }
        return static_cast<std::size_t>(std::lower_bound(keys.begin(), keys.end(), key) - keys.begin());
    }

    std::size_t nonemptyCellCount() const { return keys.size(); }

    std::uint64_t cellStart(std::uint32_t c, std::size_t n) const {
        return (static_cast<std::uint64_t>(c) * n + resolution - 1) / resolution;
    }

    std::uint64_t cellEnd(std::uint32_t c, std::size_t n) const {
        return ((static_cast<std::uint64_t>(c) + 1) * n - 1) / resolution;
    }

private:
    std::vector<std::uint32_t> firstOccupied, directory_;
    unsigned bucket_shift_=0;
};

class GridBoxIndex {
public:
    struct QueryStats {
        std::vector<std::uint32_t> ids;
        double touchedCells = 0.0;
        double nonemptyCells = 0.0;
        double fullCells = 0.0;
        double pointTests = 0.0;
    };

    GridBoxIndex(const double* points, std::size_t n, std::size_t d, double volume) : n_(n), d_(d) {
        if (n > std::size_t(std::numeric_limits<std::int32_t>::max()))
            throw std::invalid_argument("grid supports at most INT32_MAX points");
        const auto g = chooseResolution(volume, d);
        detail::Ranks source(points, n, d);
        columns_ = std::move(source.columns);
        const auto& ranks = source.by_id;
        grid_ = SparseRankGrid(ranks, n, d, g);
        cellRanks_.resize(n * d);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < d; ++j)
                cellRanks_[i * d + j] = static_cast<std::uint32_t>(ranks[j][grid_.ids[i]]);
    }

    std::size_t dim() const { return d_; }
    const SparseRankGrid& grid() const { return grid_; }

    QueryStats query(const double* lo, const double* hi, std::vector<Id>* sink=nullptr) const {
        if (!lo || !hi)
            throw std::invalid_argument("box bounds must be non-null");
        QueryStats stats;
        auto& found=sink ? *sink : stats.ids;
        std::vector<Axis> axes(d_);
        bool coversAll = true;
        double prefixes = 1;
        for (std::size_t j = 0; j < d_; ++j) {
            if (!(lo[j] <= hi[j]))
                return stats;
            const auto& column = columns_[j];
            const auto lower = std::lower_bound(column.begin(), column.end(), lo[j]);
            const auto upper = std::upper_bound(column.begin(), column.end(), hi[j]);
            if (lower >= upper)
                return stats;
            Axis& a = axes[j];
            a.lower = static_cast<std::uint32_t>(lower - column.begin());
            const auto high = static_cast<std::uint32_t>(upper - column.begin() - 1);
            a.width = high - a.lower;
            coversAll = coversAll && static_cast<std::size_t>(a.width) + 1 == n_;
            a.first = static_cast<std::uint32_t>((std::uint64_t(a.lower) * grid_.resolution) / n_);
            a.last = static_cast<std::uint32_t>((std::uint64_t(high) * grid_.resolution) / n_);
            if (j + 1 < d_)
                prefixes *= std::uint64_t(a.last) - a.first + 1;
            a.leftFull = grid_.cellStart(a.first, n_) >= a.lower;
            a.rightFull = grid_.cellEnd(a.last, n_) <= high;
        }
        if (coversAll) {
            stats.touchedCells = 1;
            for (const auto& axis : axes)
                stats.touchedCells *= std::uint64_t(axis.last) - axis.first + 1;
            stats.fullCells = stats.nonemptyCells = static_cast<double>(grid_.keys.size());
            found.insert(found.end(),grid_.ids.begin(),grid_.ids.end());
            return stats;
        }
        // Cap traversal when the theoretical number of empty prefixes is huge.
        if (prefixes > 4.0 * grid_.keys.size() + 128.0) {
            stats.touchedCells = prefixes * (std::uint64_t(axes.back().last) - axes.back().first + 1);
            for (std::size_t i = 0; i < grid_.keys.size(); ++i) {
                auto key = grid_.keys[i];
                bool inside = true, full = true;
                for (std::size_t j = d_; j-- > 0;) {
                    const auto cell = static_cast<std::uint32_t>(key % grid_.resolution);
                    key /= grid_.resolution;
                    if (cell < axes[j].first || cell > axes[j].last) {
                        inside = false;
                        break;
                    }
                    full = full && axes[j].full(cell);
                }
                if (inside)
                    appendCell(i, full, axes, stats, found);
            }
        } else {
            visitRuns(axes, 0, 0, true, stats, found);
        }
        return stats;
    }

private:
    struct Axis {
        std::uint32_t lower, width, first, last;
        bool leftFull, rightFull;
        bool full(std::uint32_t c) const { return (c != first || leftFull) && (c != last || rightFull); }
    };

    void visitRuns(const std::vector<Axis>& axes, std::size_t j, std::uint64_t prefix, bool prefixFull,
                   QueryStats& stats,std::vector<Id>& found) const {
        const auto& axis = axes[j];
        const auto base = prefix * grid_.resolution;
        if (j + 1 < d_) {
            for (auto c = axis.first; c <= axis.last; ++c)
                visitRuns(axes, j + 1, base + c, prefixFull && axis.full(c), stats, found);
            return;
        }
        stats.touchedCells += std::uint64_t(axis.last) - axis.first + 1;
        const auto lastKey = base + axis.last;
        for (auto i = grid_.lowerCell(base + axis.first);
             i < grid_.keys.size() && grid_.keys[i] <= lastKey; ++i) {
            const auto c = static_cast<std::uint32_t>(grid_.keys[i] - base);
            appendCell(i, prefixFull && axis.full(c), axes, stats, found);
        }
    }

    void appendCell(std::size_t i, bool full, const std::vector<Axis>& axes, QueryStats& stats, std::vector<Id>& found) const {
        ++stats.nonemptyCells;
        const auto range = grid_.ranges[i];
        if (full) {
            ++stats.fullCells;
            found.insert(found.end(), grid_.ids.begin() + range.begin,
                             grid_.ids.begin() + range.end);
            return;
        }
        stats.pointTests += range.end - range.begin;
        const auto* row = cellRanks_.data() + std::size_t(range.begin) * d_;
        if (d_ == 4) {
            for (auto p = range.begin; p < range.end; ++p, row += 4) {
                if ((row[0] - axes[0].lower <= axes[0].width) &
                    (row[1] - axes[1].lower <= axes[1].width) &
                    (row[2] - axes[2].lower <= axes[2].width) &
                    (row[3] - axes[3].lower <= axes[3].width))
                    found.push_back(grid_.ids[p]);
            }
            return;
        }
        for (auto p = range.begin; p < range.end; ++p, row += d_) {
            std::size_t j = 0;
            while (j < d_ && row[j] - axes[j].lower <= axes[j].width)
                ++j;
            if (j == d_)
                found.push_back(grid_.ids[p]);
        }
    }

    std::vector<std::uint32_t> cellRanks_;
    std::size_t n_;
    std::size_t d_;
    std::vector<std::vector<double>> columns_;
    SparseRankGrid grid_;

    // Keep grid-cell side length comparable to the query side. For fixed d
    // and v this gives Theta(nv) occupancy and a boundary shell Theta(n).
    static std::uint32_t chooseResolution(double volume, std::size_t d) {
        if (!d || !(volume > 0) || !std::isfinite(volume))
            throw std::invalid_argument("query volume must be positive and finite");
        const double cells = std::ceil(std::pow(volume, -1.0 / static_cast<double>(d)));
        if (cells > std::numeric_limits<std::uint32_t>::max())
            throw std::length_error("resolution exceeds 32 bits");
        return std::max<std::uint32_t>(2, static_cast<std::uint32_t>(cells));
    }
};

} // namespace fcmaes::rankbox
