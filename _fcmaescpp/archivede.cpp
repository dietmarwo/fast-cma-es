// Archive gate, rank-sum ranking and rank-space spread, derived from
// archive_de.cpp in Aleksey Vaneev's BiteOpt repository (commit e8d6785).
// Original code: Copyright (c) 2026 Aleksey Vaneev.
// fcmaes adaptations: Copyright (c) 2026 Dietmar Wolz.
// MIT license; see the notice in include/archivede.hpp.
#include "archivede.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace fcmaes::archivede {
namespace {
bool dominates(const double* a, const double* b, std::size_t n) {
    if (n == 3) {
        if (a[0] > b[0] || a[1] > b[1] || a[2] > b[2])
            return false;
        return a[0] < b[0] || a[1] < b[1] || a[2] < b[2];
    }
    if (n >= 8) {
        unsigned worse = 0, better = 0;
        for (std::size_t j = 0; j < n; ++j) {
            worse |= a[j] > b[j];
            better |= a[j] < b[j];
        }
        return !worse && better;
    }
    bool strict = false;
    for (std::size_t k = 0; k < n; ++k) {
        if (a[k] > b[k])
            return false;
        strict |= a[k] < b[k];
    }
    return strict;
}

void finite(const double* values, std::size_t count) {
    if (count && !values)
        throw std::invalid_argument("null objective buffer");
    for (std::size_t i = 0; i < count; ++i)
        if (!std::isfinite(values[i]))
            throw std::invalid_argument("objectives must be finite");
}

// Directed bounds on the mathematical sum of finite coordinates. If any
// intermediate overflows, leave the conventional archive scan in charge.
std::pair<double,double> sum_interval(const double* y, std::size_t dim) {
    double lo = y[0], hi = y[0];
    for (std::size_t j = 1; j < dim; ++j) {
        lo = std::nextafter(lo + y[j], -std::numeric_limits<double>::infinity());
        hi = std::nextafter(hi + y[j], std::numeric_limits<double>::infinity());
    }
    return {lo, hi};
}

constexpr std::size_t max_indexed_objectives = 8;

bool narrow_interval(double lo, double hi) {
    const double allowance = 128 * std::numeric_limits<double>::epsilon()
                             * std::max({1., std::abs(lo), std::abs(hi)});
    return std::isfinite(lo) && std::isfinite(hi) && hi - lo <= allowance;
}

bool exact_sum(double a, double b, double sum) {
    if (std::abs(a) < std::abs(b))
        std::swap(a, b);
    return std::isfinite(sum) && sum - a == b;
}

// std::generate_canonical<double, 53> on two 32-bit words, low word first, as
// libstdc++ computes it. Written out so that every standard library gives the
// same stream; fcmaes-rust's ArchiveDe draws the same numbers.
double unit(std::mt19937& rng) {
    const double low = static_cast<double>(rng());
    const double high = static_cast<double>(rng());
    const double r = (low + high * 4294967296.) / 18446744073709551616.;
    return r < 1. ? r : std::nextafter(1., 0.);
}

std::size_t matrix_size(std::size_t rows, std::size_t cols) {
    if (cols && rows > std::vector<double>().max_size() / cols)
        throw std::length_error("matrix too large");
    return rows * cols;
}

std::vector<std::size_t> select(const Population& values, std::size_t np, bool tiered) {
    const std::size_t count = values.size(), objectives = values.front().size();
    std::vector<std::size_t> order(count);
    std::vector<std::uint64_t> scores(count);
    Population ranks;
    std::vector<unsigned char> used;
    std::vector<std::size_t> keep;
    if (tiered) {
        ranks.resize(count, Point(objectives));
        used.resize(count);
        keep.reserve(np);
    }
    for (std::size_t j = 0; j < objectives; ++j) {
        std::iota(order.begin(), order.end(), std::size_t{0});
        std::stable_sort(order.begin(), order.end(),
                         [&](auto a, auto b) { return values[a][j] < values[b][j]; });
        for (std::size_t i = 0; i < count; ++i)
            scores[order[i]] += i;
        if (tiered) {
            const auto champion = *std::find_if(order.begin(), order.end(),
                                                [&](auto id) { return !used[id]; });
            used[champion] = 1;
            keep.push_back(champion);
            for (std::size_t i = 0; i < count; ++i)
                ranks[order[i]][j] = double(i);
        }
    }
    std::iota(order.begin(), order.end(), std::size_t{0});
    if (count < 128) {
        std::stable_sort(order.begin(),order.end(),[&](auto a,auto b) { return scores[a]<scores[b]; });
    } else {
        const auto maxscore = *std::max_element(scores.begin(),scores.end());
        std::vector<std::size_t> scratch(count);
        for (unsigned shift=0; shift<64 && (maxscore >> shift)!=0; shift+=8) {
            std::array<std::size_t,256> counts{};
            for (auto id:order) ++counts[(scores[id]>>shift)&255];
            std::size_t next=0;
            for (auto& c:counts) { const auto old=c; c=next; next+=old; }
            for (auto id:order) scratch[counts[(scores[id]>>shift)&255]++]=id;
            order.swap(scratch);
        }
    }
    if (!tiered) {
        order.resize(np);
        return order;
    }

    auto take = [&](std::size_t id) {
        keep.push_back(id);
        used[id] = 1;
    };
    struct Candidate {
        std::size_t id;
        double nearest;
    };
    std::vector<Candidate> active;
    std::size_t seed = count;
    for (std::size_t k = 0; k < 6 * count / 10; ++k) {
        auto id = order[k];
        if (used[id])
            continue;
        if (seed == count) {
            seed = id;
            take(id);
        } else
            active.push_back({id, std::numeric_limits<double>::infinity()});
    }
    const bool squared = double(objectives) * double(count) * double(count) < 4503599627370496.;
    for (std::size_t k = 1; k < 3 * np / 5 && !active.empty() && keep.size() < np; ++k) {
        auto best = active.begin();
        for (auto it = active.begin(); it != active.end(); ++it) {
            double distance = 0;
            for (std::size_t j = 0; j < objectives; ++j) {
                auto delta = ranks[seed][j] - ranks[it->id][j];
                distance += delta * delta;
            }
            it->nearest = std::min(it->nearest, squared ? distance : std::sqrt(distance));
            if (it->nearest > best->nearest)
                best = it;
        }
        seed = best->id;
        take(seed);
        active.erase(best);
    }
    for (auto id : order) {
        if (keep.size() == np)
            break;
        if (!used[id])
            take(id);
    }
    return keep;
}
} // namespace

Archive::Archive(std::size_t objectives, std::size_t payload_dim)
    : nobj_(objectives), payload_dim_(payload_dim), exact_line_(objectives == 2) {
    if (!objectives)
        throw std::invalid_argument("nobj must be positive");
    witness_.reserve(objectives);
}

void Archive::clear() {
    entries_.clear();
    skyline_.clear();
    witness_.clear();
    serial_ = 0;
    exact_line_ = nobj_ == 2;
    narrow_sums_ = false;
    next_sum_check_ = 128;
}

bool Archive::add(const double* y, std::size_t objectives, const double* x, std::size_t width) {
    if (objectives != nobj_ || width != payload_dim_ || (width && !x))
        throw std::invalid_argument("objective or payload width mismatch");
    finite(y, objectives);
    if (!witness_.empty() && dominates(witness_.data(), y, nobj_))
        return false;
    const double sum = std::accumulate(y, y + nobj_, 0.0);
    auto make_entry = [&] {
        return Entry{sum, serial_, Point(y, y + nobj_), width ? Point(x, x + width) : Point()};
    };
    if (exact_line_) {
        if ((entries_.empty() || entries_.front().sum == sum) && exact_sum(y[0], y[1], sum)) {
            entries_.push_back(make_entry());
            ++serial_;
            return true;
        }
        // Build off to the side so allocation failure leaves the original front usable.
        decltype(skyline_) tree;
        for (const auto& e : entries_)
            tree.emplace(e.y[0], e);
        skyline_.swap(tree);
        entries_.clear();
        exact_line_ = false;
    }
    if (nobj_ > 2 && nobj_ <= max_indexed_objectives && entries_.size() >= next_sum_check_) {
        next_sum_check_ = entries_.size() > SIZE_MAX / 2 ? SIZE_MAX : entries_.size() * 2;
        const double low_sum = std::min(entries_.front().sum, sum);
        const double high_sum = std::max(entries_.back().sum, sum);
        if (narrow_interval(low_sum, high_sum)) {
            auto [lo, hi] = sum_interval(y, nobj_);
            std::array<double, max_indexed_objectives> lower, upper;
            std::copy_n(y, nobj_, lower.begin());
            std::copy_n(y, nobj_, upper.begin());
            for (const auto& entry : entries_) {
                const auto [a, b] = sum_interval(entry.y.data(), nobj_);
                lo = std::min(lo, a);
                hi = std::max(hi, b);
                for (std::size_t j = 0; j < nobj_; ++j) {
                    lower[j] = std::min(lower[j], entry.y[j]);
                    upper[j] = std::max(upper[j], entry.y[j]);
                }
            }
            std::size_t axis = 0;
            for (std::size_t j = 1; j < nobj_; ++j)
                if (upper[j] - lower[j] > upper[axis] - lower[axis])
                    axis = j;
            const double spread = upper[axis] - lower[axis];
            if (narrow_interval(lo, hi) && spread > 64 * (hi - lo)) {
                decltype(skyline_) index;
                for (const auto& entry : entries_)
                    index.emplace(entry.y[axis], entry);
                skyline_.swap(index);
                entries_.clear();
                sum_lower_ = lo;
                sum_upper_ = hi;
                narrow_axis_ = axis;
                narrow_sums_ = true;
            }
        }
    }
    if (narrow_sums_) {
        const auto [point_lo, point_hi] = sum_interval(y, nobj_);
        const double lo = std::min(sum_lower_, point_lo), hi = std::max(sum_upper_, point_hi);
        if (!narrow_interval(lo, hi)) {
            // Build the original sum-ordered representation on the side.
            decltype(entries_) sorted;
            sorted.reserve(skyline_.size());
            for (const auto& pair : skyline_)
                sorted.push_back(pair.second);
            std::sort(sorted.begin(), sorted.end(), [](const Entry& a, const Entry& b) {
                return a.sum != b.sum ? a.sum < b.sum : a.serial < b.serial;
            });
            entries_.swap(sorted);
            skyline_.clear();
            narrow_sums_ = false;
            next_sum_check_ = std::max<std::size_t>(128, entries_.size());
        } else {
            const double infinity = std::numeric_limits<double>::infinity();
            const double band = std::nextafter(hi - lo, infinity);
            // When a dominates b, every coordinate difference b[j] - a[j]
            // is at most sum(b) - sum(a). The band bounds both directions.
            const auto first = skyline_.lower_bound(std::nextafter(y[narrow_axis_] - band, -infinity));
            const auto last = skyline_.upper_bound(std::nextafter(y[narrow_axis_] + band, infinity));
            for (auto it = first; it != last; ++it) {
                if (dominates(it->second.y.data(), y, nobj_)) {
                    witness_ = it->second.y;
                    return false;
                }
            }
            auto added = skyline_.emplace_hint(last, y[narrow_axis_], make_entry());
            for (auto it = first; it != last;) {
                if (it != added && dominates(y, it->second.y.data(), nobj_))
                    it = skyline_.erase(it);
                else
                    ++it;
            }
            sum_lower_ = lo;
            sum_upper_ = hi;
            ++serial_;
            return true;
        }
    }
    if (nobj_ == 2) {
        auto first = skyline_.lower_bound(y[0]);
        if (first != skyline_.end() && first->first == y[0] && first->second.y[1] <= y[1]) {
            if (first->second.y[1] < y[1]) {
                witness_ = first->second.y;
                return false;
            }
            skyline_.emplace_hint(first, y[0], make_entry());
            ++serial_;
            return true;
        }
        if (first != skyline_.begin()) {
            const auto& pred = std::prev(first)->second;
            if (pred.y[1] <= y[1]) {
                witness_ = pred.y;
                return false;
            }
        }
        auto stop = first;
        while (stop != skyline_.end() && stop->second.y[1] >= y[1])
            ++stop;
        // A valid insertion hint puts the new key immediately before the erased range.
        skyline_.emplace_hint(first, y[0], make_entry());
        skyline_.erase(first, stop);
    } else {
        auto lower = std::lower_bound(entries_.begin(), entries_.end(), sum,
                                      [](const Entry& e, double v) { return e.sum < v; });
        auto upper = std::upper_bound(lower, entries_.end(), sum,
                                      [](double v, const Entry& e) { return v < e.sum; });
        for (auto it = entries_.begin(); it != upper; ++it) {
            if (dominates(it->y.data(), y, nobj_)) {
                witness_ = it->y;
                return false;
            }
        }
        const auto offset = lower - entries_.begin();
        entries_.insert(upper, make_entry());
        // Equal rounded sums can contain strictly dominated points, even with finite inputs.
        entries_.erase(std::remove_if(entries_.begin() + offset, entries_.end(),
                                      [&](const Entry& p) { return dominates(y, p.y.data(), nobj_); }),
                       entries_.end());
    }
    ++serial_;
    return true;
}

Snapshot Archive::snapshot() const {
    Snapshot out;
    out.rows = size();
    out.payloads.reserve(matrix_size(out.rows, payload_dim_));
    out.objectives.reserve(matrix_size(out.rows, nobj_));
    auto append = [&](const Entry& e) {
        out.payloads.insert(out.payloads.end(), e.x.begin(), e.x.end());
        out.objectives.insert(out.objectives.end(), e.y.begin(), e.y.end());
    };
    if (skyline_.empty()) {
        for (const auto& e : entries_)
            append(e);
    } else {
        std::vector<const Entry*> ordered;
        ordered.reserve(out.rows);
        for (const auto& pair : skyline_)
            ordered.push_back(&pair.second);
        std::sort(ordered.begin(), ordered.end(), [](const Entry* a, const Entry* b) {
            return a->sum != b->sum ? a->sum < b->sum : a->serial < b->serial;
        });
        for (auto e : ordered)
            append(*e);
    }
    return out;
}

State::State(Point lower, Point upper, std::size_t nobj, std::size_t np, std::uint64_t budget,
             std::uint32_t seed, bool gate, bool tiered)
    : lower_(std::move(lower)), upper_(std::move(upper)), popsize_(np), budget_(budget), gate_(gate),
      tiered_(tiered), rng_(seed), archive_(nobj, lower_.size()) {
    if (np < 4 || np <= nobj || np > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("popsize must be at least four and greater than nobj");
    if (!dim() || dim() != upper_.size() || dim() > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("bounds must be matching nonempty vectors");
    widths_.resize(dim());
    for (std::size_t j = 0; j < dim(); ++j) {
        if (!std::isfinite(lower_[j]) || !std::isfinite(upper_[j]) || lower_[j] > upper_[j])
            throw std::invalid_argument("bounds must be finite and ordered");
        widths_[j] = upper_[j] - lower_[j];
    }
}

Point State::decode(const Point& unit) const {
    Point x(dim());
    for (std::size_t j = 0; j < dim(); ++j) {
        const auto lo = lower_[j], hi = upper_[j], width = widths_[j], t = unit[j];
        const auto value = std::isfinite(width) ? lo + t * width : (1. - t) * lo + t * hi;
        x[j] = t == 1. ? hi : std::clamp(value, lo, hi);
    }
    return x;
}

const Point& State::ask() {
    if (!pending_.empty() || evaluations_ == budget_)
        return physical_;
    const auto n = static_cast<std::size_t>(std::min<std::uint64_t>(budget_ - evaluations_, popsize_));
    auto rng = rng_;
    Population pending(n, Point(dim()));
    Point physical;
    physical.reserve(matrix_size(n, dim()));
    for (std::size_t i = 0; i < n; ++i) {
        auto& u = pending[i];
        if (parents_.empty()) {
            for (auto& v : u)
                v = unit(rng);
        } else {
            std::size_t a, b, c;
            do {
                a = rng() % popsize_;
            } while (a == i);
            do {
                b = rng() % popsize_;
            } while (b == i || b == a);
            do {
                c = rng() % popsize_;
            } while (c == i || c == a || c == b);
            const auto forced = rng() % dim();
            for (std::size_t j = 0; j < dim(); ++j) {
                const auto v = parents_[a][j] + 0.5 * (parents_[b][j] - parents_[c][j]);
                u[j] = (unit(rng) < 0.3 || j == forced) ? std::clamp(v, 0., 1.) : parents_[i][j];
            }
        }
        const auto x = decode(u);
        physical.insert(physical.end(), x.begin(), x.end());
    }
    pending_.swap(pending);
    physical_.swap(physical);
    rng_ = rng;
    return physical_;
}

void State::tell(const double* y, std::size_t rows, std::size_t cols) {
    if (pending_.empty())
        throw std::logic_error("tell requires an outstanding ask");
    if (rows != pending_.size() || cols != nobj())
        throw std::invalid_argument("tell shape must be (asked rows, nobj)");
    const auto count = matrix_size(rows, cols);
    finite(y, count);
    if (submitted_.empty()) {
        accepted_.resize(rows);
        submitted_.assign(y, y + count);
    } else if (!std::equal(submitted_.begin(), submitted_.end(), y)) {
        throw std::invalid_argument("retry this batch with the same objective rows");
    }
    // Progress survives a failed allocation. A retry inserts each accepted duplicate once.
    for (; processed_ < rows; ++processed_)
        accepted_[processed_] =
            archive_.add(y + processed_ * cols, cols, physical_.data() + processed_ * dim(), dim());
    Population pool = parents_, pool_y = values_;
    for (std::size_t i = 0; i < rows; ++i) {
        if (parents_.empty() || !gate_ || accepted_[i]) {
            pool.push_back(pending_[i]);
            pool_y.emplace_back(y + i * cols, y + (i + 1) * cols);
        }
    }
    if (parents_.empty()) {
        parents_.swap(pool);
        values_.swap(pool_y);
    } else {
        // Rank selection also reorders a pool containing only parents.
        auto keep = select(pool_y, popsize_, tiered_);
        Population next, next_y;
        next.reserve(keep.size());
        next_y.reserve(keep.size());
        for (auto id : keep) {
            next.push_back(std::move(pool[id]));
            next_y.push_back(std::move(pool_y[id]));
        }
        parents_.swap(next);
        values_.swap(next_y);
    }
    evaluations_ += rows;
    pending_.clear();
    physical_.clear();
    submitted_.clear();
    processed_ = 0;
}

Snapshot State::population() const {
    Snapshot out;
    out.rows = parents_.size();
    for (std::size_t i = 0; i < out.rows; ++i) {
        auto x = decode(parents_[i]);
        out.payloads.insert(out.payloads.end(), x.begin(), x.end());
        out.objectives.insert(out.objectives.end(), values_[i].begin(), values_[i].end());
    }
    return out;
}
} // namespace fcmaes::archivede
