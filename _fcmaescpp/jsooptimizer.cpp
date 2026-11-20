// Copyright (c) Dietmar Wolz. Licensed under the MIT license in the repository root.

// Stock jSO port from fcmaes-rust. See Brest, Maučec and Bošković, CEC 2017,
// doi:10.1109/CEC.2017.7969456.
#include "jsooptimizer.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace fcmaes::jso {
namespace {

constexpr double infinity = std::numeric_limits<double>::infinity();
constexpr std::size_t max_count = std::numeric_limits<std::int32_t>::max();
constexpr double pi = 3.14159265358979323846;

std::size_t rounded(double value) {
    return static_cast<std::size_t>(std::floor(value + 0.5));
}

std::vector<std::size_t> ranked(const Point &fitness) {
    std::vector<std::size_t> order(fitness.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return fitness[a] < fitness[b];
    });
    return order;
}

double repair(double value, double parent, double lo, double hi) {
    if (std::isnan(value))
        return parent;
    if (value < lo)
        value = lo + 0.5 * (parent - lo);
    else if (value > hi)
        value = parent + 0.5 * (hi - parent);
    return std::clamp(value, lo, hi);
}

}  // namespace

State::State(Point lower, Point upper, Parameters params, std::uint64_t seed)
    : lower_(std::move(lower)), upper_(std::move(upper)), params_(params), rng_(seed) {
    if (dim() == 0 || dim() != upper_.size())
        throw std::invalid_argument("jSO needs matching, non-empty bounds");
    for (std::size_t j = 0; j < dim(); ++j) {
        const double lo = lower_[j], hi = upper_[j];
        if (!std::isfinite(lo) || !std::isfinite(hi) || lo > hi || !std::isfinite(hi - lo))
            throw std::invalid_argument("jSO bounds must be finite and ordered with finite widths");
    }
    if (params_.max_evaluations == 0 || std::isnan(params_.stop_fitness))
        throw std::invalid_argument("jSO needs a positive budget and a non-NaN target");
    if (!(0 < params_.p_min && params_.p_min <= params_.p_max && params_.p_max <= 1))
        throw std::invalid_argument("jSO requires 0 < p_min <= p_max <= 1");
    params_.min_popsize = std::max<std::size_t>(4, params_.min_popsize);
    params_.memory_size = std::max<std::size_t>(2, params_.memory_size);
    if (params_.popsize == 0) {
        const double d = static_cast<double>(dim());
        params_.popsize = rounded(25.0 * std::log(d) * std::sqrt(d));
    }
    params_.popsize = std::max(params_.popsize, params_.min_popsize);
    if (std::max(params_.popsize, params_.memory_size) > max_count)
        throw std::invalid_argument("jSO population and memory sizes are too large");
    const double archive_size = params_.archive_rate * params_.popsize;
    if (!(0 <= archive_size && archive_size <= max_count))
        throw std::invalid_argument("jSO archive capacity must be finite and non-negative");

    const auto count = std::min<std::uint64_t>(params_.popsize, params_.max_evaluations);
    pop_.resize(count, Point(dim()));
    for (auto &point : pop_) {
        for (std::size_t j = 0; j < dim(); ++j) {
            const double lo = lower_[j], hi = upper_[j];
            point[j] = std::clamp(lo + uniform() * (hi - lo), lo, hi);
        }
    }
    memory_f_.assign(params_.memory_size, 0.3);
    memory_cr_.assign(params_.memory_size, 0.8);
}

double State::uniform() {
    return static_cast<double>(rng_() >> 11) * 0x1.0p-53;
}

std::size_t State::draw(std::size_t count) {
    return std::uniform_int_distribution<std::size_t>(0, count - 1)(rng_);
}

const Population &State::ask() {
    if (!pending_.empty())
        throw std::logic_error("jSO ask called before telling the pending batch");
    if (stop_ != 0 || evaluations_ == params_.max_evaluations)
        return pending_;
    if (fitness_.empty())
        pending_ = pop_;
    else
        generate(std::min<std::uint64_t>(popsize(), params_.max_evaluations - evaluations_));
    return pending_;
}

void State::generate(std::size_t count) {
    const std::size_t n = popsize();
    const double progress = static_cast<double>(evaluations_) / params_.max_evaluations;
    const double p = params_.p_max - (params_.p_max - params_.p_min) * progress;
    const auto order = ranked(fitness_);
    const auto top = std::max<std::size_t>(2, rounded(p * n));
    const double fw = progress < 0.2 ? 0.7 : (progress < 0.4 ? 0.8 : 1.2);
    pending_.reserve(count);
    trial_f_.resize(count);
    trial_cr_.resize(count);

    for (std::size_t i = 0; i < count; ++i) {
        const auto cell = draw(params_.memory_size);
        const bool fixed = cell + 1 == params_.memory_size;
        const double mean_f = fixed ? 0.9 : memory_f_[cell];
        const double mean_cr = fixed ? 0.9 : memory_cr_[cell];
        double cr = mean_cr < 0 ? 0 : std::clamp(mean_cr + normal_(rng_), 0.0, 1.0);
        if (progress < 0.25)
            cr = std::max(cr, 0.7);
        else if (progress < 0.5)
            cr = std::max(cr, 0.6);
        double f;
        do {
            f = mean_f + 0.1 * std::tan(pi * (uniform() - 0.5));
        } while (!(f > 0));
        f = std::min(f, progress < 0.6 ? 0.7 : 1.0);

        auto pbest = order[draw(top)];
        while (progress < 0.5 && pbest == i)
            pbest = order[draw(top)];
        auto r1 = draw(n - 1);
        r1 += r1 >= i;
        auto r2 = draw(n + archive_.size() - 2);
        r2 += r2 >= std::min(i, r1);
        r2 += r2 >= std::max(i, r1);
        const Point &donor = r2 < n ? pop_[r2] : archive_[r2 - n];
        const auto forced = draw(dim());
        const Point &parent = pop_[i];
        Point trial = parent;
        for (std::size_t j = 0; j < dim(); ++j) {
            if (j == forced || uniform() <= cr) {
                const double value = parent[j] + (fw * f) * (pop_[pbest][j] - parent[j])
                    + f * (pop_[r1][j] - donor[j]);
                trial[j] = repair(value, parent[j], lower_[j], upper_[j]);
            }
        }
        pending_.push_back(std::move(trial));
        trial_f_[i] = f;
        trial_cr_[i] = cr;
    }
}

int State::tell(Point values) {
    return accept(std::move(values), false);
}

int State::tell_prefix(Point values) {
    return accept(std::move(values), true);
}

int State::accept(Point values, bool prefix) {
    if (pending_.empty())
        throw std::logic_error("jSO tell called without a pending ask batch");
    if (values.empty() || values.size() > pending_.size())
        throw std::invalid_argument("jSO tell batch length must equal the asked batch length");
    for (double &value : values)
        if (!std::isfinite(value))
            value = infinity;
    const auto best = std::min_element(values.begin(), values.end());
    if (values.size() != pending_.size() && !(prefix && *best < params_.stop_fitness))
        throw std::invalid_argument("jSO tell batch length must equal the asked batch length");

    if (best_x_.empty() || *best < best_y_) {
        best_x_ = pending_[best - values.begin()];
        best_y_ = *best;
    }
    evaluations_ += values.size();
    if (fitness_.empty()) {
        pop_.resize(values.size());
        fitness_ = std::move(values);
    } else {
        update_population(values);
        ++iterations_;
    }
    pending_.clear();
    stop_ = best_y_ < params_.stop_fitness ? 1 : 0;
    return stop_;
}

void State::update_population(const Point &values) {
    const auto capacity = rounded(params_.archive_rate * popsize());
    std::vector<Success> successes;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (values[i] < fitness_[i]) {
            successes.push_back({trial_f_[i], trial_cr_[i], fitness_[i] - values[i]});
            if (capacity != 0) {
                if (archive_.size() < capacity)
                    archive_.push_back(std::move(pop_[i]));
                else
                    archive_[draw(capacity)] = std::move(pop_[i]);
            }
        }
        if (values[i] <= fitness_[i]) {
            fitness_[i] = values[i];
            pop_[i] = std::move(pending_[i]);
        }
    }
    if (!successes.empty())
        update_memory(successes);
    reduce_population();
}

void State::update_memory(const std::vector<Success> &successes) {
    double scale = 0;
    for (const auto &s : successes)
        scale = std::max(scale, s.improvement);
    double fn = 0, fd = 0, cn = 0, cd = 0;
    bool positive_cr = false;
    for (const auto &s : successes) {
        const double w = std::isinf(scale) ? (std::isinf(s.improvement) ? 1.0 : 0.0)
                                           : s.improvement / scale;
        fn += w * s.f * s.f;
        fd += w * s.f;
        cn += w * s.cr * s.cr;
        cd += w * s.cr;
        positive_cr = positive_cr || s.cr > 0;
    }
    const auto cell = memory_index_;
    if (fd > 0)
        memory_f_[cell] = 0.5 * (memory_f_[cell] + fn / fd);
    if (!positive_cr)
        memory_cr_[cell] = 0.5 * (memory_cr_[cell] - 1.0);
    else if (cd > 0)
        memory_cr_[cell] = 0.5 * (memory_cr_[cell] + cn / cd);
    memory_index_ = (cell + 1) % params_.memory_size;
}

void State::reduce_population() {
    const double progress = static_cast<double>(evaluations_) / params_.max_evaluations;
    const auto size = rounded(params_.popsize - (params_.popsize - params_.min_popsize) * progress);
    if (size < popsize()) {
        const auto order = ranked(fitness_);
        Population kept;
        Point values;
        kept.reserve(size);
        values.reserve(size);
        for (std::size_t i = 0; i < size; ++i) {
            kept.push_back(std::move(pop_[order[i]]));
            values.push_back(fitness_[order[i]]);
        }
        pop_ = std::move(kept);
        fitness_ = std::move(values);
    }
    const auto capacity = rounded(params_.archive_rate * popsize());
    while (archive_.size() > capacity) {
        const auto i = draw(archive_.size());
        if (i != archive_.size() - 1)
            archive_[i] = std::move(archive_.back());
        archive_.pop_back();
    }
}

Result State::result() const {
    Point x = best_x_;
    if (x.empty()) {
        x.resize(dim());
        for (std::size_t j = 0; j < dim(); ++j)
            x[j] = lower_[j] + 0.5 * (upper_[j] - lower_[j]);
    }
    return {std::move(x), best_y_, evaluations_, iterations_, stop_};
}

}  // namespace fcmaes::jso
