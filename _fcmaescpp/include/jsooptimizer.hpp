// Copyright (c) Dietmar Wolz. Licensed under the MIT license in the repository root.
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "pcg_random.hpp"

namespace fcmaes::jso {

using Point = std::vector<double>;
using Population = std::vector<Point>;

struct Parameters {
    std::size_t popsize = 0;
    std::size_t min_popsize = 4;
    std::size_t memory_size = 5;
    std::uint64_t max_evaluations = 100000;
    double stop_fitness = -std::numeric_limits<double>::infinity();
    double p_max = 0.25;
    double p_min = 0.125;
    double archive_rate = 1.0;
};

struct Result {
    Point x;
    double y;
    std::uint64_t evaluations;
    std::uint64_t iterations;
    int stop;
};

/// Synchronous jSO on a finite box. Each batch is cut at the evaluation cap.
class State {
public:
    State(Point lower, Point upper, Parameters params, std::uint64_t seed);

    std::size_t dim() const { return lower_.size(); }
    std::size_t popsize() const { return pop_.size(); }
    const Population &population() const { return pop_; }
    Result result() const;

    /// The batch reference is valid until the next mutating call on this state.
    const Population &ask();
    int tell(Point values);
    /// Finish with a prefix when serial evaluation has already hit its target.
    int tell_prefix(Point values);

private:
    struct Success {
        double f, cr, improvement;
    };

    double uniform();
    std::size_t draw(std::size_t count);
    void generate(std::size_t count);
    int accept(Point values, bool prefix);
    void update_population(const Point &values);
    void update_memory(const std::vector<Success> &successes);
    void reduce_population();

    Point lower_, upper_;
    Parameters params_;
    pcg64 rng_;
    std::normal_distribution<double> normal_{0.0, 0.1};
    Population pop_, archive_, pending_;
    Point fitness_, memory_f_, memory_cr_, trial_f_, trial_cr_, best_x_;
    double best_y_ = std::numeric_limits<double>::infinity();
    std::size_t memory_index_ = 0;
    std::uint64_t evaluations_ = 0;
    std::uint64_t iterations_ = 0;
    int stop_ = 0;
};

}  // namespace fcmaes::jso
