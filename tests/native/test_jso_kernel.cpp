#include "jsooptimizer.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

using fcmaes::jso::Parameters;
using fcmaes::jso::Point;
using fcmaes::jso::State;

static void require(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}

static double objective(const Point &x) {
    const double d = x[0] - 1.1;
    return d * d;
}

int main() {
    try {
        Parameters params;
        params.popsize = 20;
        params.max_evaluations = 1321;
        State state({-3.0, 2.0}, {3.0, 2.0}, params, 42);
        std::uint64_t evaluations = 0;
        for (;;) {
            const auto &batch = state.ask();
            if (batch.empty())
                break;
            Point scores;
            for (const Point &x : batch) {
                require(x.size() == 2 && x[0] >= -3.0 && x[0] <= 3.0 && x[1] == 2.0,
                        "candidate bounds");
                scores.push_back(objective(x));
            }
            evaluations += scores.size();
            state.tell(scores);
            require(state.result().evaluations == evaluations, "evaluation accounting");
        }
        const auto result = state.result();
        require(evaluations == params.max_evaluations && state.popsize() == 4, "cap and reduction");
        require(result.y < 1e-8 && result.y == objective(result.x), "returned incumbent");

        params.max_evaluations = 9;
        params.stop_fitness = 5.0;
        State early({0.0}, {1.0}, params, 13);
        require(early.ask().size() == 9, "truncated initialization batch");
        try {
            early.tell({0.0});
            throw std::runtime_error("tell accepted a wrong length");
        } catch (const std::invalid_argument &) {
            require(early.result().evaluations == 0, "rejected tell changed the count");
        }
        require(early.tell_prefix({0.0}) == 1, "serial target prefix");
        require(early.result().evaluations == 1 && early.ask().empty(), "target stopped after one call");
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
