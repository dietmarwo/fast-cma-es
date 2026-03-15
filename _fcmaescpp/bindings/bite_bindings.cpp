#include "bite_bindings.hpp"

#include "gil_utils.hpp"
#include "ndarray_utils.hpp"

#include <cmath>
#include <cfloat>
#include <limits>
#include <stdexcept>

#include "biteoptimizer.hpp"

namespace fcmaes::bindings {

namespace {

using ResultVector = nb::ndarray<nb::numpy, const double, nb::shape<-1>,
                                 nb::c_contig, nb::device::cpu>;

thread_local nb::callable *g_bite_objective = nullptr;

bool bite_callback_trampoline(int n, const double *x, double *y) {
    if (g_bite_objective == nullptr)
        throw std::runtime_error("Bite objective trampoline was not initialized");

    nb::gil_scoped_acquire acquire;
    F64Vector x_view(x, { static_cast<size_t>(n) });
    double value = nb::cast<double>((*g_bite_objective)(x_view));
    y[0] = std::isfinite(value) ? value : DBL_MAX;
    return false;
}

nb::tuple to_python_result(biteopt::BiteResult result) {
    auto *x = new Vec(std::move(result.x));
    nb::capsule owner(x, [](void *ptr) noexcept {
        delete static_cast<Vec *>(ptr);
    });
    ResultVector x_array(x->data(), { static_cast<size_t>(x->size()) }, owner);
    return nb::make_tuple(x_array, result.y, result.evaluations,
                          result.iterations, result.stop);
}

class BiteObjectiveScope {
public:
    explicit BiteObjectiveScope(nb::callable &fun) : previous_(g_bite_objective) {
        g_bite_objective = &fun;
    }

    ~BiteObjectiveScope() {
        g_bite_objective = previous_;
    }

private:
    nb::callable *previous_;
};

}  // namespace

void bind_bite(nb::module_ &m) {
    m.def(
        "optimize_bite",
        [](nb::callable fun, F64Vector guess, F64Vector lower, F64Vector upper,
           uint64_t seed, long runid, int max_evaluations, double stop_fitness, int M,
           int popsize, int stall_criterion) {
            const int dim = static_cast<int>(guess.shape(0));
            if (lower.size() != 0 && static_cast<int>(lower.shape(0)) != dim)
                throw std::invalid_argument("lower must match guess size");
            if (upper.size() != 0 && static_cast<int>(upper.shape(0)) != dim)
                throw std::invalid_argument("upper must match guess size");

            const double *lower_ptr =
                lower.size() == 0 ? nullptr : static_cast<const double *>(lower.data());
            const double *upper_ptr =
                upper.size() == 0 ? nullptr : static_cast<const double *>(upper.data());

            BiteObjectiveScope scope(fun);
            auto result = without_gil([&]() {
                return biteopt::optimize_bite(
                    runid, bite_callback_trampoline, dim, static_cast<int>(seed),
                    static_cast<const double *>(guess.data()), lower_ptr, upper_ptr,
                    max_evaluations, stop_fitness, M, popsize, stall_criterion);
            });
            return to_python_result(std::move(result));
        },
        "fun"_a,
        "guess"_a.noconvert(),
        "lower"_a.noconvert(),
        "upper"_a.noconvert(),
        "seed"_a,
        "runid"_a = 0L,
        "max_evaluations"_a = 100000,
        "stop_fitness"_a = -std::numeric_limits<double>::infinity(),
        "M"_a = 1,
        "popsize"_a = 0,
        "stall_criterion"_a = 0,
        "Execute the native BiteOpt solver through nanobind."
    );
}

}  // namespace fcmaes::bindings
