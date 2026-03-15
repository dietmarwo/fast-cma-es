#include "da_bindings.hpp"

#include "gil_utils.hpp"
#include "ndarray_utils.hpp"

#include <cmath>
#include <cfloat>
#include <limits>
#include <stdexcept>

#include "daoptimizer.hpp"

namespace fcmaes::bindings {

namespace {

using ResultVector = nb::ndarray<nb::numpy, const double, nb::shape<-1>,
                                 nb::c_contig, nb::device::cpu>;

thread_local nb::callable *g_da_objective = nullptr;

double da_callback_trampoline(int n, const double *x) {
    if (g_da_objective == nullptr)
        throw std::runtime_error("DA objective trampoline was not initialized");

    nb::gil_scoped_acquire acquire;
    F64Vector x_view(x, { static_cast<size_t>(n) });
    double value = nb::cast<double>((*g_da_objective)(x_view));
    return std::isfinite(value) ? value : DBL_MAX;
}

nb::tuple to_python_result(dual_annealing::DaResult result) {
    auto *x = new Vec(std::move(result.x));
    nb::capsule owner(x, [](void *ptr) noexcept {
        delete static_cast<Vec *>(ptr);
    });
    ResultVector x_array(x->data(), { static_cast<size_t>(x->size()) }, owner);
    return nb::make_tuple(x_array, result.y, result.evaluations,
                          result.iterations, result.stop);
}

class DaObjectiveScope {
public:
    explicit DaObjectiveScope(nb::callable &fun) : previous_(g_da_objective) {
        g_da_objective = &fun;
    }

    ~DaObjectiveScope() {
        g_da_objective = previous_;
    }

private:
    nb::callable *previous_;
};

}  // namespace

void bind_da(nb::module_ &m) {
    m.def(
        "optimize_da",
        [](nb::callable fun, F64Vector guess, F64Vector lower, F64Vector upper,
           uint64_t seed, long runid, int max_evaluations, bool use_local_search) {
            const int dim = static_cast<int>(guess.shape(0));
            if (lower.size() != 0 && static_cast<int>(lower.shape(0)) != dim)
                throw std::invalid_argument("lower must match guess size");
            if (upper.size() != 0 && static_cast<int>(upper.shape(0)) != dim)
                throw std::invalid_argument("upper must match guess size");

            const double *lower_ptr =
                lower.size() == 0 ? nullptr : static_cast<const double *>(lower.data());
            const double *upper_ptr =
                upper.size() == 0 ? nullptr : static_cast<const double *>(upper.data());

            DaObjectiveScope scope(fun);
            auto result = without_gil([&]() {
                return dual_annealing::optimize_da(
                    runid, da_callback_trampoline, dim, static_cast<int>(seed),
                    static_cast<const double *>(guess.data()), lower_ptr, upper_ptr,
                    max_evaluations, use_local_search);
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
        "use_local_search"_a = true,
        "Execute the native dual annealing solver through nanobind."
    );
}

}  // namespace fcmaes::bindings
