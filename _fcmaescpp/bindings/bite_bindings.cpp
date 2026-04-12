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
using ResultMatrix = nb::ndarray<nb::numpy, const double, nb::shape<-1, -1>,
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

void validate_bounds(F64Vector guess, F64Vector lower, F64Vector upper) {
    const int dim = static_cast<int>(guess.shape(0));
    if (dim <= 0)
        throw std::invalid_argument("guess must be non-empty");
    if (lower.size() != upper.size())
        throw std::invalid_argument("lower and upper must both be empty or both be set");
    if (lower.size() != 0 && static_cast<int>(lower.shape(0)) != dim)
        throw std::invalid_argument("lower must match guess size");
    if (upper.size() != 0 && static_cast<int>(upper.shape(0)) != dim)
        throw std::invalid_argument("upper must match guess size");
}

ResultMatrix to_python_matrix(const Mat &population) {
    auto *rows = new RowMat(population.cols(), population.rows());
    for (Eigen::Index p = 0; p < population.cols(); p++)
        rows->row(p) = population.col(p).transpose();
    nb::capsule owner(rows, [](void *ptr) noexcept {
        delete static_cast<RowMat *>(ptr);
    });
    return ResultMatrix(
        rows->data(),
        {
            static_cast<size_t>(rows->rows()),
            static_cast<size_t>(rows->cols()),
        },
        owner
    );
}

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

    nb::class_<biteopt::BiteState>(m, "Bite")
        .def(
            "__init__",
            [](biteopt::BiteState *self, F64Vector guess, F64Vector lower,
               F64Vector upper, int M, int popsize, int batch_size,
               int max_evaluations, double stop_fitness, int stall_criterion,
               uint64_t seed, long runid) {
                validate_bounds(guess, lower, upper);
                new (self) biteopt::BiteState(
                    runid,
                    static_cast<int>(guess.shape(0)),
                    static_cast<const double *>(guess.data()),
                    lower.size() == 0 ? nullptr : static_cast<const double *>(lower.data()),
                    upper.size() == 0 ? nullptr : static_cast<const double *>(upper.data()),
                    static_cast<int>(seed),
                    M,
                    popsize,
                    batch_size,
                    max_evaluations,
                    stop_fitness,
                    stall_criterion
                );
            },
            "guess"_a.noconvert(),
            "lower"_a.noconvert(),
            "upper"_a.noconvert(),
            "M"_a = 1,
            "popsize"_a = 0,
            "batch_size"_a = 8,
            "max_evaluations"_a = 100000,
            "stop_fitness"_a = -std::numeric_limits<double>::infinity(),
            "stall_criterion"_a = 0,
            "seed"_a,
            "runid"_a = 0L
        )
        .def("ask", [](biteopt::BiteState &state) {
            return to_python_matrix(without_gil([&]() {
                return state.ask();
            }));
        })
        .def("tell", [](biteopt::BiteState &state, F64Vector ys) {
            if (static_cast<int>(ys.shape(0)) != state.current_batch_size())
                throw std::invalid_argument("ys must match the current batch size");
            Vec values = as_const_vector(ys);
            return without_gil([&]() {
                return state.tell(values);
            });
        }, "ys"_a.noconvert())
        .def("result", [](biteopt::BiteState &state) {
            return to_python_result(without_gil([&]() {
                return state.result();
            }));
        })
        .def_prop_ro("dim", &biteopt::BiteState::dim)
        .def_prop_ro("popsize", &biteopt::BiteState::popsize)
        .def_prop_ro("population_size", &biteopt::BiteState::population_size)
        .def_prop_ro(
            "current_batch_size", &biteopt::BiteState::current_batch_size
        )
        .def_prop_ro("stop", &biteopt::BiteState::stop);
}

}  // namespace fcmaes::bindings
