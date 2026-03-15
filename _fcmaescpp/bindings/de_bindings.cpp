#include "de_bindings.hpp"

#include "gil_utils.hpp"
#include "ndarray_utils.hpp"

#include <atomic>
#include <cfloat>
#include <cmath>
#include <limits>
#include <mutex>
#include <stdexcept>

#include "deoptimizer.hpp"

namespace fcmaes::bindings {

namespace {

using BoolVector = nb::ndarray<const bool, nb::numpy, nb::shape<-1>,
                               nb::c_contig, nb::device::cpu>;
using ResultVector = nb::ndarray<nb::numpy, const double, nb::shape<-1>,
                                 nb::c_contig, nb::device::cpu>;
using ResultMatrix = nb::ndarray<nb::numpy, const double, nb::shape<-1, -1>,
                                 nb::c_contig, nb::device::cpu>;

std::mutex g_de_callback_mutex;
std::atomic<nb::callable *> g_de_objective{ nullptr };
std::atomic<nb::object *> g_de_terminate{ nullptr };

class DeObjectiveScope {
public:
    DeObjectiveScope(nb::callable &objective, nb::object &terminate)
            : lock_(g_de_callback_mutex) {
        g_de_objective.store(&objective, std::memory_order_release);
        g_de_terminate.store(&terminate, std::memory_order_release);
    }

    ~DeObjectiveScope() {
        g_de_terminate.store(nullptr, std::memory_order_release);
        g_de_objective.store(nullptr, std::memory_order_release);
    }

private:
    std::unique_lock<std::mutex> lock_;
};

void validate_problem(int dim, F64Vector lower, F64Vector upper, F64Vector guess,
                      F64Vector sigma, BoolVector ints) {
    if (dim <= 0)
        throw std::invalid_argument("dim must be positive");
    if (lower.size() != upper.size())
        throw std::invalid_argument("lower and upper must both be empty or both be set");
    if (lower.size() != 0 && static_cast<int>(lower.shape(0)) != dim)
        throw std::invalid_argument("lower must match dim");
    if (upper.size() != 0 && static_cast<int>(upper.shape(0)) != dim)
        throw std::invalid_argument("upper must match dim");
    if (guess.size() != sigma.size())
        throw std::invalid_argument("guess and sigma must both be empty or both be set");
    if (guess.size() != 0 && static_cast<int>(guess.shape(0)) != dim)
        throw std::invalid_argument("guess must match dim");
    if (sigma.size() != 0 && static_cast<int>(sigma.shape(0)) != dim)
        throw std::invalid_argument("sigma must match dim");
    if (ints.size() != 0 && static_cast<int>(ints.shape(0)) != dim)
        throw std::invalid_argument("ints must match dim");
}

ResultVector to_python_vector(Vec &&values) {
    auto *x = new Vec(std::move(values));
    nb::capsule owner(x, [](void *ptr) noexcept {
        delete static_cast<Vec *>(ptr);
    });
    return ResultVector(x->data(), { static_cast<size_t>(x->size()) }, owner);
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

nb::tuple to_python_result(differential_evolution::DeResult result) {
    return nb::make_tuple(
        to_python_vector(std::move(result.x)),
        result.y,
        result.evaluations,
        result.iterations,
        result.stop
    );
}

bool de_callback_trampoline(int dim, const double *x, double *y) {
    auto *objective = g_de_objective.load(std::memory_order_acquire);
    if (objective == nullptr) {
        y[0] = DBL_MAX;
        return false;
    }

    try {
        nb::gil_scoped_acquire acquire;
        F64Vector x_view(x, { static_cast<size_t>(dim) });
        double value = nb::cast<double>((*objective)(x_view));
        y[0] = std::isfinite(value) ? value : DBL_MAX;

        auto *terminate = g_de_terminate.load(std::memory_order_acquire);
        if (terminate != nullptr && !terminate->is_none()) {
            ResultVector y_view(y, { size_t(1) });
            return nb::cast<bool>((*terminate)(x_view, y_view));
        }
        return false;
    } catch (const std::exception &) {
        y[0] = DBL_MAX;
        return false;
    }
}

}  // namespace

void bind_de(nb::module_ &m) {
    m.def(
        "optimize_de",
        [](nb::callable fun, int dim, F64Vector lower, F64Vector upper,
           F64Vector guess, F64Vector sigma, BoolVector ints, uint64_t seed,
           long runid, int max_evaluations, double keep, double stop_fitness,
           int popsize, double F, double CR, double min_sigma,
           double min_mutate, double max_mutate, int workers,
           nb::object terminate) {
            validate_problem(dim, lower, upper, guess, sigma, ints);

            nb::callable objective_ref = fun;
            nb::object terminate_ref = terminate;
            DeObjectiveScope scope(objective_ref, terminate_ref);
            auto result = without_gil([&]() {
                return differential_evolution::optimize_de(
                    runid,
                    de_callback_trampoline,
                    dim,
                    static_cast<int>(seed),
                    lower.size() == 0 ? nullptr : static_cast<const double *>(lower.data()),
                    upper.size() == 0 ? nullptr : static_cast<const double *>(upper.data()),
                    guess.size() == 0 ? nullptr : static_cast<const double *>(guess.data()),
                    sigma.size() == 0 ? nullptr : static_cast<const double *>(sigma.data()),
                    min_sigma,
                    ints.size() == 0
                        ? nullptr
                        : const_cast<bool *>(static_cast<const bool *>(ints.data())),
                    max_evaluations,
                    keep,
                    stop_fitness,
                    popsize,
                    F,
                    CR,
                    min_mutate,
                    max_mutate,
                    workers
                );
            });
            return to_python_result(std::move(result));
        },
        "fun"_a,
        "dim"_a,
        "lower"_a.noconvert(),
        "upper"_a.noconvert(),
        "guess"_a.noconvert(),
        "sigma"_a.noconvert(),
        "ints"_a.noconvert(),
        "seed"_a,
        "runid"_a = 0L,
        "max_evaluations"_a = 100000,
        "keep"_a = 200.0,
        "stop_fitness"_a = -std::numeric_limits<double>::infinity(),
        "popsize"_a = 31,
        "F"_a = 0.5,
        "CR"_a = 0.9,
        "min_sigma"_a = 0.0,
        "min_mutate"_a = 0.1,
        "max_mutate"_a = 0.5,
        "workers"_a = 1,
        "terminate"_a = nb::none(),
        "Execute the native DE solver through nanobind."
    );

    nb::class_<differential_evolution::DeState>(m, "DE")
        .def(
            "__init__",
            [](differential_evolution::DeState *self, int dim, F64Vector lower,
               F64Vector upper, F64Vector guess, F64Vector sigma,
               BoolVector ints, int popsize, double keep, double F, double CR,
               double min_sigma, double min_mutate, double max_mutate,
               uint64_t seed, long runid) {
                validate_problem(dim, lower, upper, guess, sigma, ints);
                new (self) differential_evolution::DeState(
                    runid,
                    dim,
                    static_cast<int>(seed),
                    lower.size() == 0 ? nullptr : static_cast<const double *>(lower.data()),
                    upper.size() == 0 ? nullptr : static_cast<const double *>(upper.data()),
                    guess.size() == 0 ? nullptr : static_cast<const double *>(guess.data()),
                    sigma.size() == 0 ? nullptr : static_cast<const double *>(sigma.data()),
                    min_sigma,
                    ints.size() == 0
                        ? nullptr
                        : const_cast<bool *>(static_cast<const bool *>(ints.data())),
                    keep,
                    popsize,
                    F,
                    CR,
                    min_mutate,
                    max_mutate
                );
            },
            "dim"_a,
            "lower"_a.noconvert(),
            "upper"_a.noconvert(),
            "guess"_a.noconvert(),
            "sigma"_a.noconvert(),
            "ints"_a.noconvert(),
            "popsize"_a = 31,
            "keep"_a = 200.0,
            "F"_a = 0.5,
            "CR"_a = 0.9,
            "min_sigma"_a = 0.0,
            "min_mutate"_a = 0.1,
            "max_mutate"_a = 0.5,
            "seed"_a,
            "runid"_a = 0L
        )
        .def("ask", [](differential_evolution::DeState &state) {
            return to_python_matrix(without_gil([&]() {
                return state.ask();
            }));
        })
        .def("tell", [](differential_evolution::DeState &state, F64Vector ys) {
            if (static_cast<int>(ys.shape(0)) != state.popsize())
                throw std::invalid_argument("ys must match the population size");
            Vec values = as_const_vector(ys);
            return without_gil([&]() {
                return state.tell(values);
            });
        }, "ys"_a.noconvert())
        .def("population", [](differential_evolution::DeState &state) {
            return to_python_matrix(without_gil([&]() {
                return state.population();
            }));
        })
        .def("result", [](differential_evolution::DeState &state) {
            return to_python_result(without_gil([&]() {
                return state.result();
            }));
        })
        .def_prop_ro("dim", &differential_evolution::DeState::dim)
        .def_prop_ro("popsize", &differential_evolution::DeState::popsize)
        .def_prop_ro("stop", &differential_evolution::DeState::stop);
}

}  // namespace fcmaes::bindings
