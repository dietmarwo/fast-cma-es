#include "acma_bindings.hpp"

#include "gil_utils.hpp"
#include "ndarray_utils.hpp"

#include <atomic>
#include <cfloat>
#include <cmath>
#include <limits>
#include <mutex>
#include <stdexcept>

#include "acmaesoptimizer.hpp"

namespace fcmaes::bindings {

namespace {

using BatchInput = nb::ndarray<
    nb::numpy, const double, nb::shape<-1, -1>, nb::c_contig, nb::device::cpu>;
using InputMatrix = nb::ndarray<
    nb::numpy, const double, nb::shape<-1, -1>, nb::c_contig, nb::device::cpu>;
using ResultVector = nb::ndarray<
    nb::numpy, const double, nb::shape<-1>, nb::c_contig, nb::device::cpu>;
using ResultMatrix = nb::ndarray<
    nb::numpy, const double, nb::shape<-1, -1>, nb::c_contig, nb::device::cpu>;

std::mutex g_acma_callback_mutex;
std::atomic<nb::callable *> g_acma_objective{ nullptr };
std::atomic<nb::object *> g_acma_batch{ nullptr };

class AcmaCallbackScope {
public:
    AcmaCallbackScope(nb::callable &objective, nb::object &batch)
            : lock_(g_acma_callback_mutex) {
        g_acma_objective.store(&objective, std::memory_order_release);
        g_acma_batch.store(&batch, std::memory_order_release);
    }

    ~AcmaCallbackScope() {
        g_acma_batch.store(nullptr, std::memory_order_release);
        g_acma_objective.store(nullptr, std::memory_order_release);
    }

private:
    std::unique_lock<std::mutex> lock_;
};

void validate_bounds(F64Vector guess, F64Vector lower, F64Vector upper,
                     F64Vector sigma) {
    const int dim = static_cast<int>(guess.shape(0));
    if (lower.size() != upper.size())
        throw std::invalid_argument("lower and upper must both be empty or both be set");
    if (lower.size() != 0 && static_cast<int>(lower.shape(0)) != dim)
        throw std::invalid_argument("lower must match guess size");
    if (upper.size() != 0 && static_cast<int>(upper.shape(0)) != dim)
        throw std::invalid_argument("upper must match guess size");
    if (static_cast<int>(sigma.shape(0)) != dim)
        throw std::invalid_argument("sigma must match guess size");
}

Mat rows_to_columns(InputMatrix rows) {
    const int popsize = static_cast<int>(rows.shape(0));
    const int dim = static_cast<int>(rows.shape(1));
    Mat columns(dim, popsize);
    const double *data = static_cast<const double *>(rows.data());
    for (int p = 0; p < popsize; p++) {
        for (int i = 0; i < dim; i++)
            columns(i, p) = data[p * dim + i];
    }
    return columns;
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

nb::tuple to_python_result(acmaes::AcmaResult result) {
    return nb::make_tuple(
        to_python_vector(std::move(result.x)),
        result.y,
        result.evaluations,
        result.iterations,
        result.stop
    );
}

bool acma_objective_trampoline(int dim, const double *x, double *y) {
    auto *objective = g_acma_objective.load(std::memory_order_acquire);
    if (objective == nullptr) {
        y[0] = DBL_MAX;
        return false;
    }

    try {
        nb::gil_scoped_acquire acquire;
        F64Vector x_view(x, { static_cast<size_t>(dim) });
        double value = nb::cast<double>((*objective)(x_view));
        y[0] = std::isfinite(value) ? value : DBL_MAX;
    } catch (const std::exception &) {
        y[0] = DBL_MAX;
    }
    return false;
}

void acma_batch_trampoline(int popsize, int dim, double *xs, double *ys) {
    auto *batch = g_acma_batch.load(std::memory_order_acquire);
    auto *objective = g_acma_objective.load(std::memory_order_acquire);
    if ((batch == nullptr || batch->is_none()) && objective == nullptr)
        throw std::runtime_error("ACMA callbacks were not initialized");

    nb::gil_scoped_acquire acquire;
    if (batch != nullptr && !batch->is_none()) {
        BatchInput xs_view(xs, {
            static_cast<size_t>(popsize),
            static_cast<size_t>(dim),
        });
        ResultVector values = nb::cast<ResultVector>((*batch)(xs_view));
        if (static_cast<int>(values.shape(0)) != popsize)
            throw std::invalid_argument("ACMA batch callback returned wrong length");
        Eigen::Map<const Vec> mapped(
            static_cast<const double *>(values.data()),
            static_cast<Eigen::Index>(values.shape(0))
        );
        for (int i = 0; i < popsize; i++) {
            double value = mapped[i];
            ys[i] = std::isfinite(value) ? value : DBL_MAX;
        }
        return;
    }

    for (int p = 0; p < popsize; p++) {
        try {
            F64Vector x_view(xs + p * dim, { static_cast<size_t>(dim) });
            double value = nb::cast<double>((*objective)(x_view));
            ys[p] = std::isfinite(value) ? value : DBL_MAX;
        } catch (const std::exception &) {
            ys[p] = DBL_MAX;
        }
    }
}

}  // namespace

void bind_acma(nb::module_ &m) {
    m.def(
        "optimize_acma",
        [](nb::callable fun, nb::object batch_fun, F64Vector guess,
           F64Vector lower, F64Vector upper, F64Vector sigma, uint64_t seed,
           long runid, int max_evaluations, double stop_fitness,
           double stop_hist, int mu, int popsize, double accuracy,
           bool normalize, bool delayed_update, int update_gap, int workers) {
            validate_bounds(guess, lower, upper, sigma);

            nb::callable objective_ref = fun;
            nb::object batch_ref = batch_fun;
            AcmaCallbackScope scope(objective_ref, batch_ref);
            auto result = without_gil([&]() {
                return acmaes::optimize_acma(
                    runid,
                    acma_objective_trampoline,
                    acma_batch_trampoline,
                    static_cast<int>(guess.shape(0)),
                    static_cast<const double *>(guess.data()),
                    lower.size() == 0 ? nullptr : static_cast<const double *>(lower.data()),
                    upper.size() == 0 ? nullptr : static_cast<const double *>(upper.data()),
                    static_cast<const double *>(sigma.data()),
                    max_evaluations,
                    stop_fitness,
                    stop_hist,
                    mu,
                    popsize,
                    accuracy,
                    static_cast<long>(seed),
                    normalize,
                    delayed_update,
                    update_gap,
                    workers
                );
            });
            return to_python_result(std::move(result));
        },
        "fun"_a,
        "batch_fun"_a = nb::none(),
        "guess"_a.noconvert(),
        "lower"_a.noconvert(),
        "upper"_a.noconvert(),
        "sigma"_a.noconvert(),
        "seed"_a,
        "runid"_a = 0L,
        "max_evaluations"_a = 100000,
        "stop_fitness"_a = -std::numeric_limits<double>::infinity(),
        "stop_hist"_a = -1.0,
        "mu"_a = 0,
        "popsize"_a = 31,
        "accuracy"_a = 1.0,
        "normalize"_a = true,
        "delayed_update"_a = true,
        "update_gap"_a = -1,
        "workers"_a = 1,
        "Execute the native ACMA solver through nanobind."
    );

    nb::class_<acmaes::AcmaState>(m, "ACMA")
        .def(
            "__init__",
            [](acmaes::AcmaState *self, F64Vector guess, F64Vector lower,
               F64Vector upper, F64Vector sigma, int max_evaluations,
               double stop_fitness, double stop_hist, int mu, int popsize,
               double accuracy, uint64_t seed, long runid, bool normalize,
               bool delayed_update, int update_gap) {
                validate_bounds(guess, lower, upper, sigma);
                new (self) acmaes::AcmaState(
                    runid,
                    static_cast<int>(guess.shape(0)),
                    static_cast<const double *>(guess.data()),
                    lower.size() == 0 ? nullptr : static_cast<const double *>(lower.data()),
                    upper.size() == 0 ? nullptr : static_cast<const double *>(upper.data()),
                    static_cast<const double *>(sigma.data()),
                    max_evaluations,
                    stop_fitness,
                    stop_hist,
                    mu,
                    popsize,
                    accuracy,
                    static_cast<long>(seed),
                    normalize,
                    delayed_update,
                    update_gap
                );
            },
            "guess"_a.noconvert(),
            "lower"_a.noconvert(),
            "upper"_a.noconvert(),
            "sigma"_a.noconvert(),
            "max_evaluations"_a = 100000,
            "stop_fitness"_a = -std::numeric_limits<double>::infinity(),
            "stop_hist"_a = -1.0,
            "mu"_a = 0,
            "popsize"_a = 31,
            "accuracy"_a = 1.0,
            "seed"_a,
            "runid"_a = 0L,
            "normalize"_a = true,
            "delayed_update"_a = true,
            "update_gap"_a = -1
        )
        .def("ask", [](acmaes::AcmaState &state) {
            return to_python_matrix(without_gil([&]() {
                return state.ask();
            }));
        })
        .def("tell", [](acmaes::AcmaState &state, F64Vector ys) {
            if (static_cast<int>(ys.shape(0)) != state.popsize())
                throw std::invalid_argument("ys must match the population size");
            Vec values = as_const_vector(ys);
            return without_gil([&]() {
                return state.tell(values);
            });
        }, "ys"_a.noconvert())
        .def("tell_x", [](acmaes::AcmaState &state, F64Vector ys, InputMatrix xs) {
            if (static_cast<int>(ys.shape(0)) != state.popsize())
                throw std::invalid_argument("ys must match the population size");
            if (static_cast<int>(xs.shape(0)) != state.popsize())
                throw std::invalid_argument("xs must match the population size");
            if (static_cast<int>(xs.shape(1)) != state.dim())
                throw std::invalid_argument("xs must match the problem dimension");
            Vec values = as_const_vector(ys);
            Mat population = rows_to_columns(xs);
            return without_gil([&]() {
                return state.tell_x(values, population);
            });
        }, "ys"_a.noconvert(), "xs"_a.noconvert())
        .def("population", [](acmaes::AcmaState &state) {
            return to_python_matrix(without_gil([&]() {
                return state.population();
            }));
        })
        .def("result", [](acmaes::AcmaState &state) {
            return to_python_result(without_gil([&]() {
                return state.result();
            }));
        })
        .def_prop_ro("dim", &acmaes::AcmaState::dim)
        .def_prop_ro("popsize", &acmaes::AcmaState::popsize)
        .def_prop_ro("stop", &acmaes::AcmaState::stop);
}

}  // namespace fcmaes::bindings
