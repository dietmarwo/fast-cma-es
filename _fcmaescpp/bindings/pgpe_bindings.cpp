#include "pgpe_bindings.hpp"

#include "gil_utils.hpp"
#include "ndarray_utils.hpp"

#include <cfloat>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "pgpeoptimizer.hpp"

namespace fcmaes::bindings {

namespace {

using BatchInput = nb::ndarray<
    nb::numpy, const double, nb::shape<-1, -1>, nb::c_contig, nb::device::cpu>;
using ResultVector = nb::ndarray<
    nb::numpy, const double, nb::shape<-1>, nb::c_contig, nb::device::cpu>;
using ResultMatrix = nb::ndarray<
    nb::numpy, const double, nb::shape<-1, -1>, nb::c_contig, nb::device::cpu>;

thread_local nb::callable *g_pgpe_batch = nullptr;

class BatchScope {
public:
    explicit BatchScope(nb::callable &batch) : previous_(g_pgpe_batch) {
        g_pgpe_batch = &batch;
    }

    ~BatchScope() {
        g_pgpe_batch = previous_;
    }

private:
    nb::callable *previous_;
};

void pgpe_batch_trampoline(int popsize, int dim, double *xs, double *ys) {
    if (g_pgpe_batch == nullptr)
        throw std::runtime_error("PGPE batch callback was not initialized");

    nb::gil_scoped_acquire acquire;
    BatchInput xs_view(xs, {
        static_cast<size_t>(popsize),
        static_cast<size_t>(dim),
    });
    ResultVector values = nb::cast<ResultVector>((*g_pgpe_batch)(xs_view));
    if (static_cast<int>(values.shape(0)) != popsize)
        throw std::invalid_argument("PGPE batch callback returned wrong length");

    Eigen::Map<const Vec> mapped(
        static_cast<const double *>(values.data()),
        static_cast<Eigen::Index>(values.shape(0))
    );
    for (int i = 0; i < popsize; i++) {
        double value = mapped[i];
        ys[i] = std::isfinite(value) ? value : DBL_MAX;
    }
}

void validate_inputs(F64Vector guess, F64Vector lower, F64Vector upper,
                     F64Vector sigma) {
    const int dim = static_cast<int>(guess.shape(0));
    if (lower.size() != 0 && static_cast<int>(lower.shape(0)) != dim)
        throw std::invalid_argument("lower must match guess size");
    if (upper.size() != 0 && static_cast<int>(upper.shape(0)) != dim)
        throw std::invalid_argument("upper must match guess size");
    if (static_cast<int>(sigma.shape(0)) != dim)
        throw std::invalid_argument("sigma must match guess size");
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

nb::tuple to_python_result(pgpe::PgpeResult result) {
    return nb::make_tuple(
        to_python_vector(std::move(result.x)),
        result.y,
        result.evaluations,
        result.iterations,
        result.stop
    );
}

}  // namespace

void bind_pgpe(nb::module_ &m) {
    m.def(
        "optimize_pgpe",
        [](nb::callable batch_fun, F64Vector guess, F64Vector lower,
           F64Vector upper, F64Vector sigma, uint64_t seed, long runid,
           int max_evaluations, double stop_fitness, int popsize,
           int lr_decay_steps, bool use_ranking, double center_learning_rate,
           double stdev_learning_rate, double stdev_max_change, double b1,
           double b2, double eps, double decay_coef, bool normalize) {
            validate_inputs(guess, lower, upper, sigma);
            const int dim = static_cast<int>(guess.shape(0));
            const double *lower_ptr =
                lower.size() == 0 ? nullptr : static_cast<const double *>(lower.data());
            const double *upper_ptr =
                upper.size() == 0 ? nullptr : static_cast<const double *>(upper.data());

            BatchScope scope(batch_fun);
            auto result = without_gil([&]() {
                return pgpe::optimize_pgpe(
                    runid, pgpe_batch_trampoline, dim,
                    static_cast<const double *>(guess.data()), lower_ptr, upper_ptr,
                    static_cast<const double *>(sigma.data()), max_evaluations,
                    stop_fitness, popsize, static_cast<int64_t>(seed),
                    lr_decay_steps, use_ranking, center_learning_rate,
                    stdev_learning_rate, stdev_max_change, b1, b2, eps,
                    decay_coef, normalize
                );
            });
            return to_python_result(std::move(result));
        },
        "batch_fun"_a,
        "guess"_a.noconvert(),
        "lower"_a.noconvert(),
        "upper"_a.noconvert(),
        "sigma"_a.noconvert(),
        "seed"_a,
        "runid"_a = 0L,
        "max_evaluations"_a = 100000,
        "stop_fitness"_a = -std::numeric_limits<double>::infinity(),
        "popsize"_a = 32,
        "lr_decay_steps"_a = 1000,
        "use_ranking"_a = true,
        "center_learning_rate"_a = 0.15,
        "stdev_learning_rate"_a = 0.1,
        "stdev_max_change"_a = 0.2,
        "b1"_a = 0.9,
        "b2"_a = 0.999,
        "eps"_a = 1e-8,
        "decay_coef"_a = 1.0,
        "normalize"_a = true,
        "Execute the native PGPE solver through nanobind."
    );

    nb::class_<pgpe::PgpeState>(m, "PGPE")
        .def(
            "__init__",
            [](pgpe::PgpeState *self, F64Vector guess, F64Vector lower,
               F64Vector upper, F64Vector sigma, int popsize, uint64_t seed,
               long runid, int lr_decay_steps, bool use_ranking,
               double center_learning_rate, double stdev_learning_rate,
               double stdev_max_change, double b1, double b2, double eps,
               double decay_coef, bool normalize) {
                validate_inputs(guess, lower, upper, sigma);
                new (self) pgpe::PgpeState(
                    runid, static_cast<int>(guess.shape(0)),
                    static_cast<const double *>(guess.data()),
                    lower.size() == 0 ? nullptr : static_cast<const double *>(lower.data()),
                    upper.size() == 0 ? nullptr : static_cast<const double *>(upper.data()),
                    static_cast<const double *>(sigma.data()), popsize,
                    static_cast<int64_t>(seed), lr_decay_steps, use_ranking,
                    center_learning_rate, stdev_learning_rate,
                    stdev_max_change, b1, b2, eps, decay_coef, normalize
                );
            },
            "guess"_a.noconvert(),
            "lower"_a.noconvert(),
            "upper"_a.noconvert(),
            "sigma"_a.noconvert(),
            "popsize"_a = 32,
            "seed"_a,
            "runid"_a = 0L,
            "lr_decay_steps"_a = 1000,
            "use_ranking"_a = false,
            "center_learning_rate"_a = 0.15,
            "stdev_learning_rate"_a = 0.1,
            "stdev_max_change"_a = 0.2,
            "b1"_a = 0.9,
            "b2"_a = 0.999,
            "eps"_a = 1e-8,
            "decay_coef"_a = 1.0,
            "normalize"_a = true
        )
        .def("ask", [](pgpe::PgpeState &state) {
            return to_python_matrix(without_gil([&]() {
                return state.ask();
            }));
        })
        .def("tell", [](pgpe::PgpeState &state, F64Vector ys) {
            if (static_cast<int>(ys.shape(0)) != state.popsize())
                throw std::invalid_argument("ys must match the population size");
            Vec values = as_const_vector(ys);
            return without_gil([&]() {
                return state.tell(values);
            });
        }, "ys"_a.noconvert())
        .def("population", [](pgpe::PgpeState &state) {
            return to_python_matrix(without_gil([&]() {
                return state.population();
            }));
        })
        .def("result", [](pgpe::PgpeState &state) {
            return to_python_result(without_gil([&]() {
                return state.result();
            }));
        })
        .def_prop_ro("dim", &pgpe::PgpeState::dim)
        .def_prop_ro("popsize", &pgpe::PgpeState::popsize);
}

}  // namespace fcmaes::bindings
