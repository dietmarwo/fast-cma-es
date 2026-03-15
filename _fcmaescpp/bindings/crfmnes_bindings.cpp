#include "crfmnes_bindings.hpp"

#include "gil_utils.hpp"
#include "ndarray_utils.hpp"

#include <cfloat>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "crfmnesoptimizer.hpp"

namespace fcmaes::bindings {

namespace {

using BatchInput = nb::ndarray<
    nb::numpy, const double, nb::shape<-1, -1>, nb::c_contig, nb::device::cpu>;
using ResultVector = nb::ndarray<
    nb::numpy, const double, nb::shape<-1>, nb::c_contig, nb::device::cpu>;
using ResultMatrix = nb::ndarray<
    nb::numpy, const double, nb::shape<-1, -1>, nb::c_contig, nb::device::cpu>;

thread_local nb::callable *g_crfmnes_batch = nullptr;

class BatchScope {
public:
    explicit BatchScope(nb::callable &batch) : previous_(g_crfmnes_batch) {
        g_crfmnes_batch = &batch;
    }

    ~BatchScope() {
        g_crfmnes_batch = previous_;
    }

private:
    nb::callable *previous_;
};

void crfmnes_batch_trampoline(int popsize, int dim, double *xs, double *ys) {
    if (g_crfmnes_batch == nullptr)
        throw std::runtime_error("CR-FM-NES batch callback was not initialized");

    nb::gil_scoped_acquire acquire;
    BatchInput xs_view(xs, {
        static_cast<size_t>(popsize),
        static_cast<size_t>(dim),
    });
    ResultVector values = nb::cast<ResultVector>((*g_crfmnes_batch)(xs_view));
    if (static_cast<int>(values.shape(0)) != popsize)
        throw std::invalid_argument("CR-FM-NES batch callback returned wrong length");

    Eigen::Map<const Vec> mapped(
        static_cast<const double *>(values.data()),
        static_cast<Eigen::Index>(values.shape(0))
    );
    for (int i = 0; i < popsize; i++) {
        double value = mapped[i];
        ys[i] = std::isfinite(value) ? value : DBL_MAX;
    }
}

void validate_bounds(F64Vector guess, F64Vector lower, F64Vector upper) {
    const int dim = static_cast<int>(guess.shape(0));
    if (lower.size() != 0 && static_cast<int>(lower.shape(0)) != dim)
        throw std::invalid_argument("lower must match guess size");
    if (upper.size() != 0 && static_cast<int>(upper.shape(0)) != dim)
        throw std::invalid_argument("upper must match guess size");
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

nb::tuple to_python_result(crmfnes::CrfmnesResult result) {
    return nb::make_tuple(
        to_python_vector(std::move(result.x)),
        result.y,
        result.evaluations,
        result.iterations,
        result.stop
    );
}

}  // namespace

void bind_crfmnes(nb::module_ &m) {
    m.def(
        "optimize_crfmnes",
        [](nb::callable batch_fun, F64Vector guess, F64Vector lower,
           F64Vector upper, double sigma, uint64_t seed, long runid,
           int max_evaluations, double stop_fitness, int popsize,
           double penalty_coef, bool use_constraint_violation, bool normalize) {
            validate_bounds(guess, lower, upper);
            const int dim = static_cast<int>(guess.shape(0));
            const double *lower_ptr =
                lower.size() == 0 ? nullptr : static_cast<const double *>(lower.data());
            const double *upper_ptr =
                upper.size() == 0 ? nullptr : static_cast<const double *>(upper.data());

            BatchScope scope(batch_fun);
            auto result = without_gil([&]() {
                return crmfnes::optimize_crfmnes(
                    runid, crfmnes_batch_trampoline, dim,
                    static_cast<const double *>(guess.data()), lower_ptr, upper_ptr,
                    sigma, max_evaluations, stop_fitness, popsize,
                    static_cast<int64_t>(seed), penalty_coef,
                    use_constraint_violation, normalize);
            });
            return to_python_result(std::move(result));
        },
        "batch_fun"_a,
        "guess"_a.noconvert(),
        "lower"_a.noconvert(),
        "upper"_a.noconvert(),
        "sigma"_a = 0.3,
        "seed"_a,
        "runid"_a = 0L,
        "max_evaluations"_a = 100000,
        "stop_fitness"_a = -std::numeric_limits<double>::infinity(),
        "popsize"_a = 32,
        "penalty_coef"_a = 1e5,
        "use_constraint_violation"_a = true,
        "normalize"_a = false,
        "Execute the native CR-FM-NES solver through nanobind."
    );

    nb::class_<crmfnes::CrfmnesState>(m, "CRFMNES")
        .def(
            "__init__",
            [](crmfnes::CrfmnesState *self, F64Vector guess, F64Vector lower,
               F64Vector upper, double sigma, int popsize, uint64_t seed,
               long runid, double penalty_coef, bool use_constraint_violation,
               bool normalize) {
                validate_bounds(guess, lower, upper);
                new (self) crmfnes::CrfmnesState(
                    runid, static_cast<int>(guess.shape(0)),
                    static_cast<const double *>(guess.data()),
                    lower.size() == 0 ? nullptr : static_cast<const double *>(lower.data()),
                    upper.size() == 0 ? nullptr : static_cast<const double *>(upper.data()),
                    sigma, popsize, static_cast<int64_t>(seed), penalty_coef,
                    use_constraint_violation, normalize
                );
            },
            "guess"_a.noconvert(),
            "lower"_a.noconvert(),
            "upper"_a.noconvert(),
            "sigma"_a = 0.3,
            "popsize"_a = 32,
            "seed"_a,
            "runid"_a = 0L,
            "penalty_coef"_a = 1e5,
            "use_constraint_violation"_a = true,
            "normalize"_a = false
        )
        .def("ask", [](crmfnes::CrfmnesState &state) {
            return to_python_matrix(without_gil([&]() {
                return state.ask();
            }));
        })
        .def("tell", [](crmfnes::CrfmnesState &state, F64Vector ys) {
            if (static_cast<int>(ys.shape(0)) != state.popsize())
                throw std::invalid_argument("ys must match the population size");
            Vec values = as_const_vector(ys);
            return without_gil([&]() {
                return state.tell(values);
            });
        }, "ys"_a.noconvert())
        .def("population", [](crmfnes::CrfmnesState &state) {
            return to_python_matrix(without_gil([&]() {
                return state.population();
            }));
        })
        .def("result", [](crmfnes::CrfmnesState &state) {
            return to_python_result(without_gil([&]() {
                return state.result();
            }));
        })
        .def_prop_ro("dim", &crmfnes::CrfmnesState::dim)
        .def_prop_ro("popsize", &crmfnes::CrfmnesState::popsize);
}

}  // namespace fcmaes::bindings
