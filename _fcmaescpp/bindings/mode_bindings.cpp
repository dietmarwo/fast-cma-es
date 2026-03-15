#include "mode_bindings.hpp"

#include "gil_utils.hpp"
#include "ndarray_utils.hpp"

#include <stdexcept>

#include "modeoptimizer.hpp"

namespace fcmaes::bindings {

namespace {

using BoolVector = nb::ndarray<const bool, nb::numpy, nb::shape<-1>,
                               nb::c_contig, nb::device::cpu>;
using F64Matrix = nb::ndarray<const double, nb::numpy, nb::shape<-1, -1>,
                              nb::c_contig, nb::device::cpu>;
using ResultMatrix = nb::ndarray<nb::numpy, const double, nb::shape<-1, -1>,
                                 nb::c_contig, nb::device::cpu>;

void validate_constructor_args(int dim, int nobj, int ncon, F64Vector lower,
                               F64Vector upper, BoolVector ints) {
    if (dim <= 0)
        throw std::invalid_argument("dim must be positive");
    if (nobj <= 0)
        throw std::invalid_argument("nobj must be positive");
    if (ncon < 0)
        throw std::invalid_argument("ncon must be non-negative");
    if (lower.size() != static_cast<size_t>(dim))
        throw std::invalid_argument("lower must match dim");
    if (upper.size() != static_cast<size_t>(dim))
        throw std::invalid_argument("upper must match dim");
    if (ints.size() != 0 && ints.size() != static_cast<size_t>(dim))
        throw std::invalid_argument("ints must be empty or match dim");
}

mode_optimizer::mat as_population_matrix(F64Matrix xs, int dim) {
    if (static_cast<int>(xs.shape(1)) != dim)
        throw std::invalid_argument("xs must have shape (popsize, dim)");

    const auto popsize = static_cast<Eigen::Index>(xs.shape(0));
    mode_optimizer::mat pop(dim, popsize);
    const auto *data = static_cast<const double *>(xs.data());
    for (Eigen::Index p = 0; p < popsize; ++p)
        for (Eigen::Index i = 0; i < dim; ++i)
            pop(i, p) = data[p * dim + i];
    return pop;
}

mode_optimizer::mat as_value_matrix(F64Matrix ys, int nobj) {
    if (static_cast<int>(ys.shape(1)) != nobj)
        throw std::invalid_argument("ys must have shape (popsize, nobj + ncon)");

    const auto popsize = static_cast<Eigen::Index>(ys.shape(0));
    mode_optimizer::mat vals(nobj, popsize);
    const auto *data = static_cast<const double *>(ys.data());
    for (Eigen::Index p = 0; p < popsize; ++p)
        for (Eigen::Index i = 0; i < nobj; ++i)
            vals(i, p) = data[p * nobj + i];
    return vals;
}

ResultMatrix to_python_matrix(const mode_optimizer::mat &matrix) {
    auto *rows = new RowMat(matrix.cols(), matrix.rows());
    for (Eigen::Index p = 0; p < matrix.cols(); ++p)
        rows->row(p) = matrix.col(p).transpose();
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

void bind_mode(nb::module_ &m) {
    nb::class_<mode_optimizer::ModeState>(m, "MODE")
        .def(
            "__init__",
            [](mode_optimizer::ModeState *self, int dim, int nobj, int ncon,
               F64Vector lower, F64Vector upper, BoolVector ints, int popsize,
               double F, double CR, double pro_c, double dis_c, double pro_m,
               double dis_m, bool nsga_update, double pareto_update,
               double min_mutate, double max_mutate, uint64_t seed,
               long runid) {
                validate_constructor_args(dim, nobj, ncon, lower, upper, ints);
                new (self) mode_optimizer::ModeState(
                    runid,
                    dim,
                    nobj,
                    ncon,
                    static_cast<int>(seed),
                    static_cast<const double *>(lower.data()),
                    static_cast<const double *>(upper.data()),
                    ints.size() == 0
                        ? nullptr
                        : const_cast<bool *>(static_cast<const bool *>(ints.data())),
                    popsize,
                    F,
                    CR,
                    pro_c,
                    dis_c,
                    pro_m,
                    dis_m,
                    nsga_update,
                    pareto_update,
                    min_mutate,
                    max_mutate
                );
            },
            "dim"_a,
            "nobj"_a,
            "ncon"_a,
            "lower"_a.noconvert(),
            "upper"_a.noconvert(),
            "ints"_a.noconvert(),
            "popsize"_a = 64,
            "F"_a = 0.5,
            "CR"_a = 0.9,
            "pro_c"_a = 0.5,
            "dis_c"_a = 15.0,
            "pro_m"_a = 0.9,
            "dis_m"_a = 20.0,
            "nsga_update"_a = true,
            "pareto_update"_a = 0.0,
            "min_mutate"_a = 0.1,
            "max_mutate"_a = 0.5,
            "seed"_a,
            "runid"_a = 0L
        )
        .def("ask", [](mode_optimizer::ModeState &state) {
            return to_python_matrix(without_gil([&]() {
                return state.ask();
            }));
        })
        .def("tell", [](mode_optimizer::ModeState &state, F64Matrix ys) {
            return without_gil([&]() {
                return state.tell(
                    as_value_matrix(ys, state.nobj() + state.ncon())
                );
            });
        }, "ys"_a.noconvert())
        .def("tell_switch",
             [](mode_optimizer::ModeState &state, F64Matrix ys,
                bool nsga_update, double pareto_update) {
                 return without_gil([&]() {
                     return state.tell_switch(
                         as_value_matrix(ys, state.nobj() + state.ncon()),
                         nsga_update,
                         pareto_update
                     );
                 });
             },
             "ys"_a.noconvert(),
             "nsga_update"_a = true,
             "pareto_update"_a = 0.0)
        .def("set_population",
             [](mode_optimizer::ModeState &state, F64Matrix xs, F64Matrix ys) {
                 if (static_cast<int>(xs.shape(0)) != static_cast<int>(ys.shape(0)))
                     throw std::invalid_argument("xs and ys must have the same popsize");
                 return without_gil([&]() {
                     return state.set_population(
                         as_population_matrix(xs, state.dim()),
                         as_value_matrix(ys, state.nobj() + state.ncon())
                     );
                 });
             },
             "xs"_a.noconvert(),
             "ys"_a.noconvert())
        .def("population", [](mode_optimizer::ModeState &state) {
            return to_python_matrix(without_gil([&]() {
                return state.population();
            }));
        })
        .def_prop_ro("dim", &mode_optimizer::ModeState::dim)
        .def_prop_ro("nobj", &mode_optimizer::ModeState::nobj)
        .def_prop_ro("ncon", &mode_optimizer::ModeState::ncon)
        .def_prop_ro("popsize", &mode_optimizer::ModeState::popsize)
        .def_prop_ro("stop", &mode_optimizer::ModeState::stop);
}

}  // namespace fcmaes::bindings
