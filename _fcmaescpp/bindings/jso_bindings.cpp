#include "jso_bindings.hpp"

#include <algorithm>
#include <memory>
#include <new>
#include <utility>

#include "jsooptimizer.hpp"
#include "ndarray_utils.hpp"

namespace fcmaes::bindings {
namespace {

using JsoVector = nb::ndarray<nb::numpy, double, nb::shape<-1>, nb::c_contig, nb::device::cpu>;
using JsoMatrix = nb::ndarray<nb::numpy, double, nb::shape<-1, -1>, nb::c_contig, nb::device::cpu>;

jso::Point copy_vector(F64Vector values) {
    if (values.size() == 0)
        return {};
    return jso::Point(values.data(), values.data() + values.size());
}

JsoVector to_vector(jso::Point values) {
    auto buffer = std::make_unique<jso::Point>(std::move(values));
    auto *data = buffer->data();
    const auto size = buffer->size();
    nb::capsule owner(buffer.get(), [](void *p) noexcept { delete static_cast<jso::Point *>(p); });
    buffer.release();
    return JsoVector(data, {size}, owner);
}

JsoMatrix to_matrix(const jso::Population &points, std::size_t dim) {
    auto buffer = std::make_unique<RowMat>(points.size(), dim);
    auto *data = buffer->data();
    for (std::size_t i = 0; i < points.size(); ++i)
        std::copy(points[i].begin(), points[i].end(), data + i * dim);
    nb::capsule owner(buffer.get(), [](void *p) noexcept { delete static_cast<RowMat *>(p); });
    buffer.release();
    return JsoMatrix(data, {points.size(), dim}, owner);
}

}  // namespace

void bind_jso(nb::module_ &m) {
    nb::class_<jso::State>(m, "Jso")
        .def("__init__", [](jso::State *state, F64Vector lower, F64Vector upper,
                            std::uint64_t seed, std::size_t popsize, std::uint64_t max_evaluations,
                            double stop_fitness, std::size_t min_popsize, std::size_t memory_size,
                            double p_max, double p_min, double archive_rate) {
            jso::Parameters parameters;
            parameters.popsize = popsize;
            parameters.max_evaluations = max_evaluations;
            parameters.stop_fitness = stop_fitness;
            parameters.min_popsize = min_popsize;
            parameters.memory_size = memory_size;
            parameters.p_max = p_max;
            parameters.p_min = p_min;
            parameters.archive_rate = archive_rate;
            new (state) jso::State(copy_vector(lower), copy_vector(upper), parameters, seed);
        }, "lower"_a.noconvert(), "upper"_a.noconvert(), "seed"_a, "popsize"_a,
           "max_evaluations"_a, "stop_fitness"_a, "min_popsize"_a, "memory_size"_a,
           "p_max"_a, "p_min"_a, "archive_rate"_a)
        .def_prop_ro("dim", &jso::State::dim)
        .def_prop_ro("popsize", &jso::State::popsize)
        .def("ask", [](jso::State &state) { return to_matrix(state.ask(), state.dim()); })
        .def("tell", [](jso::State &state, F64Vector values) {
            return state.tell(copy_vector(values));
        }, "values"_a.noconvert())
        .def("_tell_prefix", [](jso::State &state, F64Vector values) {
            return state.tell_prefix(copy_vector(values));
        }, "values"_a.noconvert())
        .def("population", [](const jso::State &state) {
            return to_matrix(state.population(), state.dim());
        })
        .def("result", [](const jso::State &state) {
            auto result = state.result();
            return nb::make_tuple(to_vector(std::move(result.x)), result.y, result.evaluations,
                                  result.iterations, result.stop);
        });
}

}  // namespace fcmaes::bindings
