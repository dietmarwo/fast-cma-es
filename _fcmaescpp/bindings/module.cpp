#include "acma_bindings.hpp"
#include "bite_bindings.hpp"
#include "common.hpp"
#include "crfmnes_bindings.hpp"
#include "da_bindings.hpp"
#include "de_bindings.hpp"
#include "front_bindings.hpp"
#include "gil_utils.hpp"
#include "gtop_bindings.hpp"
#include "jso_bindings.hpp"
#include "mode_bindings.hpp"
#include "ndarray_utils.hpp"
#include "pgpe_bindings.hpp"
#include "pareto_bindings.hpp"
#include "owned_ndarray.hpp"
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <limits>

namespace {

nb::dict phase1_build_info() {
    nb::dict info;
    info["module"] = "_fcmaes_ext";
    info["phase"] = 1;
    info["nanobind"] = true;
    info["eigen_world_version"] = EIGEN_WORLD_VERSION;
    info["eigen_major_version"] = EIGEN_MAJOR_VERSION;
    info["eigen_minor_version"] = EIGEN_MINOR_VERSION;
    return info;
}

}  // namespace

NB_MODULE(_fcmaes_ext, m) {
    m.doc() = "Phase 1 nanobind bootstrap module for fast-cma-es.";

    m.def("phase1_build_info", &phase1_build_info,
          "Return a small dict proving that the nanobind module was built.");

    m.def(
        "_phase1_probe_sum",
        [](fcmaes::bindings::F64Vector values) {
            auto vec = fcmaes::bindings::as_const_vector(values);
            return fcmaes::bindings::without_gil([&vec]() {
                return vec.sum();
            });
        },
        "values"_a.noconvert(),
        "Small internal smoke-test for the Phase 1 nanobind bootstrap."
    );

    fcmaes::bindings::bind_acma(m);
    fcmaes::bindings::bind_bite(m);
    fcmaes::bindings::bind_crfmnes(m);
    fcmaes::bindings::bind_da(m);
    fcmaes::bindings::bind_de(m);
    fcmaes::bindings::bind_gtop(m);
    fcmaes::bindings::bind_jso(m);
    fcmaes::bindings::bind_mode(m);
    fcmaes::bindings::bind_pgpe(m);
    fcmaes::bindings::bind_pareto(m);
    fcmaes::bindings::bind_front_tools(m);
    m.def("_grid_indices",
        [](fcmaes::bindings::F64Matrix descriptors, fcmaes::bindings::F64Vector lower,
           fcmaes::bindings::F64Vector span, int capacity, int side) {
            const size_t rows = descriptors.shape(0), dim = descriptors.shape(1);
            if (dim != lower.size() || dim != span.size() || capacity <= 0 || side <= 0)
                throw std::invalid_argument("invalid grid dimensions");
            std::vector<int32_t> ids(rows);
            const auto *x = descriptors.data();
            const auto *lb = lower.data();
            const auto *scales = span.data();
            fcmaes::bindings::without_gil([&] {
                for (size_t row = 0; row < rows; ++row) {
                    int64_t id = 0, stride = 1;
                    for (size_t j = 0; j < dim; ++j) {
                        const double cell = side * ((x[row*dim+j] - lb[j]) / scales[j]);
                        if (!std::isfinite(cell) || std::abs(cell) >= 0x1p30 ||
                                stride > (int64_t{1} << 30))
                            throw std::overflow_error("descriptor outside native grid range");
                        const int64_t term = stride * static_cast<int64_t>(cell);
                        if ((term > 0 && id > std::numeric_limits<int64_t>::max() - term) ||
                            (term < 0 && id < std::numeric_limits<int64_t>::min() - term))
                            throw std::overflow_error("descriptor outside native grid range");
                        id += term;
                        stride *= side;
                    }
                    ids[row] = static_cast<int32_t>(std::clamp<int64_t>(id, 0, capacity-1));
                }
            });
            return fcmaes::bindings::owned_array(std::move(ids), std::array<size_t, 1>{rows});
        }, nb::arg("descriptors").noconvert(), nb::arg("lower").noconvert(),
           nb::arg("span").noconvert(), nb::arg("capacity"), nb::arg("side"));
}
