#include "acma_bindings.hpp"
#include "bite_bindings.hpp"
#include "common.hpp"
#include "crfmnes_bindings.hpp"
#include "da_bindings.hpp"
#include "de_bindings.hpp"
#include "gil_utils.hpp"
#include "gtop_bindings.hpp"
#include "mode_bindings.hpp"
#include "ndarray_utils.hpp"
#include "pgpe_bindings.hpp"

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
    fcmaes::bindings::bind_mode(m);
    fcmaes::bindings::bind_pgpe(m);
}
