#pragma once

#include "common.hpp"

namespace fcmaes::bindings {

using F64Vector = nb::ndarray<const double, nb::numpy, nb::shape<-1>, nb::c_contig, nb::device::cpu>;

inline auto as_const_vector(F64Vector values) {
    return Eigen::Map<const Vec>(
        static_cast<const double *>(values.data()),
        static_cast<Eigen::Index>(values.shape(0))
    );
}

}  // namespace fcmaes::bindings
