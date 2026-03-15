#pragma once

#include <Eigen/Core>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace fcmaes::bindings {

using Vec = Eigen::Matrix<double, Eigen::Dynamic, 1>;
using Mat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

}  // namespace fcmaes::bindings
