// Copyright (c) Dietmar Wolz. MIT license; see the repository LICENSE.
#pragma once

#include "ndarray_utils.hpp"
#include <array>
#include <memory>
#include <utility>
#include <vector>

namespace fcmaes::bindings {
using F64Matrix = nb::ndarray<const double, nb::numpy, nb::shape<-1, -1>, nb::c_contig, nb::device::cpu>;

template <class T, std::size_t N>
auto owned_array(std::vector<T> values, const std::array<std::size_t, N>& shape) {
    if (values.empty())
        values.resize(1); // nanobind needs a non-null pointer for empty shapes
    auto storage = std::make_unique<std::vector<T>>(std::move(values));
    auto* data = storage->data();
    nb::capsule owner(storage.get(), [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
    storage.release();
    return nb::ndarray<nb::numpy, T, nb::ndim<N>, nb::c_contig>(data, N, shape.data(), owner);
}

inline auto owned_matrix(std::vector<double> values, std::size_t rows, std::size_t cols) {
    return owned_array(std::move(values), std::array<std::size_t, 2>{rows, cols});
}
inline auto owned_mask(const std::vector<unsigned char>& values) {
    auto storage = std::make_unique<bool[]>(values.size());
    for (std::size_t i = 0; i < values.size(); ++i)
        storage[i] = values[i] != 0;
    nb::capsule owner(storage.get(), [](void* p) noexcept { delete[] static_cast<bool*>(p); });
    auto* data = storage.release();
    return nb::ndarray<nb::numpy, bool, nb::shape<-1>, nb::c_contig>(data, {values.size()}, owner);
}
// Each Eigen column is contiguous and represents one candidate row.
inline auto owned_population(Mat values) {
    auto storage = std::make_unique<Mat>(std::move(values));
    auto *data = storage->data();
    const auto rows = static_cast<size_t>(storage->cols());
    const auto cols = static_cast<size_t>(storage->rows());
    nb::capsule owner(storage.get(), [](void *ptr) noexcept {
        delete static_cast<Mat *>(ptr);
    });
    storage.release();
    static const double empty = 0.;
    using Population = nb::ndarray<nb::numpy, const double, nb::shape<-1, -1>,
                                   nb::c_contig, nb::device::cpu>;
    return Population(data ? data : &empty, {rows, cols}, owner);
}

} // namespace fcmaes::bindings
