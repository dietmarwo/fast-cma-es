// Copyright (c) Dietmar Wolz. MIT license; see the repository LICENSE.
#include "pareto_bindings.hpp"
#include "ndarray_utils.hpp"
#include "paretoarchive.hpp"
#include <memory>

namespace fcmaes::bindings {
namespace {
using Matrix = nb::ndarray<const double, nb::numpy, nb::shape<-1, -1>,
                            nb::c_contig, nb::device::cpu>;
using OwnedMatrix = nb::ndarray<nb::numpy, double, nb::shape<-1, -1>, nb::c_contig>;
using Mask = nb::ndarray<nb::numpy, bool, nb::shape<-1>, nb::c_contig>;

OwnedMatrix owned_matrix(std::vector<double> values, std::size_t rows, std::size_t cols) {
    // nanobind requires a non-null data pointer even for zero-sized shapes.
    if (values.empty()) values.resize(1);
    auto storage = std::make_unique<std::vector<double>>(std::move(values));
    nb::capsule owner(storage.get(), [](void* p) noexcept {
        delete static_cast<std::vector<double>*>(p);
    });
    auto* data = storage->data();
    storage.release();
    return OwnedMatrix(data, {rows, cols}, owner);
}

Mask owned_mask(const std::vector<unsigned char>& values) {
    auto storage = std::make_unique<bool[]>(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) storage[i] = values[i] != 0;
    nb::capsule owner(storage.get(), [](void* p) noexcept { delete[] static_cast<bool*>(p); });
    auto* data = storage.release();
    return Mask(data, {values.size()}, owner);
}
}  // namespace

void bind_pareto(nb::module_ &m) {
    nb::class_<fcmaes::ParetoArchive2D>(m, "ParetoArchive2D")
        .def("__init__", [](fcmaes::ParetoArchive2D* self, F64Vector reference, int payload_dim) {
            if (reference.size() != 2 || payload_dim < 0)
                throw std::invalid_argument("reference must have length 2; payload_dim must be nonnegative");
            new (self) fcmaes::ParetoArchive2D(
                {reference.data()[0], reference.data()[1]}, static_cast<std::size_t>(payload_dim));
        }, "reference"_a.noconvert(), "payload_dim"_a = 0)
        .def("add", [](fcmaes::ParetoArchive2D& archive, F64Vector y, F64Vector x) {
            if (y.size() != 2) throw std::invalid_argument("objectives must have length 2");
            return archive.add(y.data(), x.data(), x.size());
        }, "y"_a.noconvert(), "x"_a.noconvert())
        .def("add_batch", [](fcmaes::ParetoArchive2D& archive, Matrix ys, Matrix xs) {
            if (ys.shape(1) != 2) throw std::invalid_argument("objectives must have shape (rows, 2)");
            return owned_mask(archive.add_batch(ys.data(), xs.data(), ys.shape(0),
                                               xs.shape(0), xs.shape(1)));
        }, "ys"_a.noconvert(), "xs"_a.noconvert())
        .def("front", [](const fcmaes::ParetoArchive2D& archive) {
            auto snapshot = archive.snapshot();
            return nb::make_tuple(
                owned_matrix(std::move(snapshot.payloads), snapshot.rows, archive.payload_dim()),
                owned_matrix(std::move(snapshot.objectives), snapshot.rows, 2));
        })
        .def("clear", &fcmaes::ParetoArchive2D::clear)
        .def("hypervolume", &fcmaes::ParetoArchive2D::hypervolume)
        .def_prop_ro("size", &fcmaes::ParetoArchive2D::size)
        .def_prop_ro("payload_dim", &fcmaes::ParetoArchive2D::payload_dim)
        .def("__len__", &fcmaes::ParetoArchive2D::size);
}
}  // namespace fcmaes::bindings
