// Copyright (c) Dietmar Wolz. MIT license; see the repository LICENSE.
#include "pareto_bindings.hpp"
#include "owned_ndarray.hpp"
#include "paretoarchive.hpp"

namespace fcmaes::bindings {
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
        .def("add_batch", [](fcmaes::ParetoArchive2D& archive, F64Matrix ys, F64Matrix xs) {
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
