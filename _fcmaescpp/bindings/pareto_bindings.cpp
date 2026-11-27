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
            const auto n = ys.shape(0);
            auto mask = std::unique_ptr<bool[]>(new bool[std::max<std::size_t>(1, n)]);
            archive.add_batch_into(ys.data(), xs.data(), n, xs.shape(0), xs.shape(1), mask.get());
            return owned_buffer(std::move(mask), std::array<std::size_t, 1>{n});
        }, "ys"_a.noconvert(), "xs"_a.noconvert())
        .def("front", [](const fcmaes::ParetoArchive2D& archive) {
            const auto rows = archive.size(), width = archive.payload_dim();
            if (rows > SIZE_MAX / std::max<std::size_t>(2, width))
                throw std::length_error("archive snapshot exceeds array size");
            auto payloads = std::unique_ptr<double[]>(new double[std::max<std::size_t>(1, rows * width)]);
            auto objectives = std::unique_ptr<double[]>(new double[std::max<std::size_t>(1, rows * 2)]);
            archive.write_snapshot(objectives.get(), payloads.get());
            return nb::make_tuple(
                owned_buffer(std::move(payloads), std::array<std::size_t, 2>{rows, width}),
                owned_buffer(std::move(objectives), std::array<std::size_t, 2>{rows, 2}));
        })
        .def("clear", &fcmaes::ParetoArchive2D::clear)
        .def("hypervolume", &fcmaes::ParetoArchive2D::hypervolume)
        .def_prop_ro("size", &fcmaes::ParetoArchive2D::size)
        .def_prop_ro("payload_dim", &fcmaes::ParetoArchive2D::payload_dim)
        .def("__len__", &fcmaes::ParetoArchive2D::size);
}
}  // namespace fcmaes::bindings
