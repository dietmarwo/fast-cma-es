// Copyright (c) Dietmar Wolz. MIT license; see the repository LICENSE.
#include "front_bindings.hpp"
#include "archivede.hpp"
#include "frontdistance.hpp"
#include "gil_utils.hpp"
#include "owned_ndarray.hpp"
#include "rankbox.hpp"

namespace fcmaes::bindings {
namespace {
using rankbox::BordaBoxIndex;
using rankbox::GridBoxIndex;

std::array<std::uint64_t, 2> counters(const rankbox::BordaResult& r) {
    return {r.axis_candidates, r.score_candidates};
}
std::array<double, 4> counters(const GridBoxIndex::QueryStats& r) {
    return {r.touchedCells, r.nonemptyCells, r.fullCells, r.pointTests};
}

template <class Index> void box_methods(nb::class_<Index>& cls) {
    cls.def_prop_ro("dim", &Index::dim)
        .def(
            "query",
            [](const Index& index, F64Vector lo, F64Vector hi) {
                if (lo.size() != index.dim() || hi.size() != index.dim())
                    throw std::invalid_argument("box bounds must match the index dimension");
                auto result = without_gil([&] { return index.query(lo.data(), hi.data()); });
                const auto n = result.ids.size();
                return owned_array(std::move(result.ids), std::array<std::size_t, 1>{n});
            },
            "low"_a.noconvert(), "high"_a.noconvert())
        .def(
            "query_batch",
            [](const Index& index, F64Matrix lo, F64Matrix hi) {
                const auto rows = lo.shape(0), dim = index.dim();
                if (hi.shape(0) != rows || hi.shape(1) != dim || lo.shape(1) != dim)
                    throw std::invalid_argument("bounds must both have shape (queries, dimension)");
                using Stats = decltype(counters(index.query(nullptr, nullptr)));
                constexpr auto width = std::tuple_size<Stats>::value;
                std::vector<std::uint64_t> offsets{0};
                std::vector<rankbox::Id> ids;
                std::vector<typename Stats::value_type> stats;
                without_gil([&] {
                    offsets.reserve(rows + 1);
                    stats.reserve(rows * width);
                    for (std::size_t i = 0; i < rows; ++i) {
                        auto out = index.query(lo.data() + i * dim, hi.data() + i * dim);
                        auto counts = counters(out);
                        stats.insert(stats.end(), counts.begin(), counts.end());
                        ids.insert(ids.end(), out.ids.begin(), out.ids.end());
                        offsets.push_back(ids.size());
                    }
                });
                const auto n = ids.size();
                return nb::make_tuple(
                    owned_array(std::move(offsets), std::array<std::size_t, 1>{rows + 1}),
                    owned_array(std::move(ids), std::array<std::size_t, 1>{n}),
                    owned_array(std::move(stats), std::array<std::size_t, 2>{rows, width}));
            },
            "low"_a.noconvert(), "high"_a.noconvert());
}

nb::tuple snapshot(archivede::Snapshot out, std::size_t payload_dim, std::size_t nobj) {
    return nb::make_tuple(owned_matrix(std::move(out.payloads), out.rows, payload_dim),
                          owned_matrix(std::move(out.objectives), out.rows, nobj));
}

archivede::Point copy_vector(F64Vector x) {
    if (!x.size())
        return {};
    return {x.data(), x.data() + x.size()};
}
} // namespace

void bind_front_tools(nb::module_& m) {
    nb::class_<BordaBoxIndex> borda(m, "BordaBoxIndex");
    borda.def(
        "__init__",
        [](BordaBoxIndex* self, F64Matrix p) {
            without_gil([&] { new (self) BordaBoxIndex(p.data(), p.shape(0), p.shape(1)); });
        },
        "points"_a.noconvert());
    box_methods(borda);
    nb::class_<GridBoxIndex> grid(m, "GridBoxIndex");
    grid.def(
        "__init__",
        [](GridBoxIndex* self, F64Matrix p, double volume) {
            without_gil([&] { new (self) GridBoxIndex(p.data(), p.shape(0), p.shape(1), volume); });
        },
        "points"_a.noconvert(), "query_volume"_a);
    box_methods(grid);

    using Archive = archivede::Archive;
    nb::class_<Archive>(m, "ParetoArchiveND")
        .def(nb::init<std::size_t, std::size_t>(), "nobj"_a, "payload_dim"_a = 0)
        .def_prop_ro("size", &Archive::size)
        .def_prop_ro("nobj", &Archive::nobj)
        .def_prop_ro("payload_dim", &Archive::payload_dim)
        .def("clear", &Archive::clear)
        .def("front", [](const Archive& a) { return snapshot(a.snapshot(), a.payload_dim(), a.nobj()); })
        .def(
            "add",
            [](Archive& a, F64Vector y, F64Vector x) {
                return a.add(y.data(), y.size(), x.data(), x.size());
            },
            "y"_a.noconvert(), "x"_a.noconvert())
        .def(
            "add_batch",
            [](Archive& a, F64Matrix y, F64Matrix x) {
                const auto n = y.shape(0), d = a.nobj(), w = a.payload_dim();
                if (y.shape(1) != d || x.shape(0) != n || x.shape(1) != w)
                    throw std::invalid_argument("objective or payload batch shape mismatch");
                std::vector<unsigned char> mask(n);
                for (std::size_t i = 0; i < n; ++i) {
                    try {
                        mask[i] = a.add(y.data() + i * d, d, w ? x.data() + i * w : nullptr, w);
                    } catch (const std::invalid_argument& e) {
                        throw std::invalid_argument("row " + std::to_string(i) + ": " + e.what());
                    }
                }
                return owned_mask(mask);
            },
            "ys"_a.noconvert(), "xs"_a.noconvert());

    using State = archivede::State;
    nb::class_<State>(m, "ArchiveDE")
        .def(
            "__init__",
            [](State* self, F64Vector lo, F64Vector hi, std::size_t nobj, std::size_t np,
               std::uint64_t budget, std::uint32_t seed, bool gate, bool tiered) {
                new (self) State(copy_vector(lo), copy_vector(hi), nobj, np, budget, seed, gate, tiered);
            },
            "lower"_a.noconvert(), "upper"_a.noconvert(), "nobj"_a, "popsize"_a, "budget"_a, "seed"_a,
            "gate"_a, "tiered"_a)
        .def("ask",
             [](State& s) {
                 const auto& batch = s.ask();
                 return owned_matrix(batch, batch.size() / s.dim(), s.dim());
             })
        .def(
            "tell", [](State& s, F64Matrix y) { s.tell(y.data(), y.shape(0), y.shape(1)); },
            "ys"_a.noconvert())
        .def("front", [](const State& s) { return snapshot(s.front(), s.dim(), s.nobj()); })
        .def("population", [](const State& s) { return snapshot(s.population(), s.dim(), s.nobj()); })
        .def_prop_ro("nfev", &State::nfev);
    m.def(
        "front_igd",
        [](F64Matrix front, F64Matrix reference) {
            if (reference.shape(1) != front.shape(1))
                throw std::invalid_argument("objective dimensions must match");
            return without_gil([&] {
                return front_distance::igd(front.data(), front.shape(0), reference.data(),
                                           reference.shape(0), reference.shape(1));
            });
        },
        "front"_a.noconvert(), "reference"_a.noconvert());
}
} // namespace fcmaes::bindings
