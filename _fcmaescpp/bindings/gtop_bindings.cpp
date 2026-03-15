#include "gtop_bindings.hpp"

#include "gil_utils.hpp"
#include "ndarray_utils.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include "gtop.hpp"

namespace fcmaes::bindings {

namespace {

std::vector<double> to_std_vector(F64Vector values) {
    const auto *data = static_cast<const double *>(values.data());
    return std::vector<double>(
        data,
        data + static_cast<size_t>(values.shape(0))
    );
}

std::vector<int> to_sequence(nb::sequence seq_obj) {
    const size_t size = nb::len(seq_obj);
    if (size != 5)
        throw std::invalid_argument("sequence must contain exactly 5 planet ids");

    std::vector<int> sequence;
    sequence.reserve(size);
    for (size_t i = 0; i < size; ++i)
        sequence.push_back(nb::cast<int>(seq_obj[i]));
    return sequence;
}

double sanitize(double value) {
    return std::isfinite(value) ? value : 1e10;
}

template <typename Fn>
void bind_scalar_problem(nb::module_ &m, const char *name, Fn &&fn,
                         const char *doc) {
    m.def(
        name,
        [fn = std::forward<Fn>(fn)](F64Vector x) {
            return without_gil([&]() {
                return sanitize(fn(to_std_vector(x)));
            });
        },
        "x"_a.noconvert(),
        doc
    );
}

}  // namespace

void bind_gtop(nb::module_ &m) {
    bind_scalar_problem(
        m,
        "gtop_gtoc1",
        [](const std::vector<double> &x) {
            std::vector<double> rp;
            return gtoc1(x, rp);
        },
        "Evaluate the ESA GTOP GTOC1 benchmark."
    );

    bind_scalar_problem(
        m,
        "gtop_cassini1",
        [](const std::vector<double> &x) {
            std::vector<double> rp;
            return cassini1(x, rp);
        },
        "Evaluate the ESA GTOP Cassini1 benchmark."
    );

    bind_scalar_problem(
        m,
        "gtop_messenger",
        [](const std::vector<double> &x) {
            return messenger(x);
        },
        "Evaluate the ESA GTOP reduced Messenger benchmark."
    );

    bind_scalar_problem(
        m,
        "gtop_messengerfull",
        [](const std::vector<double> &x) {
            return messengerfull(x);
        },
        "Evaluate the ESA GTOP full Messenger benchmark."
    );

    bind_scalar_problem(
        m,
        "gtop_cassini2",
        [](const std::vector<double> &x) {
            return cassini2(x);
        },
        "Evaluate the ESA GTOP Cassini2 benchmark."
    );

    bind_scalar_problem(
        m,
        "gtop_rosetta",
        [](const std::vector<double> &x) {
            return rosetta(x);
        },
        "Evaluate the ESA GTOP Rosetta benchmark."
    );

    bind_scalar_problem(
        m,
        "gtop_sagas",
        [](const std::vector<double> &x) {
            return sagas(x);
        },
        "Evaluate the ESA GTOP Sagas benchmark."
    );

    m.def(
        "gtop_tandem",
        [](F64Vector x, nb::sequence sequence) {
            std::vector<double> values = to_std_vector(x);
            std::vector<int> seq = to_sequence(sequence);
            return without_gil([&]() {
                double tof = 0.0;
                double value = tandem(values, tof, seq.data());
                if (tof > 3652.5)
                    value += 1000.0 * (tof - 3652.5);
                return sanitize(value);
            });
        },
        "x"_a.noconvert(),
        "sequence"_a,
        "Evaluate the constrained ESA GTOP TandEM benchmark."
    );

    m.def(
        "gtop_tandem_unconstrained",
        [](F64Vector x, nb::sequence sequence) {
            std::vector<double> values = to_std_vector(x);
            std::vector<int> seq = to_sequence(sequence);
            return without_gil([&]() {
                double tof = 0.0;
                return sanitize(tandem(values, tof, seq.data()));
            });
        },
        "x"_a.noconvert(),
        "sequence"_a,
        "Evaluate the unconstrained ESA GTOP TandEM benchmark."
    );

    m.def(
        "gtop_cassini1_minlp",
        [](F64Vector x) {
            std::vector<double> values = to_std_vector(x);
            auto result = without_gil([&]() {
                double launch_dv = 0.0;
                std::vector<double> rp;
                double dv = cassini1minlp(values, rp, launch_dv);
                return std::make_pair(sanitize(dv), sanitize(launch_dv));
            });
            return nb::make_tuple(result.first, result.second);
        },
        "x"_a.noconvert(),
        "Evaluate the mixed-integer ESA GTOP Cassini1 benchmark."
    );

    bind_scalar_problem(
        m,
        "gtop_cassini2_minlp",
        [](const std::vector<double> &x) {
            return cassini2minlp(x);
        },
        "Evaluate the mixed-integer ESA GTOP Cassini2 benchmark."
    );
}

}  // namespace fcmaes::bindings
