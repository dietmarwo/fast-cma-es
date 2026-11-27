// Copyright (c) Dietmar Wolz. MIT license; see the repository LICENSE.
#pragma once

#include "paretofront2d.h"
#include <array>
#include <climits>
#include <string>

namespace fcmaes {

// Exact 2D Pareto archive with fixed-width payloads, used by the Python
// bindings (which hold the GIL during every call). The archive itself is
// ParetoFront2D (paretofront2d.h), derived from bitehv2.h in Aleksey
// Vaneev's BiteOpt repository, https://github.com/avaneev/biteopt.
// Points outside the reference box are not archived. A failing batch row
// is not applied; the rows before it stay applied.
class ParetoArchive2D {
public:
    struct Snapshot {
        std::vector<double> objectives;
        std::vector<double> payloads;
        std::size_t rows;
    };

    explicit ParetoArchive2D(std::array<double, 2> reference, std::size_t payload_dim = 0)
        : payload_dim_(payload_dim), front_(reference[0], reference[1]) {
        if (payload_dim > INT_MAX)
            throw std::invalid_argument("payload_dim exceeds the native limit");
    }

    bool add(const double* point, const double* payload, std::size_t payload_size) {
        if (!point) throw std::invalid_argument("objective buffer must not be null");
        if (payload_size != payload_dim_ || (payload_dim_ && !payload))
            throw std::invalid_argument("payload width must equal payload_dim");
        return front_.addPoint(point[0], point[1], payload, static_cast<int>(payload_dim_));
    }

    std::vector<unsigned char> add_batch(const double* points, const double* payloads,
                                         std::size_t rows, std::size_t payload_rows,
                                         std::size_t payload_width) {
        std::vector<unsigned char> accepted;
        add_rows(points, payloads, rows, payload_rows, payload_width,
                 [&](std::size_t, bool inserted) { accepted.push_back(inserted); },
                 [&] { accepted.reserve(rows); });
        return accepted;
    }

    void add_batch_into(const double* points, const double* payloads, std::size_t rows,
                        std::size_t payload_rows, std::size_t payload_width, bool* output) {
        if (rows && !output)
            throw std::invalid_argument("nonempty batch requires an admission mask");
        add_rows(points, payloads, rows, payload_rows, payload_width,
                 [&](std::size_t i, bool inserted) { output[i] = inserted; }, [] {});
    }

    void write_snapshot(double* values, double* payload) const {
        front_.forEachPoint([&](double a, double b, const std::vector<double>& x) {
            *values++ = a;
            *values++ = b;
            if (payload_dim_) {
                std::copy(x.begin(), x.end(), payload);
                payload += payload_dim_;
            }
        });
    }

    Snapshot snapshot() const {
        Snapshot result;
        result.rows = front_.size();
        result.objectives.reserve(2 * result.rows);
        result.payloads.reserve(payload_dim_ * result.rows);
        front_.forEachPoint([&result](double a, double b, const std::vector<double>& x) {
            result.objectives.push_back(a);
            result.objectives.push_back(b);
            result.payloads.insert(result.payloads.end(), x.begin(), x.end());
        });
        return result;
    }

    void clear() { front_.clear(); }
    std::size_t size() const { return front_.size(); }
    std::size_t payload_dim() const { return payload_dim_; }
    double hypervolume() const { return front_.hypervolume(); }

private:
    template<class Write, class Prepare>
    void add_rows(const double* points, const double* payloads, std::size_t rows,
                  std::size_t payload_rows, std::size_t payload_width, Write output, Prepare prepare) {
        if (rows != payload_rows || payload_width != payload_dim_)
            throw std::invalid_argument("payload batch shape must be (rows, payload_dim)");
        if (rows && (!points || (payload_dim_ && !payloads)))
            throw std::invalid_argument("nonempty batch requires objective/payload buffers");
        prepare();
        for (std::size_t row = 0; row < rows; ++row) {
            try {
                output(row, front_.addPoint(points[2 * row], points[2 * row + 1],
                    payload_dim_ ? payloads + payload_dim_ * row : nullptr, static_cast<int>(payload_dim_)));
            } catch (const std::invalid_argument& e) {
                throw std::invalid_argument("row " + std::to_string(row) + ": " + e.what()
                    + "; earlier rows retained");
            } catch (const std::overflow_error& e) {
                throw std::overflow_error("row " + std::to_string(row) + ": " + e.what()
                    + "; earlier rows retained");
            }
        }
    }

    std::size_t payload_dim_;
    ParetoFront2D front_;
};
}  // namespace fcmaes
