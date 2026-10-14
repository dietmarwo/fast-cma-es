// Copyright (c) Dietmar Wolz. MIT license; see the repository LICENSE.
#pragma once

#include "bitehv2_optimized.h"
#include <array>
#include <climits>
#include <string>

namespace fcmaes {

// One update owner. Python bindings retain the GIL throughout native calls.
// Numeric row errors leave that row unapplied; earlier batch rows remain.
class ParetoArchive2D {
public:
    struct Snapshot {
        std::vector<double> objectives;
        std::vector<double> payloads;
        std::size_t rows;
    };

    ParetoArchive2D(std::array<double, 2> lower,
                    std::array<double, 2> reference, std::size_t payload_dim = 0)
        : lower_(lower), reference_(reference), payload_dim_(payload_dim),
          front_(reference[0], reference[1]) {
        if (payload_dim > INT_MAX)
            throw std::invalid_argument("payload_dim exceeds the native limit");
        for (std::size_t j = 0; j < 2; ++j)
            if (!std::isfinite(lower[j]) || lower[j] > reference[j])
                throw std::invalid_argument("lower must be finite and <= reference");
    }

    bool add(const double* point, const double* payload, std::size_t payload_size) {
        if (!point) throw std::invalid_argument("objective buffer must not be null");
        if (payload_size != payload_dim_ || (payload_dim_ && !payload))
            throw std::invalid_argument("payload width must equal payload_dim");
        for (std::size_t j = 0; j < 2; ++j)
            if (point[j] < lower_[j])
                throw std::invalid_argument("objectives are below fixed lower bounds");
        // The kernel validates finiteness and the upper reference before its
        // cached rejection test; avoid duplicating those checks here.
        return front_.addPoint(point[0], point[1], payload, static_cast<int>(payload_dim_));
    }

    std::vector<unsigned char> add_batch(const double* points, const double* payloads,
                                         std::size_t rows, std::size_t payload_rows,
                                         std::size_t payload_width) {
        if (rows != payload_rows || payload_width != payload_dim_)
            throw std::invalid_argument("payload batch shape must be (rows, payload_dim)");
        if (rows && (!points || (payload_dim_ && !payloads)))
            throw std::invalid_argument("nonempty batch requires objective/payload buffers");
        std::vector<unsigned char> accepted;
        accepted.reserve(rows);
        for (std::size_t row = 0; row < rows; ++row) {
            try {
                accepted.push_back(add(points + 2 * row,
                    payload_dim_ ? payloads + payload_dim_ * row : nullptr, payload_width));
            } catch (const std::invalid_argument& e) {
                throw std::invalid_argument("row " + std::to_string(row) + ": " + e.what()
                    + "; earlier rows retained");
            } catch (const std::overflow_error& e) {
                throw std::overflow_error("row " + std::to_string(row) + ": " + e.what()
                    + "; earlier rows retained");
            }
        }
        return accepted;
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
    std::array<double, 2> lower_, reference_;
    std::size_t payload_dim_;
    RankSumFrontHV front_;
};
}  // namespace fcmaes
