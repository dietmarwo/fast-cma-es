/**
 * @file archivede.hpp
 * Exact strict Pareto archive and archive-gated differential evolution.
 *
 * Derived from archive_de.cpp in Aleksey Vaneev's BiteOpt repository,
 * https://github.com/avaneev/biteopt (commit e8d6785, 2026-10-09). From there
 * come the archive-gated DE/rand/1/bin, the exact online archive ordered by
 * the objective sum, and survivor selection by objective champions, rank sums
 * and greedy rank-space spread. The fcmaes adaptations by Dietmar Wolz
 * compare entries whose rounded objective sums are equal (rounding can hide a
 * dominance relation there), add payloads, a two-objective skyline, input
 * validation, a retryable ask/tell interface and a portable uniform draw.
 *
 * Original code: Copyright (c) 2026 Aleksey Vaneev.
 * fcmaes adaptations: Copyright (c) 2026 Dietmar Wolz.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <random>
#include <vector>

namespace fcmaes::archivede {

using Point = std::vector<double>;
using Population = std::vector<Point>;

struct Snapshot {
    Point payloads, objectives; // row-major, including zero-width payloads
    std::size_t rows = 0;
};

class Archive {
public:
    explicit Archive(std::size_t objectives, std::size_t payload_dim = 0);
    bool add(const double* y, std::size_t objectives, const double* x, std::size_t payload_dim);
    Snapshot snapshot() const;
    void clear();
    std::size_t size() const { return entries_.size() + skyline_.size(); }
    std::size_t nobj() const { return nobj_; }
    std::size_t payload_dim() const { return payload_dim_; }

private:
    struct Entry {
        double sum;
        std::uint64_t serial;
        Point y, x;
    };
    std::size_t nobj_, payload_dim_;
    std::uint64_t serial_ = 0;
    bool exact_line_;
    std::vector<Entry> entries_;
    std::multimap<double, Entry> skyline_;
    Point witness_;
};

/// Synchronous DE/rand/1/bin on bounded continuous decisions, F=0.5, CR=0.3.
class State {
public:
    State(Point lower, Point upper, std::size_t nobj, std::size_t popsize, std::uint64_t budget,
          std::uint32_t seed, bool gate = true, bool tiered = true);
    const Point& ask(); // an outstanding batch is idempotent; empty at the budget
    void tell(const double* y, std::size_t rows, std::size_t cols);
    Snapshot population() const;
    Snapshot front() const { return archive_.snapshot(); }
    std::size_t dim() const { return lower_.size(); }
    std::size_t nobj() const { return archive_.nobj(); }
    std::uint64_t nfev() const { return evaluations_; }

private:
    Point decode(const Point& unit) const;
    Point lower_, upper_;
    std::size_t popsize_;
    std::uint64_t budget_, evaluations_ = 0;
    bool gate_, tiered_;
    std::mt19937 rng_;
    Archive archive_;
    Population parents_, values_, pending_;
    Point physical_, submitted_;
    std::vector<unsigned char> accepted_;
    std::size_t processed_ = 0;
};

} // namespace fcmaes::archivede
