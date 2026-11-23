#include "archivede.hpp"
#include "frontdistance.hpp"
#include "rankbox.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

static void require(bool valid, const char* message) {
    if (!valid)
        throw std::runtime_error(message);
}

int main() {
    using namespace fcmaes;
    const double points[] = {0, 1, 1, 0, 1, 0, 0.5, 0.5};
    const double low[] = {0.5, 0}, high[] = {1, 0.5};
    const rankbox::BordaBoxIndex borda(points, 4, 2);
    const rankbox::GridBoxIndex grid(points, 4, 2, 0.25);
    const auto b = borda.query(low, high);
    const auto g = grid.query(low, high);
    using Ids = std::vector<rankbox::Id>;
    for (Ids ids : {b.ids, g.ids}) {
        std::sort(ids.begin(), ids.end());
        require(ids == Ids{1, 2, 3}, "closed-box query");
    }
    require(b.axis_candidates == 3 && b.score_candidates >= b.ids.size(), "rank counters");
    require(g.touchedCells >= g.nonemptyCells && g.nonemptyCells >= g.fullCells, "grid counters");

    archivede::Archive archive(2, 1);
    const double match[] = {1, 0}, dominating[] = {0, 0};
    double x = 7;
    require(archive.add(match, 2, &x, 1), "first archive member");
    x = 8;
    require(archive.add(match, 2, &x, 1), "duplicate admission");
    auto snapshot = archive.snapshot();
    require(snapshot.payloads == std::vector<double>{7, 8}, "duplicate payload order");
    require(archive.add(dominating, 2, &x, 1) && archive.size() == 1, "dominated range removal");
    require(snapshot.rows == 2 && snapshot.objectives.size() == 4, "standalone snapshot");

    archivede::State state({0, 0}, {1, 1}, 2, 8, 19, 100);
    for (std::size_t expected : {8u, 8u, 3u}) {
        const auto& xs = state.ask();
        require(xs.size() == 2 * expected, "population tail");
        std::vector<double> ys;
        for (std::size_t i = 0; i < xs.size(); i += 2) {
            ys.push_back(xs[i]);
            ys.push_back(1 - xs[i]);
        }
        state.tell(ys.data(), expected, 2);
    }
    require(state.nfev() == 19 && state.ask().empty(), "hard evaluation cap");
    require(state.front().rows == 19 && state.population().rows == 8, "archive and population");
    const double r[] = {0, 0, 1, 1}, f[] = {0, 0};
    require(std::abs(front_distance::igd(f, 1, r, 2, 2) - std::sqrt(2.) / 2) < 1e-15, "IGD");
    const double m = std::numeric_limits<double>::max(), extreme[] = {-m, m};
    require(front_distance::igd(&m, 1, extreme, 2, 1) == m, "finite mean of extreme distances");

    // Ties keep row order on both sort paths (below and above 256 rows), with
    // -0.0 equal to 0.0. fcmaes-rust's indexes use the same order.
    for (std::size_t n : {40u, 300u}) {
        std::vector<double> column(n);
        const double cycle[] = {-0.0, 0.0, 1.5, -2.0, 1.0};
        for (std::size_t i = 0; i < n; ++i)
            column[i] = cycle[i % 5];
        std::vector<rankbox::Id> want(n);
        for (std::size_t i = 0; i < n; ++i)
            want[i] = static_cast<rankbox::Id>(i);
        std::stable_sort(want.begin(), want.end(),
                         [&](rankbox::Id a, rankbox::Id b) { return column[a] < column[b]; });
        const double lo = -std::numeric_limits<double>::infinity(), hi = -lo;
        const rankbox::BordaBoxIndex tied(column.data(), n, 1);
        require(tied.query(&lo, &hi).ids == want, "Borda tie order");
        const rankbox::GridBoxIndex cells(column.data(), n, 1, 1. / double(n));
        require(cells.query(&lo, &hi).ids == want, "grid tie order");
    }
}
