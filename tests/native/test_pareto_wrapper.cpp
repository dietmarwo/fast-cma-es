// fcmaes::ParetoArchive2D: batch masks, outside rows, shape and row errors,
// snapshots and area overflow.
#include "paretoarchive.hpp"
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

static int checks = 0;
static void require(bool ok, const char* what) {
    ++checks;
    if (!ok) throw std::runtime_error(what);
}

int main() {
    try {
        fcmaes::ParetoArchive2D archive({2, 2}, 1);
        // Row 3 duplicates row 0; row 4 lies outside; row 5 sits on the boundary.
        double ys[] = {-1, 1, 1, -1, 0, 0, -1, 1, 3, -5, -3, 2};
        double xs[] = {10, 11, 12, 13, 14, 15};
        const auto accepted = archive.add_batch(ys, xs, 6, 6, 1);
        require(accepted == std::vector<unsigned char>({1, 1, 1, 0, 0, 1}), "batch mask");
        const auto snapshot = archive.snapshot();
        require(snapshot.payloads == std::vector<double>({15, 10, 12, 11}), "sorted payloads");
        require(archive.hypervolume() == 6, "hypervolume");

        bool threw = false;
        try { archive.add(nullptr, xs, 1); } catch (const std::invalid_argument&) { threw = true; }
        require(threw, "null objectives");
        threw = false;
        try { archive.add_batch(ys, xs, 6, 5, 1); } catch (const std::invalid_argument&) { threw = true; }
        require(threw && archive.size() == 4, "shape error before mutation");

        // Row 0 is applied (it replaces (1, -1)); row 1 is not finite.
        double partial[] = {0.5, -2, NAN, 0, -2, -2};
        threw = false;
        try {
            archive.add_batch(partial, xs, 3, 3, 1);
        } catch (const std::invalid_argument& e) {
            threw = std::string(e.what()).find("row 1") != std::string::npos;
        }
        require(threw && archive.size() == 4, "row error keeps the prefix");

        archive.clear();
        require(archive.size() == 0 && archive.hypervolume() == 0, "clear");
        require(snapshot.rows == 4 && snapshot.objectives.size() == 8, "snapshot owns its data");

        fcmaes::ParetoArchive2D extreme({1e308, 1e308});
        double edge[] = {1e308, 0}, origin[] = {0, 0};
        require(extreme.add(edge, nullptr, 0), "edge point");
        threw = false;
        try { extreme.add(origin, nullptr, 0); } catch (const std::overflow_error&) { threw = true; }
        require(threw && extreme.size() == 1 && extreme.hypervolume() == 0, "overflow rollback");
        std::printf("wrapper checks=%d passed\n", checks);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s after %d checks\n", e.what(), checks);
        return 1;
    }
}
