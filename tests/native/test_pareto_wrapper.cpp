#include "paretoarchive.hpp"
#include <cassert>
#include <iostream>

int main() {
    fcmaes::ParetoArchive2D archive({-2, -2}, {2, 2}, 1);
    double ys[] = {-1, 1, 1, -1, 0, 0, -1, 1};
    double xs[] = {10, 11, 12, 13};
    const auto accepted = archive.add_batch(ys, xs, 4, 4, 1);
    assert((accepted == std::vector<unsigned char>{1, 1, 1, 0}));
    const auto snapshot = archive.snapshot();
    assert((snapshot.payloads == std::vector<double>{10, 12, 11}));
    assert(archive.hypervolume() == 6);
    try { archive.add(nullptr, xs, 1); assert(false); }
    catch (const std::invalid_argument&) {}
    try {
        archive.add_batch(ys, xs, 4, 3, 1);
        assert(false);
    } catch (const std::invalid_argument&) {}
    assert(archive.size() == 3);
    double partial[] = {-1, 0, -3, 0};
    try {
        archive.add_batch(partial, xs, 2, 2, 1);
        assert(false);
    } catch (const std::invalid_argument& e) {
        assert(std::string(e.what()).find("row 1") != std::string::npos);
    }
    assert(archive.size() == 2);
    archive.clear();
    assert(archive.size() == 0 && archive.hypervolume() == 0);
    assert(snapshot.rows == 3 && snapshot.objectives.size() == 6);
    fcmaes::ParetoArchive2D extreme({0, 0}, {1e308, 1e308});
    double edge[] = {1e308, 0}, origin[] = {0, 0};
    assert(extreme.add(edge, nullptr, 0));
    try { extreme.add(origin, nullptr, 0); assert(false); }
    catch (const std::overflow_error&) {}
    assert(extreme.size() == 1 && extreme.hypervolume() == 0);
    std::cout << "wrapper shape/partial-error/ownership/reset/overflow checks passed\n";
}
