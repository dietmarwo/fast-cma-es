#include "paretoarchive.hpp"
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    std::ifstream in(argv[1], std::ios::binary);
    std::uint64_t n = 0;
    in.read(reinterpret_cast<char*>(&n), sizeof(n));
    std::vector<double> points(2 * n), payloads(16 * n, 0.0);
    in.read(reinterpret_cast<char*>(points.data()), static_cast<std::streamsize>(points.size() * sizeof(double)));
    if (!in) return 3;
    for (int rep = 0; rep < 8; ++rep) {
        fcmaes::ParetoArchive2D archive({0, 0}, {1, 1}, 16);
        const auto start = std::chrono::steady_clock::now();
        const auto mask = archive.add_batch(points.data(), payloads.data(), n, n, 16);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        std::size_t accepted = 0;
        for (auto item : mask) accepted += item;
        std::cout << rep << ',' << std::setprecision(12) << ms << ',' << archive.size()
                  << ',' << std::setprecision(17) << archive.hypervolume() << ',' << accepted << '\n';
    }
}
