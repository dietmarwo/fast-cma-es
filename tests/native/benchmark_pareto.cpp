// Insertion throughput of fcmaes::ParetoArchive2D on three synthetic streams,
// with a 16-double payload per point. Usage: benchmark_pareto [points]
#include "paretoarchive.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>

int main(int argc, char** argv) {
    const std::size_t n = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 1000000;
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> unit(0, 1);
    std::vector<double> uniform(2 * n), converging(2 * n), antichain(2 * n), payloads(16 * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        uniform[2 * i] = unit(rng); uniform[2 * i + 1] = unit(rng);
        // Noise around the front x + y = 1 shrinks over the run.
        const double x = unit(rng), noise = 0.5 * (1.0 - double(i) / n) * unit(rng);
        converging[2 * i] = x; converging[2 * i + 1] = 1 - x + noise;
        // Every point is nondominated; random order exercises interior inserts.
        const double z = unit(rng);
        antichain[2 * i] = z; antichain[2 * i + 1] = 1 - z;
    }
    std::printf("stream,points,ns_per_point,archive,hypervolume\n");
    const std::pair<const char*, const std::vector<double>*> streams[] = {
        {"uniform", &uniform}, {"converging", &converging}, {"antichain", &antichain}};
    for (const auto& stream : streams) {
        fcmaes::ParetoArchive2D archive({1.5, 1.5}, 16);
        const auto start = std::chrono::steady_clock::now();
        archive.add_batch(stream.second->data(), payloads.data(), n, n, 16);
        const double ns = std::chrono::duration<double, std::nano>(
            std::chrono::steady_clock::now() - start).count() / n;
        std::printf("%s,%zu,%.1f,%zu,%.15f\n", stream.first, n, ns, archive.size(),
                    archive.hypervolume());
    }
}
