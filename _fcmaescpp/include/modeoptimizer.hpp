#pragma once

#include <Eigen/Core>

#include <cstdint>
#include <memory>

namespace mode_optimizer {

using vec = Eigen::Matrix<double, Eigen::Dynamic, 1>;
using mat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

class ModeState {
public:
    ModeState(
        long runid, int dim, int nobj, int ncon, int seed,
        const double *lower, const double *upper, bool *ints, int popsize,
        double F, double CR, double pro_c, double dis_c, double pro_m,
        double dis_m, bool nsga_update, double pareto_update,
        double min_mutate, double max_mutate
    );
    ~ModeState();

    ModeState(const ModeState &) = delete;
    ModeState &operator=(const ModeState &) = delete;

    mat ask();
    int tell(const mat &ys);
    int tell_switch(const mat &ys, bool nsga_update, double pareto_update);
    int set_population(const mat &xs, const mat &ys);
    mat population();
    int dim() const;
    int nobj() const;
    int ncon() const;
    int popsize() const;
    int stop() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace mode_optimizer
