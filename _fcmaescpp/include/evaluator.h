/*
 * evaluator.hpp
 *
 *  Created on: Jul 12, 2021
 *      Author: Dietmar Wolz
 */

#ifndef EVALUATOR_HPP_
#define EVALUATOR_HPP_

#include <Eigen/Core>
#include <atomic>
#include <iostream>
#include <algorithm>
#include <queue>
#include <mutex>
#include <thread>
#include <random>
#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <condition_variable>
#include <cfloat>
#include <limits>

#include "pcg_random.hpp"
#include "callback_context.hpp"

using Clock = std::chrono::steady_clock;
using std::chrono::time_point;
using std::chrono::duration_cast;
using std::chrono::milliseconds;

template<typename T>
class blocking_queue {

private:
    size_t _capacity;
    std::queue<T> _queue;
    std::mutex _mutex;
    std::condition_variable _not_full;
    std::condition_variable _not_empty;

public:
    inline blocking_queue(size_t capacity) :
            _capacity(capacity) {
    }

    inline size_t size() {
        std::unique_lock<std::mutex> lock(_mutex);
        return _queue.size();
    }

    //Inserts the specified element into this queue,
    // waiting if necessary for space to become available.
    inline void put(const T &elem) {
        {
            std::unique_lock<std::mutex> lock(_mutex);
            while (_queue.size() >= _capacity)
                _not_full.wait(lock);
            _queue.push(elem);
        }
        _not_empty.notify_one();
    }

    // Retrieves and removes the head of this queue,
    // waiting if necessary until an element becomes available.
    // Returns a copy: pop() destroys the element and can free its storage.
    inline T take() {
        std::unique_lock<std::mutex> lock(_mutex);
        while (_queue.size() == 0)
            _not_empty.wait(lock);
        T front = std::move(_queue.front());
        _queue.pop();
        _not_full.notify_one();
        return front;
    }
};

typedef Eigen::Matrix<double, Eigen::Dynamic, 1> vec;
typedef Eigen::Matrix<int, Eigen::Dynamic, 1> ivec;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> mat;
typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> imat;

typedef bool (*callback_type)(int, const double*, double*);

typedef void (*callback_parallel)(int, int, double*, double*);

static bool noop_callback(int popsize, const double *x, double *y) {
    return true;
}

static void noop_callback_par(int popsize, int dim, double *x, double *y) {
}

static std::uniform_real_distribution<> distr_01 =
        std::uniform_real_distribution<>(0, 1);

// std::normal_distribution caches the second value of each generated pair.
// The normal samplers below take the distribution object from the caller, so
// that every optimizer instance owns its own cache next to its own generator.
// A shared cache would make a run depend on earlier runs and on other threads.

static Eigen::MatrixXd normal(int dx, int dy, pcg64 &rs,
        std::normal_distribution<> &gauss_01) {
    return Eigen::MatrixXd::NullaryExpr(dx, dy, [&]() {
        return gauss_01(rs);
    });
}

static Eigen::MatrixXd normalVec(int dim, pcg64 &rs,
        std::normal_distribution<> &gauss_01) {
    return Eigen::MatrixXd::NullaryExpr(dim, 1, [&]() {
        return gauss_01(rs);
    });
}

static Eigen::MatrixXd uniform(int dx, int dy, pcg64 &rs) {
    return Eigen::MatrixXd::NullaryExpr(dx, dy, [&]() {
        return distr_01(rs);
    });
}

static Eigen::MatrixXd uniformVec(int dim, pcg64 &rs) {
    return Eigen::MatrixXd::NullaryExpr(dim, 1, [&]() {
        return distr_01(rs);
    });
}

static double rand01(pcg64 &rs) {
    return distr_01(rs);
}

static int randInt(pcg64 &rs, int max) {
    return (int) (max * distr_01(rs));
}

static double normreal(pcg64 &rs, std::normal_distribution<> &gauss_01,
        double mu, double sdev) {
    return gauss_01(rs) * sdev + mu;
}

static vec normalVec(const vec &mean, const vec &sdev, int dim, pcg64 &rs,
        std::normal_distribution<> &gauss_01) {
    vec nv = normalVec(dim, rs, gauss_01);
    return (nv.array() * sdev.array()).matrix() + mean;
}

static vec zeros(int n) {
    return Eigen::MatrixXd::Zero(n, 1);
}

static mat zeros(int n, int m) {
    return Eigen::MatrixXd::Zero(n, m);
}

static vec constant(int n, double val) {
    return Eigen::MatrixXd::Constant(n, 1, val);
}

static mat constant(int n, int m, double val) {
    return Eigen::MatrixXd::Constant(n, m, val);
}

struct IndexVal {
    int index;
    double val;
};

static bool compareIndexVal(IndexVal i1, IndexVal i2) {
    return (i1.val < i2.val);
}

static ivec sort_index(const vec &x) {
    int size = x.size();
    std::vector<IndexVal> ivals(size);
    for (int i = 0; i < size; i++) {
        ivals[i].index = i;
        ivals[i].val = x[i];
    }
    std::sort(ivals.begin(), ivals.end(), compareIndexVal);
    ivec indices(size);
    for (int i = 0; i < size; i++)
        indices(i) = ivals[i].index;
    return indices;
}

static int index_min(vec &v) {
    double minv = DBL_MAX;
    int mi = -1;
    for (int i = 0; i < v.size(); i++) {
        if (v[i] < minv) {
            mi = i;
            minv = v[i];
        }
    }
    return mi;
}

// wrapper around the fitness function, scales according to boundaries

class Fitness {

public:

    Fitness(callback_type func, callback_parallel func_par_, int dim, int nobj,
            const vec &lower, const vec &upper) :
            _func(func), _func_par(func_par_), _dim(dim), _nobj(nobj), _lower(
                    lower), _upper(upper) {
        if (_lower.size() > 0) {    // bounds defined
            _scale =  _upper - _lower;
            _typx = 0.5 * (_upper + _lower);
        } else {
            _scale = constant(dim, 1.0);
            _typx = zeros(dim);
        }
        _evaluationCounter = 0;
        _normalize = false;
        _terminate = false;
        _dim = dim;
    }

    bool terminate() {
        return _terminate;
    }

    // Returns false when an earlier callback requested termination.
    bool evaluateScalar(const double *x, double &y) {
        y = std::numeric_limits<double>::infinity();
        if (_terminate)
            return false;
        ++_evaluationCounter;
        if (_func(_dim, x, &y))
            _terminate = true;
        return true;
    }

    vec eval(const vec &X) {
        std::vector<double> res(_nobj);
        if (!_terminate && _func(_dim, X.data(), res.data()))
            _terminate = true;
        for (int i = 0; i < _nobj; i++) {
            if (std::isnan(res[i]) || !std::isfinite(res[i]))
                res[i] = 1E99;
        }
        _evaluationCounter++;
        vec rvec = Eigen::Map<vec, Eigen::Unaligned>(res.data(), _nobj);
        return rvec;
    }

    vec eval(const double *const p) {
        std::vector<double> res(_nobj);
        if (!_terminate && _func(_dim, p, res.data()))
            _terminate = true;
        for (int i = 0; i < _nobj; i++) {
            if (std::isnan(res[i]) || !std::isfinite(res[i]))
                res[i] = 1E99;
        }
        _evaluationCounter++;
        vec rvec = Eigen::Map<vec, Eigen::Unaligned>(res.data(), _nobj);
        return rvec;
    }

    bool decodeRequired() const {
        return _normalize;
    }

    vec getClosestFeasible(const vec &X) const {
        if (_lower.size() > 0)
            return X.cwiseMin(_upper).cwiseMax(_lower);
        return X;
    }

    double getClosestFeasible_i(int i, double x_i) {
        return _lower.size() == 0 ? x_i : std::min(_upper[i], std::max(_lower[i], x_i));
    }

    vec getClosestFeasibleNormed(const vec &X) const {
        if (_lower.size() > 0) {
            if (_normalize)
                return X.cwiseMin(1.0).cwiseMax(-1.0);
            else
                return X.cwiseMin(_upper).cwiseMax(_lower);
        }
        return X;
    }

    void setClosestFeasible(mat &X) const {
        if (_lower.size() > 0) {
            for (int i = 0; i < X.cols(); i++) {
                if (_normalize)
                    X.col(i) = X.col(i).cwiseMin(1.0).cwiseMax(-1.0);
                else
                    X.col(i) = X.col(i).cwiseMin(_upper).cwiseMax(_lower);
            }
        }
    }

    vec norm(const vec &X) const {
        if (_lower.size() == 0)
            std::cout << "no bounds error" << std::endl;
        return ((X - _lower).array() / _scale.array()).matrix();
    }

    double norm_i(int i, double x) const {
        if (_lower.size() == 0)
            std::cout << "no bounds error" << std::endl;
        return std::max(0.0, std::min(1.0, (x - _lower[i]) / _scale[i])); // avoid Nan errors
    }

    bool feasible(int i, double x) {
        return _lower.size() == 0 || (x >= _lower[i] && x <= _upper[i]);
    }

    bool hasBounds() {
        return _lower.size() != 0;
    }

    vec sample(pcg64 &rs) {
        if (_lower.size() == 0)
            std::cout << "no bounds error" << std::endl;
        vec rv = uniformVec(_dim, rs);
        return (rv.array() * _scale.array()).matrix() + _lower;
    }

    vec sample(pcg64 &rs, vec &up, vec &lo) {
         vec rv = uniformVec(_dim, rs);
         return (rv.array() * (up - lo).array()).matrix() + lo;
    }

    double sample_i(int i, pcg64 &rs) {
        if (_lower.size() == 0)
            std::cout << "no bounds error" << std::endl;
        return _lower[i] + _scale[i] * distr_01(rs);
    }

    double sample_i(int i, pcg64 &rs, vec &up, vec &lo) {
        return lo[i] + (up[i] - lo[i]) * distr_01(rs);
    }

    int evaluations() const {
        return _evaluationCounter;
    }

    void resetEvaluations() {
        _evaluationCounter = 0;
    }

    void incrEvaluations() {
        _evaluationCounter++;
    }

    vec scale() {
        return _scale;
    }

    vec typx() {
        return _typx;
    }

    void setNormalize(bool normalize) {
        _normalize = normalize;
    }

    void setTerminate() {
        _terminate = true;
    }

    vec encode(const vec &X) const {
        if (_normalize)
            return 2 * (X - _typx).array() / _scale.array();
        else
            return X;
    }

    vec decode(const vec &X) const {
        if (_normalize)
            return 0.5 * (X.array() * _scale.array()).matrix() + _typx;
        else
            return X;
    }

    // evaluatedX, if given, receives the points passed to the objective
    // (decoded and clipped to the bounds), one per column.
    void values(const mat &popX, vec &ys, mat *evaluatedX = nullptr) {
        int popsize = popX.cols();
        int n = popX.rows();
        std::vector<double> pargs(popsize * n);
        std::vector<double> res(popsize);
        if (evaluatedX != nullptr)
            evaluatedX->resize(n, popsize);
        for (int p = 0; p < popsize; p++) {
            Eigen::Map<vec> out(pargs.data() + p*n, n);
            if (_normalize)
                out = 0.5 * (popX.col(p).array() * _scale.array()).matrix() + _typx;
            else
                out = popX.col(p);
            if (_lower.size() > 0)
                out = out.cwiseMin(_upper).cwiseMax(_lower);
            if (evaluatedX != nullptr)
                evaluatedX->col(p) = out;
        }
        _func_par(popsize, n, pargs.data(), res.data());
        for (int p = 0; p < popX.cols(); p++)
            ys[p] = res[p];
        _evaluationCounter += popsize;
    }

    vec violations(const mat &X, double penalty_coef) {
        vec violations = zeros(X.cols());
        if (_lower.size() > 0) {
            for (int i = 0; i < X.cols(); i++) {
                vec x = decode(X.col(i));
                violations[i] = penalty_coef
                        * ((_lower - x).cwiseMax(0).sum()
                                + (x - _upper).cwiseMax(0).sum());
            }
        }
        return violations;
    }

    void getMinValues(double *const p) const {
        for (int i = 0; i < _lower.size(); i++)
            p[i] = _lower[i];
    }

    void getMaxValues(double *const p) const {
        for (int i = 0; i < _upper.size(); i++)
            p[i] = _upper[i];
    }

private:
    callback_type _func;
    callback_parallel _func_par;
    int _dim;
    int _nobj;
    vec _lower;
    vec _upper;
    vec _scale;
    vec _typx;
    bool _normalize;
    // written by the evaluator threads of a run with workers > 1
    std::atomic<bool> _terminate;
    std::atomic<long> _evaluationCounter;
};

// Each submitted slot has one owner until its completion is popped. A
// worker evaluates the slot in place, so an out-of-order result can never
// overwrite another request and detached tickets need no heap allocation.
class ScalarEvaluator {
public:
    struct Slot {
        vec x;
        vec decoded;
        int target = 0;
        double y = std::numeric_limits<double>::infinity();
        bool evaluated = false;
    };

    ScalarEvaluator(Fitness &fit, int workers)
        : fit(fit), requests(workers), completed(workers), slots(workers),
          context(fcmaes::callback_context) {
        threads.reserve(workers);
        try {
            for (int i = 0; i < workers; ++i)
                threads.emplace_back([this] { work(); });
        } catch (...) {
            shutdown();
            throw;
        }
    }

    ~ScalarEvaluator() { shutdown(); }

    void submit(int id, vec x, int target = 0) {
        Slot &slot = slots[id];
        slot.x = std::move(x);
        slot.target = target;
        requests.put(id);
    }

    int take() { return completed.take(); }
    const Slot &slot(int id) const { return slots[id]; }

private:
    void work() {
        fcmaes::callback_context = context;
        for (int id = requests.take(); id >= 0; id = requests.take()) {
            Slot &slot = slots[id];
            slot.evaluated = false;
            if (!stopping) {
                try {
                    if (fit.decodeRequired()) {
                        slot.decoded = fit.getClosestFeasible(fit.decode(slot.x));
                        slot.evaluated = fit.evaluateScalar(slot.decoded.data(), slot.y);
                    } else {
                        // The DE result must match the point the callback saw.
                        slot.x = fit.getClosestFeasible(slot.x);
                        slot.evaluated = fit.evaluateScalar(slot.x.data(), slot.y);
                    }
                } catch (...) {
                    slot.y = std::numeric_limits<double>::infinity();
                }
            }
            completed.put(id);
        }
    }

    void shutdown() {
        stopping = true;
        // At most one completion per slot can be queued. Workers can finish
        // without a consumer even when an optimizer exits with work pending.
        for (size_t i = 0; i < threads.size(); ++i)
            requests.put(-1);
        for (auto &thread : threads)
            thread.join();
    }

    Fitness &fit;
    blocking_queue<int> requests;
    blocking_queue<int> completed;
    std::vector<Slot> slots;
    const void *context;
    std::vector<std::thread> threads;
    std::atomic<bool> stopping{false};
};

#endif /* EVALUATOR_HPP_ */
