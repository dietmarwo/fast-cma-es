/*
 * evaluator.hpp
 *
 *  Created on: Jul 12, 2021
 *      Author: Dietmar Wolz
 */

#ifndef EVALUATOR_HPP_
#define EVALUATOR_HPP_

#include <Eigen/Core>
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

#define EIGEN_VECTORIZE_SSE2
#include <EigenRand/EigenRand>

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
    inline const T& take() {
        std::unique_lock<std::mutex> lock(_mutex);
        while (_queue.size() == 0)
            _not_empty.wait(lock);
        T &front = _queue.front();
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

static std::normal_distribution<> gauss_01 = std::normal_distribution<>(0, 1);

static Eigen::MatrixXd normal(int dx, int dy, Eigen::Rand::P8_mt19937_64 &rs) {
    return Eigen::Rand::normal<mat>(dx, dy, rs);
}

static Eigen::MatrixXd cauchy(int dx, int dy, Eigen::Rand::P8_mt19937_64 &rs) {
    return Eigen::Rand::cauchy<mat>(dx, dy, rs);
}

static Eigen::MatrixXd studentT(int dx, int dy, Eigen::Rand::P8_mt19937_64 &rs) {
    return Eigen::Rand::studentT<mat>(dx, dy, rs);
}

static double rand01(Eigen::Rand::P8_mt19937_64 &rs) {
    return distr_01(rs);
}

static int randInt(Eigen::Rand::P8_mt19937_64 &rs, int max) {
    return (int) (max * distr_01(rs));
}

static double normreal(Eigen::Rand::P8_mt19937_64 &rs, double mu, double sdev) {
    return gauss_01(rs) * sdev + mu;
}

static Eigen::MatrixXd normalVec(int dim, Eigen::Rand::P8_mt19937_64 &rs) {
    return Eigen::Rand::normal<vec>(dim, 1, rs);
}

static vec normalVec(const vec &mean, const vec &sdev, int dim, Eigen::Rand::P8_mt19937_64 &rs) {
    vec nv = Eigen::Rand::normal<vec>(dim, 1, rs);
    return (nv.array() * sdev.array()).matrix() + mean;
}

static Eigen::MatrixXd cauchyVec(int dim, Eigen::Rand::P8_mt19937_64 &rs) {
    return Eigen::Rand::cauchy<vec>(dim, 1, rs);
}

static Eigen::MatrixXd studentTVec(int dim, Eigen::Rand::P8_mt19937_64 &rs) {
    return Eigen::Rand::studentT<vec>(dim, 1, rs);
}

static Eigen::MatrixXd uniform(int dx, int dy, Eigen::Rand::P8_mt19937_64 &rs) {
    return Eigen::Rand::uniformReal<mat>(dx, dy, rs);
}

static Eigen::MatrixXd uniformVec(int dim, Eigen::Rand::P8_mt19937_64 &rs) {
    return Eigen::Rand::uniformReal<vec>(dim, 1, rs);
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
    IndexVal ivals[size];
    for (int i = 0; i < size; i++) {
        ivals[i].index = i;
        ivals[i].val = x[i];
    }
    std::sort(ivals, ivals + size, compareIndexVal);
    return Eigen::MatrixXi::NullaryExpr(size, 1, [&ivals](int i) {
        return ivals[i].index;
    });
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

    vec eval(const vec &X) {
        double res[_nobj];
        _terminate = _terminate || _func(_dim, X.data(), res);
        for (int i = 0; i < _nobj; i++) {
            if (std::isnan(res[i]) || !std::isfinite(res[i]))
                res[i] = 1E99;
        }
        _evaluationCounter++;
        vec rvec = Eigen::Map<vec, Eigen::Unaligned>(res, _nobj);
        return rvec;
    }

    vec eval(const double *const p) {
        double res[_nobj];
        _terminate = _terminate || _func(_dim, p, res);
        for (int i = 0; i < _nobj; i++) {
            if (std::isnan(res[i]) || !std::isfinite(res[i]))
                res[i] = 1E99;
        }
        _evaluationCounter++;
        vec rvec = Eigen::Map<vec, Eigen::Unaligned>(res, _nobj);
        return rvec;
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
        return (x - _lower[i]) / _scale[i];
    }

    bool feasible(int i, double x) {
        return _lower.size() == 0 || (x >= _lower[i] && x <= _upper[i]);
    }

    bool hasBounds() {
        return _lower.size() != 0;
    }

    vec sample(Eigen::Rand::P8_mt19937_64 &rs) {
        if (_lower.size() == 0)
            std::cout << "no bounds error" << std::endl;
        vec rv = uniformVec(_dim, rs);
        return (rv.array() * _scale.array()).matrix() + _lower;
    }

    vec sample(Eigen::Rand::P8_mt19937_64 &rs, vec &up, vec &lo) {
         vec rv = uniformVec(_dim, rs);
         return (rv.array() * (up - lo).array()).matrix() + lo;
    }

    double sample_i(int i, Eigen::Rand::P8_mt19937_64 &rs) {
        if (_lower.size() == 0)
            std::cout << "no bounds error" << std::endl;
        return _lower[i] + _scale[i] * distr_01(rs);
    }

    double sample_i(int i, Eigen::Rand::P8_mt19937_64 &rs, vec &up, vec &lo) {
        return lo[i] + (up[i] - lo[i]) * distr_01(rs);
    }

    int evaluations() {
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

    void values(const mat &popX, vec &ys) {
        int popsize = popX.cols();
        int n = popX.rows();
        double pargs[popsize * n];
        double res[popsize];
        for (int p = 0; p < popsize; p++) {
            vec x = getClosestFeasible(decode(popX.col(p)));
            for (int i = 0; i < n; i++)
                pargs[p * n + i] = x(i);
        }
        _func_par(popsize, n, pargs, res);
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
    bool _terminate;
    long _evaluationCounter;
};

struct vec_id {
public:

    vec_id(const vec &v, int id) :
            _id(id), _v(v) {
    }

    int _id;
    vec _v;
};

class evaluator {
public:

    evaluator(Fitness *fit, int nobj, int workers) :
            _fit(fit), _nobj(nobj), _workers(workers), _stop(false) {
        _requests = new blocking_queue<vec_id*>(2 * workers);
        _evaled = new blocking_queue<vec_id*>(2 * workers);
        _t0 = Clock::now();
        if (_workers <= 0)
            _workers = std::thread::hardware_concurrency();
        for (int thread_id = 0; thread_id < _workers; thread_id++) {
            _jobs.push_back(evaluator_job(thread_id, this));
        }
    }

    ~evaluator() {
        join();
        delete _requests;
        delete _evaled;
    }

    void evaluate(vec &x, int id) {
        _requests->put(new vec_id(x, id));
    }

    // needs to be deleted
    vec_id* result() {
        return _evaled->take();
    }

    void execute(int thread_id) {
        while (!_stop) {
            vec_id *vid = _requests->take();
            if (!_stop) {
                try {
                    vec x = _fit->getClosestFeasible(_fit->decode(vid->_v));
                    vid->_v = _fit->eval(x);
                } catch (std::exception &e) {
                    std::cout << e.what() << std::endl;
                    vid->_v = constant(_nobj, DBL_MAX);
                }
                _evaled->put(vid);
            } else
                delete vid;
        }
    }

    void join() {
        _stop = true;
        vec x(0);
        // to release all locks
        for (auto &job : _jobs) {
            _requests->put(new vec_id(x, 0));
        }
        for (auto &job : _jobs) {
            job.join();
        }
    }

private:

    class evaluator_job {

    public:
        evaluator_job(int id, evaluator *exec) {
            _thread = std::thread(&evaluator::execute, exec, id);
        }

        void join() {
            if (_thread.joinable())
                _thread.join();
        }

    private:
        std::thread _thread;
    };

    Fitness *_fit;
    int _nobj;
    int _workers;
    bool _stop;
    blocking_queue<vec_id*> *_requests;
    blocking_queue<vec_id*> *_evaled;
    std::vector<evaluator_job> _jobs;
    time_point<Clock> _t0;
};

#endif /* EVALUATOR_HPP_ */
