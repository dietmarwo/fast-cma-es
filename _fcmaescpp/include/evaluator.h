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
    inline blocking_queue(size_t capacity) : _capacity(capacity) {
    }

    inline size_t size() {
    	std::unique_lock<std::mutex> lock(_mutex);
        return _queue.size();
    }

    //Inserts the specified element into this queue,
    // waiting if necessary for space to become available.
    inline void put(const T& elem) {
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
		T& front = _queue.front();
		_queue.pop();
    	_not_full.notify_one();
        return front;
    }
};

typedef Eigen::Matrix<double, Eigen::Dynamic, 1> vec;
typedef Eigen::Matrix<int, Eigen::Dynamic, 1> ivec;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> mat;

typedef void (*callback_type)(int, const double*, double*);

static std::uniform_real_distribution<> distr_01 = std::uniform_real_distribution<>(
        0, 1);

static std::normal_distribution<> gauss_01 = std::normal_distribution<>(0, 1);

static Eigen::MatrixXd normal(int dx, int dy, pcg64 &rs) {
    return Eigen::MatrixXd::NullaryExpr(dx, dy, [&]() {
        return gauss_01(rs);
    });
}

static Eigen::MatrixXd normalVec(int dim, pcg64 &rs) {
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

static vec zeros(int n) {
    return Eigen::MatrixXd::Zero(n, 1);
}


static vec constant(int n, double val) {
	return  Eigen::MatrixXd::Constant(n, 1, val);
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

    Fitness(callback_type func, int dim, int nobj, const vec &lower,
            const vec &upper) :
            	_func(func), _dim(dim), _nobj(nobj), _lower(lower), _upper(upper) {
        _scale = _upper - _lower;
        _typx = 0.5 * (_upper + _lower);
        _evaluationCounter = 0;
        _normalize = false;
    }

    vec eval(const vec &X) {
        double res[_nobj];
        _func(_dim, X.data(), res);
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
        _func(_dim, p, res);
        for (int i = 0; i < _nobj; i++) {
            if (std::isnan(res[i]) || !std::isfinite(res[i]))
               res[i] = 1E99;      
        }
        _evaluationCounter++;
        vec rvec = Eigen::Map<vec, Eigen::Unaligned>(res, _nobj);
        return rvec;
    }

    void valuesVec(const mat &popX, int popsize, vec &ys) {
        for (int p = 0; p < popsize; p++) {
            vec x = decode(getClosestFeasible(popX.col(p)));
            ys(p) = eval(x)(0);
        }
    }

    vec getClosestFeasible(const vec &X) const {
        if (_normalize)
            return X.cwiseMin(1.0).cwiseMax(-1.0);
        else
            return X.cwiseMin(_upper).cwiseMax(_lower);
    }

    void setClosestFeasible(mat &X) const {
		for (int i = 0; i < X.cols(); i++)
			X.col(i) = X.col(i).cwiseMin(_upper).cwiseMax(_lower);
    }

    vec norm(const vec &X) const {
    	return ((X - _lower).array() / _scale.array()).matrix();
    }

    double norm_i(int i, double x) const {
    	return (x - _lower[i]) / _scale[i];
    }

    bool feasible(int i, double x) {
        return (x >= _lower[i] && x <= _upper[i]);
    }

    vec sample(pcg64 &rs) {
		vec rv = uniformVec(_dim, rs);
		return (rv.array() * _scale.array()).matrix() + _lower;
    }

    double sample_i(int i, pcg64 &rs) {
        return _lower[i] + _scale[i] * distr_01(rs);
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

    void setNormalize(bool normalize) {
        _normalize = normalize;
    }

    vec encode(const vec &X) const {
        if (_normalize)
            return 2*(X - _typx).array() / _scale.array();
        else
            return X;
    }

    vec decode(const vec &X) const {
        if (_normalize)
            return 0.5*(X.array() * _scale.array()).matrix() + _typx;
        else
            return X;
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
    int _dim;
    int _nobj;
    vec _lower;
    vec _upper;
    vec _scale;
    vec _typx;
    bool _normalize;
    long _evaluationCounter;
};

struct vec_id {
public:

	vec_id(const vec &v, int id) : _id(id), _v(v) {
	}

    int _id;
    vec _v;
};

class evaluator {
public:

	evaluator(Fitness* fit, int nobj, int workers) :
		_fit(fit), _nobj(nobj), _workers(workers), _stop(false) {
		_requests = new blocking_queue<vec_id*>(2*workers);
		_evaled = new blocking_queue<vec_id*>(2*workers);
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
        	vec_id* vid = _requests->take();
        	if (!_stop) {
				try {
					vid->_v = _fit->eval(vid->_v);
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

    Fitness* _fit;
    int _nobj;
    int _workers;
    bool _stop;
    blocking_queue<vec_id*>* _requests;
    blocking_queue<vec_id*>* _evaled;
    std::vector<evaluator_job> _jobs;
    time_point<Clock> _t0;
};

#endif /* EVALUATOR_HPP_ */
