/**
 * @file paretofront2d.h
 * Exact online 2-D Pareto front with incremental hypervolume, C++11.
 *
 * Copyright (c) 2026 Aleksey Vaneev (original archive, BiteOpt bitehv2.h,
 * https://github.com/avaneev/biteopt).
 * Copyright (c) 2026 Dietmar Wolz (fcmaes adaptation: compensated area sums,
 * error handling, payloads, append/gap shortcuts).
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
#ifndef FCMAES_PARETOFRONT2D_H
#define FCMAES_PARETOFRONT2D_H

#include <cmath>
#include <cstddef>
#include <iterator>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fcmaes {

// Both objectives are minimized. Along the front, a increases and b strictly
// decreases, so one predecessor lookup decides rejection and the dominated
// successors are contiguous. A cached front point rejects most candidates
// in O(1); it is only a shortcut, never a replacement for the dominance test.
class ParetoFront2D {
public:
    typedef std::pair<double, std::vector<double> > Mapped;
    typedef std::map<double, Mapped> Front;

    explicit ParetoFront2D(double refX, double refY)
        : refX_(refX), refY_(refY), cached_(false),
          cacheA_(0), cacheB_(0), cacheMean_(0) {
        if (!std::isfinite(refX) || !std::isfinite(refY))
            throw std::invalid_argument("nonfinite hypervolume reference");
    }

    // Coordinate caches copy safely; map iterators must belong to the copy.
    // A copied or moved archive rebuilds its gap hint on its next acceptance.
    ParetoFront2D(const ParetoFront2D& other)
        : refX_(other.refX_), refY_(other.refY_), area_(other.area_),
          cached_(other.cached_), cacheA_(other.cacheA_), cacheB_(other.cacheB_),
          cacheMean_(other.cacheMean_), front_(other.front_) {}
    ParetoFront2D& operator=(const ParetoFront2D& other) {
        if (this != &other) { ParetoFront2D copy(other); swap(copy); }
        return *this;
    }
    ParetoFront2D(ParetoFront2D&& other) noexcept
        : ParetoFront2D(other.refX_, other.refY_) { swap(other); }
    ParetoFront2D& operator=(ParetoFront2D&& other) noexcept {
        if (this != &other) { ParetoFront2D moved(std::move(other)); swap(moved); }
        return *this;
    }
    void swap(ParetoFront2D& other) noexcept {
        using std::swap;
        front_.swap(other.front_);
        swap(refX_, other.refX_); swap(refY_, other.refY_); swap(area_, other.area_);
        swap(cached_, other.cached_); swap(cacheA_, other.cacheA_);
        swap(cacheB_, other.cacheB_); swap(cacheMean_, other.cacheMean_);
        gapValid_ = false; other.gapValid_ = false;
    }
    void clear() noexcept {
        front_.clear(); area_ = Sum(); cached_ = false; gapValid_ = false;
    }

    // Returns true if (a, b) was archived. Points with a > refX or b > refY
    // are outside the reference box and return false: they add nothing to the
    // hypervolume. There is no lower bound. Duplicate objectives keep their
    // first payload. The payload x[0..xN) is copied only for archived points;
    // a null pointer or xN <= 0 means no payload. Non-finite objectives throw
    // std::invalid_argument, a non-representable area std::overflow_error;
    // both, and allocation failures, leave the front unchanged. x may point
    // into an archived payload. Not thread-safe; callbacks must not reenter.
    bool addPoint(double a, double b, const double* x = nullptr, int xN = 0) {
        if (!std::isfinite(a) || !std::isfinite(b))
            throw std::invalid_argument("objectives must be finite");
        if (a > refX_ || b > refY_) return false;
        if (a == 0) a = 0;
        if (b == 0) b = 0;
        // Explicit initialization lets an empty archive accept the reference.
        if (cached_ && cacheA_ <= a && cacheB_ <= b) return false;
        return insert_(a, b, x, xN);
    }

    double hypervolume() const { return area_.value; }
    std::size_t size() const { return front_.size(); }
    bool empty() const { return front_.empty(); }

    // Sorted traversal. Do not mutate this archive from the visitor.
    // Signature: void(double a, double b, const vector<double>&).
    template<class Visitor> void forEachPoint(Visitor visitor) const {
        for (Front::const_iterator i = front_.begin(); i != front_.end(); ++i)
            visitor(i->first, i->second.first, i->second.second);
    }

    // O(1) live view; usual std::map iterator invalidation rules apply.
    const Front& getFront() const { return front_; }

private:
    struct Sum {
        double value, correction;
        Sum() : value(0), correction(0) {}
        void add(double increment) {
            if (!std::isfinite(increment) || increment < 0)
                throw std::overflow_error("nonfinite hypervolume increment");
            const double y = increment - correction;
            const double next = value + y;
            if (!std::isfinite(next) || next < 0)
                throw std::overflow_error("nonfinite hypervolume total");
            correction = (next - value) - y;
            value = next;
        }
    };

    double refX_, refY_;
    Sum area_;
    bool cached_;
    double cacheA_, cacheB_, cacheMean_;
    Front front_;
    bool gapValid_ = false;
    double gapA_ = 0, gapB_ = 0, successorA_ = 0, successorB_ = 0;
    Front::iterator gapSuccessor_;

    // Keep validation and cached rejection small enough to inline at call sites.
    bool insert_(double a, double b, const double* x, int xN) {
        if (!front_.empty()) {
            const Front::const_reverse_iterator last = front_.rbegin();
            if (a > last->first && b < last->second.first)
                return insertGap_(a, b, x, xN, front_.end(), refX_, last->second.first);
        }
        if (gapValid_ && gapA_ < a && a < successorA_ && successorB_ < b && b < gapB_)
            return insertGap_(a, b, x, xN, gapSuccessor_, successorA_, gapB_);

        Front::iterator first = front_.lower_bound(a);
        if (first != front_.end() && first->first == a && first->second.first <= b)
            return false;
        const bool hasPred = first != front_.begin();
        const double predB = hasPred ? std::prev(first)->second.first : refY_;
        if (hasPred && predB <= b) return false;

        double oldX = a, oldY = predB;
        Sum delta;
        std::size_t removed = 0;
        Front::iterator stop = first;
        for (; stop != front_.end() && stop->second.first >= b; ++stop) {
            delta.add(rectangle_(stop->first, oldX, oldY, b));
            oldX = stop->first; oldY = stop->second.first;
            ++removed;
        }
        delta.add(rectangle_(stop == front_.end() ? refX_ : stop->first, oldX, oldY, b));
        const Sum next = nextArea_(a, b, removed, delta);
        Mapped payload(b, x == nullptr || xN <= 0 ? std::vector<double>() :
            std::vector<double>(x, x + xN));
        // Allocate before erasing. Same-key replacement reuses the existing
        // node; swapping its fully prepared payload cannot throw.
        Front::iterator pos;
        if (first != front_.end() && first->first == a) {
            pos = first;
            pos->second.swap(payload);
            front_.erase(std::next(pos), stop);
        } else {
            pos = front_.insert(first, std::make_pair(a, std::move(payload)));
            front_.erase(first, stop);
        }
        area_ = next;
        remember_(a, b);
        rememberGap_(pos);
        return true;
    }

    bool insertGap_(double a, double b, const double* x, int xN,
                    Front::iterator successor, double nextA, double prevB) {
        Sum next = area_;
        next.add(rectangle_(nextA, a, prevB, b));
        Mapped payload(b, x == nullptr || xN <= 0 ? std::vector<double>() :
            std::vector<double>(x, x + xN));
        const Front::iterator pos = front_.insert(successor, std::make_pair(a, std::move(payload)));
        area_ = next;
        remember_(a, b);
        rememberGap_(pos);
        return true;
    }
    void rememberGap_(Front::iterator pos) noexcept {
        const Front::iterator successor = std::next(pos);
        gapValid_ = successor != front_.end();
        if (gapValid_) {
            gapA_ = pos->first; gapB_ = pos->second.first;
            successorA_ = successor->first; successorB_ = successor->second.first;
            gapSuccessor_ = successor;
        }
    }

    static double rectangle_(double hx, double lx, double hy, double ly) {
        if (hx == lx || hy == ly) return 0;
        const double width = hx - lx, height = hy - ly;
        const double result = width * height;
        if (!std::isfinite(width) || !std::isfinite(height) || !std::isfinite(result))
            throw std::overflow_error("nonfinite hypervolume rectangle");
        return result;
    }
    Sum nextArea_(double a, double b, std::size_t removed, const Sum& delta) const {
        if (removed == front_.size()) {
            Sum next;
            next.add(rectangle_(refX_, a, refY_, b));
            return next;
        }
        Sum next = area_;
        next.add(delta.value);
        return next;
    }
    void remember_(double a, double b) {
        // A half-sum is finite even when the ordinary a+b would overflow.
        // This cache heuristic affects only rejection speed, never ordering
        // or membership. If the newcomer removes the cache, replace it even
        // when rounding gives equal means.
        const double mean = a * 0.5 + b * 0.5;
        if (!cached_ || (a <= cacheA_ && b <= cacheB_) || mean < cacheMean_) {
            cached_ = true; cacheA_ = a; cacheB_ = b; cacheMean_ = mean;
        }
    }

};
}  // namespace fcmaes
#endif  // FCMAES_PARETOFRONT2D_H
