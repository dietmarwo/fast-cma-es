// Ownership, cached hints, and allocation-failure transactions.
#if defined(__GNUC__) && !defined(__clang__)
// The replacement operators below pair malloc with free; GCC cannot see that.
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif
#include <cstdlib>
#include <new>
static int fail_after = -1;
void* operator new(std::size_t n) {
    if (fail_after == 0) { fail_after = -1; throw std::bad_alloc(); }
    if (fail_after > 0) --fail_after;
    void* p = std::malloc(n ? n : 1);
    if (!p) throw std::bad_alloc();
    return p;
}
void operator delete(void* p) noexcept { std::free(p); }
#if __cplusplus >= 201402L
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
#endif
#include "paretoarchive.hpp"
#include <cstdio>
#include <memory>

using fcmaes::ParetoFront2D;

static int checks = 0;
static void require(bool value) {
    ++checks;
    if (!value) throw std::runtime_error("ownership/transaction check failed");
}
static ParetoFront2D seeded() {
    ParetoFront2D h(10,10); const double x[] = {42,43};
    h.addPoint(1,9,x,2); h.addPoint(9,1,x,2); h.addPoint(3,7,x,2);
    return h;
}
static void verify_hv(const ParetoFront2D& h) {
    long double expected = 0, previous = 10;
    for (const auto& e : h.getFront()) {
        expected += (10.L-e.first)*(previous-e.second.first);
        previous = e.second.first;
    }
    require(std::fabs(expected-h.hypervolume()) < 1e-11L);
}
int main() {
    // Copy/assignment from a front whose last accepted point has a successor.
    ParetoFront2D source = seeded();
    ParetoFront2D copy(source), assigned(10,10);
    assigned = source;
    source.clear();
    require(copy.addPoint(5,5)); require(assigned.addPoint(5,5));
    verify_hv(copy); verify_hv(assigned);
    std::unique_ptr<ParetoFront2D> survivor;
    { ParetoFront2D temporary = seeded(); survivor.reset(new ParetoFront2D(temporary)); }
    require(survivor->addPoint(5,5)); verify_hv(*survivor);
    // Self-assignment through an alias, so compilers do not warn about it.
    ParetoFront2D& copy_alias = copy;
    copy = copy_alias; require(copy.size()==4); verify_hv(copy);

    ParetoFront2D moved(std::move(assigned));
    require(assigned.empty() && assigned.hypervolume()==0);
    require(assigned.addPoint(4,8)); verify_hv(assigned);
    ParetoFront2D moved_again(10,10); moved_again = std::move(moved);
    require(moved.empty() && moved.hypervolume()==0);
    require(moved.addPoint(4,8)); verify_hv(moved);
    ParetoFront2D& moved_alias = moved_again;
    moved_again = std::move(moved_alias); verify_hv(moved_again);
    ParetoFront2D swap_target = seeded(); swap_target.swap(moved_again);
    require(moved_again.addPoint(5,5)); verify_hv(moved_again);

    // Payload copying must precede destruction of the aliased map entry.
    for (bool same_key : {false,true}) {
        ParetoFront2D h = seeded();
        const double* x = h.getFront().begin()->second.second.data();
        require(h.addPoint(same_key ? 1 : 0,0,x,2));
        require(h.getFront().begin()->second.second == std::vector<double>({42,43}));
        verify_hv(h);
    }

    int failed = 0;
    const double cases[][2] = {{9.5,.5},{5,5},{0,0},{3,6},{2,2}};
    for (const auto& p : cases) for (int budget=0; budget<5; ++budget) {
        ParetoFront2D h = seeded();
        const auto before = h.getFront(); const double hv = h.hypervolume();
        const double* x = h.getFront().begin()->second.second.data();
        fail_after = budget;
        try { h.addPoint(p[0],p[1],x,2); fail_after=-1; }
        catch (const std::bad_alloc&) {
            fail_after=-1; ++failed;
            require(h.getFront()==before && h.hypervolume()==hv);
            require(h.addPoint(p[0],p[1],x,2));
        }
        verify_hv(h);
    }
    require(failed>=9); // vector + node, except same-key node reuse
    ParetoFront2D lhs(10,10); lhs.addPoint(8,8);
    const auto lhs_before = lhs.getFront();
    fail_after=0;
    try { lhs=*survivor; fail_after=-1; require(false); }
    catch (const std::bad_alloc&) { fail_after=-1; require(lhs.getFront()==lhs_before); }
    verify_hv(lhs);

    // Public wrapper copies retain independent payloads and clear safely.
    fcmaes::ParetoArchive2D wrapper({10,10},1);
    double p[]={3,7}, x=9; wrapper.add(p,&x,1);
    auto second=wrapper; wrapper.clear(); p[0]=5;p[1]=5;
    second.add(p,&x,1); require(second.size()==2 && wrapper.size()==0);
    auto snap=second.snapshot(); second.clear(); require(snap.rows==2);
    std::printf("safety checks=%d; injected allocation failures=%d; passed\n",checks,failed);
}
