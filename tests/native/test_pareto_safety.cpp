// Ownership, cached hints, and allocation-failure transactions.
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

static int checks = 0;
static void require(bool value) {
    ++checks;
    if (!value) throw std::runtime_error("ownership/transaction check failed");
}
static RankSumFrontHV seeded() {
    RankSumFrontHV h(10,10); const double x[] = {42,43};
    h.addPoint(1,9,x,2); h.addPoint(9,1,x,2); h.addPoint(3,7,x,2);
    return h;
}
static void verify_hv(const RankSumFrontHV& h) {
    long double expected = 0, previous = 10;
    for (const auto& e : h.getFront()) {
        expected += (10.L-e.first)*(previous-e.second.first);
        previous = e.second.first;
    }
    require(std::fabs(expected-h.hypervolume()) < 1e-11L);
}
int main() {
    // Copy/assignment from a front whose last accepted point has a successor.
    RankSumFrontHV source = seeded();
    RankSumFrontHV copy(source), assigned(10,10);
    assigned = source;
    source.clear();
    require(copy.addPoint(5,5)); require(assigned.addPoint(5,5));
    verify_hv(copy); verify_hv(assigned);
    std::unique_ptr<RankSumFrontHV> survivor;
    { RankSumFrontHV temporary = seeded(); survivor.reset(new RankSumFrontHV(temporary)); }
    require(survivor->addPoint(5,5)); verify_hv(*survivor);
    copy = copy; require(copy.size()==4); verify_hv(copy);

    RankSumFrontHV moved(std::move(assigned));
    require(assigned.empty() && assigned.hypervolume()==0);
    require(assigned.addPoint(4,8)); verify_hv(assigned);
    RankSumFrontHV moved_again(10,10); moved_again = std::move(moved);
    require(moved.empty() && moved.hypervolume()==0);
    require(moved.addPoint(4,8)); verify_hv(moved);
    moved_again = std::move(moved_again); verify_hv(moved_again);
    RankSumFrontHV swap_target = seeded(); swap_target.swap(moved_again);
    require(moved_again.addPoint(5,5)); verify_hv(moved_again);

    // Payload copying must precede destruction of the aliased map entry.
    for (bool same_key : {false,true}) {
        RankSumFrontHV h = seeded();
        const double* x = h.getFront().begin()->second.second.data();
        require(h.addPoint(same_key ? 1 : 0,0,x,2));
        require(h.getFront().begin()->second.second == std::vector<double>({42,43}));
        verify_hv(h);
    }

    int failed = 0;
    const double cases[][2] = {{9.5,.5},{5,5},{0,0},{3,6},{2,2}};
    for (const auto& p : cases) for (int budget=0; budget<5; ++budget) {
        RankSumFrontHV h = seeded();
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
    RankSumFrontHV lhs(10,10); lhs.addPoint(8,8);
    const auto lhs_before = lhs.getFront();
    fail_after=0;
    try { lhs=*survivor; fail_after=-1; require(false); }
    catch (const std::bad_alloc&) { fail_after=-1; require(lhs.getFront()==lhs_before); }
    verify_hv(lhs);

    // Public wrapper copies retain independent payloads and clear safely.
    fcmaes::ParetoArchive2D wrapper({0,0},{10,10},1);
    double p[]={3,7}, x=9; wrapper.add(p,&x,1);
    auto second=wrapper; wrapper.clear(); p[0]=5;p[1]=5;
    second.add(p,&x,1); require(second.size()==2 && wrapper.size()==0);
    auto snap=second.snapshot(); second.clear(); require(snap.rows==2);
    std::printf("safety checks=%d; injected allocation failures=%d; passed\n",checks,failed);
}
