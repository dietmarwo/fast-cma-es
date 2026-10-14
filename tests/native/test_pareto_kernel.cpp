#include "bitehv2_optimized.h"
#include <algorithm>
#include <limits>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>

struct Point { double a, b; int id; };
static std::size_t checks = 0;
static void require(bool ok, const char* message) {
    ++checks;
    if (!ok) throw std::runtime_error(message);
}
static bool weak(Point a, Point b) { return a.a <= b.a && a.b <= b.b; }
static bool oracle(std::vector<Point>& front, Point p) {
    for (std::size_t i = 0; i < front.size(); ++i) if (weak(front[i], p)) return false;
    front.erase(std::remove_if(front.begin(), front.end(), [p](Point q) { return weak(p, q); }), front.end());
    front.push_back(p);
    return true;
}
static std::vector<Point> observed(const RankSumFrontHV& archive) {
    std::vector<Point> out;
    archive.forEachPoint([&out](double a, double b, const std::vector<double>& x) {
        require(x.size() == 1, "payload size");
        out.push_back(Point{a,b,static_cast<int>(x[0])});
    });
    return out;
}
static void verify(const RankSumFrontHV& archive, std::vector<Point> front, double rx, double ry) {
    std::sort(front.begin(), front.end(), [](Point a, Point b) { return a.a < b.a; });
    const std::vector<Point> actual = observed(archive);
    require(actual.size() == front.size() && archive.size() == front.size(), "front size");
    long double hv = 0, bestB = ry, lastA = front.empty() ? rx : front[0].a;
    for (std::size_t i = 0; i < front.size(); ++i) {
        require(actual[i].a == front[i].a && actual[i].b == front[i].b && actual[i].id == front[i].id, "front/payload membership");
        hv += (static_cast<long double>(front[i].a) - lastA) * (static_cast<long double>(ry) - bestB);
        bestB = std::min(bestB, static_cast<long double>(front[i].b));
        lastA = front[i].a;
    }
    hv += (static_cast<long double>(rx) - lastA) * (static_cast<long double>(ry) - bestB);
    require(std::fabs(static_cast<long double>(archive.hypervolume()) - hv) <= 1e-11L * std::max(1.0L, std::fabs(hv)), "independent union HV");
}
static std::size_t updates = 0;
static void stream(const std::vector<Point>& points, double rx = 1, double ry = 1) {
    std::vector<RankSumFrontHV> archives;
    for (int k = 0; k < 4; ++k) archives.push_back(RankSumFrontHV(rx, ry));
    archives[1].getFront();
    std::vector<Point> front;
    for (std::size_t i = 0; i < points.size(); ++i) {
        const Point p = points[i];
        const bool accepted = oracle(front, p);
        if (i == 47) archives[2].getFront();
        if (i == 130) archives[3].getFront();
        for (std::size_t k = 0; k < archives.size(); ++k) {
            const double payload = p.id;
            require(archives[k].addPoint(p.a, p.b, &payload, 1) == accepted, "acceptance");
            verify(archives[k], front, rx, ry);
            ++updates;
        }
    }
}

int main() {
    try {
        std::mt19937_64 rng(20261010);
        for (int trial = 0; trial < 60; ++trial) {
            std::vector<Point> points;
            for (int i = 0; i < 1000; ++i) {
                const double a = trial % 2 == 0 ? (static_cast<int>(rng()%257)-128)/128.0 : std::uniform_real_distribution<double>(-1,1)(rng);
                const double b = trial % 2 == 0 ? (static_cast<int>(rng()%257)-128)/128.0 : std::uniform_real_distribution<double>(-1,1)(rng);
                points.push_back(Point{a,b,i});
            }
            stream(points);
        }
        std::vector<Point> anti;
        for (int i = 0; i < 1024; ++i) anti.push_back(Point{(i+.5)/1024,1-(i+.5)/1024,i});
        std::shuffle(anti.begin(), anti.end(), rng);
        // Cross-page erasure, split/merge, same-a replacement and full reset.
        anti.push_back(Point{.2,.2,1024});
        anti.push_back(Point{.2,.1,1025});
        anti.push_back(Point{.8,.05,1026});
        anti.push_back(Point{.05,.8,1027});
        anti.push_back(Point{0,0,1028});
        for (int i = 0; i < 192; ++i) anti.push_back(Point{-1+(i+.5)/192,-(i+.5)/192,1029+i});
        stream(anti);
        std::reverse(anti.begin(), anti.end());
        stream(anti);
        stream(std::vector<Point>{{1,0,0},{0,1,1},{.5,.5,2},{.5,.5,3},{-0.,.5,4},{0.,.4,5},{0.,0.,6},{-0.,-0.,7}});

        // Held std::map reference and surviving iterator remain usable.
        RankSumFrontHV live(1,1);
        const RankSumFrontHV::Front& view = live.getFront();
        double id = 1;
        require(live.addPoint(.1,.9,&id,1), "live insert");
        RankSumFrontHV::Front::const_iterator survivor = view.begin();
        id=2; live.addPoint(.8,.1,&id,1);
        require(view.size()==2 && survivor->first==.1 && survivor->second.second[0]==1, "live reference/iterator");
        id=3; live.addPoint(0,0,&id,1);
        require(view.size()==1 && view.begin()->first==0, "live full clear");

        RankSumFrontHV invalid(1,1);
        id=42; invalid.addPoint(.5,.5,&id,1);
        const std::vector<Point> saved = observed(invalid);
        const double savedHV = invalid.hypervolume();
        const double bad[][2] = {{NAN,0},{0,INFINITY},{2,0},{0,2}};
        for (std::size_t i=0;i<4;++i) {
            bool threw=false;
            try { invalid.addPoint(bad[i][0],bad[i][1]); } catch (const std::invalid_argument&) { threw=true; }
            require(threw && invalid.hypervolume()==savedHV, "invalid input rollback");
            verify(invalid,saved,1,1);
        }
        // Rejection must not inspect an oversized caller payload.
        require(!invalid.addPoint(.6,.6,&id,1000000), "deferred payload");
        require(observed(invalid)[0].id==42,"first payload copied");

        for (int mode=0;mode<2;++mode) {
            RankSumFrontHV extreme(1e308,1e308);
            if (mode) extreme.getFront();
            extreme.addPoint(1e308,1e308);
            bool threw=false;
            try { extreme.addPoint(-1e308,0); } catch (const std::overflow_error&) { threw=true; }
            require(threw && extreme.size()==1 && extreme.hypervolume()==0, "area overflow rollback");
            require(!extreme.addPoint(1e308,1e308), "cache valid after failure");
        }
        RankSumFrontHV zero(1e308,1e308);
        require(zero.addPoint(-1e308,1e308) && zero.hypervolume()==0,"zero-height rectangle");
        bool badRef=false;
        try { RankSumFrontHV bad(INFINITY,1); } catch (const std::invalid_argument&) { badRef=true; }
        require(badRef,"finite reference");
        RankSumFrontHV payload(1,1);
        double values[3]={1,2,3};
        require(payload.addPoint(.3,.3,values,3),"payload acceptance");
        values[0]=99;
        require(payload.getFront().begin()->second.second==std::vector<double>({1,2,3}),"payload independent copy");
        require(!payload.addPoint(.3,.3,values,3),"duplicate rejected");
        require(payload.getFront().begin()->second.second[0]==1,"duplicate first payload");
        // Large contiguous deletion and independent copies of the ordered map.
        RankSumFrontHV large(1,1);
        for (int i=0;i<50000;++i) large.addPoint((i+.5)/50000,1-(i+.5)/50000);
        RankSumFrontHV copy=large;
        require(large.size()==50000 && copy.size()==50000,"archive copy");
        copy.addPoint(0,0);
        require(copy.size()==1 && copy.hypervolume()==1 && large.size()==50000,"independent full clear");
        large.addPoint(0,0);
        require(large.size()==1 && large.hypervolume()==1,"large full clear");
        std::printf("PASS: %zu oracle-checked prefix updates; %zu assertions; payload/live-view/rollback/copy/50000-point deletion tests\n",updates,checks);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr,"FAIL: %s after %zu checks\n",e.what(),checks);
        return 1;
    }
}
