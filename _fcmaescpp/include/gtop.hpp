#pragma once

#include <vector>

double gtoc1(const std::vector<double> &x, std::vector<double> &rp);
double cassini1(const std::vector<double> &x, std::vector<double> &rp);
double sagas(const std::vector<double> &x);
double rosetta(const std::vector<double> &x);
double cassini2(const std::vector<double> &x);
double messenger(const std::vector<double> &x);
double messengerfull(const std::vector<double> &x);
double cassini1minlp(const std::vector<double> &x, std::vector<double> &rp,
                     double &launchDV);
double cassini2minlp(const std::vector<double> &x);
double tandem(const std::vector<double> &x, double &tof,
              const int sequence_[]);
