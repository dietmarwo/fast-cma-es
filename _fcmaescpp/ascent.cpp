#include "ascent/Ascent.h"

using namespace asc;

struct F8 {

	static constexpr double ksi = 0.05236;
	static constexpr double ksi_2 = ksi * ksi;
	static constexpr double ksi_3 = ksi * ksi_2;
	double w;

	void operator()(const state_t &y, state_t &yDot, const double) {

			double y0 = y[0];
			double y0_2 = y0 * y0;
			double y0_3 = y0_2 * y0;

			double y1 = y[1];
			double y1_2 = y1 * y1;

			double y2 = y[2];

        	yDot[0] = -0.877 * y0 + y2 - 0.088 * y0 * y2 + 0.47 * y0_2 - 0.019 * y1_2 - y0_2 * y2 + 3.846 * y0_3
							+ 0.215 * ksi - 0.28 * y0_2 * ksi + 0.47 * y0 * ksi_2 - 0.63 * ksi_3
							- (0.215 * ksi - 0.28 * y0_2 * ksi - 0.63 * ksi_3) * 2 * w;
			yDot[1] = y2;
			yDot[2] = -4.208 * y0 - 0.396 * y2 - 0.47 * y0_2 - 3.564 * y0_3 + 20.967 * ksi - 6.265 * y0_2 * ksi
							+ 46 * y0 * ksi_2 - 61.4 * ksi_3
							- (20.967 * ksi - 6.265 * y0_2 * ksi - 61.4 * ksi_3) * 2 * w;
	 	}
};

extern "C" {
double* integrateF8_C(double* yd, double w, double dt, double step) {

     state_t y(3);
	for (int i = 0; i < 3; i++) 
		y[i] = yd[i];

//    RK4 integrator;
    DOPRI45 integrator;
    F8 f8;
    f8.w = w;
    int steps = 0;
    double t = 0.0;
    while (t < dt) {
        steps++;
        if (t + step >= dt) {
            integrator(f8, y, t, dt - t);
            break;
        } else
            integrator(f8, y, t, step);
    }
    double *res = new double[3];
	for (int i = 0; i < 3; i++)
		res[i] = y[i];
    return res;
};
}



