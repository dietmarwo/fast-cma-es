#include "ascent/Ascent.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "keplerian_toolbox/core_functions/propagate_lagrangian.hpp"
#include "keplerian_toolbox/core_functions/propagate_taylor.hpp"
#include "keplerian_toolbox/core_functions/propagate_taylor_J2.hpp"
#include "keplerian_toolbox/core_functions/par2ic.hpp"
#include "keplerian_toolbox/core_functions/ic2par.hpp"
#include "keplerian_toolbox/core_functions/fb_vel.hpp"
#include "keplerian_toolbox/core_functions/fb_prop.hpp"
#include "keplerian_toolbox/core_functions/lambert_find_N.hpp"
#include "keplerian_toolbox/planet/base.hpp"
#include "keplerian_toolbox/planet/jpl_low_precision.hpp"
#include "keplerian_toolbox/lambert_problem.hpp"

using namespace asc;

struct PVThrust {

    double _veff;
    double _mu;
    double _ux;
    double _uy;
    double _uz;
    double _unorm;

    void operator()(const state_t &pv, state_t &yDot, const double) {

        double x = pv[0];  // position
        double y = pv[1];
        double z = pv[2];
        double m = pv[6];
        double r = sqrt(x * x + y * y + z * z);
        double rrr = r * r * r;
        yDot[0] = pv[3]; // velocity
        yDot[1] = pv[4];
        yDot[2] = pv[5];
        yDot[3] = -_mu * x / (rrr);
        yDot[4] = -_mu * y / (rrr);
        yDot[5] = -_mu * z / (rrr);
        if (m > 0) {
            yDot[3] += _ux / m;
            yDot[4] += _uy / m;
            yDot[5] += _uz / m;
        }
        yDot[6] = -_unorm / _veff;
    }
};

extern "C" {
double* integratePVthrust(double *yd, double mu, double ux, double uy, double uz, double veff, double dt, double step) {

    state_t y(7);
    for (int i = 0; i < 7; i++)
        y[i] = yd[i];

      RK4 integrator;
//    DOPRI45 integrator;
//    ABM4 integrator;
//    PC233 integrator;
//    VABM integrator;

    PVThrust pvt;
    pvt._veff = veff;
    pvt._mu = mu;
    pvt._ux = ux;
    pvt._uy = uy;
    pvt._uz = uz;
    pvt._unorm = sqrt(ux * ux + uy * uy + uz * uz);

//    Recorder recorder;

    int steps = 0;
    double t = 0.0;
    if (dt > 0) {
        while (t < dt) {
    //      recorder({ t, y[0], y[1], y[2], y[3], y[4], y[5], y[6] });
            steps++;
            if (t + step >= dt) {
                integrator(pvt, y, t, dt - t);
                break;
            } else
                integrator(pvt, y, t, step);
    //      recorder.csv("PVThrust", { "t", "x", "y", "z", "vx", "vy", "vz", "m" });
        }   
    } else {
        while (t > dt) {
            steps++;
            if (t - step <= dt) {
                integrator(pvt, y, t, dt - t);
                break;
            } else
                integrator(pvt, y, t, -step);
        }
    }

    double *res = new double[7];
    for (int i = 0; i < 7; i++)
        res[i] = y[i];
    return res;
}
;
}

struct Damp {

    double alpha;

    void operator()(const state_t &y, state_t &yDot, const double) {

        double x1 = y[0];
        double x2 = y[1];
        yDot[0] = x2;
        yDot[1] = -x1 + alpha;
    }
};

extern "C" {
double* integrateDamp_C(double *yd, double alpha, double dt, double step) {

    state_t y(2);
    for (int i = 0; i < 2; i++)
        y[i] = yd[i];

//    RK4 integrator;
//    DOPRI45 integrator;
//    ABM4 integrator;
    PC233 integrator;
//    VABM integrator;
    Damp damp;
    damp.alpha = alpha;
    int steps = 0;
    double t = 0.0;
    while (t < dt) {
        steps++;
        if (t + step >= dt) {
            integrator(damp, y, t, dt - t);
            break;
        } else
            integrator(damp, y, t, step);
    }
    double *res = new double[2];
    for (int i = 0; i < 2; i++)
        res[i] = y[i];
    return res;
}
;
}

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

        yDot[0] = -0.877 * y0 + y2 - 0.088 * y0 * y2 + 0.47 * y0_2
                - 0.019 * y1_2 - y0_2 * y2 + 3.846 * y0_3 + 0.215 * ksi
                - 0.28 * y0_2 * ksi + 0.47 * y0 * ksi_2 - 0.63 * ksi_3
                - (0.215 * ksi - 0.28 * y0_2 * ksi - 0.63 * ksi_3) * 2 * w;
        yDot[1] = y2;
        yDot[2] = -4.208 * y0 - 0.396 * y2 - 0.47 * y0_2 - 3.564 * y0_3
                + 20.967 * ksi - 6.265 * y0_2 * ksi + 46 * y0 * ksi_2
                - 61.4 * ksi_3
                - (20.967 * ksi - 6.265 * y0_2 * ksi - 61.4 * ksi_3) * 2 * w;
    }
};

//#include "ascent/integrators_modular/VABM.h"
//#using VABM = VABM<state_t>;

extern "C" {
double* integrateF8_C(double *yd, double w, double dt, double step) {

    state_t y(3);
    for (int i = 0; i < 3; i++)
        y[i] = yd[i];

//    RK4 integrator;
//    DOPRI45 integrator;
//    ABM4 integrator;
    PC233 integrator;
//    VABM integrator;
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
}
;
}

void wic2par(state_t rv, double* kep, double mu) {
	kep_toolbox::array3D rk;
	kep_toolbox::array3D vk;
	for (int i = 0; i < 3; i++) {
	     rk[i] = rv[i];
	     vk[i] = rv[i+3];
	}
	kep_toolbox::ic2par(rk, vk, mu, kep);
}

struct PVTwaste {

	static constexpr double muE = 3.986004407799724E5;
	static constexpr double muS = 1.32712440018E11;
	static constexpr double muM = 4.9028E3;

	static constexpr double Re = 6378.1363;
	static constexpr double C20 = -4.84165371736E-4;
	static constexpr double C22 = 2.43914352398E-6;
	static constexpr double S22 = -1.40016683654E-6;

	static constexpr double muERe2V5C20 = muE*Re*Re*sqrt(5)*C20;
	static constexpr double muERe2V15C22 = muE*Re*Re*sqrt(15)*C22;
	static constexpr double muERe2V15S22 = muE*Re*Re*sqrt(15)*S22;

	static constexpr double thetaG = 280.4606*M_PI/180.0;
	static constexpr double vE = 4.178074622024230E-3*M_PI/180.0;
	static constexpr double vMa = 1.512151961904581E-4*M_PI/180.0;
	static constexpr double vMp = 1.2893925235125941E-6*M_PI/180.0;
	static constexpr double vMs = 6.128913003523574E-7*M_PI/180.0;
	static constexpr double aS = 1.49619E8;
	static constexpr double Psrp = 4.56E-3;
	static constexpr double Psrpas2 = Psrp*aS*aS;

	static constexpr double phiS = 357.5256*M_PI/180.0;
	static constexpr double OMomS = 282.94*M_PI/180.0;
	static constexpr double vS = 1.1407410259335311E-5*M_PI/180.0;
	static constexpr double ep = 23.4392911*M_PI/180.0;
	static constexpr double cosep = cos(ep);
	static constexpr double sinep = sin(ep);

    double _cram;

    void sunr(double t, double& xsun, double& ysun, double& zsun) {
		double lsun = phiS + vS * t;
		double rsun = 1E6 * (149.619 - 2.499*cos(lsun)
								- 0.021*cos(2*lsun));
		double lambdas = OMomS + lsun +
				(6892.0/3600.0*sin(lsun) +
						72.0/3600.0*sin(2*lsun))*M_PI/180.0;
		xsun = rsun * cos(lambdas);
		ysun = rsun * sin(lambdas) * cosep;
		zsun = rsun * sin(lambdas) * sinep;
    }

    void moonr(double t, double& xM, double& yM, double& zM) {
		double phiM = vS * t;
		double phiMa = vMa * t;
		double phiMp = vMp * t;
		double phiMs = vMs * t;
		double L0 = phiMp + phiMa + 218.31617*M_PI/180.0;
		double lM = phiMa + 134.96292*M_PI/180.0;
		double lsM = phiM + 357.5256*M_PI/180.0;
		double FM = phiMp + phiMa + phiMs + 93.27283*M_PI/180.0;
		double DM = phiMp + phiMa - phiM + 297.85027*M_PI/180.0;

		double rM = 385000.0 - 20905.0 * cos(lM) - 3699.0 * cos(2*DM - lM)
				- 2956.0 * cos(2*DM) - 570.0 * cos(2*lM)
				+ 246.0 * cos(2*lM - 2*DM) - 205.0 * cos(lsM - 2*DM)
				- 171.0 * cos(lM + 2*DM)
				- 152.0 * cos(lM + lsM - 2*DM);

		double lambdaM = L0
				+ (22640.0 / 3600.0 * sin(lM) + 769.0 / 3600.0 * sin(2 * lM)
				- 4856.0 / 3600.0 * sin(lM - 2*DM) + 2370.0 / 3600.0 * sin(2*DM)
				- 668.0 / 3600.0 * sin(lsM) - 412.0 / 3600.0 * sin(2*FM)
				- 212.0 / 3600.0 * sin(2*lM - 2*DM) - 206.0 / 3600.0 * sin(lM + lsM - 2*DM)
				+ 192.0 / 3600.0 * sin(lM + 2*DM) - 165.0 / 3600.0 * sin(lsM - 2*DM)
				+ 148.0 / 3600.0 * sin(lM - lsM) - 125.0 / 3600.0 * sin(DM)
				- 110.0 / 3600.0 * sin(lM + lsM) - 55.0 / 3600.0 * sin(2*FM - 2*DM))
				* M_PI/180.0;

		double betaM =  (18520.0/3600.0 * sin(FM + lambdaM - L0 + (412.0 / 3600.0 * sin(2*FM)
				+ 541.0 / 3600.0 * sin(lsM))) * M_PI/180.0
				- 526.0 / 3600.0 * sin(FM - 2*DM)
				+ 44.0 / 3600.0 * sin(lM + FM - 2*DM) - 31.0 / 3600.0 * sin(-lM + FM - 2*DM)
				- 25.0 / 3600.0 * sin(-2*lM + FM) - 23.0 / 3600.0 * sin(lsM + FM - 2*DM)
				+ 21.0 / 3600.0 * sin(-lM + FM) + 11.0 / 3600.0 * sin(-lsM + FM - 2*DM))
				* M_PI/180.0;

		double M1 = rM * cos(lambdaM) * cos(betaM);
		double M2 = rM * sin(lambdaM) * cos(betaM);
		double M3 = rM * sin(betaM);

		xM = M1;
		yM = cosep * M2 - sinep * M3;
		zM = sinep * M2 + cosep * M3;
    }

    void operator()(const state_t &pv, state_t &yDot, const double t) {

		// position
		double x = pv[0];
		double y = pv[1];
		double z = pv[2];

		// velocity
		yDot[0] = pv[3];
		yDot[1] = pv[4];
		yDot[2] = pv[5];

		double rq = x * x + y * y + z * z;
		double r = sqrt(rq);
		double rq2 = rq * rq;
		double rq3 = rq2 * rq;
		double rrr = r*r*r;

		double z15 = 15.0*z*z / rq3;
		double rq23 = 3.0 / rq2;
		double muERe2V5C20q2r = muERe2V5C20 / (2*r);

		// earth gravity
		double fkepX = -muE * x / rrr;
		double fkepY = -muE * y / rrr;
		double fkepZ = -muE * z / rrr;

		// J2 perturbation
		double fJ2X = muERe2V5C20q2r * x * (rq23 - z15);
		double fJ2Y = muERe2V5C20q2r * y * (rq23 - z15);
		double fJ2Z = muERe2V5C20q2r * z * (3*rq23 - z15);

		// solar radiation pressure
		double xsun, ysun, zsun;
		sunr(t, xsun, ysun, zsun);

		double rsun = sqrt(xsun * xsun + ysun * ysun + zsun * zsun);
		double rsun3 = rsun*rsun*rsun;

		double xsd = x - xsun;
		double ysd = y - ysun;
		double zsd = z - zsun;

		double rsd = sqrt(xsd * xsd + ysd * ysd + zsd * zsd);
		double rsd3 = rsd*rsd*rsd;

		double fSRPX = _cram * xsd * Psrpas2 / rsd3;
		double fSRPY = _cram * ysd * Psrpas2 / rsd3;
		double fSRPZ = _cram * zsd * Psrpas2 / rsd3;

		// C22
		double tGvEt = thetaG + vE * t;
		double costGvEt = cos(tGvEt);
		double sintGvEt = sin(tGvEt);

		double xc = x * costGvEt + y * sintGvEt;
		double yc = -x * sintGvEt + y * costGvEt;
		double zc = z;

		double xcq = xc*xc;
		double ycq = yc*yc;
		double zcq = zc*zc;

		double rc = xcq + ycq + zcq;
		double rc5 = rc*rc*rc*rc*rc;
		double rc5s = sqrt(rc5);
		double rc7s = sqrt(rc5*rc*rc);

		double fc22x = 5*muERe2V15C22*xc*(ycq-xcq) / (2*rc7s) + muERe2V15C22*x / rc5s;
		double fc22y = 5*muERe2V15C22*yc*(ycq-xcq) / (2*rc7s) + muERe2V15C22*y / rc5s;
		double fc22Z = 5*muERe2V15C22*zc*(ycq-xcq) / (2*rc7s);
		double fc22X = fc22x * costGvEt - fc22y * sintGvEt;
		double fc22Y = fc22x * sintGvEt - fc22y * costGvEt;

		// S22
		double fs22x = -5*muERe2V15S22*xcq*y / rc7s + muERe2V15S22*y / rc5s;
		double fs22y = -5*muERe2V15S22*xc*ycq / rc7s + muERe2V15S22*x / rc5s;
		double fs22Z = -5*muERe2V15S22*zc*(ycq-xcq) / rc7s;
		double fs22X = fs22x * costGvEt - fs22y * sintGvEt;
		double fs22Y = fs22x * sintGvEt - fs22y * costGvEt;

		// solar gravity
		double fsunX = -muS * (xsd / rsd3 + xsun / rsun3);
		double fsunY = -muS * (ysd / rsd3 + ysun / rsun3);
		double fsunZ = -muS * (zsd / rsd3 + zsun / rsun3);

		// lunar gravity
		// solar radiation pressure
		double xmoo, ymoo, zmoo;
		moonr(t, xmoo, ymoo, zmoo);

		double rmoo = sqrt(xmoo * xmoo + ymoo * ymoo + zmoo * zmoo);
		double rmoo3 = rmoo*rmoo*rmoo;

		double xmd = x - xmoo;
		double ymd = y - ymoo;
		double zmd = z - zmoo;

		double rmd = sqrt(xmd * xmd + ymd * ymd + zmd * zmd);
		double rmd3 = rmd*rmd*rmd;

		double fmooX = -muM * (xmd / rmd3 + xmoo / rmoo3);
		double fmooY = -muM * (ymd / rmd3 + ymoo / rmoo3);
		double fmooZ = -muM * (zmd / rmd3 + zmoo / rmoo3);

		// acceleration
		yDot[3] = fkepX + fJ2X + fSRPX + fc22X + fs22X + fsunX + fmooX;
		yDot[4] = fkepY + fJ2Y + fSRPY + fc22Y + fs22Y + fsunY + fmooY;
		yDot[5] = fkepZ + fJ2Z + fSRPZ + fc22Z + fs22Z + fsunZ + fmooZ;
    }
};

using namespace std;

extern "C" {

int integratePVTwaste_C(double *rvt, double dt, double step,
		double cram, bool dopri) {

    state_t y(6);
    for (int i = 0; i < 6; i++) // r, v
        y[i] = rvt[i];

    PVTwaste pvt;
    pvt._cram = cram;

    int steps = 0;
    double t = rvt[7];
    double tN = t + dt;

    if (dt > 0) {
		if (dopri) {
			DOPRI45 integrator;
			while (t < tN) {
				steps++;
				if (t + step >= tN) {
					integrator(pvt, y, t, tN - t);
					break;
				} else
					integrator(pvt, y, t, step);
			}
		} else {
			RK4 integrator;
			while (t < tN) {
				steps++;
				if (t + step >= tN) {
					integrator(pvt, y, t, tN - t);
					break;
				} else
					integrator(pvt, y, t, step);
			}
		}
    } else {
    	step = -abs(step); // step negative
        if (dopri) {
            DOPRI45 integrator;
            while (t > dt) {
                steps++;
                if (t + step <= tN) {
                    integrator(pvt, y, t, tN - t);
                    break;
                } else
                    integrator(pvt, y, t, step);
            }
        } else {
            RK4 integrator;
            while (t > dt) {
                steps++;
                if (t + step <= tN) {
                    integrator(pvt, y, t, tN - t);
                    break;
                } else
                    integrator(pvt, y, t, step);
            }
        }
    }
    for (int i = 0; i < 6; i++)
        rvt[i] = y[i];
    rvt[7] = tN;
    return steps;
};

void integratePVTwasteN_C(double *rvt, double dtN, double step,
		double cram, int N, bool dopri, double* res) {
    try {
		state_t y(6);
		for (int i = 0; i < 6; i++) // r, v
			y[i] = rvt[i];

		PVTwaste pvt;
		pvt._cram = cram;

		// time[1], r[3], v[3], kep[6]
	    double t = rvt[7];
	    double tN = t + dtN;
		double dt = (tN - t) / N;

	    if (dtN > 0) {
			if (dopri) {
				DOPRI45 integrator;
				int j = 0;
				for (int i = 0; i < N+1; i++) {
					res[j++] = t;
					for (int k = 0; k < 6; k++)
						res[j++] = y[k];
					double kep[6];
					wic2par(y, kep, pvt.muE);
					for (int k = 0; k < 6; k++)
						res[j++] = kep[k];
					if (i == N) break;
					double nextT = t + dt;
					while (t < nextT) {
						if (t + step >= nextT) {
							integrator(pvt, y, t, nextT - t);
							break;
						} else
							integrator(pvt, y, t, step);
					}
				}
			} else {
				RK4 integrator;
				int j = 0;
				for (int i = 0; i < N+1; i++) {
					res[j++] = t;
					for (int k = 0; k < 6; k++)
						res[j++] = y[k];
					double kep[6];
					wic2par(y, kep, pvt.muE);
					for (int k = 0; k < 6; k++)
						res[j++] = kep[k];
					if (i == N) break;
					double nextT = t + dt;
					while (t < nextT) {
						if (t + step >= nextT) {
							integrator(pvt, y, t, nextT - t);
							break;
						} else
							integrator(pvt, y, t, step);
					}
				}
			}
	    } else {
			step = -abs(step); // step negative
			if (dopri) {
				DOPRI45 integrator;
				int j = 0;
				for (int i = 0; i < N+1; i++) {
					res[j++] = t;
					for (int k = 0; k < 6; k++)
						res[j++] = y[k];
					double kep[6];
					wic2par(y, kep, pvt.muE);
					for (int k = 0; k < 6; k++)
						res[j++] = kep[k];
					if (i == N) break;
					double nextT = t + dt;
					while (t > nextT) {
						if (t + step <= nextT) {
							integrator(pvt, y, t, nextT - t);
							break;
						} else
							integrator(pvt, y, t, step);
					}
				}
			} else {
				RK4 integrator;
				int j = 0;
				for (int i = 0; i < N+1; i++) {
					res[j++] = t;
					for (int k = 0; k < 6; k++)
						res[j++] = y[k];
					double kep[6];
					wic2par(y, kep, pvt.muE);
					for (int k = 0; k < 6; k++)
						res[j++] = kep[k];
					if (i == N) break;
					double nextT = t + dt;
					while (t > nextT) {
						if (t + step <= nextT) {
							integrator(pvt, y, t, nextT - t);
							break;
						} else
							integrator(pvt, y, t, step);
					}
				}
			}
		}
		for (int i = 0; i < 6; i++)
			rvt[i] = y[i];
		rvt[7] = tN;
     } catch (std::exception &e) {
         cout << e.what() << endl;
     }
};

}

