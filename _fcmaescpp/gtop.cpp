/***********************************************************************/
/*

 For information on ESA's original GTOP, see here:

 https://www.esa.int/gsp/ACT/projects/gtop/

 */
/***********************************************************************/
// ------------------------------------------------------------------------ //
// This source file is part of the 'ESA Advanced Concepts Team's            //
// Space Mechanics Toolbox' software.                                       //
//                                                                          //
// The source files are for research use only,                              //
// and are distributed WITHOUT ANY WARRANTY. Use them on your own risk.     //
//                                                                          //
// Copyright (c) 2004-2007 European Space Agency                            //
// ------------------------------------------------------------------------ //
#ifndef ASTRO_FUNCTIONS_H
#define ASTRO_FUNCTIONS_H

#include <math.h>
#include <float.h>
#include <iostream>
#include <vector>
#include <cctype>
// #include "ZeroFinder.h"

using namespace std;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Conversion from Mean Anomaly to Eccentric Anomaly via Kepler's equation
double Mean2Eccentric(const double, const double);

void Conversion(const double*, double*, double*, const double);

double norm(const double*, const double*);

double norm2(const double*);

void vett(const double*, const double*, double*);

double gtop_asinh(double);

double gtop_acosh(double);

double tofabn(const double&, const double&, const double&);

void vers(const double*, double*);

double x2tof(const double&, const double&, const double&, const int);

#endif

// ------------------------------------------------------------------------ //
// This source file is part of the 'ESA Advanced Concepts Team's            //
// Space Mechanics Toolbox' software.                                       //
//                                                                          //
// The source files are for research use only,                              //
// and are distributed WITHOUT ANY WARRANTY. Use them on your own risk.     //
//                                                                          //
// Copyright (c) 2004-2007 European Space Agency                            //
// ------------------------------------------------------------------------ //

#ifndef LAMBERT_H
#define LAMBERT_H

// #include "Astro_Functions.h"

void LambertI(const double*, const double*, double, const double, const int, //INPUT
        double*, double*, double&, double&, double&, int&);  //OUTPUT

#endif

// ------------------------------------------------------------------------ //
// This source file is part of the 'ESA Advanced Concepts Team's            //
// Space Mechanics Toolbox' software.                                       //
//                                                                          //
// The source files are for research use only,                              //
// and are distributed WITHOUT ANY WARRANTY. Use them on your own risk.     //
//                                                                          //
// Copyright (c) 2004-2007 European Space Agency                            //
// ------------------------------------------------------------------------ //

#ifndef MISSION_H
#define MISSION_H

#include <vector>
// #include "Pl_Eph_An.h"

using namespace std;

// problem types
#define orbit_insertion          0 // Tandem
#define total_DV_orbit_insertion 1 // Cassini 1
#define rndv                     2 // Rosetta
#define total_DV_rndv            3 // Cassini 2 and Messenger
#define asteroid_impact          4 // gtoc1
#define time2AUs                 5 // SAGAS 

struct customobject {
    double keplerian[6];
    double epoch;
    double mu;
};

struct mgaproblem {
    int type;                           //problem type
    vector<int> sequence; //fly-by sequence (ex: 3,2,3,3,5,is Earth-Venus-Earth-Earth-Jupiter)
    vector<int> rev_flag;               //vector of flags for clockwise legs
    double e;              //insertion e (only in case total_DV_orbit_insertion)
    double rp;      //insertion rp in km (only in case total_DV_orbit_insertion)
    customobject asteroid; //asteroid data (in case fly-by sequence has a final number = 10)
    double Isp;
    double mass;
    double DVlaunch;
};

int MGA(
//INPUTS
        vector<double>, mgaproblem,

        //OUTPUTS
        vector<double>&, vector<double>&, double&);

int MGAM(
//INPUTS
        vector<double>, mgaproblem,

        //OUTPUTS
        vector<double>&, vector<double>&, double&, double&);

#endif

// ------------------------------------------------------------------------ //
// This source file is part of the 'ESA Advanced Concepts Team's            //
// Space Mechanics Toolbox' software.                                       //
//                                                                          //
// The source files are for research use only,                              //
// and are distributed WITHOUT ANY WARRANTY. Use them on your own risk.     //
//                                                                          //
// Copyright (c) 2004-2007 European Space Agency                            //
// ------------------------------------------------------------------------ //

#ifndef MGA_DSM_H
#define MGA_DSM_H

#include <vector>
// #include "Pl_Eph_An.h"
// #include "Astro_Functions.h"
// #include "propagateKEP.h"
// #include "Lambert.h"
// #include "time2distance.h"
// #include "mga.h"

using namespace std;

struct mgadsmproblem {
    int type;                           //problem type
    vector<int> sequence; //fly-by sequence (ex: 3,2,3,3,5,is Earth-Venus-Earth-Earth-Jupiter)
    double e;              //insertion e (only in case total_DV_orbit_insertion)
    double rp;      //insertion rp in km (only in case total_DV_orbit_insertion)
    customobject asteroid; //asteroid data (in case fly-by sequence has a final number = 10)
    double AUdist;         //Distance to reach in AUs (only in case of time2AUs)
    double DVtotal;        //Total DV allowed in km/s (only in case of time2AUs)
    double DVonboard; //Total DV on the spacecraft in km/s (only in case of time2AUs)

    //Pre-allocated memory, in order to remove allocation of heap space in MGA_DSM calls
    //The DV vector serves also as output containing all the values for the impulsive DV 
    std::vector<double*> r;    // = std::vector<double*>(n);
    std::vector<double*> v;    // = std::vector<double*>(n);
    std::vector<double> DV;    // = std::vector<double>(n+1);
};

int MGA_DSM(
/* INPUT values: */
vector<double> x,  // it is the decision vector
        mgadsmproblem &mgadsm, // contains the problem specific data, passed as reference as mgadsm.DV is an output

        /* OUTPUT values: */
        double &J  // J output
        );

#endif

// ------------------------------------------------------------------------ //
// This source file is part of the 'ESA Advanced Concepts Team's            //
// Space Mechanics Toolbox' software.                                       //
//                                                                          //
// The source files are for research use only,                              //
// and are distributed WITHOUT ANY WARRANTY. Use them on your own risk.     //
//                                                                          //
// Copyright (c) 2004-2007 European Space Agency                            //
// ------------------------------------------------------------------------ //

#ifndef miscTandem_H
#define miscTandem_H

int xant(const double (&X)[5], const double &x);
int yant(const double (&Y)[15], const double &y);
int xantA5(const double (&X)[9], const double &x);
int yantA5(const double (&Y)[13], const double &y);
double interp2SF(const double (&X)[5], double (&Y)[15], const double &VINF,
        const double &declination);
double interp2A5(const double (&X)[5], double (&Y)[15], const double &VINF,
        const double &declination);
double SoyuzFregat(const double &VINF, const double &declination);
double Atlas501(const double &VINF, const double &declination);
void ecl2equ(double (&ecl)[3], double (&equ)[3]);

#endif

// ------------------------------------------------------------------------ //
// This source file is part of the 'ESA Advanced Concepts Team's            //
// Space Mechanics Toolbox' software.                                       //
//                                                                          //
// The source files are for research use only,                              //
// and are distributed WITHOUT ANY WARRANTY. Use them on your own risk.     //
//                                                                          //
// Copyright (c) 2004-2007 European Space Agency                            //
// ------------------------------------------------------------------------ //

#ifndef PL_EPH_AN_H
#define PL_EPH_AN_H

#include <vector>

using namespace std;

void Planet_Ephemerides_Analytical(const double, const int, double*, double*);

void Custom_Eph(const double, const double, const double[], double*, double*);

#endif

// ------------------------------------------------------------------------ //
// This source file is part of the 'ESA Advanced Concepts Team's            //
// Space Mechanics Toolbox' software.                                       //
//                                                                          //
// The source files are for research use only,                              //
// and are distributed WITHOUT ANY WARRANTY. Use them on your own risk.     //
//                                                                          //
// Copyright (c) 2004-2007 European Space Agency                            //
// ------------------------------------------------------------------------ //

#ifndef POWSWINGBYINV_H
#define POWSWINGBYINV_H

void PowSwingByInv2(const double, const double, const double, double&, double&);

#endif

// ------------------------------------------------------------------------ //
// This source file is part of the 'ESA Advanced Concepts Team's            //
// Space Mechanics Toolbox' software.                                       //
//                                                                          //
// The source files are for research use only,                              //
// and are distributed WITHOUT ANY WARRANTY. Use them on your own risk.     //
//                                                                          //
// Copyright (c) 2004-2006 European Space Agency                            //
// ------------------------------------------------------------------------ //

#ifndef PROPAGATEKEP_H
#define PROPAGATEKEP_H

// #include "Astro_Functions.h"

void propagateKEP(const double*, const double*, double, double, double*,
        double*);

void IC2par(const double*, const double*, double, double*);

void par2IC(const double*, double, double*, double*);

// Returns the cross product of the vectors X and Y.
// That is, z = X x Y.  X and Y must be 3 element
// vectors.
void cross(const double*, const double*, double*);

#endif

// ------------------------------------------------------------------------ //
// This source file is part of the 'ESA Advanced Concepts Team's            //
// Space Mechanics Toolbox' software.                                       //
//                                                                          //
// The source files are for research use only,                              //
// and are distributed WITHOUT ANY WARRANTY. Use them on your own risk.     //
//                                                                          //
// Copyright (c) 2004-2007 European Space Agency                            //
// ------------------------------------------------------------------------ //

#ifndef TIME_2_DISTANCE_H
#define TIME_2_DISTANCE_H

#include <math.h>
#include <float.h>
#include <iostream>
#include <vector>
#include <cctype>
// #include "propagateKEP.h"

using namespace std;

double time2distance(const double*, const double*, double);

#endif

/*
 *  GOProblems.h
 *  SeGMO, a Sequential Global Multiobjective Optimiser
 *
 *  Created by Dario Izzo on 5/17/08.
 *  Copyright 2008 ¿dvanced Concepts Team (European Space Agency). All rights reserved.
 *
 */

#ifndef TRAJOBJFUNS_H
#define TRAJOBJFUNS_H
#include<vector>

//NOTE: the functions here have passing by reference + const as they are called a lot of time during execution and thus
//it is worth trying to save time by avoiding to make a copy of the variable passed

double gtoc1(const std::vector<double> &x, std::vector<double> &rp);
double cassini1(const std::vector<double> &x, std::vector<double> &rp);
double sagas(const std::vector<double> &x);
double rosetta(const std::vector<double> &x);
double cassini2(const std::vector<double> &x);
double messenger(const std::vector<double> &x);
double messengerfull(const std::vector<double> &x);
double cassini1minlp(const std::vector<double> &x, std::vector<double> &rp,
        double &launchDV);
double tandem(const std::vector<double> &x, double &tof, const int sequence_[]);

#endif

// ------------------------------------------------------------------------ //
// This source file is part of the 'ESA Advanced Concepts Team's            //
// Space Mechanics Toolbox' software.                                       //
//                                                                          //
// The source files are for research use only,                              //
// and are distributed WITHOUT ANY WARRANTY. Use them on your own risk.     //
//                                                                          //
// Copyright (c) 2004-2007 European Space Agency                            //
// ------------------------------------------------------------------------ //

#ifndef ZEROFINDER_H
#define ZEROFINDER_H

using namespace std;

/// Namespace
namespace ZeroFinder {

/// Class for one dimensional functions
//  with some parameters
/**  The ()-operator with one double argument
 *  has to be overloaded for a derived class
 *  The return value is the ordinate computed for
 *  the abscissa-argument.
 */
class Function1D {
public:
    //virtual double Compute(double x)=0;
    virtual double operator()(double x)=0;
    // parameters
    double p1, p2;
    void SetParameters(double a, double b);
};

class Function1D_7param {
public:
    //virtual double Compute(double x)=0;
    virtual double operator()(double x)=0;
    // parameters
    double p1, p2, p3, p4, p5, p6, p7;
    void SetParameters(double a, double b, double c, double d, double e,
            double f, double g);
};

class FZero {
private:
    double a, c; // lower and upper bound

public:
    FZero(double a, double b); // constructor
    // fzero procedure
    double FindZero(Function1D &f);
    double FindZero7(Function1D_7param &f);
    void SetBounds(double a, double b);
};

}
#endif

#include <vector>
#include <stdlib.h>
#include "math.h"
#define VENUS 2
#define EARTH 3
#define MARS 4
#define JUPITER 5
#define SATURN 6
/***********************************************************************/
/***********************************************************************/
/***********************************************************************/
void gtopx(int GTOP_PROBLEM, double *f, double *g, double *x) {
    double DVtot, DVonboard, launchDV, flightDV, arrivalDV, dummy;
    std::vector<double> solution, rp;

    int n;
    if (GTOP_PROBLEM == 1) {
        n = 6;
    }
    if (GTOP_PROBLEM == 2) {
        n = 22;
    }
    if (GTOP_PROBLEM == 3) {
        n = 18;
    }
    if (GTOP_PROBLEM == 4) {
        n = 26;
    }
    if (GTOP_PROBLEM == 5) {
        n = 8;
    }
    if (GTOP_PROBLEM == 6) {
        n = 22;
    }
    if (GTOP_PROBLEM == 7) {
        n = 12;
    }
    if (GTOP_PROBLEM == 8) {
        n = 10;
    }
    if (GTOP_PROBLEM == 9) {
        n = 6;
    }
    if (GTOP_PROBLEM == 10) {
        n = 10;
    }

    for (int i = 0; i < n; ++i) {
        solution.push_back(x[i]);
    }

    if (GTOP_PROBLEM == 1) {
        /* Objective function value F(X) (denoted here as f[0]) */
        f[0] = cassini1(solution, rp);
        /* Constraints */
        g[0] = rp[0] - 6351.8;
        g[1] = rp[1] - 6351.8;
        g[2] = rp[2] - 6778.1;
        g[3] = rp[3] - 671492.0;
    }

    if (GTOP_PROBLEM == 2) {
        f[0] = cassini2(solution);
    }
    if (GTOP_PROBLEM == 3) {
        f[0] = messenger(solution);
    }
    if (GTOP_PROBLEM == 4) {
        f[0] = messengerfull(solution);
    }

    if (GTOP_PROBLEM == 5) {
        /* Objective function value F(X) (denoted here as f[0]) */
        f[0] = gtoc1(solution, rp);

        f[0] = f[0] - 2000000.0;

        /* Constraints */
        g[0] = rp[0] - 6351.8;
        g[1] = rp[1] - 6778.1;
        g[2] = rp[2] - 6351.8;
        g[3] = rp[3] - 6778.1;
        g[4] = rp[4] - 600000.0;
        g[5] = rp[5] - 70000.0;
    }

    if (GTOP_PROBLEM == 6) {
        f[0] = rosetta(solution);
    }

    if (GTOP_PROBLEM == 7) {
        /* Objective function value F(X) (denoted here as f[0]) */
        f[0] = sagas(solution);
        /* Constraints */
        g[0] = 6.782 - DVtot;
        g[1] = 1.782 - DVonboard;
    }

    if (GTOP_PROBLEM == 8) {
        /* Objective function value F(X) (denoted here as f[0]) */
        dummy = cassini1minlp(solution, rp, launchDV);
        /* Constraints */
        g[0] = rp[0] - 6351.8;
        g[1] = rp[1] - 6351.8;
        g[2] = rp[2] - 6778.1;
        g[3] = rp[3] - 671492.0;

        f[0] = dummy;
    }

    if (GTOP_PROBLEM == 9) {
        /* Objective function value F(X) (denoted here as f[0]) */
        dummy = cassini1(solution, rp);
        /* Constraints */
        g[0] = rp[0] - 6351.8;
        g[1] = rp[1] - 6351.8;
        g[2] = rp[2] - 6778.1;
        g[3] = rp[3] - 671492.0;
        g[4] = 7.0 - dummy;

        f[0] = dummy;
        f[1] = x[1] + x[2] + x[3] + x[4] + x[5]; //Total time of mission
    }

    if (GTOP_PROBLEM == 10) {
        /* Objective function value F(X) (denoted here as f[0]) */
        dummy = cassini1minlp(solution, rp, launchDV);
        /* Constraints */
        g[0] = rp[0] - 6351.8;
        g[1] = rp[1] - 6351.8;
        g[2] = rp[2] - 6778.1;
        g[3] = rp[3] - 671492.0;
        g[4] = 6.0 - dummy;

        f[0] = dummy;
        f[1] = x[1] + x[2] + x[3] + x[4] + x[5]; //Total time of mission

    }
}

// ------------------------------------------------------------------------ //
// This source file is part of the 'ESA Advanced Concepts Team's            //
// Space Mechanics Toolbox' software.                                       //
//                                                                          //
// The source files are for research use only,                              //
// and are distributed WITHOUT ANY WARRANTY. Use them on your own risk.     //
//                                                                          //
// Copyright (c) 2004-2007 European Space Agency                            //
// ------------------------------------------------------------------------ //

#include <iostream>
#include <iomanip>
// #include "Astro_Functions.h"

class CZF: public ZeroFinder::Function1D {
public:
    double operator()(double x) {
        return p1 * tan(x) - log(tan(0.5 * x + 0.25 * M_PI)) - p2;
    }
};

double Mean2Eccentric(const double M, const double e) {

    double tolerance = 1e-13;
    int n_of_it = 0; // Number of iterations
    double Eccentric, Ecc_New;
    double err = 1.0;

    if (e < 1.0) {
        Eccentric = M + e * cos(M); // Initial guess
        while ((err > tolerance) && (n_of_it < 100)) {
            Ecc_New = Eccentric
                    - (Eccentric - e * sin(Eccentric) - M)
                            / (1.0 - e * cos(Eccentric));
            err = fabs(Eccentric - Ecc_New);
            Eccentric = Ecc_New;
            n_of_it++;
        }
    } else {
        CZF FF;  // function to find its zero point
        ZeroFinder::FZero fz(-0.5 * 3.14159265358979 + 1e-8,
                0.5 * 3.14159265358979 - 1e-8);
        FF.SetParameters(e, M);
        Ecc_New = fz.FindZero(FF);
        Eccentric = Ecc_New;
    }
    return Eccentric;
}

void Conversion(const double *E, double *pos, double *vel, const double mu) {
    double a, e, ii, omg, omp, theta;
    double b, n;
    double X_per[3], X_dotper[3];
    double R[3][3];

    a = E[0];
    e = E[1];
    ii = E[2];
    omg = E[3];
    omp = E[4];
    theta = E[5];

    b = a * sqrt(1 - e * e);
    n = sqrt(mu / pow(a, 3));

    const double sin_theta = sin(theta);
    const double cos_theta = cos(theta);

    X_per[0] = a * (cos_theta - e);
    X_per[1] = b * sin_theta;

    X_dotper[0] = -(a * n * sin_theta) / (1 - e * cos_theta);
    X_dotper[1] = (b * n * cos_theta) / (1 - e * cos_theta);

    const double cosomg = cos(omg);
    const double cosomp = cos(omp);
    const double sinomg = sin(omg);
    const double sinomp = sin(omp);
    const double cosi = cos(ii);
    const double sini = sin(ii);

    R[0][0] = cosomg * cosomp - sinomg * sinomp * cosi;
    R[0][1] = -cosomg * sinomp - sinomg * cosomp * cosi;

    R[1][0] = sinomg * cosomp + cosomg * sinomp * cosi;
    R[1][1] = -sinomg * sinomp + cosomg * cosomp * cosi;

    R[2][0] = sinomp * sini;
    R[2][1] = cosomp * sini;

    // evaluate position and velocity
    for (int i = 0; i < 3; i++) {
        pos[i] = 0;
        vel[i] = 0;
        for (int j = 0; j < 2; j++) {
            pos[i] += R[i][j] * X_per[j];
            vel[i] += R[i][j] * X_dotper[j];
        }
    }
    return;
}

double norm(const double *vet1, const double *vet2) {
    double Vin = 0;
    for (int i = 0; i < 3; i++) {
        Vin += (vet1[i] - vet2[i]) * (vet1[i] - vet2[i]);
    }
    return sqrt(Vin);
}

double norm2(const double *vet1) {
    double temp = 0.0;
    for (int i = 0; i < 3; i++) {
        temp += vet1[i] * vet1[i];
    }
    return sqrt(temp);
}

//subfunction that evaluates vector product
void vett(const double *vet1, const double *vet2, double *prod) {
    prod[0] = (vet1[1] * vet2[2] - vet1[2] * vet2[1]);
    prod[1] = (vet1[2] * vet2[0] - vet1[0] * vet2[2]);
    prod[2] = (vet1[0] * vet2[1] - vet1[1] * vet2[0]);
}

double gtop_asinh(double x) {
    return log(x + sqrt(x * x + 1));
}

double gtop_acosh(double x) {
    return log(x + sqrt(x * x - 1));
}

double x2tof(const double &x, const double &s, const double &c, const int lw) {
    double am, a, alfa, beta;

    am = s / 2;
    a = am / (1 - x * x);
    if (x < 1)  //ellpise
            {
        beta = 2 * asin(sqrt((s - c) / (2 * a)));
        if (lw)
            beta = -beta;
        alfa = 2 * acos(x);
    } else {
        alfa = 2 * gtop_acosh(x);
        beta = 2 * gtop_asinh(sqrt((s - c) / (-2 * a)));
        if (lw)
            beta = -beta;
    }

    if (a > 0) {
        return (a * sqrt(a) * ((alfa - sin(alfa)) - (beta - sin(beta))));
    } else {
        return (-a * sqrt(-a) * ((sinh(alfa) - alfa) - (sinh(beta) - beta)));
    }

}

// Subfunction that evaluates the time of flight as a function of x
double tofabn(const double &sigma, const double &alfa, const double &beta) {
    if (sigma > 0) {
        return (sigma * sqrt(sigma) * ((alfa - sin(alfa)) - (beta - sin(beta))));
    } else {
        return (-sigma * sqrt(-sigma)
                * ((sinh(alfa) - alfa) - (sinh(beta) - beta)));
    }
}

// subfunction that evaluates unit vectors
void vers(const double *V_in, double *Ver_out) {
    double v_mod = 0;
    int i;

    for (i = 0; i < 3; i++) {
        v_mod += V_in[i] * V_in[i];
    }

    double sqrtv_mod = sqrt(v_mod);

    for (i = 0; i < 3; i++) {
        Ver_out[i] = V_in[i] / sqrtv_mod;
    }
}

// ------------------------------------------------------------------------ //
// This source file is part of the 'ESA Advanced Concepts Team's            //
// Space Mechanics Toolbox' software.                                       //
//                                                                          //
// The source files are for research use only,                              //
// and are distributed WITHOUT ANY WARRANTY. Use them on your own risk.     //
//                                                                          //
// Copyright (c) 2004-2007 European Space Agency                            //
// ------------------------------------------------------------------------ //

/*
 This routine implements a new algorithm that solves Lambert's problem. The
 algorithm has two major characteristics that makes it favorable to other
 existing ones.

 1) It describes the generic orbit solution of the boundary condition
 problem through the variable X=log(1+cos(alpha/2)). By doing so the
 graphs of the time of flight become defined in the entire real axis and
 resembles a straight line. Convergence is granted within few iterations
 for all the possible geometries (except, of course, when the transfer
 angle is zero). When multiple revolutions are considered the variable is
 X=tan(cos(alpha/2)*pi/2).

 2) Once the orbit has been determined in the plane, this routine
 evaluates the velocity vectors at the two points in a way that is not
 singular for the transfer angle approaching to pi (Lagrange coefficient
 based methods are numerically not well suited for this purpose).

 As a result Lambert's problem is solved (with multiple revolutions
 being accounted for) with the same computational effort for all
 possible geometries. The case of near 180 transfers is also solved
 efficiently.

 We note here that even when the transfer angle is exactly equal to pi
 the algorithm does solve the problem in the plane (it finds X), but it
 is not able to evaluate the plane in which the orbit lies. A solution
 to this would be to provide the direction of the plane containing the
 transfer orbit from outside. This has not been implemented in this
 routine since such a direction would depend on which application the
 transfer is going to be used in.

 Inputs:
 r1=Position vector at departure (column)
 r2=Position vector at arrival (column, same units as r1)
 t=Transfer time (scalar)
 mu=gravitational parameter (scalar, units have to be
 consistent with r1,t units)
 lw=1 if long way is chosen


 Outputs:
 v1=Velocity at departure        (consistent units)
 v2=Velocity at arrival
 a=semi major axis of the solution
 p=semi latus rectum of the solution
 theta=transfer angle in rad
 iter=number of iteration made by the newton solver (usually 6)
 */

#include <math.h>
#include <iostream>
// #include "Lambert.h"

using namespace std;

void LambertI(const double *r1_in, const double *r2_in, double t,
        const double mu, //INPUT
        const int lw, //INPUT
        double *v1, double *v2, double &a, double &p, double &theta, int &iter) //OUTPUT
        {
    double V, T,
            r2_mod = 0.0,    // R2 module
            dot_prod = 0.0, // dot product
            c,              // non-dimensional chord
            s,              // non dimesnional semi-perimeter
            am,             // minimum energy ellipse semi major axis
            lambda,         //lambda parameter defined in Battin's Book
            x, x1, x2, y1, y2, x_new = 0, y_new, err, alfa, beta, psi, eta,
            eta2, sigma1, vr1, vt1, vt2, vr2, R = 0.0;
    int i_count, i;
    const double tolerance = 1e-11;
    double r1[3], r2[3], r2_vers[3];
    double ih_dum[3], ih[3], dum[3];

    // Increasing the tolerance does not bring any advantage as the
    // precision is usually greater anyway (due to the rectification of the tof
    // graph) except near particular cases such as parabolas in which cases a
    // lower precision allow for usual convergence.

    if (t <= 0) {
        cout << "ERROR in Lambert Solver: Negative Time in input." << endl;
        return;
    }

    for (i = 0; i < 3; i++) {
        r1[i] = r1_in[i];
        r2[i] = r2_in[i];
        R += r1[i] * r1[i];
    }

    R = sqrt(R);
    V = sqrt(mu / R);
    T = R / V;

    // working with non-dimensional radii and time-of-flight
    t /= T;
    for (i = 0; i < 3; i++)  // r1 dimension is 3
            {
        r1[i] /= R;
        r2[i] /= R;
        r2_mod += r2[i] * r2[i];
    }

    // Evaluation of the relevant geometry parameters in non dimensional units
    r2_mod = sqrt(r2_mod);

    for (i = 0; i < 3; i++)
        dot_prod += (r1[i] * r2[i]);

    theta = acos(dot_prod / r2_mod);

    if (lw)
        theta = 2 * acos(-1.0) - theta;

    c = sqrt(1 + r2_mod * (r2_mod - 2.0 * cos(theta)));
    s = (1 + r2_mod + c) / 2.0;
    am = s / 2.0;
    lambda = sqrt(r2_mod) * cos(theta / 2.0) / s;

    // We start finding the log(x+1) value of the solution conic:
    // NO MULTI REV --> (1 SOL)
    //  inn1=-.5233;    //first guess point
    //  inn2=.5233;     //second guess point
    x1 = log(0.4767);
    x2 = log(1.5233);
    y1 = log(x2tof(-.5233, s, c, lw)) - log(t);
    y2 = log(x2tof(.5233, s, c, lw)) - log(t);

    // Newton iterations
    err = 1;
    i_count = 0;
    while ((err > tolerance) && (y1 != y2)) {
        i_count++;
        x_new = (x1 * y2 - y1 * x2) / (y2 - y1);
        y_new = logf(x2tof(expf(x_new) - 1, s, c, lw)) - logf(t); //[MR] Why ...f() functions? Loss of data!
        x1 = x2;
        y1 = y2;
        x2 = x_new;
        y2 = y_new;
        err = fabs(x1 - x_new);
    }
    iter = i_count;
    x = expf(x_new) - 1; //[MR] Same here... expf -> exp

    // The solution has been evaluated in terms of log(x+1) or tan(x*pi/2), we
    // now need the conic. As for transfer angles near to pi the lagrange
    // coefficient technique goes singular (dg approaches a zero/zero that is
    // numerically bad) we here use a different technique for those cases. When
    // the transfer angle is exactly equal to pi, then the ih unit vector is not
    // determined. The remaining equations, though, are still valid.

    a = am / (1 - x * x);

    // psi evaluation
    if (x < 1)  // ellipse
            {
        beta = 2 * asin(sqrt((s - c) / (2 * a)));
        if (lw)
            beta = -beta;
        alfa = 2 * acos(x);
        psi = (alfa - beta) / 2;
        eta2 = 2 * a * pow(sin(psi), 2) / s;
        eta = sqrt(eta2);
    } else       // hyperbola
    {
        beta = 2 * gtop_asinh(sqrt((c - s) / (2 * a)));
        if (lw)
            beta = -beta;
        alfa = 2 * gtop_acosh(x);
        psi = (alfa - beta) / 2;
        eta2 = -2 * a * pow(sinh(psi), 2) / s;
        eta = sqrt(eta2);
    }

    // parameter of the solution
    p = (r2_mod / (am * eta2)) * pow(sin(theta / 2), 2);
    sigma1 = (1 / (eta * sqrt(am))) * (2 * lambda * am - (lambda + x * eta));
    vett(r1, r2, ih_dum);
    vers(ih_dum, ih);

    if (lw) {
        for (i = 0; i < 3; i++)
            ih[i] = -ih[i];
    }

    vr1 = sigma1;
    vt1 = sqrt(p);
    vett(ih, r1, dum);

    for (i = 0; i < 3; i++)
        v1[i] = vr1 * r1[i] + vt1 * dum[i];

    vt2 = vt1 / r2_mod;
    vr2 = -vr1 + (vt1 - vt2) / tan(theta / 2);
    vers(r2, r2_vers);
    vett(ih, r2_vers, dum);
    for (i = 0; i < 3; i++)
        v2[i] = vr2 * r2[i] / r2_mod + vt2 * dum[i];

    for (i = 0; i < 3; i++) {
        v1[i] *= V;
        v2[i] *= V;
    }
    a *= R;
    p *= R;
}

// ------------------------------------------------------------------------ //
// This source file is part of the 'ESA Advanced Concepts Team's            //
// Space Mechanics Toolbox' software.                                       //
//                                                                          //
// The source files are for research use only,                              //
// and are distributed WITHOUT ANY WARRANTY. Use them on your own risk.     //
//                                                                          //
// Copyright (c) 2004-2007 European Space Agency                            //
// ------------------------------------------------------------------------ //

#include <iomanip>
#include <fstream>
#include <math.h>
#include <vector>
#include <iostream>
// #include "Pl_Eph_An.h"
// #include "mga.h"
// #include "Lambert.h"
// #include "PowSwingByInv.h"
// #include "Astro_Functions.h"
#include <stdio.h>
#define MAX(a, b) (a > b ? a : b)

using namespace std;

//the function return 0 if the input is right or -1 it there is something wrong

int MGA(vector<double> t, // it is the vector which provides time in modified julian date 2000.
                          // The first entry is launch date, the next entries represent the time needed to
                          // fly from last swing-by to current swing-by.
        mgaproblem problem,

        /* OUTPUT values: */
        vector<double> &rp,  // periplanets radius
        vector<double> &DV,  // final delta-Vs
        double &obj_funct) //objective function
        //double &martin_launch)   //objective function

        {
    const int n = problem.sequence.size();
    const vector<int> sequence = problem.sequence;
    const vector<int> rev_flag = problem.rev_flag; // array containing 0 clockwise, 1 un-clockwise
    customobject cust_obj = problem.asteroid;

    double MU[9] = {            //1.32712440018e11, //SUN = 0
            1.32712428e11, 22321,    // Gravitational constant of Mercury    = 1
                    324860,     // Gravitational constant of Venus      = 2
                    398601.19,  // Gravitational constant of Earth      = 3
                    42828.3,    // Gravitational constant of Mars       = 4
                    126.7e6,    // Gravitational constant of Jupiter    = 5
                    37.9e6,     // Gravitational constant of Saturn     = 6
                    5.78e6,     // Gravitational constant of Uranus     = 7
                    6.8e6       // Gravitational constant of Neptune    = 8
            };
    double penalty[9] = { 0, 0,        // Mercury
            6351.8,   // Venus
            6778.1,   // Earth
            6000,     // Mars
            //671492, // Jupiter
            600000,// Jupiter
            70000,    // Saturn
            0,        // Uranus
            0         // Neptune
            };

    double penalty_coeffs[9] = { 0, 0,      // Mercury
            0.01,   // Venus
            0.01,   // Earth
            0.01,   // Mars
            0.001,  // Jupiter
            0.01,   // Saturn
            0,      // Uranus
            0       // Neptune
            };

    double DVtot = 0;
    double Dum_Vec[3], Vin, Vout;
    double V_Lamb[2][2][3], dot_prod;
    double a, p, theta, alfa;
    double DVrel, DVarr = 0;

    //only used for orbit insertion (ex: cassini)
    double DVper, DVper2;
    const double rp_target = problem.rp;
    const double e_target = problem.e;
    const double DVlaunch = problem.DVlaunch;

    //only used for asteroid impact (ex: gtoc1)
    const double initial_mass = problem.mass;   // Satellite initial mass [Kg]
    double final_mass;                          // satelite final mass
    const double Isp = problem.Isp;            // Satellite specific impulse [s]
    const double g = 9.80665 / 1000.0;          // Gravity

    double *vec, *rec;
    vector<double*> r;    // {0...n-1} position
    vector<double*> v;    // {0...n-1} velocity

    double T = 0.0;         // total time

    int i_count, j_count, lw;

    int iter = 0;

    if (n >= 2) {
        for (i_count = 0; i_count < n; i_count++) {
            vec = new double[3];  // velocity and position are 3 D vector
            rec = new double[3];
            r.push_back(vec);
            v.push_back(rec);

            DV[i_count] = 0.0;
        }

        T = 0;
        for (i_count = 0; i_count < n; i_count++) {
            T += t[i_count];
            if (sequence[i_count] < 10)
                Planet_Ephemerides_Analytical(T, sequence[i_count], r[i_count],
                        v[i_count]); //r and  v in heliocentric coordinate system
            else {
                Custom_Eph(T + 2451544.5, cust_obj.epoch, cust_obj.keplerian,
                        r[i_count], v[i_count]);
            }
        }

        vett(r[0], r[1], Dum_Vec);

        if (Dum_Vec[2] > 0)
            lw = (rev_flag[0] == 0) ? 0 : 1;
        else
            lw = (rev_flag[0] == 0) ? 1 : 0;

        LambertI(r[0], r[1], t[1] * 24 * 60 * 60, MU[0], lw,          // INPUT
                V_Lamb[0][0], V_Lamb[0][1], a, p, theta, iter); // OUTPUT
        DV[0] = norm(V_Lamb[0][0], v[0]);                // Earth launch

        for (i_count = 1; i_count <= n - 2; i_count++) {
            vett(r[i_count], r[i_count + 1], Dum_Vec);

            if (Dum_Vec[2] > 0)
                lw = (rev_flag[i_count] == 0) ? 0 : 1;
            else
                lw = (rev_flag[i_count] == 0) ? 1 : 0;

            /*if (i_count%2 != 0)   {*/
            LambertI(r[i_count], r[i_count + 1], t[i_count + 1] * 24 * 60 * 60,
                    MU[0], lw, // INPUT
                    V_Lamb[1][0], V_Lamb[1][1], a, p, theta, iter);    // OUTPUT

            // norm first perform the subtraction of vet1-vet2 and the evaluate ||...||
            Vin = norm(V_Lamb[0][1], v[i_count]);
            Vout = norm(V_Lamb[1][0], v[i_count]);

            dot_prod = 0.0;
            for (int i = 0; i < 3; i++) {
                dot_prod += (V_Lamb[0][1][i] - v[i_count][i])
                        * (V_Lamb[1][0][i] - v[i_count][i]);
            }
            alfa = acos(dot_prod / (Vin * Vout));

            // calculation of delta V at pericenter
            PowSwingByInv2(Vin, Vout, alfa, DV[i_count], rp[i_count - 1]);

            rp[i_count - 1] *= MU[sequence[i_count]];

            if (i_count != n - 2)  //swap
                for (j_count = 0; j_count < 3; j_count++) {
                    V_Lamb[0][0][j_count] = V_Lamb[1][0][j_count]; // [j_count];
                    V_Lamb[0][1][j_count] = V_Lamb[1][1][j_count]; // [j_count];
                }
        }
    } else {
        return -1;
    }

    for (i_count = 0; i_count < 3; i_count++)
        Dum_Vec[i_count] = v[n - 1][i_count] - V_Lamb[1][1][i_count];

    DVrel = norm2(Dum_Vec);

    if (problem.type == total_DV_orbit_insertion) {

        DVper = sqrt(DVrel * DVrel + 2 * MU[sequence[n - 1]] / rp_target);
        DVper2 = sqrt(
                2 * MU[sequence[n - 1]] / rp_target
                        - MU[sequence[n - 1]] / rp_target * (1 - e_target));
        DVarr = fabs(DVper - DVper2);
    }

    else if (problem.type == asteroid_impact) {

        DVarr = DVrel;
    }

    DVtot = 0;

    for (i_count = 1; i_count < n - 1; i_count++)
        DVtot += DV[i_count];

    if (problem.type == total_DV_orbit_insertion) {

        DVtot += DVarr;
    }

    // Build Penalty
    for (i_count = 0; i_count < n - 2; i_count++)
        if (rp[i_count] < penalty[sequence[i_count + 1]])
            DVtot += penalty_coeffs[sequence[i_count + 1]]
                    * fabs(rp[i_count] - penalty[sequence[i_count + 1]]);

    // Launcher Constraint
    if (DV[0] > DVlaunch)
        DVtot += (DV[0] - DVlaunch);

    if (problem.type == total_DV_orbit_insertion) {

        obj_funct = DVtot;
    }

    else if (problem.type == asteroid_impact) {

        // Evaluation of satellite final mass
        obj_funct = final_mass = initial_mass * exp(-DVtot / (Isp * g));

        // V asteroid - V satellite
        for (i_count = 0; i_count < 3; i_count++)
            Dum_Vec[i_count] = v[n - 1][i_count] - V_Lamb[1][1][i_count]; // arrival relative velocity at the asteroid;

        dot_prod = 0;
        for (i_count = 0; i_count < 3; i_count++)
            dot_prod += Dum_Vec[i_count] * v[n - 1][i_count];

        obj_funct = 2000000 - (final_mass) * fabs(dot_prod);
    }

    // final clean
    for (i_count = 0; i_count < n; i_count++) {
        delete[] r[i_count];
        delete[] v[i_count];
    }
    r.clear();
    v.clear();

    //martin_launch = DV[0];

//     printf("\n DV[0]    = %f",DV[0]);
//     printf("\n DVlaunch = %f",DVlaunch);
//     printf("\n DVarr    = %f",DVarr);
//     printf("\n DVtot    = %f \n",DVtot);               

    return 0;
}

int MGAM(vector<double> t, // it is the vector which provides time in modified julian date 2000.
                           // The first entry is launch date, the next entries represent the time needed to
                           // fly from last swing-by to current swing-by.
        mgaproblem problem,

        /* OUTPUT values: */
        vector<double> &rp,  // periplanets radius
        vector<double> &DV,  // final delta-Vs
        double &obj_funct, //objective function
        double &martin_launch)   //objective function

        {
    const int n = problem.sequence.size();
    const vector<int> sequence = problem.sequence;
    const vector<int> rev_flag = problem.rev_flag; // array containing 0 clockwise, 1 un-clockwise
    customobject cust_obj = problem.asteroid;

    double MU[9] = {   //1.32712440018e11, //SUN = 0
            1.32712428e11, 22321,    // Gravitational constant of Mercury    = 1
                    324860,     // Gravitational constant of Venus      = 2
                    398601.19,  // Gravitational constant of Earth      = 3
                    42828.3,    // Gravitational constant of Mars       = 4
                    126.7e6,    // Gravitational constant of Jupiter    = 5
                    37.9e6,     // Gravitational constant of Saturn     = 6
                    5.78e6,     // Gravitational constant of Uranus     = 7
                    6.8e6       // Gravitational constant of Neptune    = 8
            };
    double penalty[9] = { 0, 0,        // Mercury
            6351.8,   // Venus
            6778.1,   // Earth
            6000,     // Mars
            //671492, // Jupiter
            600000,// Jupiter
            70000,    // Saturn
            0,        // Uranus
            0         // Neptune
            };

    double penalty_coeffs[9] = { 0, 0,      // Mercury
            0.01,   // Venus
            0.01,   // Earth
            0.01,   // Mars
            0.001,  // Jupiter
            0.01,   // Saturn
            0,      // Uranus
            0       // Neptune
            };

    double DVtot = 0;
    double Dum_Vec[3], Vin, Vout;
    double V_Lamb[2][2][3], dot_prod;
    double a, p, theta, alfa;
    double DVrel, DVarr = 0;

    //only used for orbit insertion (ex: cassini)
    double DVper, DVper2;
    const double rp_target = problem.rp;
    const double e_target = problem.e;
    const double DVlaunch = problem.DVlaunch;

    //only used for asteroid impact (ex: gtoc1)
    const double initial_mass = problem.mass;   // Satellite initial mass [Kg]
    double final_mass;                          // satelite final mass
    const double Isp = problem.Isp;            // Satellite specific impulse [s]
    const double g = 9.80665 / 1000.0;          // Gravity

    double *vec, *rec;
    vector<double*> r;    // {0...n-1} position
    vector<double*> v;    // {0...n-1} velocity

    double T = 0.0;         // total time

    int i_count, j_count, lw;

    int iter = 0;

    if (n >= 2) {
        for (i_count = 0; i_count < n; i_count++) {
            vec = new double[3];  // velocity and position are 3 D vector
            rec = new double[3];
            r.push_back(vec);
            v.push_back(rec);

            DV[i_count] = 0.0;
        }

        T = 0;
        for (i_count = 0; i_count < n; i_count++) {
            T += t[i_count];
            if (sequence[i_count] < 10)
                Planet_Ephemerides_Analytical(T, sequence[i_count], r[i_count],
                        v[i_count]); //r and  v in heliocentric coordinate system
            else {
                Custom_Eph(T + 2451544.5, cust_obj.epoch, cust_obj.keplerian,
                        r[i_count], v[i_count]);
            }
        }

        vett(r[0], r[1], Dum_Vec);

        if (Dum_Vec[2] > 0)
            lw = (rev_flag[0] == 0) ? 0 : 1;
        else
            lw = (rev_flag[0] == 0) ? 1 : 0;

        LambertI(r[0], r[1], t[1] * 24 * 60 * 60, MU[0], lw,          // INPUT
                V_Lamb[0][0], V_Lamb[0][1], a, p, theta, iter); // OUTPUT
        DV[0] = norm(V_Lamb[0][0], v[0]);                // Earth launch

        for (i_count = 1; i_count <= n - 2; i_count++) {
            vett(r[i_count], r[i_count + 1], Dum_Vec);

            if (Dum_Vec[2] > 0)
                lw = (rev_flag[i_count] == 0) ? 0 : 1;
            else
                lw = (rev_flag[i_count] == 0) ? 1 : 0;

            /*if (i_count%2 != 0)   {*/
            LambertI(r[i_count], r[i_count + 1], t[i_count + 1] * 24 * 60 * 60,
                    MU[0], lw, // INPUT
                    V_Lamb[1][0], V_Lamb[1][1], a, p, theta, iter);    // OUTPUT

            // norm first perform the subtraction of vet1-vet2 and the evaluate ||...||
            Vin = norm(V_Lamb[0][1], v[i_count]);
            Vout = norm(V_Lamb[1][0], v[i_count]);

            dot_prod = 0.0;
            for (int i = 0; i < 3; i++) {
                dot_prod += (V_Lamb[0][1][i] - v[i_count][i])
                        * (V_Lamb[1][0][i] - v[i_count][i]);
            }
            alfa = acos(dot_prod / (Vin * Vout));

            // calculation of delta V at pericenter
            PowSwingByInv2(Vin, Vout, alfa, DV[i_count], rp[i_count - 1]);

            rp[i_count - 1] *= MU[sequence[i_count]];

            if (i_count != n - 2)  //swap
                for (j_count = 0; j_count < 3; j_count++) {
                    V_Lamb[0][0][j_count] = V_Lamb[1][0][j_count]; // [j_count];
                    V_Lamb[0][1][j_count] = V_Lamb[1][1][j_count]; // [j_count];
                }
        }
    } else {
        return -1;
    }

    for (i_count = 0; i_count < 3; i_count++)
        Dum_Vec[i_count] = v[n - 1][i_count] - V_Lamb[1][1][i_count];

    DVrel = norm2(Dum_Vec);

    if (problem.type == total_DV_orbit_insertion) {

        DVper = sqrt(DVrel * DVrel + 2 * MU[sequence[n - 1]] / rp_target);
        DVper2 = sqrt(
                2 * MU[sequence[n - 1]] / rp_target
                        - MU[sequence[n - 1]] / rp_target * (1 - e_target));
        DVarr = fabs(DVper - DVper2);
    }

    else if (problem.type == asteroid_impact) {

        DVarr = DVrel;
    }

    DVtot = 0;

    for (i_count = 1; i_count < n - 1; i_count++)
        DVtot += DV[i_count];

    if (problem.type == total_DV_orbit_insertion) {

        DVtot += DVarr;
    }

    // Build Penalty
    for (i_count = 0; i_count < n - 2; i_count++)
        if (rp[i_count] < penalty[sequence[i_count + 1]])
            DVtot += penalty_coeffs[sequence[i_count + 1]]
                    * fabs(rp[i_count] - penalty[sequence[i_count + 1]]);

    // Launcher Constraint
    if (DV[0] > DVlaunch)
        DVtot += (DV[0] - DVlaunch);

    if (problem.type == total_DV_orbit_insertion) {

        obj_funct = DVtot;
    }

    else if (problem.type == asteroid_impact) {

        // Evaluation of satellite final mass
        obj_funct = final_mass = initial_mass * exp(-DVtot / (Isp * g));

        // V asteroid - V satellite
        for (i_count = 0; i_count < 3; i_count++)
            Dum_Vec[i_count] = v[n - 1][i_count] - V_Lamb[1][1][i_count]; // arrival relative velocity at the asteroid;

        dot_prod = 0;
        for (i_count = 0; i_count < 3; i_count++)
            dot_prod += Dum_Vec[i_count] * v[n - 1][i_count];

        obj_funct = 2000000 - (final_mass) * fabs(dot_prod);
    }

    // final clean
    for (i_count = 0; i_count < n; i_count++) {
        delete[] r[i_count];
        delete[] v[i_count];
    }
    r.clear();
    v.clear();

    martin_launch = DV[0];

//     printf("\n DV[0]    = %f",DV[0]);
//     printf("\n DVlaunch = %f",DVlaunch);
//     printf("\n DVarr    = %f",DVarr);
//     printf("\n DVtot    = %f \n",DVtot);               

    return 0;
}

// ------------------------------------------------------------------------ //
// This source file is part of the 'ESA Advanced Concepts Team's            //
// Space Mechanics Toolbox' software.                                       //
//                                                                          //
// The source files are for research use only,                              //
// and are distributed WITHOUT ANY WARRANTY. Use them on your own risk.     //
//                                                                          //
// Copyright (c) 2004-2007 European Space Agency                            //
// ------------------------------------------------------------------------ //

#include <iostream>
#include <iomanip>
#include <fstream>
// #include "Pl_Eph_An.h"
// #include "mga_dsm.h"

// [MR] TODO: exctract these somewhere...
const double MU[9] = { 1.32712428e11, // SUN                                  = 0
        22321,                  // Gravitational constant of Mercury    = 1
        324860,                 // Gravitational constant of Venus      = 2
        398601.19,              // Gravitational constant of Earth      = 3
        42828.3,                // Gravitational constant of Mars       = 4
        126.7e6,                // Gravitational constant of Jupiter    = 5
        0.37939519708830e8,     // Gravitational constant of Saturn     = 6
        5.78e6,                 // Gravitational constant of Uranus     = 7
        6.8e6                   // Gravitational constant of Neptune    = 8
        };

//  Definition of planetari radii
//  TODO: maybe put missing values here so that indices correspond to those from MU[]?
const double RPL[6] = { 2440,   // Mercury
        6052,   // Venus
        6378,   // Earth
        3397,   // Mars
        71492,  // Jupiter
        60330   // Saturn
        };

//TODO: move somewhere else
void vector_normalize(const double in[3], double out[3]) {
    double norm = norm2(in);
    for (int i = 0; i < 3; i++) {
        out[i] = in[i] / norm;
    }
}

/**
 * Compute velocity and position of an celestial object of interest at specified time.
 *
 * problem - concerned problem
 * T       - time
 * i_count - hop number (starting from 0)
 * r       - [output] object's position
 * v       - [output] object's velocity
 */
void get_celobj_r_and_v(const mgadsmproblem &problem, const double T,
        const int i_count, double *r, double *v) {
    if (problem.sequence[i_count] < 10) { //normal planet
        Planet_Ephemerides_Analytical(T, problem.sequence[i_count], r, v); // r and  v in heliocentric coordinate system
    } else { //asteroid
        Custom_Eph(T + 2451544.5, problem.asteroid.epoch,
                problem.asteroid.keplerian, r, v);
    }
}

/**
 * Precomputes all velocities and positions of celestial objects of interest for the problem.
 * Before calling this function, r and v verctors must be pre-allocated with sufficient amount of entries.
 *
 * problem - concerned problem
 * r       - [output] array of position vectors
 * v       - [output] array of velocity vectors
 */
void precalculate_ers_and_vees(const vector<double> &t,
        const mgadsmproblem &problem, std::vector<double*> &r,
        std::vector<double*> &v) {
    double T = t[0]; //time of departure

    for (unsigned int i_count = 0; i_count < problem.sequence.size();
            i_count++) {
        get_celobj_r_and_v(problem, T, i_count, r[i_count], v[i_count]);
        T += t[4 + i_count]; //time of flight
    }
}

/**
 * Get gravitational constant of an celestial object of interest.
 *
 * problem - concerned problem
 * i_count - hop number (starting from 0)
 */
double get_celobj_mu(const mgadsmproblem &problem, const int i_count) {
    if (problem.sequence[i_count] < 10) { //normal planet
        return MU[problem.sequence[i_count]];
    } else { //asteroid
        return problem.asteroid.mu;
    }
}

// FIRST BLOCK (P1 to P2)
/**
 * t          - decision vector
 * problem    - problem parameters
 * r          - planet positions
 * v          - planet velocities
 * DV         - [output] velocity contributions table
 * v_sc_pl_in - [output] next hop input speed
 */
void first_block(const vector<double> &t, const mgadsmproblem &problem,
        const std::vector<double*> &r, std::vector<double*> &v,
        std::vector<double> &DV, double v_sc_nextpl_in[3]) {
    //First, some helper constants to make code more readable
    const int n = problem.sequence.size();
    const double VINF = t[1];         // Hyperbolic escape velocity (km/sec)
    const double udir = t[2];       // Hyperbolic escape velocity var1 (non dim)
    const double vdir = t[3];       // Hyperbolic escape velocity var2 (non dim)
    // [MR] {LITTLE HACKER TRICK} Instead of copying (!) arrays let's just introduce pointers to appropriate positions in the decision vector.
    const double *tof = &t[4];
    const double *alpha = &t[n + 3];

    int i; //loop counter

    // Spacecraft position and velocity at departure
    double vtemp[3];
    cross(r[0], v[0], vtemp);

    double zP1[3];
    vector_normalize(vtemp, zP1);

    double iP1[3];
    vector_normalize(v[0], iP1);

    double jP1[3];
    cross(zP1, iP1, jP1);

    double theta, phi;
    theta = 2 * M_PI * udir;             // See Picking a Point on a Sphere
    phi = acos(2 * vdir - 1) - M_PI / 2; // In this way: -pi/2<phi<pi/2 so phi can be used as out-of-plane rotation

    double vinf[3];
    for (i = 0; i < 3; i++)
        vinf[i] = VINF
                * (cos(theta) * cos(phi) * iP1[i]
                        + sin(theta) * cos(phi) * jP1[i] + sin(phi) * zP1[i]);

    //double v_sc_pl_in[3];  // Spacecraft absolute incoming velocity at P1 [MR] not needed?
    double v_sc_pl_out[3]; // Spacecraft absolute outgoing velocity at P1

    for (i = 0; i < 3; i++) {
        //v_sc_pl_in[i]  = v[0][i];
        v_sc_pl_out[i] = v[0][i] + vinf[i];
    }

    // Computing S/C position and absolute incoming velocity at DSM1
    double rd[3], v_sc_dsm_in[3];

    propagateKEP(r[0], v_sc_pl_out, alpha[0] * tof[0] * 86400, MU[0], rd,
            v_sc_dsm_in); // [MR] last two are output.

    // Evaluating the Lambert arc from DSM1 to P2
    double Dum_Vec[3]; // [MR] Rename it to something sensible...
    vett(rd, r[1], Dum_Vec);

    int lw = (Dum_Vec[2] > 0) ? 0 : 1;
    double a, p, theta2;
    int iter_unused; // [MR] unused variable

    double v_sc_dsm_out[3]; // DSM output speed

    LambertI(rd, r[1], tof[0] * (1 - alpha[0]) * 86400, MU[0], lw, v_sc_dsm_out,
            v_sc_nextpl_in, a, p, theta2, iter_unused); // [MR] last 6 are output

    // First Contribution to DV (the 1st deep space maneuver)
    for (i = 0; i < 3; i++) {
        Dum_Vec[i] = v_sc_dsm_out[i] - v_sc_dsm_in[i]; // [MR] Temporary variable reused. Dirty.
    }

    DV[0] = norm2(Dum_Vec);
}

// ------
// INTERMEDIATE BLOCK
// WARNING: i_count starts from 0
void intermediate_block(const vector<double> &t, const mgadsmproblem &problem,
        const std::vector<double*> &r, const std::vector<double*> &v,
        int i_count, const double v_sc_pl_in[], std::vector<double> &DV,
        double *v_sc_nextpl_in) {
    //[MR] A bunch of helper variables to simplify the code
    const int n = problem.sequence.size();
    // [MR] {LITTLE HACKER TRICK} Instead of copying (!) arrays let's just introduce pointers to appropriate positions in the decision vector.
    const double *tof = &t[4];
    const double *alpha = &t[n + 3];
    const double *rp_non_dim = &t[2 * n + 2]; // non-dim perigee fly-by radius of planets P2..Pn(-1) (i=1 refers to the second planet)
    const double *gamma = &t[3 * n]; // rotation of the bplane-component of the swingby outgoing
    const vector<int> &sequence = problem.sequence;

    int i; //loop counter

    // Evaluation of the state immediately after Pi
    double v_rel_in[3];
    double vrelin = 0.0;

    for (i = 0; i < 3; i++) {
        v_rel_in[i] = v_sc_pl_in[i] - v[i_count + 1][i];
        vrelin += v_rel_in[i] * v_rel_in[i];
    }

    // Hop object's gravitional constant
    double hopobj_mu = get_celobj_mu(problem, i_count + 1);

    double e = 1.0
            + rp_non_dim[i_count] * RPL[sequence[i_count + 1] - 1] * vrelin
                    / hopobj_mu;

    double beta_rot = 2 * asin(1 / e); // velocity rotation

    double ix[3];
    vector_normalize(v_rel_in, ix);

    double vpervnorm[3];
    vector_normalize(v[i_count + 1], vpervnorm);

    double iy[3];
    vett(ix, vpervnorm, iy);
    vector_normalize(iy, iy); // [MR]this *might* not work properly...

    double iz[3];
    vett(ix, iy, iz);

    double v_rel_in_norm = norm2(v_rel_in);

    double v_sc_pl_out[3]; // TODO: document me!

    for (i = 0; i < 3; i++) {
        double iVout = cos(beta_rot) * ix[i]
                + cos(gamma[i_count]) * sin(beta_rot) * iy[i]
                + sin(gamma[i_count]) * sin(beta_rot) * iz[i];
        double v_rel_out = v_rel_in_norm * iVout;
        v_sc_pl_out[i] = v[i_count + 1][i] + v_rel_out;
    }

    // Computing S/C position and absolute incoming velocity at DSMi
    double rd[3], v_sc_dsm_in[3];

    propagateKEP(r[i_count + 1], v_sc_pl_out,
            alpha[i_count + 1] * tof[i_count + 1] * 86400, MU[0], rd,
            v_sc_dsm_in); // [MR] last two are output

    // Evaluating the Lambert arc from DSMi to Pi+1
    double Dum_Vec[3]; // [MR] Rename it to something sensible...
    vett(rd, r[i_count + 2], Dum_Vec);

    int lw = (Dum_Vec[2] > 0) ? 0 : 1;
    double a, p, theta;
    int iter_unused; // [MR] unused variable

    double v_sc_dsm_out[3]; // DSM output speed

    LambertI(rd, r[i_count + 2],
            tof[i_count + 1] * (1 - alpha[i_count + 1]) * 86400, MU[0], lw,
            v_sc_dsm_out, v_sc_nextpl_in, a, p, theta, iter_unused); // [MR] last 6 are output.

    // DV contribution
    for (i = 0; i < 3; i++) {
        Dum_Vec[i] = v_sc_dsm_out[i] - v_sc_dsm_in[i]; // [MR] Temporary variable reused. Dirty.
    }

    DV[i_count + 1] = norm2(Dum_Vec);
}

// FINAL BLOCK
//
void final_block(const mgadsmproblem &problem, const std::vector<double*> &v,
        const double v_sc_pl_in[], std::vector<double> &DV) {
    //[MR] A bunch of helper variables to simplify the code
    const int n = problem.sequence.size();
    const double rp_target = problem.rp;
    const double e_target = problem.e;
    const vector<int> &sequence = problem.sequence;

    int i; //loop counter

    // Evaluation of the arrival DV
    double Dum_Vec[3];
    for (i = 0; i < 3; i++) {
        Dum_Vec[i] = v[n - 1][i] - v_sc_pl_in[i];
    }

    double DVrel, DVarr;
    DVrel = norm2(Dum_Vec);  // Relative velocity at target planet

    if ((problem.type == orbit_insertion)
            || (problem.type == total_DV_orbit_insertion)) {
        double DVper = sqrt(
                DVrel * DVrel + 2 * MU[sequence[n - 1]] / rp_target); //[MR] should MU be changed to get_... ?
        double DVper2 = sqrt(
                2 * MU[sequence[n - 1]] / rp_target
                        - MU[sequence[n - 1]] / rp_target * (1 - e_target));
        DVarr = fabs(DVper - DVper2);
    } else if (problem.type == rndv) {
        DVarr = DVrel;
    } else if (problem.type == total_DV_rndv) {
        DVarr = DVrel;
    } else {
        DVarr = 0.0;  // no DVarr is considered
    }

    DV[n - 1] = DVarr;
}

int MGA_DSM(
/* INPUT values: */ //[MR] make this parameters const, if they are not modified and possibly references (especially 'problem').
        vector<double> t, // it is the vector which provides time in modified julian date 2000. [MR] ??? Isn't it the decision vetor ???
        mgadsmproblem &problem,

        /* OUTPUT values: */
        double &J) // , double &launch_dv, double &flight_dv, double &arrival_dv )
        {
    //[MR] A bunch of helper variables to simplify the code
    const int n = problem.sequence.size();

    int i; //loop counter

    //References to objects pre-allocated in the mgadsm struct
    std::vector<double*> &r = problem.r;
    std::vector<double*> &v = problem.v;

    std::vector<double> &DV = problem.DV; //DV contributions

    precalculate_ers_and_vees(t, problem, r, v);

    double inter_pl_in_v[3], inter_pl_out_v[3]; //inter-hop velocities

    // FIRST BLOCK
    first_block(t, problem, r, v, DV, inter_pl_out_v); // [MR] output

    // INTERMEDIATE BLOCK
    for (int i_count = 0; i_count < n - 2; i_count++) {
        //copy previous output velocity to current input velocity
        inter_pl_in_v[0] = inter_pl_out_v[0];
        inter_pl_in_v[1] = inter_pl_out_v[1];
        inter_pl_in_v[2] = inter_pl_out_v[2];

        intermediate_block(t, problem, r, v, i_count, inter_pl_in_v, DV,
                inter_pl_out_v);
    }

    //copy previous output velocity to current input velocity
    inter_pl_in_v[0] = inter_pl_out_v[0];
    inter_pl_in_v[1] = inter_pl_out_v[1];
    inter_pl_in_v[2] = inter_pl_out_v[2];
    // FINAL BLOCK
    final_block(problem, v, inter_pl_in_v, DV);

    // **************************************************************************
    // Evaluation of total DV spent by the propulsion system
    // **************************************************************************
    double DVtot = 0.0;

    for (i = 0; i < n; i++) {
        DVtot += DV[i];
    }

    // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    //[MR] Calculation of the actual procedure output (DVvec and J)

    const double &VINF = t[1]; // Variable renaming: Hyperbolic escape velocity (km/sec)

    for (i = n; i > 0; i--) {
        DV[i] = DV[i - 1];
    }
    DV[0] = VINF;

    // Finally our objective function (J) is:

    if (problem.type == orbit_insertion) {
        J = DVtot;
    } else if (problem.type == total_DV_orbit_insertion) {
        J = DVtot + VINF;
    } else if (problem.type == rndv) {
        J = DVtot;
    } else if (problem.type == total_DV_rndv) {
        J = DVtot + VINF;
    } else if (problem.type == time2AUs) { // [MR] TODO: extract method
        // [MR] helper constants
        const vector<int> &sequence = problem.sequence;
        const double *rp_non_dim = &t[2 * n + 2]; // non-dim perigee fly-by radius of planets P2..Pn(-1) (i=1 refers to the second planet)
        const double *gamma = &t[3 * n]; // rotation of the bplane-component of the swingby outgoing
        const double AUdist = problem.AUdist;
        const double DVtotal = problem.DVtotal;
        const double DVonboard = problem.DVonboard;
        const double *tof = &t[4];

        // non dimensional units
        const double AU = 149597870.66;
        const double V = sqrt(MU[0] / AU);
        const double T = AU / V;

        //evaluate the state of the spacecraft after the last fly-by
        double vrelin = 0.0;
        double v_rel_in[3];
        for (i = 0; i < 3; i++) {
            v_rel_in[i] = inter_pl_in_v[i] - v[n - 1][i];
            vrelin += v_rel_in[i] * v_rel_in[i];
        }

        double e = 1.0
                + rp_non_dim[n - 2] * RPL[sequence[n - 2 + 1] - 1] * vrelin
                        / get_celobj_mu(problem, n - 1); //I hope the planet index (n - 1) is OK

        double beta_rot = 2 * asin(1 / e);              // velocity rotation

        double vrelinn = norm2(v_rel_in);
        double ix[3];
        for (i = 0; i < 3; i++)
            ix[i] = v_rel_in[i] / vrelinn;
        // ix=r_rel_in/norm(v_rel_in);  // activating this line and disactivating the one above
        // shifts the singularity for r_rel_in parallel to v_rel_in

        double vnorm = norm2(v[n - 1]);

        double vtemp[3];
        for (i = 0; i < 3; i++)
            vtemp[i] = v[n - 1][i] / vnorm;

        double iy[3];
        vett(ix, vtemp, iy);

        double iynorm = norm2(iy);
        for (i = 0; i < 3; i++)
            iy[i] = iy[i] / iynorm;

        double iz[3];
        vett(ix, iy, iz);
        double v_rel_in_norm = norm2(v_rel_in);

        double v_sc_pl_out[3]; // TODO: document me!
        for (i = 0; i < 3; i++) {
            double iVout = cos(beta_rot) * ix[i]
                    + cos(gamma[n - 2]) * sin(beta_rot) * iy[i]
                    + sin(gamma[n - 2]) * sin(beta_rot) * iz[i];
            double v_rel_out = v_rel_in_norm * iVout;
            v_sc_pl_out[i] = v[n - 1][i] + v_rel_out;
        }

        double r_per_AU[3];
        double v_sc_pl_out_per_V[3];
        for (i = 0; i < 3; i++) {
            r_per_AU[i] = r[n - 1][i] / AU;
            v_sc_pl_out_per_V[i] = v_sc_pl_out[i] / V;
        }

        double time = time2distance(r_per_AU, v_sc_pl_out_per_V, AUdist);
        // if (time == -1) cout << " ERROR" << endl;

        if (time != -1) {
            double DVpen = 0;
            double sum = 0.0;

            for (i = 0; i < n + 1; i++)
                sum += DV[i];

            if (sum > DVtotal)
                //DVpen += DVpen+(sum-DVtotal);

                sum = 0.0;
            for (i = 1; i < n + 1; i++)
                sum += DV[i];

            if (sum > DVonboard)
                //DVpen = DVpen + (sum - DVonboard);

                sum = 0.0;
            for (i = 0; i < n - 1; i++)
                sum += tof[i];

            J = (time * T / 60 / 60 / 24 + sum) / 365.25 + DVpen * 100;
        } else
            J = 100000;   // there was an ERROR in time2distance
    } // time2AU

//     launch_dv = DV[0];
//     for (i=1; i<n; i++){ flight_dv += DV[i]; }
//     arrival_dv = DV[n];
//
    // printf("\n DV[0]    = %f",DV[0]);
    // printf("\n DV[n]    = %f",DV[n]);
    // printf("\n DVtot    = %f ",DVtot);   
    // printf("\n flight_dv    = %f ",flight_dv);    
    // printf("\n flight_dv + arrival_dv = %f \n",flight_dv+arrival_dv); 

    return 0;
}

// ------------------------------------------------------------------------ //
// This source file is part of the 'ESA Advanced Concepts Team's            //
// Space Mechanics Toolbox' software.                                       //
//                                                                          //
// The source files are for research use only,                              //
// and are distributed WITHOUT ANY WARRANTY. Use them on your own risk.     //
//                                                                          //
// Copyright (c) 2004-2007 European Space Agency                            //
// ------------------------------------------------------------------------ //

// #include "misc4Tandem.h"
#include "math.h"
#include <vector>

//The following data refer to the Launcher Atlas501 performances as given by NASA
static const double x_atl[9] = { 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 5.75, 6 };
static const double y_atl[13] = { -40, -30, -29, -28.5, -20, -10, 0, 10, 20,
        28.5, 29, 30, 40 };
static const double data_atl[13][9] = { { 0, 0, 0, 0, 0, 0, 0, 0, 0 }, { 0, 0,
        0, 0, 0, 0, 0, 0, 0 },
        { 1160, 1100, 1010, 930, 830, 740, 630, 590, 550 }, { 2335.0, 2195.0,
                2035.0, 1865.0, 1675.0, 1480.0, 1275.0, 1175.0, 1075.0 }, {
                2335.0, 2195.0, 2035.0, 1865.0, 1675.0, 1480.0, 1275.0, 1175.0,
                1075.0 }, { 2335.0, 2195.0, 2035.0, 1865.0, 1675.0, 1480.0,
                1275.0, 1175.0, 1075.0 }, { 2335.0, 2195.0, 2035.0, 1865.0,
                1675.0, 1480.0, 1275.0, 1175.0, 1075.0 }, { 2335.0, 2195.0,
                2035.0, 1865.0, 1675.0, 1480.0, 1275.0, 1175.0, 1075.0 }, {
                2335.0, 2195.0, 2035.0, 1865.0, 1675.0, 1480.0, 1275.0, 1175.0,
                1075.0 }, { 2335.0, 2195.0, 2035.0, 1865.0, 1675.0, 1480.0,
                1275.0, 1175.0, 1075.0 }, { 1160, 1100, 1010, 930, 830, 740,
                630, 590, 550 }, { 0, 0, 0, 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0,
                0, 0, 0, 0 } };

//The following data refer to the Launcher Soyuz-Fregat performances as given by ESOC. The data here and consider
// an elaborated five impulse strategy to exploit the launcher performances as much as possible.
static const double x_sf[5] = { 1, 2, 3, 4, 5 };
static const double y_sf[15] = { -90, -65, -50, -40, -30, -20, -10, 0, 10, 20,
        30, 40, 50, 65, 90 };
static const double data_sf[15][5] = { { 0.00000, 0.00000, 0.00000, 0.00000,
        0.00000 }, { 100.00000, 100.00000, 100.00000, 100.00000, 100.00000 }, {
        1830.50000, 1815.90000, 1737.70000, 1588.00000, 1344.30000 }, {
        1910.80000, 1901.90000, 1819.00000, 1636.40000, 1369.30000 }, {
        2001.80000, 1995.30000, 1891.30000, 1673.90000, 1391.90000 }, {
        2108.80000, 2088.60000, 1947.90000, 1708.00000, 1409.50000 }, {
        2204.00000, 2167.30000, 1995.50000, 1734.50000, 1419.60000 }, {
        2270.80000, 2205.80000, 2013.60000, 1745.10000, 1435.20000 }, {
        2204.70000, 2133.60000, 1965.40000, 1712.80000, 1413.60000 }, {
        2087.90000, 2060.60000, 1917.70000, 1681.10000, 1392.50000 }, {
        1979.17000, 1975.40000, 1866.50000, 1649.00000, 1371.70000 }, {
        1886.90000, 1882.20000, 1801.00000, 1614.60000, 1350.50000 }, {
        1805.90000, 1796.00000, 1722.70000, 1571.60000, 1327.60000 }, {
        100.00000, 100.00000, 100.00000, 100.00000, 100.00000 }, { 0.00000,
        0.00000, 0.00000, 0.00000, 0.00000 } };

int xant(const double &x) {
    int i;
    for (i = 1; i < 4; i++) {
        if (x_sf[i] > x)
            break;
    }
    return i - 1;
}

int yant(const double &y) {
    int i;
    for (i = 1; i < 14; i++) {
        if (y_sf[i] > y)
            break;
    }
    return i - 1;
}

int xantA5(const double &x) {
    int i;
    for (i = 1; i < 8; i++) {
        if (x_atl[i] > x)
            break;
    }
    return i - 1;
}

int yantA5(const double &y) {
    int i;
    for (i = 1; i < 12; i++) {
        if (y_atl[i] > y)
            break;
    }
    return i - 1;
}

double interp2SF(const double &VINF, const double &declination) {

    double retval;
    int v_indx = xant(VINF);
    int dec_indx = yant(declination);
    if (fabs(declination) >= 90)
        return 0;
    if ((VINF < 1) || (VINF > 5))
        return 0;

    double dx = x_sf[v_indx + 1] - x_sf[v_indx];
    double dydx = dx * (y_sf[dec_indx + 1] - y_sf[dec_indx]);
    retval = data_sf[dec_indx][v_indx] / dydx * (x_sf[v_indx + 1] - VINF)
            * (y_sf[dec_indx + 1] - declination);
    retval += data_sf[dec_indx][v_indx + 1] / dydx * (VINF - x_sf[v_indx])
            * (y_sf[dec_indx + 1] - declination);
    retval += data_sf[dec_indx + 1][v_indx] / dydx * (x_sf[v_indx + 1] - VINF)
            * (declination - y_sf[dec_indx]);
    retval += data_sf[dec_indx + 1][v_indx + 1] / dydx * (VINF - x_sf[v_indx])
            * (declination - y_sf[dec_indx]);
    return retval;
}

double interp2A5(const double &VINF, const double &declination) {

    double retval;
    int v_indx = xantA5(VINF);
    int dec_indx = yantA5(declination);
    if ((VINF < 2.5) || (VINF > 6))
        return 0;
    if (fabs(declination) > 40)
        return 0;

    double dx = x_atl[v_indx + 1] - x_atl[v_indx];
    double dydx = dx * (y_atl[dec_indx + 1] - y_atl[dec_indx]);
    retval = data_atl[dec_indx][v_indx] / dydx * (x_atl[v_indx + 1] - VINF)
            * (y_atl[dec_indx + 1] - declination);
    retval += data_atl[dec_indx][v_indx + 1] / dydx * (VINF - x_atl[v_indx])
            * (y_atl[dec_indx + 1] - declination);
    retval += data_atl[dec_indx + 1][v_indx] / dydx * (x_atl[v_indx + 1] - VINF)
            * (declination - y_atl[dec_indx]);
    retval += data_atl[dec_indx + 1][v_indx + 1] / dydx * (VINF - x_atl[v_indx])
            * (declination - y_atl[dec_indx]);
    return retval;
}

double SoyuzFregat(const double &VINF, const double &declination) {
//This function returns the mass that a Soyuz-Fregat launcher can inject
//into a given escape velocity and asymptote declination. The data here
//refer to ESOC WP-521 and consider an elaborated five impulse strategy to
//exploit the launcher performances as much as possible.
    return interp2SF(VINF, declination);
}

double Atlas501(const double &VINF, const double &declination) {
//This function returns the mass that a Atlas501 Launcher
    return interp2A5(VINF, declination);
}
void ecl2equ(double (&ecl)[3], double (&equ)[3]) {
    static const double incl = 0.409072975;
    double temp[3];
    temp[0] = ecl[0];
    temp[1] = ecl[1];
    temp[2] = ecl[2];
    equ[0] = temp[0];
    equ[1] = temp[1] * cos(incl) - temp[2] * sin(incl);
    equ[2] = temp[1] * sin(incl) + temp[2] * cos(incl);
}

// ------------------------------------------------------------------------ //
// This source file is part of the 'ESA Advanced Concepts Team's            //
// Space Mechanics Toolbox' software.                                       //
//                                                                          //
// The source files are for research use only,                              //
// and are distributed WITHOUT ANY WARRANTY. Use them on your own risk.     //
//                                                                          //
// Copyright (c) 2004-2007 European Space Agency                            //
// ------------------------------------------------------------------------ //

#include <iostream>
#include <iomanip>
// #include "Pl_Eph_An.h"
// #include "Astro_Functions.h"

void Planet_Ephemerides_Analytical(const double mjd2000, const int planet,
        double *position, double *velocity) {
    const double pi = acos(-1.0);
    const double RAD = pi / 180.0;
    const double AU = 149597870.66; // Astronomical Unit
    const double KM = AU;
    //const double MuSun = 1.32712440018e+11; //Gravitational constant of Sun);
    const double MuSun = 1.327124280000000e+011; //Gravitational constant of Sun);
    double Kepl_Par[6];
    double XM;

    double T = (mjd2000 + 36525.00) / 36525.00;

    switch (planet) {
    case (1):  // Mercury
        Kepl_Par[0] = (0.38709860);
        Kepl_Par[1] = (0.205614210 + 0.000020460 * T - 0.000000030 * T * T);
        Kepl_Par[2] = (7.002880555555555560 + 1.86083333333333333e-3 * T
                - 1.83333333333333333e-5 * T * T);
        Kepl_Par[3] = (4.71459444444444444e+1 + 1.185208333333333330 * T
                + 1.73888888888888889e-4 * T * T);
        Kepl_Par[4] = (2.87537527777777778e+1 + 3.70280555555555556e-1 * T
                + 1.20833333333333333e-4 * T * T);
        XM = 1.49472515288888889e+5 + 6.38888888888888889e-6 * T;
        Kepl_Par[5] = (1.02279380555555556e2 + XM * T);
        break;
    case (2):  // Venus
        Kepl_Par[0] = (0.72333160);
        Kepl_Par[1] = (0.006820690 - 0.000047740 * T + 0.0000000910 * T * T);
        Kepl_Par[2] = (3.393630555555555560 + 1.00583333333333333e-3 * T
                - 9.72222222222222222e-7 * T * T);
        Kepl_Par[3] = (7.57796472222222222e+1 + 8.9985e-1 * T + 4.1e-4 * T * T);
        Kepl_Par[4] = (5.43841861111111111e+1 + 5.08186111111111111e-1 * T
                - 1.38638888888888889e-3 * T * T);
        XM = 5.8517803875e+4 + 1.28605555555555556e-3 * T;
        Kepl_Par[5] = (2.12603219444444444e2 + XM * T);
        break;
    case (3):  // Earth
        Kepl_Par[0] = (1.000000230);
        Kepl_Par[1] = (0.016751040 - 0.000041800 * T - 0.0000001260 * T * T);
        Kepl_Par[2] = (0.00);
        Kepl_Par[3] = (0.00);
        Kepl_Par[4] = (1.01220833333333333e+2 + 1.7191750 * T
                + 4.52777777777777778e-4 * T * T
                + 3.33333333333333333e-6 * T * T * T);
        XM = 3.599904975e+4 - 1.50277777777777778e-4 * T
                - 3.33333333333333333e-6 * T * T;
        Kepl_Par[5] = (3.58475844444444444e2 + XM * T);
        break;
    case (4):  // Mars
        Kepl_Par[0] = (1.5236883990);
        Kepl_Par[1] = (0.093312900 + 0.0000920640 * T - 0.0000000770 * T * T);
        Kepl_Par[2] = (1.850333333333333330 - 6.75e-4 * T
                + 1.26111111111111111e-5 * T * T);
        Kepl_Par[3] = (4.87864416666666667e+1 + 7.70991666666666667e-1 * T
                - 1.38888888888888889e-6 * T * T
                - 5.33333333333333333e-6 * T * T * T);
        Kepl_Par[4] = (2.85431761111111111e+2 + 1.069766666666666670 * T
                + 1.3125e-4 * T * T + 4.13888888888888889e-6 * T * T * T);
        XM = 1.91398585e+4 + 1.80805555555555556e-4 * T
                + 1.19444444444444444e-6 * T * T;
        Kepl_Par[5] = (3.19529425e2 + XM * T);
        break;
    case (5):  // Jupiter
        Kepl_Par[0] = (5.2025610);
        Kepl_Par[1] = (0.048334750 + 0.000164180 * T - 0.00000046760 * T * T
                - 0.00000000170 * T * T * T);
        Kepl_Par[2] = (1.308736111111111110 - 5.69611111111111111e-3 * T
                + 3.88888888888888889e-6 * T * T);
        Kepl_Par[3] = (9.94433861111111111e+1 + 1.010530 * T
                + 3.52222222222222222e-4 * T * T
                - 8.51111111111111111e-6 * T * T * T);
        Kepl_Par[4] = (2.73277541666666667e+2 + 5.99431666666666667e-1 * T
                + 7.0405e-4 * T * T + 5.07777777777777778e-6 * T * T * T);
        XM = 3.03469202388888889e+3 - 7.21588888888888889e-4 * T
                + 1.78444444444444444e-6 * T * T;
        Kepl_Par[5] = (2.25328327777777778e2 + XM * T);
        break;
    case (6):  // Saturn
        Kepl_Par[0] = (9.5547470);
        Kepl_Par[1] = (0.055892320 - 0.00034550 * T - 0.0000007280 * T * T
                + 0.000000000740 * T * T * T);
        Kepl_Par[2] = (2.492519444444444440 - 3.91888888888888889e-3 * T
                - 1.54888888888888889e-5 * T * T
                + 4.44444444444444444e-8 * T * T * T);
        Kepl_Par[3] = (1.12790388888888889e+2 + 8.73195138888888889e-1 * T
                - 1.52180555555555556e-4 * T * T
                - 5.30555555555555556e-6 * T * T * T);
        Kepl_Par[4] = (3.38307772222222222e+2 + 1.085220694444444440 * T
                + 9.78541666666666667e-4 * T * T
                + 9.91666666666666667e-6 * T * T * T);
        XM = 1.22155146777777778e+3 - 5.01819444444444444e-4 * T
                - 5.19444444444444444e-6 * T * T;
        Kepl_Par[5] = (1.75466216666666667e2 + XM * T);
        break;
    case (7):  // Uranus
        Kepl_Par[0] = (19.218140);
        Kepl_Par[1] = (0.04634440 - 0.000026580 * T + 0.0000000770 * T * T);
        Kepl_Par[2] = (7.72463888888888889e-1 + 6.25277777777777778e-4 * T
                + 3.95e-5 * T * T);
        Kepl_Par[3] = (7.34770972222222222e+1 + 4.98667777777777778e-1 * T
                + 1.31166666666666667e-3 * T * T);
        Kepl_Par[4] = (9.80715527777777778e+1 + 9.85765e-1 * T
                - 1.07447222222222222e-3 * T * T
                - 6.05555555555555556e-7 * T * T * T);
        XM = 4.28379113055555556e+2 + 7.88444444444444444e-5 * T
                + 1.11111111111111111e-9 * T * T;
        Kepl_Par[5] = (7.26488194444444444e1 + XM * T);
        break;
    case (8):  //Neptune
        Kepl_Par[0] = (30.109570);
        Kepl_Par[1] = (0.008997040 + 0.0000063300 * T - 0.0000000020 * T * T);
        Kepl_Par[2] = (1.779241666666666670 - 9.54361111111111111e-3 * T
                - 9.11111111111111111e-6 * T * T);
        Kepl_Par[3] = (1.30681358333333333e+2 + 1.0989350 * T
                + 2.49866666666666667e-4 * T * T
                - 4.71777777777777778e-6 * T * T * T);
        Kepl_Par[4] = (2.76045966666666667e+2 + 3.25639444444444444e-1 * T
                + 1.4095e-4 * T * T + 4.11333333333333333e-6 * T * T * T);
        XM = 2.18461339722222222e+2 - 7.03333333333333333e-5 * T;
        Kepl_Par[5] = (3.77306694444444444e1 + XM * T);
        break;
    case (9):  // Pluto
        //Fifth order polynomial least square fit generated by Dario Izzo
        //(ESA ACT). JPL405 ephemerides (Charon-Pluto barycenter) have been used to produce the coefficients.
        //This approximation should not be used outside the range 2000-2100;
        T = mjd2000 / 36525.00;
        Kepl_Par[0] = (39.34041961252520 + 4.33305138120726 * T
                - 22.93749932403733 * T * T + 48.76336720791873 * T * T * T
                - 45.52494862462379 * T * T * T * T
                + 15.55134951783384 * T * T * T * T * T);
        Kepl_Par[1] = (0.24617365396517 + 0.09198001742190 * T
                - 0.57262288991447 * T * T + 1.39163022881098 * T * T * T
                - 1.46948451587683 * T * T * T * T
                + 0.56164158721620 * T * T * T * T * T);
        Kepl_Par[2] = (17.16690003784702 - 0.49770248790479 * T
                + 2.73751901890829 * T * T - 6.26973695197547 * T * T * T
                + 6.36276927397430 * T * T * T * T
                - 2.37006911673031 * T * T * T * T * T);
        Kepl_Par[3] = (110.222019291707 + 1.551579150048 * T
                - 9.701771291171 * T * T + 25.730756810615 * T * T * T
                - 30.140401383522 * T * T * T * T
                + 12.796598193159 * T * T * T * T * T);
        Kepl_Par[4] = (113.368933916592 + 9.436835192183 * T
                - 35.762300003726 * T * T + 48.966118351549 * T * T * T
                - 19.384576636609 * T * T * T * T
                - 3.362714022614 * T * T * T * T * T);
        Kepl_Par[5] = (15.17008631634665 + 137.023166578486 * T
                + 28.362805871736 * T * T - 29.677368415909 * T * T * T
                - 3.585159909117 * T * T * T * T
                + 13.406844652829 * T * T * T * T * T);
        break;

    }

    // conversion of AU into KM
    Kepl_Par[0] *= KM;

    // conversion of DEG into RAD
    Kepl_Par[2] *= RAD;
    Kepl_Par[3] *= RAD;
    Kepl_Par[4] *= RAD;
    Kepl_Par[5] *= RAD;
    Kepl_Par[5] = fmod(Kepl_Par[5], 2.0 * pi);

    // Conversion from Mean Anomaly to Eccentric Anomaly via Kepler's equation
    Kepl_Par[5] = Mean2Eccentric(Kepl_Par[5], Kepl_Par[1]);

    // Position and Velocity evaluation according to j2000 system
    Conversion(Kepl_Par, position, velocity, MuSun);
}

void Custom_Eph(const double jd, const double epoch, const double keplerian[],
        double *position, double *velocity) {
    const double pi = acos(-1.0);
    const double RAD = pi / 180.0;
    const double AU = 149597870.66; // Astronomical Unit
    const double muSUN = 1.32712428e+11;    // Gravitational constant of Sun
    double a, e, i, W, w, M, jdepoch, DT, n, E;
    double V[6];

    a = keplerian[0] * AU; // in km
    e = keplerian[1];
    i = keplerian[2];
    W = keplerian[3];
    w = keplerian[4];
    M = keplerian[5];
    jdepoch = epoch + 2400000.5;
    DT = (jd - jdepoch) * 86400;
    n = sqrt(muSUN / pow(a, 3));

    M = M / 180.0 * pi;
    M += n * DT;
    M = fmod(M, 2 * pi);
    E = Mean2Eccentric(M, e);
    V[0] = a;
    V[1] = e;
    V[2] = i * RAD;
    V[3] = W * RAD;
    V[4] = w * RAD;
    V[5] = E;

    Conversion(V, position, velocity, muSUN);
}

// ------------------------------------------------------------------------ //
// This source file is part of the 'ESA Advanced Concepts Team's            //
// Space Mechanics Toolbox' software.                                       //
//                                                                          //
// The source files are for research use only,                              //
// and are distributed WITHOUT ANY WARRANTY. Use them on your own risk.     //
//                                                                          //
// Copyright (c) 2004-2007 European Space Agency                            //
// ------------------------------------------------------------------------ //

// #include "PowSwingByInv.h"
#include <math.h>

void PowSwingByInv2 (const double Vin,const double Vout,const double alpha,
                    double &DV,double &rp)
{
    const int maxiter = 30;
    int i = 0;
    double err = 1.0;
    double f,df;                    // function and its derivative
    double rp_new;
    const double tolerance = 1e-8;

    double aIN  = 1.0/pow(Vin,2);     // semimajor axis of the incoming hyperbola
    double aOUT = 1.0/pow(Vout,2);    // semimajor axis of the incoming hyperbola

    rp = 1.0;
    while ((err > tolerance)&&(i < maxiter))
    {
        i++;
        f = asin(aIN/(aIN + rp)) + asin(aOUT/(aOUT + rp)) - alpha;
        df = -aIN/sqrt((rp + 2 * aIN) * rp)/(aIN+rp) -
             aOUT/sqrt((rp + 2 * aOUT) * rp)/(aOUT+rp);
        rp_new = rp - f/df;
        if (rp_new > 0)
        {
            err = fabs(rp_new - rp);
            rp = rp_new;
        }
        else
            rp /= 2.0;
    }

    // Evaluation of the DV
    DV = fabs (sqrt(Vout*Vout + (2.0/rp)) - sqrt(Vin*Vin + (2.0/rp)));
}

// ------------------------------------------------------------------------ //
// This source file is part of the 'ESA Advanced Concepts Team's            //
// Space Mechanics Toolbox' software.                                       //
//                                                                          //
// The source files are for research use only,                              //
// and are distributed WITHOUT ANY WARRANTY. Use them on your own risk.     //
//                                                                          //
// Copyright (c) 2004-2007 European Space Agency                            //
// ------------------------------------------------------------------------ //
//
// File: propageteKEP.cpp
//

// #include "propagateKEP.h"

/*
 Origin: MATLAB code programmed by Dario Izzo (ESA/ACT)

 C++ version by Tamas Vinko (ESA/ACT)

 Inputs:
 r0:    column vector for the non dimensional position
 v0:    column vector for the non dimensional velocity
 t:     non dimensional time

 Outputs:
 r:    column vector for the non dimensional position
 v:    column vector for the non dimensional velocity

 Comments:  The function works in non dimensional units, it takes an
 initial condition and it propagates it as in a kepler motion analytically.
 */

void propagateKEP(const double *r0_in, const double *v0_in, double t, double mu,
        double *r, double *v) {

    /*
     The matrix DD will be almost always the unit matrix, except for orbits
     with little inclination in which cases a rotation is performed so that
     par2IC is always defined
     */

    double DD[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
    double h[3];
    double ih[3] = { 0, 0, 0 };
    double temp1[3] = { 0, 0, 0 }, temp2[3] = { 0, 0, 0 };
    double E[6];
    double normh, M, M0;
    double r0[3], v0[3];

    int i;

    for (i = 0; i < 3; i++) {
        r0[i] = r0_in[i];
        v0[i] = v0_in[i];
    }

    vett(r0, v0, h);

    normh = norm2(h);

    for (i = 0; i < 3; i++)
        ih[i] = h[i] / normh;

    if (fabs(fabs(ih[2]) - 1.0) < 1e-3) // the abs is needed in cases in which the orbit is retrograde,
            {                                 // that would held ih=[0,0,-1]!!
        DD[0] = 1;
        DD[1] = 0;
        DD[2] = 0;
        DD[3] = 0;
        DD[4] = 0;
        DD[5] = 1;
        DD[6] = 0;
        DD[7] = -1;
        DD[8] = 0;

        // Random rotation matrix that make the Euler angles well defined for the case
        // For orbits with little inclination another ref. frame is used.

        for (i = 0; i < 3; i++) {
            temp1[0] += DD[i] * r0[i];
            temp1[1] += DD[i + 3] * r0[i];
            temp1[2] += DD[i + 6] * r0[i];
            temp2[0] += DD[i] * v0[i];
            temp2[1] += DD[i + 3] * v0[i];
            temp2[2] += DD[i + 6] * v0[i];
        }
        for (i = 0; i < 3; i++) {
            r0[i] = temp1[i];
            temp1[i] = 0.0;
            v0[i] = temp2[i];
            temp2[i] = 0.0;
        }
        // for practical reason we take the transpose of the matrix DD here (will be used at the end of the function)
        DD[0] = 1;
        DD[1] = 0;
        DD[2] = 0;
        DD[3] = 0;
        DD[4] = 0;
        DD[5] = -1;
        DD[6] = 0;
        DD[7] = 1;
        DD[8] = 0;
    }

    IC2par(r0, v0, mu, E);
    if (E[1] < 1.0) {
        M0 = E[5] - E[1] * sin(E[5]);
        M = M0 + sqrt(mu / pow(E[0], 3)) * t;
    } else {
        M0 = E[1] * tan(E[5]) - log(tan(0.5 * E[5] + 0.25 * M_PI));
        M = M0 + sqrt(mu / pow(-E[0], 3)) * t;
    }

    E[5] = Mean2Eccentric(M, E[1]);
    par2IC(E, mu, r, v);

    for (int j = 0; j < 3; j++) {
        temp1[0] += DD[j] * r[j];
        temp1[1] += DD[j + 3] * r[j];
        temp1[2] += DD[j + 6] * r[j];
        temp2[0] += DD[j] * v[j];
        temp2[1] += DD[j + 3] * v[j];
        temp2[2] += DD[j + 6] * v[j];
    }
    for (i = 0; i < 3; i++) {
        r[i] = temp1[i];
        v[i] = temp2[i];
    }

    return;
}

/*
 Origin: MATLAB code programmed by Dario Izzo (ESA/ACT)

 C++ version by Tamas Vinko (ESA/ACT) 12/09/2006

 Inputs:
 r0:    column vector for the position
 v0:    column vector for the velocity

 Outputs:
 E:     Column Vectors containing the six keplerian parameters,
 (a,e,i,OM,om,Eccentric Anomaly (or Gudermannian whenever e>1))

 Comments:  The parameters returned are, of course, referred to the same
 ref. frame in which r0,v0 are given. Units have to be consistent, and
 output angles are in radians
 The algorithm used is quite common and can be found as an example in Bate,
 Mueller and White. It goes singular for zero inclination
 */

void IC2par(const double *r0, const double *v0, double mu, double *E) {
    double k[3];
    double h[3];
    double Dum_Vec[3];
    double n[3];
    double evett[3];

    double p = 0.0;
    double temp = 0.0;
    double R0, ni;
    int i;

    vett(r0, v0, h);

    for (i = 0; i < 3; i++)
        p += h[i] * h[i];

    p /= mu;

    k[0] = 0;
    k[1] = 0;
    k[2] = 1;
    vett(k, h, n);

    for (i = 0; i < 3; i++)
        temp += pow(n[i], 2);

    temp = sqrt(temp);

    for (i = 0; i < 3; i++)
        n[i] /= temp;

    R0 = norm2(r0);

    vett(v0, h, Dum_Vec);

    for (i = 0; i < 3; i++)
        evett[i] = Dum_Vec[i] / mu - r0[i] / R0;

    double e = 0.0;
    for (i = 0; i < 3; i++)
        e += pow(evett[i], 2);

    E[0] = p / (1 - e);
    E[1] = sqrt(e);
    e = E[1];

    E[2] = acos(h[2] / norm2(h));

    temp = 0.0;
    for (i = 0; i < 3; i++)
        temp += n[i] * evett[i];

    E[4] = acos(temp / e);

    if (evett[2] < 0)
        E[4] = 2 * M_PI - E[4];

    E[3] = acos(n[0]);
    if (n[1] < 0)
        E[3] = 2 * M_PI - E[3];

    temp = 0.0;
    for (i = 0; i < 3; i++)
        temp += evett[i] * r0[i];

    ni = acos(temp / e / R0);  // danger, the argument could be improper.

    temp = 0.0;
    for (i = 0; i < 3; i++)
        temp += r0[i] * v0[i];

    if (temp < 0.0)
        ni = 2 * M_PI - ni;

    if (e < 1.0)
        E[5] = 2.0 * atan(sqrt((1 - e) / (1 + e)) * tan(ni / 2.0)); // algebraic kepler's equation
    else
        E[5] = 2.0 * atan(sqrt((e - 1) / (e + 1)) * tan(ni / 2.0)); // algebraic equivalent of kepler's equation in terms of the Gudermannian
}

/*
 Origin: MATLAB code programmed by Dario Izzo (ESA/ACT)

 C++ version by Tamas Vinko (ESA/ACT)

 Usage: [r0,v0] = IC2par(E,mu)

 Outputs:
 r0:    column vector for the position
 v0:    column vector for the velocity

 Inputs:
 E:     Column Vectors containing the six keplerian parameters,
 (a (negative for hyperbolas),e,i,OM,om,Eccentric Anomaly)
 mu:    gravitational constant

 Comments:  The parameters returned are, of course, referred to the same
 ref. frame in which r0,v0 are given. a can be given either in kms or AUs,
 but has to be consistent with mu.All the angles must be given in radians.
 This function does work for hyperbolas as well.
 */

void par2IC(const double *E, double mu, double *r0, double *v0) {
    double a = E[0];
    double e = E[1];
    double i = E[2];
    double omg = E[3];
    double omp = E[4];
    double EA = E[5];
    double b, n, xper, yper, xdotper, ydotper;
    double R[3][3];
    double cosomg, cosomp, sinomg, sinomp, cosi, sini;
    double dNdZeta;

    // Grandezze definite nel piano dell'orbita

    if (e < 1.0) {
        b = a * sqrt(1 - e * e);
        n = sqrt(mu / (a * a * a));
        xper = a * (cos(EA) - e);
        yper = b * sin(EA);

        xdotper = -(a * n * sin(EA)) / (1 - e * cos(EA));
        ydotper = (b * n * cos(EA)) / (1 - e * cos(EA));
    } else {
        b = -a * sqrt(e * e - 1);
        n = sqrt(-mu / (a * a * a));

        dNdZeta = e * (1 + tan(EA) * tan(EA))
                - (0.5 + 0.5 * pow(tan(0.5 * EA + 0.25 * M_PI), 2))
                        / tan(0.5 * EA + 0.25 * M_PI);

        xper = a / cos(EA) - a * e;
        yper = b * tan(EA);

        xdotper = a * tan(EA) / cos(EA) * n / dNdZeta;
        ydotper = b / pow(cos(EA), 2) * n / dNdZeta;
    }

    // Matrice di trasformazione da perifocale a ECI

    cosomg = cos(omg);
    cosomp = cos(omp);
    sinomg = sin(omg);
    sinomp = sin(omp);
    cosi = cos(i);
    sini = sin(i);

    R[0][0] = cosomg * cosomp - sinomg * sinomp * cosi;
    R[0][1] = -cosomg * sinomp - sinomg * cosomp * cosi;
    R[0][2] = sinomg * sini;
    R[1][0] = sinomg * cosomp + cosomg * sinomp * cosi;
    R[1][1] = -sinomg * sinomp + cosomg * cosomp * cosi;
    R[1][2] = -cosomg * sini;
    R[2][0] = sinomp * sini;
    R[2][1] = cosomp * sini;
    R[2][2] = cosi;

    // Posizione nel sistema inerziale

    double temp[3] = { xper, yper, 0.0 };
    double temp2[3] = { xdotper, ydotper, 0 };

    for (int j = 0; j < 3; j++) {
        r0[j] = 0.0;
        v0[j] = 0.0;
        for (int k = 0; k < 3; k++) {
            r0[j] += R[j][k] * temp[k];
            v0[j] += R[j][k] * temp2[k];
        }
    }
    return;
}

// Returns the cross product of the vectors X and Y.
// That is, z = X x Y.  X and Y must be 3 element
// vectors.
void cross(const double *x, const double *y, double *z) {
    z[0] = x[1] * y[2] - x[2] * y[1];
    z[1] = x[2] * y[0] - x[0] * y[2];
    z[2] = x[0] * y[1] - x[1] * y[0];
}

// ------------------------------------------------------------------------ //
// This source file is part of the 'ESA Advanced Concepts Team's            //
// Space Mechanics Toolbox' software.                                       //
//                                                                          //
// The source files are for research use only,                              //
// and are distributed WITHOUT ANY WARRANTY. Use them on your own risk.     //
//                                                                          //
// Copyright (c) 2004-2007 European Space Agency                            //
// ------------------------------------------------------------------------ //

/*
 %Inputs:
 %           r0:    column vector for the position (mu=1)
 %           v0:    column vector for the velocity (mu=1)
 %           rtarget: distance to be reached
 %
 %Outputs:
 %           t:     time taken to reach a given distance
 %
 %Comments:  everything works in non dimensional units
 */

// #include "time2distance.h"
#include <complex>

double time2distance(const double *r0, const double *v0, double rtarget) {
    double temp = 0.0;
    double E[6];
    double r0norm = norm2(r0);
    double a, e, E0, p, ni, Et;
    int i;

    if (r0norm < rtarget) {
        for (i = 0; i < 3; i++)
            temp += r0[i] * v0[i];

        IC2par(r0, v0, 1, E);
        a = E[0];
        e = E[1];
        E0 = E[5];
        p = a * (1 - e * e);
        // If the solution is an ellipse
        if (e < 1) {
            double ra = a * (1 + e);
            if (rtarget > ra)
                return -1; // NaN;
            else // we find the anomaly where the target distance is reached
            {
                ni = acos((p / rtarget - 1) / e);         //in 0-pi
                Et = 2 * atan(sqrt((1 - e) / (1 + e)) * tan(ni / 2)); // algebraic kepler's equation

                if (temp > 0)
                    return sqrt(pow(a, 3))
                            * (Et - e * sin(Et) - E0 + e * sin(E0));
                else {
                    E0 = -E0;
                    return sqrt(pow(a, 3))
                            * (Et - e * sin(Et) + E0 - e * sin(E0));
                }
            }
        } else // the solution is a hyperbolae
        {
            ni = acos((p / rtarget - 1) / e);         // in 0-pi
            Et = 2 * atan(sqrt((e - 1) / (e + 1)) * tan(ni / 2)); // algebraic equivalent of kepler's equation in terms of the Gudermannian

            if (temp > 0) // out==1
                return sqrt(pow((-a), 3))
                        * (e * tan(Et) - log(tan(Et / 2 + M_PI / 4))
                                - e * tan(E0) + log(tan(E0 / 2 + M_PI / 4)));
            else {
                E0 = -E0;
                return sqrt(pow((-a), 3))
                        * (e * tan(Et) - log(tan(Et / 2 + M_PI / 4))
                                + e * tan(E0) - log(tan(E0 / 2 + M_PI / 4)));
            }
        }
    } else
        return 12;
}

/*
 *  GOProblems.cpp
 *  SeGMO, a Sequential Global Multiobjective Optimiser
 *
 *  Created by Dario Izzo on 5/17/08.
 *  Copyright 2008 ¿dvanced Concepts Team (European Space Agency). All rights reserved.
 *
 */

#include <math.h>
// #include "mga.h"
// #include "mga_dsm.h"
// #include "misc4Tandem.h"

using namespace std;

double gtoc1(const vector<double> &x, std::vector<double> &rp) {
    const int GTOC1_DIM = 8;
    vector<double> Delta_V(GTOC1_DIM);
    rp.resize(GTOC1_DIM - 2);
    vector<double> t(GTOC1_DIM);
    mgaproblem problem;

    //Filling up the problem parameters
    problem.type = asteroid_impact;
    problem.mass = 1500.0;              // Satellite initial mass [Kg]
    problem.Isp = 2500.0;               // Satellite specific impulse [s]
    problem.DVlaunch = 2.5;             // Launcher DV in km/s

    int sequence_[GTOC1_DIM] = { 3, 2, 3, 2, 3, 5, 6, 10 }; // sequence of planets
    vector<int> sequence(GTOC1_DIM);
    problem.sequence.insert(problem.sequence.begin(), sequence_,
            sequence_ + GTOC1_DIM);

    const int rev_[GTOC1_DIM] = { 0, 0, 0, 0, 0, 0, 1, 0 }; // sequence of clockwise legs
    vector<int> rev(GTOC1_DIM);
    problem.rev_flag.insert(problem.rev_flag.begin(), rev_, rev_ + GTOC1_DIM);

    problem.asteroid.keplerian[0] = 2.5897261;    // Asteroid data
    problem.asteroid.keplerian[1] = 0.2734625;
    problem.asteroid.keplerian[2] = 6.40734;
    problem.asteroid.keplerian[3] = 128.34711;
    problem.asteroid.keplerian[4] = 264.78691;
    problem.asteroid.keplerian[5] = 320.479555;
    problem.asteroid.epoch = 53600;

    double obj = 0;
    //double launch_dv = 0;   

    MGA(x, problem, rp, Delta_V, obj);

    return obj;
}

double cassini1(const vector<double> &x, std::vector<double> &rp) {
    const int CASSINI_DIM = 6;
    vector<double> Delta_V(CASSINI_DIM);
    rp.resize(CASSINI_DIM - 2);
    vector<double> t(CASSINI_DIM);
    mgaproblem problem;

    //Filling up the problem parameters
    problem.type = total_DV_orbit_insertion;

    int sequence_[CASSINI_DIM] = { 3, 2, 2, 3, 5, 6 }; // sequence of planets
    vector<int> sequence(CASSINI_DIM);
    problem.sequence.insert(problem.sequence.begin(), sequence_,
            sequence_ + CASSINI_DIM);

    const int rev_[CASSINI_DIM] = { 0, 0, 0, 0, 0, 0 }; // sequence of clockwise legs
    vector<int> rev(CASSINI_DIM);
    problem.rev_flag.insert(problem.rev_flag.begin(), rev_, rev_ + CASSINI_DIM);

    problem.e = 0.98;      // Final orbit eccentricity
    problem.rp = 108950;    // Final orbit pericenter
    problem.DVlaunch = 0;   // Launcher DV

    double obj = 0;
    //double launch_dv = 0;

    MGA(x, problem, rp, Delta_V, obj);

    //launchDV = launch_dv;

    return obj;
}

double cassini1minlp(const vector<double> &x, std::vector<double> &rp,
        double &launchDV) {
    const int CASSINI_DIM = 6;
    vector<double> Delta_V(CASSINI_DIM);
    rp.resize(CASSINI_DIM - 2);
    vector<double> t(CASSINI_DIM);
    mgaproblem problem;

    //Filling up the problem parameters
    problem.type = total_DV_orbit_insertion;

    int sequence_[CASSINI_DIM] = { 3, (int) x[6], (int) x[7],
            (int) x[8], (int) x[9], 6 }; // sequence of planets

    vector<int> sequence(CASSINI_DIM);
    problem.sequence.insert(problem.sequence.begin(), sequence_,
            sequence_ + CASSINI_DIM);

    const int rev_[CASSINI_DIM] = { 0, 0, 0, 0, 0, 0 }; // sequence of clockwise legs
    vector<int> rev(CASSINI_DIM);
    problem.rev_flag.insert(problem.rev_flag.begin(), rev_, rev_ + CASSINI_DIM);

    problem.e = 0.98;      // Final orbit eccentricity
    problem.rp = 108950;    // Final orbit pericenter
    problem.DVlaunch = 0;   // Launcher DV

    double obj = 0;
    double launch_dv = 0;

    MGAM(x, problem, rp, Delta_V, obj, launch_dv);

    launchDV = launch_dv;

    return obj;
}

double messenger(const vector<double> &x) {
    mgadsmproblem problem;

    int sequence_[5] = { 3, 3, 2, 2, 1 }; // sequence of planets
    problem.sequence.insert(problem.sequence.begin(), sequence_, sequence_ + 5);
    problem.type = total_DV_rndv;

    //Memory allocation
    problem.r = std::vector<double*>(5);
    problem.v = std::vector<double*>(5);
    problem.DV = std::vector<double>(5 + 1);
    for (int i = 0; i < 5; i++) {
        problem.r[i] = new double[3];
        problem.v[i] = new double[3];
    }

    double obj = 0;
    double launch_dv = 0;
    double flight_dv = 0;
    double arrival_dv = 0;

    MGA_DSM(
    /* INPUT values: */
    x, problem,

    /* OUTPUT values: */
    obj);

    //Memory release
    for (int i = 0; i < 5; i++) {
        delete[] problem.r[i];
        delete[] problem.v[i];
    }
    problem.r.clear();
    problem.v.clear();

    return obj;
}

double messengerfull(const vector<double> &x) {
    mgadsmproblem problem;

    int sequence_[7] = { 3, 2, 2, 1, 1, 1, 1 };
    ; // sequence of planets
    problem.sequence.insert(problem.sequence.begin(), sequence_, sequence_ + 7);
    problem.type = orbit_insertion;
    problem.e = 0.704;
    problem.rp = 2640.0;

    //Memory allocation
    problem.r = std::vector<double*>(7);
    problem.v = std::vector<double*>(7);
    problem.DV = std::vector<double>(7 + 1);
    for (int i = 0; i < 7; i++) {
        problem.r[i] = new double[3];
        problem.v[i] = new double[3];
    }

    double obj = 0;
    double launch_dv = 0;
    double flight_dv = 0;
    double arrival_dv = 0;
    MGA_DSM(
    /* INPUT values: */
    x, problem,

    /* OUTPUT values: */
    obj);

    //Memory release
    for (int i = 0; i < 7; i++) {
        delete[] problem.r[i];
        delete[] problem.v[i];
    }
    problem.r.clear();
    problem.v.clear();

    return obj;
}

double cassini2(const vector<double> &x) {
    mgadsmproblem problem;

    int sequence_[6] = { 3, 2, 2, 3, 5, 6 }; // sequence of planets
    problem.sequence.insert(problem.sequence.begin(), sequence_, sequence_ + 6);
    problem.type = total_DV_rndv;

    double obj = 0;
    double launch_dv = 0;
    double flight_dv = 0;
    double arrival_dv = 0;

    //Allocate temporary memory for MGA_DSM
    problem.r = std::vector<double*>(6);
    problem.v = std::vector<double*>(6);
    problem.DV = std::vector<double>(6 + 1);

    for (int i = 0; i < 6; i++) {
        problem.r[i] = new double[3];
        problem.v[i] = new double[3];
    }

    MGA_DSM(
    /* INPUT values: */
    x, problem,

    /* OUTPUT values: */
    obj);

    //Free temporary memory for MGA_DSM
    for (int i = 0; i < 6; i++) {
        delete[] problem.r[i];
        delete[] problem.v[i];
    }
    problem.r.clear();
    problem.v.clear();

    return obj;
}

double cassini2minlp(const vector<double> &x) {
    mgadsmproblem problem;

    //int sequence_[6] = { 3, 2, 2, 3, 5, 6 }; // sequence of planets
    int sequence_[6] = { 3, (int) x[22], (int) x[23],
            (int) x[24], (int) x[25], 6 }; // sequence of planets
    problem.sequence.insert(problem.sequence.begin(), sequence_, sequence_ + 6);
    problem.type = total_DV_rndv;

    double obj = 0;
    double launch_dv = 0;
    double flight_dv = 0;
    double arrival_dv = 0;

    //Allocate temporary memory for MGA_DSM
    problem.r = std::vector<double*>(6);
    problem.v = std::vector<double*>(6);
    problem.DV = std::vector<double>(6 + 1);

    for (int i = 0; i < 6; i++) {
        problem.r[i] = new double[3];
        problem.v[i] = new double[3];
    }

    MGA_DSM(
    /* INPUT values: */
    x, problem,

    /* OUTPUT values: */
    obj);

    //Free temporary memory for MGA_DSM
    for (int i = 0; i < 6; i++) {
        delete[] problem.r[i];
        delete[] problem.v[i];
    }
    problem.r.clear();
    problem.v.clear();

    return obj;
}


double rosetta(const vector<double> &x) {
    mgadsmproblem problem;

    int sequence_[6] = { 3, 3, 4, 3, 3, 10 }; // sequence of planets
    problem.sequence.insert(problem.sequence.begin(), sequence_, sequence_ + 6);
    problem.type = rndv;
    problem.asteroid.keplerian[0] = 3.50294972836275;
    problem.asteroid.keplerian[1] = 0.6319356;
    problem.asteroid.keplerian[2] = 7.12723;
    problem.asteroid.keplerian[3] = 50.92302;
    problem.asteroid.keplerian[4] = 11.36788;
    problem.asteroid.keplerian[5] = 0.0;
    problem.asteroid.epoch = 52504.23754000012;
    problem.asteroid.mu = 0.0;

    //Allocate temporary memory for MGA_DSM
    problem.r = std::vector<double*>(6);
    problem.v = std::vector<double*>(6);
    problem.DV = std::vector<double>(6 + 1);

    for (int i = 0; i < 6; i++) {
        problem.r[i] = new double[3];
        problem.v[i] = new double[3];
    }

    double obj = 0;
    double launch_dv = 0;
    double flight_dv = 0;
    double arrival_dv = 0;
    MGA_DSM(
    /* INPUT values: */
    x, problem,

    /* OUTPUT values: */
    obj);

    //Free temporary memory for MGA_DSM
    for (int i = 0; i < 6; i++) {
        delete[] problem.r[i];
        delete[] problem.v[i];
    }
    problem.r.clear();
    problem.v.clear();

    return obj;
}

double sagas(const vector<double> &x) {
    mgadsmproblem problem;

    int sequence_[3] = { 3, 3, 5 }; // sequence of planets
    problem.sequence.insert(problem.sequence.begin(), sequence_, sequence_ + 3);
    problem.type = time2AUs;
    problem.AUdist = 50.0;
    problem.DVtotal = 6.782;
    problem.DVonboard = 1.782;

    //Allocate temporary memory for MGA_DSM
    problem.r = std::vector<double*>(3);
    problem.v = std::vector<double*>(3);
    problem.DV = std::vector<double>(3 + 2);

    for (int i = 0; i < 3; i++) {
        problem.r[i] = new double[3];
        problem.v[i] = new double[3];
    }

    double obj = 0;
    double launch_dv = 0;
    double flight_dv = 0;
    double arrival_dv = 0;
    MGA_DSM(
    /* INPUT values: */
    x, problem,

    /* OUTPUT values: */
    obj);

    double DVtot = problem.DV[0]+problem.DV[1]+problem.DV[2]+problem.DV[3]+problem.DV[4];
    double DVonboard = DVtot - problem.DV[0];

    if (DVtot > problem.DVtotal)
        obj += 10 + 10*DVtot;
    if (DVonboard > problem.DVonboard)
        obj += 10 + 10*DVonboard;

    //Free temporary memory for MGA_DSM
    for (int i = 0; i < 3; i++) {
        delete[] problem.r[i];
        delete[] problem.v[i];
    }
    problem.r.clear();
    problem.v.clear();

    return obj;
}

double tandem(const vector<double> &x, double &tof, const int sequence_[]) {
    mgadsmproblem problem;
    problem.sequence.insert(problem.sequence.begin(), sequence_, sequence_ + 5);
    problem.type = orbit_insertion;
    problem.rp = 80330.0;
    problem.e = 0.98531407996358;

    //Allocate temporary memory for MGA_DSM
    problem.r = std::vector<double*>(5);
    problem.v = std::vector<double*>(5);
    problem.DV = std::vector<double>(5 + 1);

    for (int i = 0; i < 5; i++) {
        problem.r[i] = new double[3];
        problem.v[i] = new double[3];
    }

    double obj = 0;
    MGA_DSM(
    /* INPUT values: */
    x, problem,

    /* OUTPUT values: */
    obj);

    //evaluating the mass from the dvs
    double rE[3];
    double vE[3];
    Planet_Ephemerides_Analytical(x[0], 3, rE, vE);
    double VINFE = x[1];
    double udir = x[2];
    double vdir = x[3];
    double vtemp[3];
    vtemp[0] = rE[1] * vE[2] - rE[2] * vE[1];
    vtemp[1] = rE[2] * vE[0] - rE[0] * vE[2];
    vtemp[2] = rE[0] * vE[1] - rE[1] * vE[0];
    double iP1[3];
    double normvE = sqrt(vE[0] * vE[0] + vE[1] * vE[1] + vE[2] * vE[2]);
    iP1[0] = vE[0] / normvE;
    iP1[1] = vE[1] / normvE;
    iP1[2] = vE[2] / normvE;
    double zP1[3];
    double normvtemp = sqrt(
            vtemp[0] * vtemp[0] + vtemp[1] * vtemp[1] + vtemp[2] * vtemp[2]);
    zP1[0] = vtemp[0] / normvtemp;
    zP1[1] = vtemp[1] / normvtemp;
    zP1[2] = vtemp[2] / normvtemp;
    double jP1[3];
    jP1[0] = zP1[1] * iP1[2] - zP1[2] * iP1[1];
    jP1[1] = zP1[2] * iP1[0] - zP1[0] * iP1[2];
    jP1[2] = zP1[0] * iP1[1] - zP1[1] * iP1[0];
    double theta = 2 * M_PI * udir;       //See Picking a Point on a Sphere
    double phi = acos(2 * vdir - 1) - M_PI / 2; //In this way: -pi/2<phi<pi/2 so phi can be used as out-of-plane rotation
    double vinf[3];
    vinf[0] = VINFE
            * (cos(theta) * cos(phi) * iP1[0] + sin(theta) * cos(phi) * jP1[0]
                    + sin(phi) * zP1[0]);
    vinf[1] = VINFE
            * (cos(theta) * cos(phi) * iP1[1] + sin(theta) * cos(phi) * jP1[1]
                    + sin(phi) * zP1[1]);
    vinf[2] = VINFE
            * (cos(theta) * cos(phi) * iP1[2] + sin(theta) * cos(phi) * jP1[2]
                    + sin(phi) * zP1[2]);
    //We rotate it to the equatorial plane
    ecl2equ(vinf, vinf);
    //And we find the declination in degrees
    double normvinf = sqrt(
            vinf[0] * vinf[0] + vinf[1] * vinf[1] + vinf[2] * vinf[2]);
    double sindelta = vinf[2] / normvinf;
    double declination = asin(sindelta) / M_PI * 180;

    //double m_initial = SoyuzFregat(VINFE,declination);
    double m_initial = Atlas501(VINFE, declination);

    //We evaluate the final mass
    double Isp = 312;
    double g0 = 9.80665;
    double sumDVvec = 0;
    tof = x[4] + x[5] + x[6] + x[7];
    for (unsigned int i = 1; i <= 5; i++) {
        sumDVvec = sumDVvec + problem.DV[i];
    }
    double m_final;
    sumDVvec = sumDVvec + 0.165; //losses for 3 swgbys + insertion

    m_final = m_initial * exp(-sumDVvec / Isp / g0 * 1000);

    //Free temporary memory for MGA_DSM
    for (int i = 0; i < 5; i++) {
        delete[] problem.r[i];
        delete[] problem.v[i];
    }
    problem.r.clear();
    problem.v.clear();

    return -m_final;
}

// ------------------------------------------------------------------------ //
// This source file is part of the 'ESA Advanced Concepts Team's            //
// Space Mechanics Toolbox' software.                                       //
//                                                                          //
// The source files are for research use only,                              //
// and are distributed WITHOUT ANY WARRANTY. Use them on your own risk.     //
//                                                                          //
// Copyright (c) 2004-2007 European Space Agency                            //
// ------------------------------------------------------------------------ //

#include <valarray>
#include <vector>
// #include "ZeroFinder.h"
#include <math.h>

#define NUMERATOR(dab,dcb,fa,fb,fc) fb*(dab*fc*(fc-fb)-fa*dcb*(fa-fb))
#define DENOMINATOR(fa,fb,fc) (fc-fb)*(fa-fb)*(fa-fc)

void ZeroFinder::Function1D::SetParameters(double a, double b) {
    p1 = a;
    p2 = b;
}

void ZeroFinder::Function1D_7param::SetParameters(double a, double b, double c,
        double d, double e, double f, double g) {
    p1 = a;
    p2 = b;
    p3 = c;
    p4 = d;
    p5 = e;
    p6 = f;
    p7 = g;
}

//
ZeroFinder::FZero::FZero(double a, double b) {
    SetBounds(a, b);
}

//
void ZeroFinder::FZero::SetBounds(double lb, double ub) {
    a = lb;
    c = ub;
}

//-------------------------------------------------------------------------------------//
// This part is an adaption of the 'Amsterdam method', which is an inversre quadratic  //
// interpolation - bisection method                                                    //
// See http://mymathlib.webtrellis.net/roots/amsterdam.html                            //
//
//-------------------------------------------------------------------------------------//
double ZeroFinder::FZero::FindZero(Function1D &f) {
    int max_iterations = 500;
    double tolerance = 1e-15;

    double fa = f(a), b = 0.5 * (a + c), fc = f(c), fb = fa * fc;
    double delta, dab, dcb;
    int i;

    // If the initial estimates do not bracket a root, set the err flag and //
    // return.  If an initial estimate is a root, then return the root.     //

    //double err = 0;
    if (fb >= 0.0) {
        if (fb > 0.0) {
            return 0.0;
        } else
            return (fa == 0.0) ? a : c;
    }

    // Insure that the initial estimate a < c. //

    if (a > c) {
        delta = a;
        a = c;
        c = delta;
        delta = fa;
        fa = fc;
        fc = delta;
    }

    // If the function at the left endpoint is positive, and the function //
    // at the right endpoint is negative.  Iterate reducing the length    //
    // of the interval by either bisection or quadratic inverse           //
    // interpolation.  Note that the function at the left endpoint        //
    // remains nonnegative and the function at the right endpoint remains //
    // nonpositive.                                                       //

    if (fa > 0.0)
        for (i = 0; i < max_iterations; i++) {

            // Are the two endpoints within the user specified tolerance ?

            if ((c - a) < tolerance)
                return 0.5 * (a + c);

            // No, Continue iteration.

            fb = f(b);

            // Check that we are converging or that we have converged near //
            // the left endpoint of the inverval.  If it appears that the  //
            // interval is not decreasing fast enough, use bisection.      //
            if ((c - a) < tolerance)
                return 0.5 * (a + c);
            if ((b - a) < tolerance) {
                if (fb > 0) {
                    a = b;
                    fa = fb;
                    b = 0.5 * (a + c);
                    continue;
                } else
                    return b;
            }

            // Check that we are converging or that we have converged near //
            // the right endpoint of the inverval.  If it appears that the //
            // interval is not decreasing fast enough, use bisection.      //

            if ((c - b) < tolerance) {
                if (fb < 0) {
                    c = b;
                    fc = fb;
                    b = 0.5 * (a + c);
                    continue;
                } else
                    return b;
            }

            // If quadratic inverse interpolation is feasible, try it. //

            if ((fa > fb) && (fb > fc)) {
                delta = DENOMINATOR(fa, fb, fc);
                if (delta != 0.0) {
                    dab = a - b;
                    dcb = c - b;
                    delta = NUMERATOR(dab,dcb,fa,fb,fc) / delta;

                    // Will the new estimate of the root be within the   //
                    // interval?  If yes, use it and update interval.    //
                    // If no, use the bisection method.                  //

                    if (delta > dab && delta < dcb) {
                        if (fb > 0.0) {
                            a = b;
                            fa = fb;
                        } else if (fb < 0.0) {
                            c = b;
                            fc = fb;
                        } else
                            return b;
                        b += delta;
                        continue;
                    }
                }
            }

            // If not, use the bisection method. //

            fb > 0 ? (a = b, fa = fb) : (c = b, fc = fb);
            b = 0.5 * (a + c);
        }
    else

        // If the function at the left endpoint is negative, and the function //
        // at the right endpoint is positive.  Iterate reducing the length    //
        // of the interval by either bisection or quadratic inverse           //
        // interpolation.  Note that the function at the left endpoint        //
        // remains nonpositive and the function at the right endpoint remains //
        // nonnegative.                                                       //

        for (i = 0; i < max_iterations; i++) {
            if ((c - a) < tolerance)
                return 0.5 * (a + c);
            fb = f(b);

            if ((b - a) < tolerance) {
                if (fb < 0) {
                    a = b;
                    fa = fb;
                    b = 0.5 * (a + c);
                    continue;
                } else
                    return b;
            }

            if ((c - b) < tolerance) {
                if (fb > 0) {
                    c = b;
                    fc = fb;
                    b = 0.5 * (a + c);
                    continue;
                } else
                    return b;
            }

            if ((fa < fb) && (fb < fc)) {
                delta = DENOMINATOR(fa, fb, fc);
                if (delta != 0.0) {
                    dab = a - b;
                    dcb = c - b;
                    delta = NUMERATOR(dab,dcb,fa,fb,fc) / delta;
                    if (delta > dab && delta < dcb) {
                        if (fb < 0.0) {
                            a = b;
                            fa = fb;
                        } else if (fb > 0.0) {
                            c = b;
                            fc = fb;
                        } else
                            return b;
                        b += delta;
                        continue;
                    }
                }
            }
            fb < 0 ? (a = b, fa = fb) : (c = b, fc = fb);
            b = 0.5 * (a + c);
        }
    //err = -2;
    return b;
}

//-----------------------------------------------------------------------------------
double ZeroFinder::FZero::FindZero7(Function1D_7param &f) {
    int max_iterations = 500;
    double tolerance = 1e-15;

    double fa = f(a), b = 0.5 * (a + c), fc = f(c), fb = fa * fc;
    double delta, dab, dcb;
    int i;

    // If the initial estimates do not bracket a root, set the err flag and //
    // return.  If an initial estimate is a root, then return the root.     //

    //double err = 0;
    if (fb >= 0.0) {
        if (fb > 0.0) {
            return 0.0;
        } else
            return (fa == 0.0) ? a : c;
    }

    // Insure that the initial estimate a < c. //

    if (a > c) {
        delta = a;
        a = c;
        c = delta;
        delta = fa;
        fa = fc;
        fc = delta;
    }

    // If the function at the left endpoint is positive, and the function //
    // at the right endpoint is negative.  Iterate reducing the length    //
    // of the interval by either bisection or quadratic inverse           //
    // interpolation.  Note that the function at the left endpoint        //
    // remains nonnegative and the function at the right endpoint remains //
    // nonpositive.                                                       //

    if (fa > 0.0)
        for (i = 0; i < max_iterations; i++) {

            // Are the two endpoints within the user specified tolerance ?

            if ((c - a) < tolerance)
                return 0.5 * (a + c);

            // No, Continue iteration.

            fb = f(b);

            // Check that we are converging or that we have converged near //
            // the left endpoint of the inverval.  If it appears that the  //
            // interval is not decreasing fast enough, use bisection.      //
            if ((c - a) < tolerance)
                return 0.5 * (a + c);
            if ((b - a) < tolerance) {
                if (fb > 0) {
                    a = b;
                    fa = fb;
                    b = 0.5 * (a + c);
                    continue;
                } else
                    return b;
            }

            // Check that we are converging or that we have converged near //
            // the right endpoint of the inverval.  If it appears that the //
            // interval is not decreasing fast enough, use bisection.      //

            if ((c - b) < tolerance) {
                if (fb < 0) {
                    c = b;
                    fc = fb;
                    b = 0.5 * (a + c);
                    continue;
                } else
                    return b;
            }

            // If quadratic inverse interpolation is feasible, try it. //

            if ((fa > fb) && (fb > fc)) {
                delta = DENOMINATOR(fa, fb, fc);
                if (delta != 0.0) {
                    dab = a - b;
                    dcb = c - b;
                    delta = NUMERATOR(dab,dcb,fa,fb,fc) / delta;

                    // Will the new estimate of the root be within the   //
                    // interval?  If yes, use it and update interval.    //
                    // If no, use the bisection method.                  //

                    if (delta > dab && delta < dcb) {
                        if (fb > 0.0) {
                            a = b;
                            fa = fb;
                        } else if (fb < 0.0) {
                            c = b;
                            fc = fb;
                        } else
                            return b;
                        b += delta;
                        continue;
                    }
                }
            }

            // If not, use the bisection method. //

            fb > 0 ? (a = b, fa = fb) : (c = b, fc = fb);
            b = 0.5 * (a + c);
        }
    else

        // If the function at the left endpoint is negative, and the function //
        // at the right endpoint is positive.  Iterate reducing the length    //
        // of the interval by either bisection or quadratic inverse           //
        // interpolation.  Note that the function at the left endpoint        //
        // remains nonpositive and the function at the right endpoint remains //
        // nonnegative.                                                       //

        for (i = 0; i < max_iterations; i++) {
            if ((c - a) < tolerance)
                return 0.5 * (a + c);
            fb = f(b);

            if ((b - a) < tolerance) {
                if (fb < 0) {
                    a = b;
                    fa = fb;
                    b = 0.5 * (a + c);
                    continue;
                } else
                    return b;
            }

            if ((c - b) < tolerance) {
                if (fb > 0) {
                    c = b;
                    fc = fb;
                    b = 0.5 * (a + c);
                    continue;
                } else
                    return b;
            }

            if ((fa < fb) && (fb < fc)) {
                delta = DENOMINATOR(fa, fb, fc);
                if (delta != 0.0) {
                    dab = a - b;
                    dcb = c - b;
                    delta = NUMERATOR(dab,dcb,fa,fb,fc) / delta;
                    if (delta > dab && delta < dcb) {
                        if (fb < 0.0) {
                            a = b;
                            fa = fb;
                        } else if (fb > 0.0) {
                            c = b;
                            fc = fb;
                        } else
                            return b;
                        b += delta;
                        continue;
                    }
                }
            }
            fb < 0 ? (a = b, fa = fb) : (c = b, fc = fb);
            b = 0.5 * (a + c);
        }
    //err = -2;
    return b;
}

