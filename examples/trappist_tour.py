import numpy as np
from numpy.linalg import norm
from pykep.core import epoch, epoch_from_iso_string, DAY2SEC, EARTH_RADIUS, lambert_problem, propagate_lagrangian, fb_prop, AU, DEG2RAD, ic2par
from pykep.planet import keplerian
from pykep.orbit_plots import plot_planet, plot_lambert, plot_kepler

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


### constants

# Cavendish constant (m^3/s^2/kg)
G = 6.67430E-11
# Sun_mass (kg)
SM = 1.989E30
# Earth mass (kg)
EM = 5.9722E24
# Mass of the Trappist-1 star
MS = 8.98266512E-2 * SM
# Gravitational parameter of the Trappist system
MU_TRAPPIST = G * MS
# Radius of Trappist-1 [m]
RADIUS_TRAPPIST = 83484000.0
# Maximal allowed distance to Trappist-1 [m]
SAFE_RADIUS_TRAPPIST = 10 * RADIUS_TRAPPIST
# A death penalty for infeasible solutions not directly covered by the constraint handling, i.e. singularities
DV_PENALTY = 1e10
# Starting time of mission
T_START = epoch_from_iso_string("20190302T000000")

# Masses and Keplerian elements a,e,i,W,w,M of the various planets
masses = np.array([1.36004499e+00,1.29688971e+00,3.85884170e-01,6.88627613e-01,1.03510927e+00,1.31679654e+00,3.20988718e-01]) * EM
elements = np.array([
[0.01154035,0.0158073,0.02226718,0.02926548,0.0385054,0.04683828,0.06191385],
[0.00455247,0.00107698,0.00624362,0.00577129,0.00861154,0.00400202,0.00378361],
[1.1,1.3,0.5,-0.3,0.01,1.2,2.3],
[238.35940145,221.02196847,90.26523049,164.16001523,171.42668515,166.63349296,282.808809],
[126.22225948,80.78036351,147.44525146,315.09376191,179.82855007,25.31242216,176.19682451],
[33.40998384,-89.79506054,-28.38482427,45.00185452,-145.16404128,12.59883066,-12.12072197]])
elements = np.transpose(elements)
elements[:,0] = elements[:,0] * AU
elements[:,2:] = elements[:,2:] * DEG2RAD

safe_radius = np.array([1.016,1.197,1.978,1.020,1.145,1.229,1.875]) * EARTH_RADIUS

# make the planets
planets = []
names = ["b", "c", "d", "e", "f", "g", "h"]
for i in range(7):
    planets.append(keplerian(T_START, 
    elements[i,:6], MU_TRAPPIST, G*masses[i], EARTH_RADIUS, safe_radius[i],"planet_"+names[i]))

#Colors for plotting
pl2c = {'planet_b':'coral',
        'planet_c':'seagreen',
        'planet_d': 'purple',
        'planet_e': 'steelblue',
        'planet_f': 'firebrick',
        'planet_g': 'gray',
        'planet_h': 'olive' }


class trappist_tour:
    """
    TOF encoded tour, allowing to constrain time for each leg.
    The decision vector contains of: 
    * [u, v, T0] continous part, u and v starting direction, T0 length of first lambert leg
    * for each of the k planetary encounter [beta, rp/rP, eta, T] with beta and rp/rP defining the fly-by, eta the time of the deep space maneuver and T the total duration of the leg. All continuous.
    * [s0, s1, s2, s3, s4, s5, s6] integer part, permutation of [0..6] indicating the visiting sequence

    The initial conditions in starting radius (R_START), velocity magnitude (V_START) and starting epoch (T_START) are
    fixed for this problem (not part of the optimization).
    """
    def __init__(self):
        # there is one leg between each planet plus one additional leg for entering the system
        self.n_legs = len(planets)
        
        # the travel time between planetary encounters can never be shorter than 5 or longer than 2000 days
        self.tof = [[5.0, 2000.0]] * len(planets)
        
        # the initial starting conditions of the spacecraft
        self.R_START = 10 * AU
        self.V_START = 1e4
        self.T_START = epoch_from_iso_string("20190302T000000")
        self.common_mu = MU_TRAPPIST


    def get_nobj(self):
        # Our objectives are to minimize DV and total time of flight
        return 2
        
    def get_nix(self):
        # Integer dimension of the decision vector representing the visiting sequence for the planets
        return 7

    def get_nec(self):
        # sequence of planetary encounters needs to be a permutation
        return 1

    def get_nic(self):
        # checks for keeping a safe distance to the star
        return 1


    def get_bounds(self):
        # I. continuous part (initial leg)
        # we limit v in [0.25, 0.75] which corresponds to a maximum of +/-30 deg Lattitude
        lb = [0.0, 0.25, self.tof[0][0]]
        ub = [1.0, 0.75, self.tof[0][1]]
        
        # II. continuous part (planetary encounters)
        # encoded as blocks of [beta, rp/rP, eta, T]
        for lower_tof, upper_tof in self.tof[1:]:
            lb += [0, 1.1, 1e-3, lower_tof]
            ub += [2 * np.pi, 100.0, 1.0 - 1e-3, upper_tof]
            
        # III. integer part
        lb += [0, 0, 0, 0, 0, 0, 0]
        ub += [6, 6, 6, 6, 6, 6, 6]

        return (lb, ub)
        

    def _periapsis_passed(self, E0, E, dt, period):
        """ Given two anomaly, the time and the period, is the spacecraft passing the periapsis? 
        * E0: starting anomaly         
        * E: final anomaly 
        * dt: time of flight (in days) 
        * period: orbital period (in days) 
        """
        if dt > period:
            return True # over a whole period we pass it for sure
        if E0 > 0:  # spacecraft flying away from body
            return 0 < E < E0
        else:  # flying towards body
            return E > 0 or (dt > period / 2)


    def _check_distance(self, r0, v0, dt, safe_radius=SAFE_RADIUS_TRAPPIST, mu=MU_TRAPPIST):
        """ Computes the periapsis and whether a spacecraft on a certain orbit passed it.
        * r0, v0: initial state of the spacecraft
        * dt: time of flight (in days)
        * safe_radius: minimal allowed distance to central body
        * mu: gravity of central body
        
        returns (True/False, difference between safe_radius and periapsis)
        """
        # get orbital parameters
        a, e, _, _, _, E0 = ic2par(r0, v0, mu)
        if e <= 1.0:
            rp = a * (1 - e)    # circular orbit
        else: 
            rp = -a * (1 - e)   # hyperbolic orbit

        try:
            r, v = propagate_lagrangian(r0, v0, dt * DAY2SEC, mu)
        except RuntimeError:
            print('ERROR: Trajectory infeasible.')
            # if propagation fails, it typically means that the trajectory passes
            # through the star - consequently we return an extremely high penalty
            return (True, 10e16)
            
        # get current orbital period [days]
        period = 2 * np.pi * (a ** 3 / mu) ** .5 / DAY2SEC
            
        # calculate new anomaly
        _, _, _, _, _, E = ic2par(r, v, mu)
            
        return (self._periapsis_passed(E0, E, dt, period), safe_radius - rp)


    # computation of the objective function
    def fitness(self, x, logging=False, plotting=False, ax=None):
        """ Actual computation of the fitness function
            * x is the chromosome to be evaluated
            * logging toggles detailed output about the encoded trajectory
            * plotting, ax: toggle plotting of the trajectory on the corresponding matplotlib axis
        """
        # split chromosome in continous and integer part
        xc, xi = x[:-7], x[-7:]
        
        # decode integer part
        seq = [planets[int(i)] for i in xi]
        
        # check for valid sequences
        eq_constraint = len(set([int(i) for i in xi])) - 7
        
        # decode continuous part
        u, v = xc[:2]
        T = xc[2::4]
        betas = xc[3::4]
        rps = xc[4::4]
        etas = xc[5::4]
        
        # starting point on sphere
        theta = 2 * np.pi * u
        phi = np.arccos(2 * v - 1) - np.pi / 2
        rx = self.R_START * np.cos(phi) * np.cos(theta)
        ry = self.R_START * np.cos(phi) * np.sin(theta)
        rz = self.R_START * np.sin(phi)

        # epochs and ephemerides of the planetary encounters
        t_P = list([None] * (self.n_legs))
        r_P = list([None] * (self.n_legs))
        v_P = list([None] * (self.n_legs))
        lamberts = list([None] * (self.n_legs - 1))
        v_outs = list([None] * (self.n_legs - 1))
        DV = list([0.0] * (self.n_legs - 1))
        
        # violation of distance constraint gets accumulated here
        iq_constraint = -10e16

        # initial starting point
        r_init = [rx, ry, rz]
        
        for i, planet in enumerate(seq):
            t_P[i] = epoch(self.T_START.mjd2000 + sum(T[0:i+1]))
            r_P[i], v_P[i] = seq[i].eph(t_P[i])

        # first leg: pure lambert
        lambert_init = lambert_problem(r_init, r_P[0], T[0] * DAY2SEC, self.common_mu, False, 0)
        
        v_beg_l = lambert_init.get_v1()[0]
        v_end_l = lambert_init.get_v2()[0]
        
        # the first impulse is discounted by the fact that the spacecraft already starts with a velocity
        DV_init = np.abs(norm(v_beg_l) - self.V_START)

        # checking violation of constraint after first DSM
        close_encounter, d = self._check_distance(r_init, v_beg_l, T[0])
        if close_encounter:
            iq_constraint = max(iq_constraint, d)

        # successive legs
        for i in range(0, self.n_legs - 1):
            # Fly-by
            v_outs[i] = fb_prop(v_end_l, v_P[i], rps[i] * seq[i].radius, betas[i], seq[i].mu_self)
            
            # updating inequality constraint if necessary 
            close_encounter, d = self._check_distance(r_P[i], v_outs[i], etas[i] * T[i+1])
            if close_encounter:
                iq_constraint = max(iq_constraint, d)            

            # s/c propagation before the DSM
            r, v = propagate_lagrangian(r_P[i], v_outs[i], etas[i] * T[i+1] * DAY2SEC, self.common_mu)
            # Lambert arc to reach next body
            dt = (1 - etas[i]) * T[i+1] * DAY2SEC
            lamberts[i] = lambert_problem(r, r_P[i+1], dt, self.common_mu, False, 0)
                        
            v_end_l = lamberts[i].get_v2()[0]
            v_beg_l = lamberts[i].get_v1()[0]
            
            # DSM occuring at time eta_i*T_i
            if np.isnan(v_beg_l[0]):
                # in rare occassions, the lambert problem is singular or results in unreasonably
                # high velocities. We apply a death penalty to the solution in this case
                print('WARNING: death penalty applied')
                DV[i] = DV_PENALTY
                return (DV_init, DV, T, lamberts, eq_constraint, iq_constraint)

            # updating inequality constraint if necessary 
            close_encounter, d = self._check_distance(r, v_beg_l, (1 - etas[i]) * T[i+1])
            if close_encounter:
                iq_constraint = max(iq_constraint, d)   
                
            DV[i] += norm([a - b for a, b in zip(v_beg_l, v)])

        # pretty printing
        if logging:
            print(f"== 1 : starting point -> {seq[0].name}")
            print(f"Duration: {T[0]:0.6f}d")
            print(f"DV: {DV_init:0.6f}m/s\n")

            for i in range(self.n_legs - 1):
                print(f"== {i+2} : {seq[i].name} -> {seq[i+1].name} ==")
                print(f"Duration: {T[i+1]:0.6f}d")
                print(f"Fly-by epoch: {t_P[i].mjd2000:0.6f} mjd2000")
                print(f"Fly-by radius: {rps[i]:0.6f} planetary radii")
                print(f"DSM after {etas[i] * T[i+1]:0.6f}d")
                print(f"DSM magnitude: {DV[i]:0.6f}m/s\n")

            print(f"Total Delta V: {sum(DV) + DV_init:0.3f}m/s")
            print(f"Total mission time: {sum(T):0.6f}d ({sum(T) / 365.25:0.3f} years)")
            

        # plotting
        if plotting:
            ax.scatter(0, 0, 0, color='chocolate')
            for i, planet in enumerate(seq):
                plot_planet(planet, t0=t_P[i], color=pl2c[planet.name], legend=False, units=AU, axes=ax)
            
            # leg for entering the system
            plot_lambert(lambert_init, sol=0, color='g', legend=False, units=AU, N=10000, axes=ax)
            
            # intersystem legs
            for i in range(self.n_legs - 1):
                plot_kepler(r_P[i], v_outs[i], etas[i] * T[i+1] * DAY2SEC, self.common_mu, N=5000, color=(0, 0, 1.0), label=False, units=AU, axes=ax)
            for l in lamberts:
                plot_lambert(l, sol=0, color='r', legend=False, units=AU, N=5000, axes=ax)
            
            # some settings for better viewing
            zoom = 0.15
            ax.set_xlim(-zoom,zoom)
            ax.set_ylim(-zoom,zoom)
            ax.set_zlim(-zoom,zoom) 
            ax.margins(x=0)
            ax.view_init(elev=90, azim=90)
            ax.grid(False)
            ax.axis(False)            

        # return objectives and constraint violations
        return (DV_init + sum(DV), sum(T), eq_constraint, iq_constraint)


    def pretty(self, x):
        """ Prints out details about the encoded trajectory """
        _ = self.fitness(x, logging=True)


    def plot(self, x, ax=None):
        """ Plots the encoded trajectory in 3d. If no existing matplotlib axis is provided, a new figure is generated """
        if ax is None:
            fig = plt.figure()
            axis = fig.gca(projection='3d')
        else:
            axis = ax

        _ = self.fitness(x, logging=False, plotting=True, ax=axis)
        return axis
        
        
    def example(self):
        """ Returns an example solution. """
        return [0.829814333836995, 0.49439092971262544, 1501.6919241751443, 
                1.5873030906083387, 1.512021386175137, 0.23024558713584858, 
                255.87994897371408,   4.161192563429589,  8.278275659200725, 0.10265621329161562,
                318.6140876998619, 4.546432641418882, 4.078585717496654, 0.33605586368271645,
                374.6287841767241, 0.9461693914845313, 19.13985248269904, 0.4834964750829982,
                343.8204904698013, 2.1935168149524964, 32.48727508444479, 0.40957309270588427,
                325.1537638293067, 4.743610880353654, 49.95062395827792, 0.3668039400426466,
                350.0377039343523, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def combine_scores(points):
    """ Function for aggregating single solutions into one score using hypervolume indicator """
    import pygmo as pg
    ref_point = np.array([2500, 4000])
    
    # solutions that not dominate the reference point are excluded
    filtered_points = [s for s in points if pg.pareto_dominance(s, ref_point)]
    
    if len(filtered_points) == 0:
        return 0.0
    else:
        hv = pg.hypervolume(filtered_points)
        return -hv.compute(ref_point)

# Optimize udp        
udp = trappist_tour()

import sys
import matplotlib as mpl
import matplotlib.pyplot as plt

def checkb(): 
    xs0 = [
        #[0.8103052143514196, 0.4958438921601793, 1391.3751923384077, 1.5705031061326133, 1.10152681364635, 0.0016095258443698383, 82.18405609752253, 3.4828254359241826, 5.805283740928708, 0.12951302122350888, 87.2679747681332, 1.3094355042435961, 1.3332858776534566, 0.12277759964845775, 53.86655101466599, 3.2547267548850134, 63.96710375276724, 0.20216877217210844, 53.92350060574348, 2.1786978596502466, 3.522291383513096, 0.41636247760550216, 49.81026670423341, 4.745153497778955, 1.1017651872095273, 0.3760793684039778, 58.755535060506936],
        [0.6648787750928142, 0.5029037392083677, 1501.4596305877558, 1.5638313711592389, 1.1012539261497651, 0.041914796416672714, 131.60174019793945, 2.745449244610581, 89.25937663913041, 0.5623250537530655, 132.44893337730156, 3.0812966442142518, 72.65894552647971, 0.48335860579653267, 130.33731726802853, 1.7698985581347955, 6.615275378212581, 0.2646730965238785, 113.52378795452492, 1.0535821296178647, 1.7892851065678994, 0.29640801240898296, 89.09034624380266, 4.2376108038408855, 16.677264725681784, 0.4983360800786258, 88.98946998703931],
        #[0.20018375867543092, 0.5023851451731149, 1458.4595090832045, 1.5429744503569454, 1.1000318280176042, 0.0011616454056745888, 112.9406012197096, 4.869407423298176, 6.643882058409057, 0.0837653720395933, 142.57763714820246, 3.1327567353108368, 9.88979652601552, 0.44780726253537506, 143.7247945126893, 2.6886136136130308, 62.49103201344211, 0.29945294667575206, 142.7797515073766, 2.7712720612074353, 5.667799121188566, 0.31070075656754437, 136.62832544498468, 3.5844213146927144, 52.78193949547493, 0.24297164140851896, 135.5425455208572],
        #[0.7285732324089172, 0.4985790010166462, 1500.407210285844, 1.5419267996987513, 1.1276068595511894, 0.47855871782031056, 148.34188831226726, 2.109891384427789, 14.713583487904497, 0.4629376418817439, 133.19940792979685, 6.161173286619677, 1.958879664099828, 0.49417205648715334, 144.89303037037965, 3.1204215811584057, 2.464758495152999, 0.36344185224770154, 143.813623126781, 1.4943916370358166, 39.821645830832374, 0.13502492256118542, 141.34646616034348, 4.912022439874159, 10.76002054102991, 0.6441016702734828, 145.23608745913216],
        #[0.20340652551204996, 0.5056853274356777, 1500.770701362955, 1.1850607484071398, 1.1181188244945226, 0.08244776727426395, 162.66980172211422, 0.013505253484817015, 61.838084746956426, 0.10413945127915052, 161.78145518881792, 3.097135533366906, 5.487704166956004, 0.45938759513212535, 161.436157709332, 2.408776064568739, 92.04719806869508, 0.2834835571120545, 160.27426560738198, 3.178427612276295, 53.1738153817593, 0.36568693438486166, 160.25545002231294, 1.3017811946948619, 12.957521893838916, 0.2714354334018979, 153.30459230166628],
        # [0.49502862846255713, 0.5048111452754893, 1499.6848094460584, 1.5513269137642092, 1.103285246112547, 0.47226712054105574, 136.41535700876074, 5.840188039370229, 36.26873214297542, 0.5232420780296914, 139.70305916521465, 4.071039061556977, 3.054515353606252, 0.48767006433058757, 192.10646154293636, 4.399537463983878, 46.88026328675744, 0.60865336264806, 196.72954963694443, 2.3916830724541605, 49.33956745922373, 0.39976170143018824, 194.49158661684288, 3.1571846808944635, 69.21780860312921, 0.6260233401945711, 196.05919755859702],
        [0.23727350647397294, 0.49979314550316456, 1499.735184928535, 1.8993953961558216, 1.1229778567220705, 0.32726976313094636, 196.3260416419087, 0.571808313061152, 51.176370287812844, 0.31896331938984324, 190.95009750381843, 3.121433997583582, 5.05687604396001, 0.43277720393319696, 191.7178752795583, 2.885925716578056, 32.834255791631165, 0.3668091309488611, 190.18734329493995, 3.3807710867067837, 17.490656609074144, 0.4918378988267123, 192.085850257355, 2.5448983752032657, 15.408726945793642, 0.4560054008228409, 187.31446229550676],
        # [0.6955541883724188, 0.5019039870064873, 1500.0458430413805, 1.08948092114163, 1.19337938680055, 0.26972584421402324, 217.88903529847704, 1.841433354770671, 27.098061650345798, 0.47349578849668617, 199.84991356699652, 6.234259153529268, 1.5588130735140142, 0.48191144044166456, 211.59538781678563, 3.1279474135224383, 1.4359649586475074, 0.4715774241024832, 209.70132033840684, 2.9859128716475927, 45.42378803285263, 0.3959608297493462, 208.99067444072142, 1.734268145317678, 27.649751699272286, 0.11800307997611785, 203.30268764365508],
        # [0.6863421051215592, 0.4990808730170854, 1502.9673386010597, 1.396815590325504, 1.3872488437578252, 0.09327854021725566, 224.56534586450576, 3.8297309946924756, 97.41060278426741, 0.3026968069517748, 227.06605840502309, 6.23577534043258, 10.651812232902692, 0.4940935672897167, 229.82930296621316, 2.5014221538956583, 81.47996005058462, 0.3065975520996918, 227.71300459513756, 1.689869883125991, 73.05349057830941, 0.36382071061961685, 224.89368637714517, 5.031089036304736, 36.19704165520255, 0.2986895316180104, 226.64158294515127],
        #[0.2084713078236328, 0.49980993436887833, 1500.7691912090618, 1.815225303779698, 1.456715522299409, 0.2665239350566243, 249.8563369227068, 0.17933267297170866, 77.97836323725956, 0.29294056615451447, 247.56706610515369, 3.2752020600112215, 8.538968195138164, 0.3186725317677196, 255.5044115577448, 1.9622219682355178, 37.386340545977504, 0.3185099880458807, 247.08369224799097, 2.3872338612207007, 50.37423373477901, 0.3986083237063697, 243.69190186205932, 2.9631226019881574, 61.84989651919031, 0.21643351626145135, 241.71079764526877],
        ]
    seq = [0, 1, 5, 4, 3, 2, 6]
    xs = [np.array(x + seq) for x in xs0]
    ys = []
    for x in xs:
        #ts = (x[:-7])[2::4]
        t = sum(x[2::4])
        #x0 = x[-7:] = np.argsort(x[-7:])
        mpl.rcParams["legend.fontsize"] = 10
        fig = plt.figure()
        axes = fig.gca(projection="3d")

        y = udp.fitness(x, plotting=True, ax=axes)
        plt.show()
        udp.pretty(x)
        sys.exit()
        print(y, t)
        ys.append([y[0],y[1]])
        #print(t)
    print(combine_scores(ys))
    
if __name__ == '__main__':
    checkb()




