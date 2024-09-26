# See original code at
# https://optimize.esa.int/challenge/spoc-2-quantum-communications-constellations/About
# https://optimize.esa.int/challenge/spoc-2-quantum-communications-constellations/p/quantum-communications-constellations
# 
# Changes: 
# - Factor 30 speedup using numba and igraph
# - Added competitive algorithms
# - See corresponding tutorial 
#   https://github.com/dietmarwo/fast-cma-es/blob/master/tutorials/ESAChallenge.adoc

# Requires pykep which needs python 3.8, 
# Create an python 3.8 environment:

# mamba create -n env38 python=3.8
# conda activate env38

# Install dependencies:

# mamba install pykep
# mamba install pygmo
# pip install networkx
# pip install sgp4
# pip install seaborn
# pip install matplotlib
# pip install igraph
# pip install pymoo

# Tested using https://docs.conda.io/en/main/miniconda.html on Linux Mint 21.2

# Basic imports
import pykep as pk
import pygmo as pg
import numpy as np
import scipy
import os, time
from matplotlib import pyplot as plt

import sys 
from loguru import logger

logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss.SS} | {process} | {level} | {message}", level="INFO")
logger.add("log_{time}.txt", format="{time:HH:mm:ss.SS} | {process} | {level} | {message}", level="INFO")

import seaborn as sns
# SGP4 - we use SPG4 to propagate orbits around New Mars as a proxy
# for a plausible orbital positions propagator around a habitable planet
from sgp4.api import Satrec, SatrecArray
from sgp4.api import WGS72

# Networkx
import igraph as ig # for speed
import networkx as nx

# Static data
def get_mothership_satellites():
    """Construct array of mothership orbital elements
    (the TLEs of actual Earth-orbiting satellites below are used as a proxy for
    plausible orbital dynamics around a habitable planet)
    """
    mothership_tles = [
        [
            "1 39634U 14016A   22349.82483685  .00000056  00000-0  21508-4 0  9992",
            "2 39634  98.1813 354.7934 0001199  83.3324 276.7993 14.59201191463475"
        ],
        [
            "1 26400U 00037A   00208.84261022 +.00077745 +00000-0 +00000-0 0  9995",
            "2 26400 051.5790 297.6001 0012791 171.3037 188.7763 15.69818870002328"
        ],
        [
            "1 36508U 10013A   22349.92638064  .00000262  00000-0  64328-4 0  9992",
            "2 36508  92.0240 328.0627 0004726  21.3451 338.7953 14.51905975672463"
        ],
        [
            "1 40128U 14050A   22349.31276420 -.00000077  00000-0  00000-0 0  9995",
            "2 40128  50.1564 325.0733 1614819 130.5958 244.6527  1.85519534 54574"
        ],
        [
            "1 49810U 21116B   23065.71091236 -.00000083  00000+0  00000+0 0  9998",
            "2 49810  57.2480  13.9949 0001242 301.4399 239.8890  1.70475839  7777"
        ],
        [
            "1 44878U 19092F   22349.75758852  .00015493  00000-0  00000-0 0  9998",
            "2 44878  97.4767 172.6133 0012815  68.6990 291.5614 15.23910904165768"
        ],
        [
            "1 04382U 70034A   22349.88472104  .00001138  00000-0  18306-3 0  9999",
            "2 04382  68.4200 140.9159 1043234  48.2283 320.3286 13.08911192477908"
        ]
    ]

    # Assembling the list of Satrec motherships
    motherships = []
    for tle in mothership_tles:
        motherships.append(Satrec.twoline2rv(tle[0], tle[1]))
    return motherships

from numba import njit

@njit(fastmath=True)
def line_of_sight(r1,r2):
    """Given two position vectors returns the distance of the line of sight to the origin

    Args:
        r1 (numpy array): first point
        r2 (numpy array): second point
    """
    denom = np.linalg.norm(r2-r1)
    if denom < 1e-6:
        # if r1 ~= r2, it will return the norm of r1
        return np.linalg.norm(r1)
    else:
        r21 = (r2-r1) / denom
        h1 = np.dot(r1,r21)
        arg = np.linalg.norm(r1)**2 - h1**2
        # We check for a positive arg in case r1 and r2 are near collinearity
        return np.sqrt(arg) if arg > 1e-6 else 0.0

@njit(fastmath=True)
def zenith_angle(src, dst):
    """Computes the cosine of the zenith angle (theta_z) of the LOS between source and destination node
    
    Args:
        src (numpy array, N_r x 3): rover x, y, z positions
        dst (numpy array, N_s x 3): mothership x, y, z positions
    
    Returns:
        float: cosine of the zenith angle
    """
    dpos = dst - src
    if np.linalg.norm(dpos) < 1e-6:
        cos_theta_z = 0
    else:
        cos_theta_z = np.dot(dpos, src) / (np.linalg.norm(dpos) * np.linalg.norm(src))
    return cos_theta_z

@njit(fastmath=True)
def qkd_metric(idx, src, dst, cos_theta_z, eta, eps_z, n_rovers):
    """Computes the edge weight according to QKD probabilities
    
        Args:
            idx (int): index of node in the graph
            src (numpy array, 1x3): position of source node
            dst (numpy array, 1x3): position of destination node
            cos_theta_z (float): cosine of the zenith angle of qkd link
            eta (int): satellite quality indicator for corresponding constellation
    
        Returns:
            float: edge weight
            float: communications link distance between src and dst
    """
    edge_weight = -np.log(eta) # constellation quality score
    d_link = np.linalg.norm(src - dst) # distance of communications link
    edge_weight += 2 * np.log(d_link) # final edge weight
    if edge_weight < 0:
        # Safeguard: whenever this happens, the collision-avoidance constraint is
        # also not satisfied. Nevertheless, we must return a value for the edge weight
        # to ensure that the fitness does not throw (a negative valued edge would also
        # imply the non-existence of a shortest path)
        edge_weight = 1e3
    
    if idx <= n_rovers:
        if cos_theta_z >= eps_z: # Apply max zenith angle constraint to mothership-rover link
            edge_weight += 1.0 / np.sin(np.pi / 2 - np.arccos(cos_theta_z))
        else:
            edge_weight = 0
    return edge_weight, d_link

@njit(fastmath=True)
def get_adjmatrix(pos, ep_idx, eta, num_w1_sats, LOS, N, eps_z, n_rovers):
    adjmatrix = np.zeros((N, N))
    d_min = np.inf
    for i in range(N):
        for j in range(i):
            # Ensure there is LOS
            los = line_of_sight(pos[i, ep_idx, :], pos[j, ep_idx, :])
            cos_theta_z = zenith_angle(pos[i, ep_idx, :], pos[j, ep_idx, :])
            if los >= LOS or cos_theta_z > 0:
                # Eta based on j because it is the destination satellite in the link
                eta_j = eta[0] if j < num_w1_sats else eta[1]
                adjmatrix[i,j], d_link = \
                    qkd_metric(N-i, pos[i, ep_idx, :], pos[j, ep_idx, :], cos_theta_z, eta_j, eps_z, n_rovers)
                if d_link < d_min:
                    d_min = d_link
                adjmatrix[j,i] = adjmatrix[i,j]
    return adjmatrix, d_min

@njit(fastmath=True)
def construct_rover_pos(lambda0, phi0, n_rovers, jds, R_p, w_p):
    """Computes the position of the rovers at time t based on the initial latitude and longitude
    
    Args:
        lambda0 (float, N_r x 1): initial latitudes of the rovers
        phi0 (float, N_r x 1): initial longitudes of the rovers

    Returns:
        float, n_rovers x n_epochs x 3: rover x, y, z positions
    """
    pos_r = np.zeros((n_rovers, jds.shape[0], 3))
    time_range = (jds - jds[0]) * 24 * 3600 # in seconds
    for i, t in enumerate(time_range):
        pos_r[:, i, 0] = R_p * np.cos(lambda0) * np.cos(phi0 + w_p * t) # x
        pos_r[:, i, 1] = R_p * np.cos(lambda0) * np.sin(phi0 + w_p * t) # y
        pos_r[:, i, 2] = R_p * np.sin(lambda0) # z
    return pos_r

class constellation_udp:
    """A Pygmo compatible UDP (User Defined Problem) representing the constellation design problem for SpOC 2023.

    Two Walker constellations are defined in a mixint chromosome:
        x = [a1,ei,i1,w1,eta1] + [a2,e2,i2,w2,eta2] + [S1,P1,F1] + [S2,P2,F2] + [r1,r2,r3,r4]

    The constellations must relay information between 7 motherships in orbit and 4 rovers on the surface of New Mars
    """
    def __init__(self):
        """Constructor"""

        # Define the time grid over which to optimize the communications network
        self._t0 = 10000 # starting epoch in mjd2000
        self.n_epochs = 11 # number of epochs in time grid to consider
        self._duration = 10 # difference between the first and last epoch considered in years
        jd0, fr = pk.epoch(self._t0, 'mjd2000').jd, 0.0 # reference Julian date
        self.jds = np.linspace(jd0, jd0 + self._duration * 365.25, self.n_epochs) # time grid in JD
        self.frs = self.jds * fr # date fractions (defaults to 0)

        # Reference epoch for SGP4 is 1949 December 31 00:00 UT
        self.ep_ref = pk.epoch_from_iso_string("19491231T000000")

        # SGP4-ready mothership satellites
        mothership_satellites = get_mothership_satellites()
        self.pos_m = self.construct_mothership_pos(SatrecArray(mothership_satellites))
        self.n_motherships = len(mothership_satellites)

        # Latitudes and longitudes of rovers
        rovers_db_path = os.path.join(".", "data", "spoc2", "constellations", "rovers.txt")
        self.rovers_db = np.loadtxt(rovers_db_path)
        self.lambdas = self.rovers_db[:, 0] # latitudes
        self.phis = self.rovers_db[:, 1] # longitudes
        self._min_rover_dist = 3000 # minimum inter-rover distance (km)
        self.n_rovers = 4

        # Minimum line-of-sight parameter (km)
        # Radius amplification factor: 1.05
        self.LOS = 1.05 * pk.EARTH_RADIUS / 1000.0
        # Radius of the New-Mars planet (km)
        self.R_p = pk.EARTH_RADIUS / 1000.0
        # Angular velocity of New Mars (rad/s)
        self.w_p = 7.29e-5 # 2 * pi / (23 hours 56 minutes 4 seconds)
        # Threshold zenith angle constraint for rover-sat link (rad)
        self._zenith_angle = np.pi / 3
        self.eps_z = np.cos(self._zenith_angle)
        # Minimum inter-satellite distance (km)
        self._min_sat_dist = 50

    def get_bounds(self):
        """Get bounds for the decision variables.

        Returns:
            Tuple of lists: bounds for the decision variables.
        """
        lb = [1.06, 0., 0., 0., 1.0] + [2.0, 0., 0., 0., 1.0] + [4, 2, 0] + [4, 2, 0] + [0, 0, 0, 0]
        #ub = [1.8, 0.02, np.pi, 2*np.pi, 1000.0] + [3.5, 0.1, np.pi, 2*np.pi, 1000.0] + [10, 10, 9] + [10, 10, 9] + [99, 99, 99, 99]
        # we adapt the boundaries so that they work with a continuous optimizer
        ub = [1.8, 0.02, np.pi, 2*np.pi, 1000.0] + [3.5, 0.1, np.pi, 2*np.pi, 1000.0] + \
                [10.9999, 10.9999, 9.9999] + [10.9999, 10.9999, 9.9999] + [99.9999, 99.9999, 99.9999, 99.9999]
        return (lb, ub)
    
    def get_ints(self):
        return np.array([False]*10 + [True]*10)

    def get_nix(self):
        """Get number of integer variables.

        Returns:
            int: number of integer variables.
        """
        return 6 + 4

    def get_nobj(self):
        """Get number of objectives.

        Returns:
            int: the number of objectives
        """
        return 2
    
    def get_nic(self):
        """Get number of inequality constraints.

        Returns:
            int: the number of constraints
        """
        return 2
    
    def get_rover_constraint(self, lambda0, phi0):
        """Evaluate the rover constraint (minimum distance between any two rovers)

        Args:
            lambda0 (float, N_r x 1): latitudes of the rovers
            phi0 (float, N_r x 1): longitudes of the rovers

        Returns
            float: the difference between the actual and allowable minimum distance between rovers
        """
        # Compute rover positions on the planet
        pos = np.zeros((self.n_rovers, 3))
        pos[:, 0] = np.sin(lambda0) * np.cos(phi0)
        pos[:, 1] = np.cos(lambda0) * np.cos(phi0)
        pos[:, 2] = np.sin(phi0)
        def safe_arccos(u, v):
            inner_product = np.dot(u, v)
            if inner_product > 1:
                return 0
            if inner_product < -1:
                return np.pi
            return np.arccos(inner_product)
        d = scipy.spatial.distance.cdist(pos, pos, lambda u, v: pk.EARTH_RADIUS/1000 * safe_arccos(u, v))
        d = d + np.diag([np.inf]*4)
        min_d = np.min(d)
        # Will be negative if min(d) is larger than the min allowable inter-rover distance
        return self._min_rover_dist - min_d, min_d
    
    def get_sat_constraint(self, d_min):
        """Evaluate the satellite constraint (minimum distance between any two satellites)

        Args:
            d_min (float): the minimum distance between any two satellites at any epoch

        Returns:
            float: the difference between the actual and allowable minimum distance between satellites
        """
        # Will be negative if d_min is larger than the min allowable inter-satellite distance
        return self._min_sat_dist - d_min


    def generate_walker(self, S,P,F,a,e,incl,w,t0):
        """Generates a Walker constallation as a SatrecArray

        Args:
            S (int): number of satellites per plane
            P (int): number of planes        
            F (int): spacing parameter (i.e. if 2 phasing repeats each 2 planes)
            a (float): semi-major axis
            e (float): eccentricity
            incl (float): inclination
            w (float): argument of perigee
            t0 (float): epoch

        Returns:
            SatrecArray: satellites ready to be SGP4 propagated
        """
        walker_l = []
        mean_motion = np.sqrt(pk.MU_EARTH/a**3/pk.EARTH_RADIUS**3)
        # planes
        for i in range(P):
            #satellites
            for j in range(S):
                satellite = Satrec()
                satellite.sgp4init(
                    WGS72,                            # gravity model
                    'i',                              # 'a' = old AFSPC mode, 'i' = improved mode
                    j + i*S,                          # satnum: Satellite number
                    t0-self.ep_ref.mjd2000,           # epoch: days since 1949 December 31 00:00 UT
                    0.0,                              # bstar: drag coefficient (1/earth radii) - 3.8792e-05
                    0.0,                              # ndot: ballistic coefficient (revs/day)
                    0.0,                               # nddot: mean motion 2nd derivative (revs/day^3)
                    e,                                # ecco: eccentricity
                    w,                                # argpo: argument of perigee (radians)
                    incl,                             # inclo: inclination (radians)
                    2*np.pi/P/S*F*i+2.*np.pi/S*j,     # mo: mean anomaly (radians)
                    mean_motion*60,                   # no_kozai: mean motion (radians/minute)
                    2.*np.pi/P*i                      # nodeo: R.A. of ascending node (radians)
                )
                walker_l.append(satellite)
        # Creating the vectorized list
        return SatrecArray(walker_l)
    
    def build_graph(self, ep_idx, pos, num_w1_sats, eta):
        """Builds a networkx graph from the satellite positions. Links are weighted via a "QKD-inspired metric
        and only exist when motherships/constellation satellites/rovers have line-of-sight

        Args:
            ep_idx (int): idx of the epoch in the time grid 
            pos (numpy array 3xN): position vector of the satellites
            num_w1_sats (int): number of satellites in the first Walker constellation
            eta (tuple): satellite quality indicator for each Walker constellation

        Returns:
            igraph graph: nodes are motherships/Walker satellites/rovers; links are distances when there is LOS
        """
        N = pos[:, ep_idx, :].shape[0] # number of vertices
        adjmatrix, d_min = get_adjmatrix(pos, ep_idx, eta, num_w1_sats, self.LOS, N, self.eps_z, self.n_rovers)
        g = ig.Graph.Adjacency((adjmatrix > 0).tolist())
        g.es['weight'] = adjmatrix[adjmatrix.nonzero()]
        return g, adjmatrix, d_min

    def average_shortest_path(self, G, src, dst):
        """Computes the average shortest path length between the source and destination *partitions* of nodes in the graph *G*
        (the source is assumed to be the motherships and the destination the rovers)

        Args:
            G (networkx graph): The graph
            src (int): the number of motherships (to be used as a negative index in G)
            dst (int): the number of rovers (to be used as a negative index in G)

        Returns:
            float: average shortest path
        """
        n_nodes = G.vcount()
        src0 = n_nodes - src - dst
        dst0 = n_nodes - dst
        sp = np.array(G.distances(list(range(src0, src0+src)), \
                                  list(range(dst0, dst0+dst)), weights=G.es["weight"]))
        return np.mean(sp)

    def construct_walkers(self, x):
        """Generates two Walker constellations according to specifications
        
        Args:
            x (list): chromosome describing the New Mars communications infrastructure

        Returns:
            SatrecArray: Walker1 constellation satellites ready to be SGP4 propagated
            SatrecArray: Walker2 constellation satellites ready to be SGP4 propagated
        """
        # Parse the chromosome
        a1,e1,i1,w1,_,a2,e2,i2,w2,_,S1,P1,F1,S2,P2,F2,_,_,_,_ = x
        # Construct the 1st walker constellation as a SatrecArray
        walker1 = self.generate_walker(int(S1),int(P1),int(F1),a1,e1,i1,w1,self._t0)
        # Construct the 2nd walker constellation as a SatrecArray
        walker2 = self.generate_walker(int(S2),int(P2),int(F2),a2,e2,i2,w2,self._t0)
        return walker1, walker2

    def construct_mothership_pos(self, motherships):
        """Computes the position of the motherships over a predefined time grid
        
        Args:
            motherships (SatrecArray): mothership satellites ready to be SGP4 propagated

        Returns:
            float, n_motherships x n_epochs x 3: mothership x, y, z positions
        """

        err, pos, _ = motherships.sgp4(self.jds, self.frs)
        # Check propagation went well
        if not np.all(err == 0):
            raise ValueError("The motherships cannot be propagated succesfully on the defined time grid")
        return pos

    def construct_pos(self, walker1, walker2, pos_r):
        """Construct cumulative position of Walker satellites, motherships and rovers

        Args:
            walker1 (SatrecArray): Walker1 constellation satellites ready to be SGP4 propagated
            walker2 (SatrecArray): Walker2 constellation satellites ready to be SGP4 propagated
            pos_r (float, n_rovers x n_epochs x 3): rover x, y, z positions

        Returns:
            float, (S1xP1 + S2xP2 + n_motherships + n_rovers) x n_epochs x 3: overall position vector
        """
        # Compute ephemerides for Walker1 satellites at all epochs)
        err_w1, pos_w1, _ = walker1.sgp4(self.jds, self.frs)
        # Compute ephemerides for Walker2 satellites at all epochs)
        err_w2, pos_w2, _ = walker2.sgp4(self.jds, self.frs)
        # Check propagation went well
        if not (np.all(err_w1 == 0) and np.all(err_w2 == 0)):
            raise ValueError("The walker constellations cannot be propagated successfully on the defined time grid")
        # Position vector for Walker constellation satellites, motherships and rovers)
        cum_pos = np.concatenate((pos_w1,pos_w2, self.pos_m, pos_r))
        return cum_pos

    def fitness(self, x, verbose=False):
        """Evaluate the fitness of the decision variables.

        Args:
            x (list): chromosome describing the New Mars communications infrastructure
            verbose (bool): If True, print some info.

        Returns:
            float: fitness for average shortest path
            float: fitness for total number of satellites
            float: constraint for rover positioning
        """
        # Construct the Walker constellations based on input chromosome 
        walker1, walker2 = self.construct_walkers(x)
        # Extract the quality factors and the number of satellites in the Walkers
        _,_,_,_,eta1,_,_,_,_,eta2,S1,P1,_,S2,P2,_,_,_,_,_ = x
        N1 = S1 * P1
        N2 = S2 * P2
        # Extract the rover indices from the input chromosome
        rovers_idx = np.array(x[-4:]).astype(int)
        # Look up latitude and longitudes corresponding to rover indices
        lambda0 = self.lambdas[rovers_idx]
        phi0 = self.phis[rovers_idx]
        # Construct the rover positions
        rovers = construct_rover_pos(lambda0, phi0, self.n_rovers, self.jds, self.R_p, self.w_p)
        # Concatenate the position of the Walkers, motherships and rover
        cum_pos = self.construct_pos(walker1, walker2, rovers)

        # Evaluating the fitness function
        if verbose:
            print("FITNESS EVALUATION:")

        # First objective (minimize):
        # Compute the average shortest path between any mothership-rover pair
        # Iterate over epochs
        f1 = 0
        nf1 = 34 # f1 normalization factor
        d_sat_min_ep = np.inf
        for ep_idx in range(1, self.n_epochs):
            # Constructs the graph:
            # Nodes: Walker sats + motherships + rovers
            # Edges: LOS communication
            G, _, d_sat_min = self.build_graph(ep_idx, cum_pos, N1, (eta1, eta2))
            if d_sat_min < d_sat_min_ep:
                d_sat_min_ep = d_sat_min
            f1 += self.average_shortest_path(G, self.n_motherships, self.n_rovers)

        # Average over the number of epochs
        f1 /= (self.n_epochs - 1)

        # Second objective (minimize):
        # Compute the total number of satellites (weighted by the quality factors)
        f2 = eta1 * N1 + eta2 * N2
        nf2 = 100000 # f2 normalization factor

        # Constraints:
        # The minimum distance between any two rovers needs to be at least 3000km
        # to ensure good coverage of the surface of New Mars
        min_rover_d, d_rover_min = self.get_rover_constraint(lambda0, phi0)
        # The minimum distance between any two nodes of the graph across all epochs 
        # needs to be at least 50km to ensure a collision-free communications network
        min_sat_d = self.get_sat_constraint(d_sat_min_ep) 

        # Additional information on the fitness of the input chromosome
        if verbose:
            print(100 * "-")
            print("RESULTS:")
            print("Total number of satellites (W1: {}, W2: {}): {}".format(N1, N2, N1+N2))
            print("OBJECTIVE #1 - Average communications cost: {}".format(f1/nf1))
            print("OBJECTIVE #2 - Cost of infrastructure: {}".format(f2/nf2))
            print("CONSTRAINT - Minimum distance between rovers ({}): {} km".format("NOK" if min_rover_d > 0 else "OK", d_rover_min))
            print("CONSTRAINT - Minimum distance between sats ({}): {} km".format("NOK" if min_sat_d > 0 else "OK", d_sat_min_ep))
            print(100 * "-")
        return [f1/nf1, f2/nf2, min_rover_d, min_sat_d]
    
    def pretty(self, x):
        """A verbose evaluation of the fitness functions

        Args:
            x (list): chromosome describing the New Mars communications infrastructure

        Returns:
            float: fitness for average shortest path
            float: fitness for total number of satellites
            float: constraint for rover positioning
            float: constraint for satellite positioning
        """
        f1, f2, c1, c2 = self.fitness(x, verbose=True)
        return f1, f2, c1, c2

    def example(self, verbose=False):
        """A random chromosome example for the constellation optimization

        Returns:
            list: a valid chromosome representing a possible constellation design
        """
        if verbose:
            print("CHROMOSOME:")
            print("x = [a1, e1, i1, w1, eta1] + [a2, e2, i2, w2, eta2] + [S1, P1, F1] + [S2, P2, F2] + [r1, r2, r3, r4]")
            print(100 * "-")
            print("a1: float representing the normalized semi-major axis of Walker1 satellite orbits (in km) [1.05,1.8]")
            print("e1: float representing the eccentricity [0, 0.1]")
            print("i1: float representing the inclination [0, pi]")
            print("w1: float representing the argument of the perigee [0, 2pi]")
            print("eta1: float defined as the quality indicator of satellites in the first walker constellation [0, 100]")
            print(100 * "-")
            print("a2: float representing the normalized semi-major axis of Walker2 satellite orbits (in km) [2.0,3.5]")
            print("e2: float representing the eccentricity [0, 0.1]")
            print("i2: float representing the inclination [0, pi]")
            print("w2: float representing the argument of the perigee [0, 2pi]")
            print("eta2: float defined as the quality indicator of satellites in the first walker constellation [0, 100]")
            print(100 * "-")
            print("S1: integer corresponding to the number of satellites per plane [4, 10]")
            print("P1: integer corresponding to the number of planes [2, 10]")
            print("F1: integer defining the phasing of the constellation [0, 9]")
            print(100 * "-")
            print("S2: integer corresponding to the number of satellites per plane [4, 10]")
            print("P2: integer corresponding to the number of planes [2, 10]")
            print("F2: integer defining the phasing of the constellation [0, 9]")
            print(100 * "-")
            print("r1: index of rover 1 [0, 99]")
            print("r2: index of rover 2 [0, 99]")
            print("r3: index of rover 3 [0, 99]")
            print("r4: index of rover 4 [0, 99]")
            print(100 * "-")

        return [1.8, 0.0, 1.2, 0.0, 55.0] + [2.3, 0.0, 1.2, 0.0, 15.0] + [10, 2, 1] + [10, 2, 1] + [13, 21, 34, 55]
    
    def compute_orbit_walker(self, walker, ep0, sma):
        """Compute one full-orbit of the Walker constellation planes (for plots)

        Args:
            walker (sgp4.SatrecArray): the array of Walker satellites to plot
            ep0 (float): Julian date denoting starting epoch
            sma (float): semi-major axis of orbit

        Returns:
            pos (numpy array, P x N x 3): N orbital x, y, z positions for P planes
        """
        
        # Extract mean motion
        mean_motion = np.sqrt(pk.MU_EARTH / sma**3 / pk.EARTH_RADIUS**3) * 24 * 60 * 60 / (2 * np.pi)
        # Compute time range for one full orbit
        jds = np.linspace(ep0, ep0 + 1 / mean_motion, 100)
        frs = jds * 0.0
        # Propagate using SGP4
        err, pos, _ = walker.sgp4(jds, frs)
        if not np.all(err == 0):
            raise ValueError("The satellite cannot be propagated successfully on the defined time grid")
        
        return pos
    
    def compute_orbit_motherships(self, ep0):
        """Compute one full-orbit of the motherships from epoch ep0 (for plots)

        Args:
            ep0 (float): Julian date denoting starting epoch

        Returns:
            orbits (numpy array, S x N x 3): N orbital x, y, z positions for S satellites
        """
        
        # Pre-allocate return array
        N = 100 # number of samples along orbit
        # Get SGP4-ready motherships
        motherships = get_mothership_satellites()
        orbits = np.zeros((len(motherships), N, 3))
        for i, usr in enumerate(motherships):
            # Extract mean motion
            mean_motion = usr.no_kozai * 24 * 60 / (2 * np.pi) # revolutions per day
            # Compute time range for one full orbit
            jds = np.linspace(ep0, ep0 + 1 / mean_motion, N)
            frs = jds * 0.0
            # Propagate using SGP4
            err, pos, _ = usr.sgp4_array(jds, frs)
            if not np.all(err == 0):
                raise ValueError("The satellite cannot be propagated successfully on the defined time grid")
            orbits[i] = pos
        
        return orbits
    
    def plot(self, x, src, dst, ep=1, lims=10000, ax=None, dark_mode=True):
        """Plot the full constellations with solution path and optional orbits

        Args:
            x (list): chromosome describing the communications network
            src (int): mothership index denoting path source
            dst (int): rover index denoting path destination
            ep (int): index of the epoch in the predefined time grid
            lims (float, optional): plot limits. Defaults to 10000.
            ax (matplotlib 3D axis, optional): plot axis.
            dark_mode (bool, optional): dark background for plot (recommended)

        Returns:
            matplotlib.axis: the 3D plot axes
            list: indices of the graph nodes on the communications path (if one is found, otherwise [])
        """
        
        # Create the plotting axis if needed
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')

        # Apply a dark background for better visualization
        if dark_mode:
            sns.set(style="darkgrid")
            plt.style.use("dark_background")
            
        # Construct the two Walker constellations from the specifications 
        walker1, walker2 = self.construct_walkers(x)
        # Construct the rover positions
        rovers_idx = np.array(x[-4:]).astype(int)
        lambda0 = self.lambdas[rovers_idx]
        phi0 = self.phis[rovers_idx]
        rovers = construct_rover_pos(lambda0, phi0, self.n_rovers, self.jds, self.R_p, self.w_p)
        # Construct the Walker satellite positions
        pos = self.construct_pos(walker1, walker2, rovers)
        # Compute and plot the orbits of the Walker and mothership satellites at the epoch ep
        # Walker 1
        N1 = x[10] * x[11]
        w1_orb = self.compute_orbit_walker(walker1, self.jds[ep], x[0])
        for i in range(N1):
            ax.plot(w1_orb[i, :, 0], w1_orb[i, :, 1], w1_orb[i, :, 2], 'r-', linewidth=0.5)
        # Walker 2
        N2 = x[13] * x[14]
        w2_orb = self.compute_orbit_walker(walker2, self.jds[ep], x[5])
        for i in range(N2):
            ax.plot(w2_orb[i, :, 0], w2_orb[i, :, 1], w2_orb[i, :, 2], 'b-', linewidth=0.5)
        # Motherships
        mothership_orb = self.compute_orbit_motherships(self.jds[ep])
        for i in range(len(mothership_orb)):
            ax.plot(mothership_orb[i, :, 0], mothership_orb[i, :, 1], mothership_orb[i, :, 2], 'w-', linewidth=0.5)

        # Overlay the Walker satellite and mothership positions at epoch ep 
        # Walker1: red, Walker2: blue, motherships: white, rovers: yellow
        ax.scatter(pos[:len(walker1),ep,0], pos[:len(walker1),ep,1], pos[:len(walker1),ep,2], c='r', marker="1", s=200)
        ax.scatter(pos[len(walker1):-self.n_motherships-self.n_rovers,ep,0], pos[len(walker1):-self.n_motherships-self.n_rovers,ep,1], pos[len(walker1):-self.n_motherships-self.n_rovers,ep,2], c='b', marker="1", s=200)
        ax.scatter(pos[-self.n_motherships-self.n_rovers:-self.n_rovers,ep,0], pos[-self.n_motherships-self.n_rovers:-self.n_rovers,ep,1], pos[-self.n_motherships-self.n_rovers:-self.n_rovers,ep,2], c='w', marker="1", s=300)
        # Annotate source nodes (motherships)
        for i in range(self.n_motherships):
            ax.text(pos[-self.n_motherships-self.n_rovers+i,ep,0], pos[-self.n_motherships-self.n_rovers+i,ep,1], pos[-self.n_motherships-self.n_rovers+i,ep,2],  '%s' % (str(i+1)), size=20, zorder=1,  color='w')         
        
        # Annotate destination nodes (rovers)
        ax.scatter(pos[-self.n_rovers:,ep,0], pos[-self.n_rovers:,ep,1], pos[-self.n_rovers:,ep,2], c='y', marker="^", s=200)
        for i in range(self.n_rovers):
            ax.text(pos[-self.n_rovers+i,ep,0], pos[-self.n_rovers+i,ep,1], pos[-self.n_rovers+i,ep,2],  '%s' % (str(i+1)), size=20, zorder=1,  color='y') 

        # Build the communications network
        path = []
        eta1, eta2 = x[4], x[9]
        G, _, _ = self.build_graph(ep, pos, N1, (eta1, eta2))
        N = len(G)
        src_node = N1 + N2 + src - 1
        dst_node = N1 + N2 + self.n_motherships + dst - 1
        # Find the shortest path (if one exists)
        try:
            path = nx.shortest_path(G, src_node, dst_node, weight="weight", method="dijkstra")
            for i,j in zip(path[:-1], path[1:]):
                ax.plot([pos[i,ep,0], pos[j,ep,0]], [pos[i,ep,1], pos[j,ep,1]], [pos[i,ep,2], pos[j,ep,2]], 'g-.', linewidth=3)
            print("Mothership {} (node {}) communicates with rover {} (node {}) at epoch {} via: {}".format(\
                src, src_node, dst,  dst_node, ep, path))
        except nx.exception.NetworkXNoPath as e:
            print("Mothership {} (node {}) cannot reach rover {} (node {}) at epoch {}".format(\
                src, src_node, dst,  dst_node, ep))

        # Plot the New Mars planet
        r = pk.EARTH_RADIUS/1000
        u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
        x = r * np.cos(u) * np.sin(v)
        y = r * np.sin(u) * np.sin(v)
        z = r * np.cos(v)
        ax.plot_surface(x, y, z, alpha=0.3, color="purple", linewidth=0)
        ax.set_axis_off()
        ax.set_xlim(-lims,lims)
        ax.set_ylim(-lims,lims)
        ax.set_zlim(-lims,lims)
        ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
        return ax, path

def combine_scores(points):
    """ Function for aggregating single solutions into one score using hypervolume indicator """

    ref_point = np.array([1.2, 1.4])
    
    # solutions that not dominate the reference point are excluded
    filtered_points = [s[:2] for s in points if pg.pareto_dominance(s[:2], ref_point)]
    
    if len(filtered_points) == 0:
        return 0.0
    else:
        hv = pg.hypervolume(filtered_points)
        #return -hv.computborderse(ref_point) * 10000
        return -hv.compute(ref_point) * 10000

    
import warnings
warnings.filterwarnings("ignore")

from fcmaes.optimizer import wrapper, dtime, Bite_cpp, De_cpp, Crfmnes_cpp
import fcmaes
from fcmaes import retry, mode, modecpp, moretry, mapelites, diversifier
from scipy.optimize import Bounds    
from os import walk
import multiprocessing as mp
import ctypes as ct
from functools import partial
from fcmaes import bitecpp
from multiprocessing import Manager

udp = constellation_udp() 
nobj = 2
ncon = 2
dim = 20
ref_point = np.array([1.2, 1.4])
ubs = udp.get_bounds()
bounds = Bounds(ubs[0], ubs[1]) 
    
def fitness(x): # fitness wrapper converting the last ten arguments into integer values
    x[10:] = x[10:].astype(int)
    return np.array(udp.fitness(x))

def select_valid(xs, ys):
    cv = np.array([np.amax(y[nobj:], 0) for y in ys])
    valid = (cv <= 0)
    ys = ys.T[:nobj].T
    ys = ys[valid]
    xs = xs[valid]                        
    xs, ys = moretry.pareto(xs, ys)
    return xs, ys

def read_solution(fname):
    with np.load(fname) as data:
        xs = data['xs']
        ys = data['ys']                      
    return xs, ys

from fcmaes.evaluator import parallel_mo

#+++++++ Apply fcmaes multi objective optimization using NSGA-II population update ++++++++++++++++++++++++
# Uses fcmaes multi objective optimization to optimize the pareto front
# Uses parallel function evaluation, but cannot pass score 6400 even when executed many times, 
# using a large number of iterations and a big population. 
# Which is the reason only one team - "ML Actonauts" achieved this goal during the GECCO competition 
# https://www.esa.int/gsp/ACT/projects/spoc-2023/ 

def mo_par():
            
    guess = None
    #guess, _ = read_solution("res/quantcomm_1_100_6372134.npz") # inject an existing pareto front   
    popsize = 512

    es = mode.MODE(nobj, ncon, bounds, popsize = popsize, nsga_update=True) # Python MOO optimizer
    #es = modecpp.MODE_C(nobj, ncon, bounds, popsize = popsize, nsga_update=True) # C++ MOO optimizer
   
    fit = parallel_mo(fitness, nobj+ncon, workers = mp.cpu_count())
    iters = 0
    stop = 0
    max_hv = 0
    time_0 = time.perf_counter()
    if not guess is None:
        es.set_guess(guess, fitness)

    while stop == 0 and iters < 100000:               
        xs = es.ask()
        ys = fit(xs)        
        es.tell(ys) # tell evaluated x             
        iters += 1
        valid = [y[:2] for y in ys if np.less_equal(y , np.array([1.2, 1.4, 0, 0])).all()]
        hv = pg.hypervolume(valid).compute(ref_point)
        if hv > max_hv:
            max_hv = hv
        if hv > 0.9999*max_hv: # show stagnation
            logger.info(f'time: {dtime(time_0)} iter: {iters} hv: {hv * 10000}')
            np.savez_compressed("quantcomm_" + str(int(hv * 1000000)), xs=xs, ys=ys)
    fit.stop()
    return xs, ys

#+++++++ Apply fcmaes diversifier quality diversity algorithm ++++++++++++++++++++++++
# The initial archive is created using an existing pareto front. This way the QD-algorithm
# dosn't need to find this "hard to reach" area of the solution space by its own. 

def mo_to_qd(y):
    f1, f2, c1, c2 = y
    # weight the objectives and constraints
    return f1/0.5 + f2/1.4 + c1/3000 + c2/50, \
                np.minimum(ref_point, np.array([f1, f2])) # use the objectives as descriptive space

def qd_fun(x):
    return mo_to_qd(fitness(x)) # convert the MO result into a QD result

def get_arch(qd_bounds, niche_num, samples_per_niche):
    xs, _ = read_solution("res/quantcomm_1_100_6372134.npz") # inject an existing pareto front      
    arch = mapelites.empty_archive(dim, qd_bounds, niche_num, samples_per_niche)
    mapelites.update_archive(arch, xs, qd_fun)
    return arch

def nd_par(niche_num = 10000):
    udp = constellation_udp()
    ubs = udp.get_bounds()
    qd_bounds = Bounds([0.7, 0.], [1.2, 1.4])
    samples_per_niche = 20
    arch = get_arch(qd_bounds, niche_num, samples_per_niche)
    opt_params0 = {'solver':'elites', 'popsize':100, 'use':2}
    opt_params1 = {'solver':'CRMFNES_CPP', 'max_evals':2000, 'popsize':32, 'stall_criterion':3}
    archive = diversifier.minimize(
         mapelites.wrapper(qd_fun, 2, interval=10000, save_interval=100000),
         bounds, qd_bounds,
         workers = 32, opt_params=[opt_params0, opt_params1], archive = arch,
         niche_num = niche_num, samples_per_niche = samples_per_niche,
         max_evals = 1000000)

    print('final archive:', archive.info())
    archive.save('final archive')
    
    
#++++++++++++++++++++++ Apply BiteOpt Single-Objective Optimization ++++++++++++++++++++++++++++++++++++++++
# Uses https://github.com/avaneev/biteopt applied to a fitness function maximizing the hypervolume
# to find the best target_num solutions maximizing the pareto front
# Easily surpasses score 6400 when given enough time. The final result needs to be reduced to 100 solutions. 

def so_par():

    target_num = 512 # desired size of the pareto front  

    # hypervolume replacing one solution of the pareto front  
    def fit_hyper(i, ys, x):
        y = fitness(x)
        c = sum([10000 + c for c in y[2:] if c > 0])
        if c > 0: # constraint violation
            return c
        if pg.pareto_dominance(y[:2], ref_point):           
            ys[i] = y[:2]
            return -pg.hypervolume(ys).compute(ref_point) * 10000  
        else:
            return 0
    
    # parallel optimization of the whole pareto front    
    class OptSo(object):
           
        def __init__(self, 
                     max_evals,
                     xs,
                     ys
                    ): 
            self.max_evals = max_evals  
            self.manager = Manager()   
            self.ys = self.manager.list(ys)
            self.ys0 = list(ys)
            self.xs = self.manager.list(xs)
            self.min_ys = np.amin(ys, axis=0)
            self.count = mp.RawValue(ct.c_int, 0) 
            self.mutex = mp.Lock() 
            self.n = len(ys)
        
        def incr(self):
            with self.mutex:
                next = self.count.value
                self.count.value += 1
                return next
    
        def eval(self, workers=mp.cpu_count()):
            proc=[mp.Process(target=self.eval_loop) for pid in range(workers)]
            [p.start() for p in proc]
            [p.join() for p in proc]             
            return np.array(self.xs), np.array(self.ys)
        
        def eval_loop(self):
            while True:
                i = self.incr()
                if i >= self.n:
                    return
                logger.info(f'optimizing solution {i}')
                fit = wrapper(partial(fit_hyper, i, list(self.ys)))
                x0 = self.xs[i]                                
                ret = bitecpp.minimize(fit, bounds, x0, max_evaluations = self.max_evals)
                if ret.fun < 0: # no constraint violation?
                    y = fitness(ret.x)[:2]
                    self.ys[i] = y
                    self.xs[i] = ret.x
     
    def opt_so(max_evals, xs, ys, workers=mp.cpu_count()):
        eval = OptSo(max_evals, xs, ys)
        return eval.eval(workers)
    
    max_evals = 2000    
      
    # random initialization
    # rg = Generator(MT19937()) 
    # xs = [rg.uniform(ubs[0], ubs[1]) for _ in range(target_num)]
    # ys = [ref_point-0.000001 for _ in range(target_num)]

    # initialization with a given pareto front 
    xs, ys = read_solution("res/quantcomm_1_100_6372134.npz") # inject an existing pareto front
    
    last_xs = []
    last_ys = []
    for i in range(1, 1000):
        xs, ys = opt_so(max_evals, xs, ys)
        xs, ys = moretry.pareto(np.array(list(xs) + last_xs), 
                                    np.array(list(ys) + last_ys))
        if len(ys) > target_num:
            xs, ys = reduce(xs, ys, target_num)        
        hv = int(pg.hypervolume(ys).compute(ref_point) * 10000000)  

        np.savez_compressed("quantcomm_" + str(i) + "_" + str(len(ys)) + "_" 
                            + str(max_evals) + "_" + str(hv), xs=xs, ys=ys)
        last_xs = list(xs)
        last_ys = list(ys)
        
    return xs


#++++++++++++++++++++++ Apply PYMOO NSGA-II ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Uses PYMOO and https://github.com/avaneev/biteopt to find the best target_num solutions
# maximizing the pareto front.
# Easily surpasses score 6400 when given enough time. The result is already reduced to 100 solutions. 
# Note that instead of relying on PYMOOs parallelization mechanisms (partial) optimization runs
# are executed in parallel achieving maximal scaling with the number of cores. 
# The fitness function is "hijacked" collecting solutions later joined for the resulting pareto front.

def pymoo_par():

    from pymoo.core.problem import ElementwiseProblem 
    from pymoo.algorithms.moo.nsga2 import NSGA2 
    from pymoo.termination import get_termination
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.constraints.eps import AdaptiveEpsilonConstraintHandling
    from pymoo.optimize import minimize
    from itertools import chain

    target_num = 100 # desired size of the pareto front 
    n_eval = 10000
    popsize = 300
    time_0 = time.perf_counter()
    guess = None
    guess, ys = read_solution("res/quantcomm_1_100_6372134.npz") # inject an existing pareto front
    guess, ys = reduce(guess, ys, target_num)   
    
    class fitness_wrapper():
        
        def __init__(self, 
                     pid,
                     xs_out,
                     ys_out
                    ): 
            self.max_hv = 0
            self.xs = []
            self.ys = []
            self.count = 1
            self.evals = 0
            self.pid = pid
            self.xs_out = xs_out
            self.ys_out = ys_out
        
        # fitness accumulates valid solutions and monitors their hypervolume
        def __call__(self, x):
            y = fitness(x)
            self.evals += 1    
            if np.amax(y[2:]) <= 0 and np.less_equal(y[:2], ref_point).all() : # add only valid solutions
                self.ys.append(y[:2]) # exclude constraint values because solution is valid
                self.xs.append(x)
            if len(self.ys) >= 2*popsize:
                self.count += 1      
                xs, ys = moretry.pareto(np.array(self.xs), np.array(self.ys)) # reduce to pareto front
                self.xs, self.ys = list(xs), list(ys)
                hv = pg.hypervolume(self.ys).compute(ref_point)
                if hv > self.max_hv * 1.0001: # significant improvement: register solutions at managed dicts
                    self.max_hv = hv
                    self.xs_out[self.pid] = self.xs
                    self.ys_out[self.pid] = self.ys
                    logger.info(f'time: {dtime(time_0)} ev: {self.evals} hv: {hv * 10000} n: {len(ys)}')
            return y    
      
    class OptPymoo(object):
        
        def eval_loop(self, workers=mp.cpu_count()):
            xs = guess
            for i in range(1, 1000):
                xs, ys = self.eval(i, xs, workers)
            return xs, ys
            
        def eval(self, i, guess, workers):
            manager = Manager()
            xs_out = manager.dict() # for inter process communication
            ys_out = manager.dict() # collects solutions generated in the sub processes
            fits = [fitness_wrapper(pid, xs_out, ys_out) for pid in range(workers)]
            proc=[mp.Process(target=self.optimize, args=(guess, fits[pid], pid)) for pid in range(workers)]
            [p.start() for p in proc] # spawn NSGAII optimization workers
            [p.join() for p in proc]    
            xs = np.array(list(chain.from_iterable(xs_out.values()))) # join collected solutions
            ys = np.array(list(chain.from_iterable(ys_out.values()))) # we ignore the pymoo optimization result
            xs, ys = moretry.pareto(xs, ys)
            if len(ys) > target_num:
                xs, ys = reduce(xs, ys, target_num)        
            hv = int(pg.hypervolume(ys).compute(ref_point) * 10000000)  
            np.savez_compressed("quantcomm_" + str(i) + "_" + str(len(ys)) + 
                                "_" + str(hv), xs=xs, ys=ys)       
            return xs, ys
        
        def optimize(self, guess, fit, pid):
            
            class MyProblem(ElementwiseProblem):
    
                def __init__(self, **kwargs):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
                    super().__init__(n_var=dim,
                                     n_obj=nobj,
                                     n_constr=ncon,
                                     xl=np.array(bounds.lb),
                                     xu=np.array(bounds.ub), **kwargs)
            
                def _evaluate(self, x, out, *args, **kwargs):   
                    y = fit(x)
                    out["F"] = y[:nobj]
                    out["G"] = y[nobj:]

            problem = MyProblem()
            algorithm = NSGA2(
                pop_size=popsize,
                n_offsprings=10,
                sampling=FloatRandomSampling() if guess is None else guess,
                crossover=SBX(prob=0.9, eta=15), # simulated binary crossover
                mutation=PM(eta=20), # polynomial mutation     
                eliminate_duplicates=True,
            )    
            algorithm = AdaptiveEpsilonConstraintHandling(algorithm, perc_eps_until=0.5)        
            minimize(problem, algorithm, get_termination("n_eval", n_eval), verbose=False, seed=pid*677)
    
    opt = OptPymoo()
    return opt.eval_loop()

#++++++++++++++++++++++ Reduction to 100 solutions ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Uses https://github.com/avaneev/biteopt / parallel optimization to find the best num solutions
# maximizing the pareto front
   
def reduce(xs, ys, num, evals = 50000, retries = mp.cpu_count()): 
    if len(ys) <= num:
        return xs, ys
    bounds = Bounds([0]*num, [len(ys)-1E-9]*num) # select best num from xs, ys
    
    def fit(x): # selects 100 solutions and returns the negated pareto front of this selection
        selected = x.astype(int) 
        ys_sel = ys[selected]      
        hv = pg.hypervolume(ys_sel)
        return -hv.compute(ref_point) * 10000

    res = fcmaes.retry.minimize(wrapper(fit), # parallel optimization restart / retry
                         bounds, 
                         optimizer=Bite_cpp(evals), 
                         num_retries=retries)
    
    selected = res.x.astype(int)
    return xs[selected], ys[selected]
  
if __name__ == '__main__':

    #pymoo_par()
    #so_par()    
    mo_par()
    #nd_par()

