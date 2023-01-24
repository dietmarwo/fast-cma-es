import numpy as np
import os
import json
import matplotlib.pyplot as plt

from scipy.optimize import Bounds
from fcmaes import retry
from fcmaes.optimizer import Bite_cpp, wrapper, logger
from fcmaes.optimizer import Crfmnes_cpp, Crfmnes, de_cma, Bite_cpp, Cma_cpp, LDe_cpp, dtime,  De_cpp, random_search, wrapper, logger
import ctypes as ct
import multiprocessing as mp  
 
# see https://optimize.esa.int/challenge/spoc-delivery-scheduling/About
# see https://optimize.esa.int/challenge/spoc-delivery-scheduling/p/delivery-scheduling

"""
    Extends ESAs verification code for the SpOC: Delivery Scheduling competition held in 2022.
    Original code
    https://api.optimize.esa.int/media/problems/spoc-delivery-scheduling-delivery-scheduling-1649698262955.py
    is unmodified. 
    A second class fitness is added, which removes redundancies in the decision variable vector reducing
    dimension to 352. Instead of only considering the station with minimal resources it uses a weighted
    sum as fitness value. Performance is greatly improved by using https://numba.pydata.org/ enabling about
    500000 fitness evaluations per second on a 16 core CPU.  
    Applies the https://github.com/avaneev/biteopt BiteOpt algorithm iteratively 
    using parallel restarts. 
    """

class trappist_schedule:
    """
    UDP (User-Defined Problem) for the Trappist-1 scheduling optimisation problem.
    This corresponds to the third challenge in SpOC (Space Optimisation Competition)
    conceived by the ACT for the GECCO 2022 conference.

    The challenge involves optimising the schedule for delivering asteroids to
    hypothetical processing stations in a differnt orbit in the far future.
    A more detailed overview of the problem scenario and the three challenges can be found here:

    https://www.esa.int/gsp/ACT/projects/gecco-2022-competition/

    This class conforms to the pygmo UDP format.
    """

    def __init__(
        self,
        path=os.path.join(".", "trappist/candidates.txt"),
        n_stations=12,
        start_time=0.0,
        end_time=80.0,
        station_gap=1.0,
        materials=(
            "Material A",
            "Material B",
            "Material C",
        ),
    ):
        # Database of asteroid-to-station visit opportunities,
        self.db = self._load(path)

        # Number of stations
        self.n_stations = n_stations

        # Number of asteroids
        self.n_asteroids = len(self.db)

        # A flattened version of the database with
        # (asteroid ID, station ID, opportunity ID) tuples as keys and
        # (arrival time, mass A, mass B, mass C) tuples as values.
        # This method also computes the maximum number of opportunities
        # in the database (relevant for bounds checks)
        (self.flat_db, self.max_opportunities) = self._flatten(self.db)

        # The start and end times for the whole problem.
        # Units: days
        self.start_time = start_time
        self.end_time = end_time

        # Station gap (minimum time allowed between activating two consecutive stations).
        # Units: days
        self.station_gap = station_gap

        # List of material names
        self.materials = materials

    def get_nobj(self):
        """
        There is only one objective for this challenge:
        to maximise the minimum amount of material collected per station.

        Returns:
            Number of objectives.
        """
        return 1

    def get_nix(self):
        """
        Each assignment consists of a pair of asteroid ID and station ID,
        hence the total number is 2 x the number of asteroids.

        Returns:
            Number of integer components of the chromosome.
        """
        return self.n_asteroids * 2

    def get_nec(self):
        """
        There are two equality constraints (cf. _fitness_impl() for details).

        Returns:
            Number of equality constraints.
        """
        return 2

    def get_nic(self):
        """
        There are two equality constraints
        (cf. _fitness_impl() for details).

        Returns:
            Number of inequality constraints.
        """
        return 2

    def get_bounds(self):
        """
        Bounds for chromosome elements.

        Returns:
            Bounds for each element in the chromosome.
        """

        lb = [self.start_time] * (2 * self.n_stations)
        lb.extend([1, 0, 0] * self.n_asteroids)

        ub = [self.end_time] * (2 * self.n_stations)
        ub.extend(
            [self.n_asteroids, self.n_stations, self.max_opportunities]
            * self.n_asteroids
        )

        return (lb, ub)

    def _load(
        self,
        path,
    ):
        """

        Load the database from an external JSON file.

        Args:
            path: The path to the database file

        Returns:
            The path to the database file.
        """

        with open(path) as db:
            _db = json.loads(db.read())

        db = {}
        for ast_id, stations in _db.items():

            opportunities = {}
            for stat_id, opps in stations.items():
                # Convert the station ID from str to int.
                opportunities[int(stat_id)] = list(opps)

            # Convert the asteroid ID from str to int.
            db[int(ast_id)] = opportunities

        return db

    def _flatten(
        self,
        db,
    ):

        """
        Flatten the database.

        Args:
            db: The database of possible asteroid / station assignment opportunities.

        Returns:
            A flat version of the database with (asteroid ID, station ID, opportunity ID)
            tuples as keys and (arrival time, mass A, mass B, mass C) tuples as values and
            the maximum number of opportunities for any asteroid / station pair in the database.
        """

        flat_db = {}
        max_opps = 0
        for ast_id, stations in db.items():
            for stat_id, opps in stations.items():
                if len(opps) > max_opps:
                    max_opps = len(opps)
                for idx, opp in enumerate(opps):
                    flat_db[(ast_id, stat_id, idx)] = opp

        return (flat_db, max_opps - 1)

    def _plot(
        self,
        masses,
        schedule,
        ax=None,
        path=None,
    ):
        """
        Plot the total material masses at each station and
        the schedule vs. opportunities for each station.

        Args:
            masses: A 2D array containing the masses corresponding to all assignment opportunities.
            schedule: The actual scheduled asteroid / station assignments and their corresponding masses.
            ax: Plot axes. Defaults to None.
            path: A file to save the plot to. Defaults to None.

        Returns:
            Plot axes.
        """

        if ax is None:
            fig, ax = plt.subplots(2, 1, figsize=(12, 18))

        (m_ax, w_ax) = ax[0], ax[1]

        # ==[ Plot mass distribution ]==

        indices = np.arange(1, self.n_stations + 1)
        bar_width = 0.2

        m_ax.bar(
            indices - bar_width,
            masses[:, 0],
            bar_width,
            color="r",
            label="Material A",
        )
        m_ax.bar(
            indices,
            masses[:, 1],
            bar_width,
            color="g",
            label="Material B",
        )
        m_ax.bar(
            indices + bar_width,
            masses[:, 2],
            bar_width,
            color="b",
            label="Material C",
        )

        # ==[ Plot minimum masses for each material]==
        min_masses = masses.min(axis=0)
        m_ax.plot(
            [0, self.n_stations + 1],
            [min_masses[0], min_masses[0]],
            "r--",
            label=f"Minimum mass of {self.materials[0]}",
        )
        m_ax.plot(
            [0, self.n_stations + 1],
            [min_masses[1], min_masses[1]],
            "g--",
            label=f"Minimum mass of {self.materials[1]}",
        )
        m_ax.plot(
            [0, self.n_stations + 1],
            [min_masses[2], min_masses[2]],
            "b--",
            label=f"Minimum mass of {self.materials[2]}",
        )

        m_ax.set_xlim((0.4, 12.6))
        m_ax.set_ylim((0.4, masses.max() + 4))
        m_ax.set_xticks(list(range(1, self.n_stations + 1)))
        m_ax.set_xticklabels(list(range(1, self.n_stations + 1)))
        m_ax.set_xlabel("Station")
        m_ax.set_ylabel("Material masses")
        m_ax.legend()

        # ==[ Plot schedule ]==

        # Plot all opportunities
        for stat_id, data in schedule.items():

            if len(data) == 0:
                continue

            opportunities, atimes, window = (
                data["opportunities"],
                data["assignments"],
                data["window"],
            )

            # Opportunities
            w_ax.plot(
                opportunities,
                np.ones((len(opportunities),)) * stat_id,
                "r.",
                ms=1,
                label="Opportunities" if stat_id == self.n_stations else None,
            )

            # Arrival times
            w_ax.plot(
                atimes,
                np.ones((len(atimes),)) * stat_id,
                "c|",
                ms=6,
                label="Assignments" if stat_id == self.n_stations else None,
            )

            # Window
            w_ax.plot(
                [window[0], window[0]],
                [1, self.n_stations],
                "--",
                color="lightgray",
                lw=0.5,
            )
            w_ax.plot(
                [window[1], window[1]],
                [1, self.n_stations],
                "--",
                color="darkgray",
                lw=0.5,
            )

        w_ax.plot([self.start_time, self.start_time], [1, self.n_stations], "k-")
        w_ax.plot([self.end_time, self.end_time], [1, self.n_stations], "k-")

        w_ax.set_xlabel("Time [days]")

        w_ax.set_yticks(list(range(1, self.n_stations + 1)))
        w_ax.set_yticklabels(list(range(1, self.n_stations + 1)))
        w_ax.set_ylabel("Station")
        w_ax.set_ylim((0, 14))

        w_ax.legend(loc=1)

        if path is not None:
            fig.savefig(path, dpi=100)

        return ax

    def _fitness_impl(
        self,
        x,
        logging=False,
        plotting=False,
        ax=None,
        path=None,
    ):
        """
        Computes the constraints and the fitness of the provided chromosome.

        1. Equality constraints:

        1.1. Asteroid IDs: all asteroids in the database must be present in the chromosome.
        1.2. Opportunity IDs: all opportunity IDs in the chromosome must correspond to opportunities in the database.

        2. Inequality constraints:

        2.1. Station gaps: all station gaps must be greater than a minimal time period (self.station_gap)
        2.2. Arrival times: all asteroid arrival times must be between the start and end times of the corresponding station

        3. Fitness:

        3.1 Iterate over the chromosome and add the masses of the materials for all assigned asteroids with valid transfers
        3.2 Find the minimum mass of each material per station. This is the final fitness.

        Args:
            x: A list of integers and floats in the following format:
                - Station start and end times (2 x self.n_stations)
                - The following items for all selected asteroids (3 x self.n_asteroids integers in total):
                    - Asteroid ID
                    - Station ID
                    - Opportunity ID

                NOTE: The triplets do not have to be ordered by asteroid ID.

            logging: Logging switch. Defaults to False.
            plotting: Plotting switch. Defaults to False.
            ax: Plot axes. Defaults to None.
            path: File path for saving the plots. Defaults to None.

        Returns:
            A tuple containing:
                - The fitness
                - A list of equality constraints
                - A list of inequality constraints
                - Plot axes

        """

        eq_constraints = []
        ineq_constraints = []

        # Offset representing the time windows for stations (for convenience only)
        station_times_offset = 2 * self.n_stations

        # Extract a set of tuples of all the selections as tuples
        asteroid_ids = [int(a_id) for a_id in x[station_times_offset::3]]
        station_ids = [int(s_id) for s_id in x[(station_times_offset + 1) :: 3]]
        assignments = [
            int(assignment - 1) for assignment in x[(station_times_offset + 2) :: 3]
        ]

        triplets = tuple(zip(asteroid_ids, station_ids, assignments))

        # Extract the start and end times for all stations from the chromosome
        station_times = x[:station_times_offset]
        station_start_times = station_times[0 : len(station_times) : 2]
        station_end_times = station_times[1 : len(station_times) : 2]

        """
        1. Equality constraints.
        """

        # ==[ 1.1 Check asteroid IDs ]==

        asteroid_id_violations = set(asteroid_ids).symmetric_difference(
            set(self.db.keys())
        )

        eq_constraints.append(len(asteroid_id_violations))

        # ==[ 1.2. Check opportunities ]==

        # Check the opportunity IDs  for asteroid / station assignments
        # where the station ID is *not* 0 (i.e., the asteroid has
        # been activated and selected for transfer to a valid station).
        opportunity_id_violations = [
            triplet
            for triplet in triplets
            if triplet not in self.flat_db and triplet[1] > 0
        ]

        eq_constraints.append(len(opportunity_id_violations))

        """
        2. Inequality constraints.
        """

        # First we sort and index station activation windows.
        # We produce a list where each entry has the following format:
        #
        # [idx, [start, end]]
        #
        # where
        # - idx: index in the *original* chromosome (= station ID)
        # - (start, end): start and end times for visiting the corresponding station

        station_windows = zip(station_start_times, station_end_times)
        sorted_indexed_windows = sorted(
            enumerate(station_windows, 1), key=lambda i: i[1]
        )

        # ==[ 2.1. Check station gaps ]==

        # Compute the gaps between stations from the sorted array of station windows
        gaps = [
            s2[1][0] - s1[1][1]
            for s1, s2 in zip(sorted_indexed_windows[:-1], sorted_indexed_windows[1:])
        ]

        gap_violations = self.station_gap - np.array(gaps, dtype=np.float32)

        ineq_constraints.append(max(gap_violations))

        # ==[ 2.2. Arrival times ]==

        # Collect all asteroid   times for each station
        station_window_start_times = []
        station_window_end_times = []

        for s_id in station_ids:
            if s_id == 0:
                # Trick to make sure that unassigned asteroids don't trigger
                # and inequality constraint violation
                station_window_start_times.append(-1.0)
                station_window_end_times.append(self.end_time + 1.0)
            else:
                station_window_start_times.append(station_start_times[s_id - 1])
                station_window_end_times.append(station_end_times[s_id - 1])

        # Find any instances where the arrival time is before the start time or
        # after the end time of the corresponding station's window.
        arrival_times = np.array(
            [
                self.flat_db[triplet][0]
                if triplet in self.flat_db
                else (self.end_time - self.start_time) / 2
                for triplet in triplets
            ],
            dtype=np.float32,
        )
        station_window_start_times = np.array(
            station_window_start_times, dtype=np.float32
        )
        station_window_end_times = np.array(station_window_end_times, dtype=np.float32)

        # Find instances where the arrival times are earier than the
        # start time for the corresponding station
        arrival_time_violations = np.where(
            arrival_times - station_window_start_times < 0.0, 1, 0
        )

        # Add to those any instances where the arrival times are later than the
        # end time for the corresponding station
        arrival_time_violations += np.where(
            station_window_end_times - arrival_times < 0.0, 1, 0
        )

        # Find any violations (1s indicate a violation in either start time or end time)
        arrival_time_violations = np.where(arrival_time_violations > 0, 1, 0)

        ineq_constraints.append(arrival_time_violations.sum())

        """
        3. Fitness
        """

        # Compute the masses of all materials accumulated at each station
        masses_per_station = {
            s_id: np.array([0.0, 0.0, 0.0], dtype=np.float32)
            for s_id in range(1, self.n_stations + 1)
        }

        for (triplet, violation) in zip(
            triplets, arrival_time_violations.astype(bool).tolist()
        ):
            # If the asteroid is assigned to a valid station and
            # its arrival time is within bounds...
            if triplet[1] > 0 and not violation:
                # Add the material masses to the stations
                masses_per_station[triplet[1]] += np.array(
                    self.flat_db[triplet][1:],
                    dtype=np.float32,
                )

        # Collect all the masses per station into a single 2D array that is easy to manipulate
        masses = np.array(
            [masses_per_station[s] for s in range(1, self.n_stations + 1)]
        )

        # Final fitness computation.
        # The objective is to maximise the minimum mass
        # of any material across all stations.
        fitness = -masses.min()

        if logging:

            print(
                f"==[ Invalid asteroid IDs: {eq_constraints[0]} out of {len(self.db)}"
            )
            print(f"==[ Invalid arrival times: {eq_constraints[1]}")
            print(f"==[ Minimal inter-station gap: {min(gaps):<2.4}")
            print(f"==[ Invalid assignments: {ineq_constraints[1]}")
            print(f"==[ Masses per station:")
            print(
                f"{'Station ID':>12} {self.materials[0]:>12} {self.materials[1]:>12} {self.materials[2]:>12}"
            )
            for stat_id, mass_dist in masses_per_station.items():
                print(
                    f"{stat_id:>12} {mass_dist[0]:>12.6f} {mass_dist[1]:>12.6f} {mass_dist[2]:>12.6f}"
                )

            for idx, item in enumerate(gap_violations):
                if item > 0:
                    # Get the station IDs in the original chromosome
                    station_1_id = sorted_indexed_windows[idx][0]
                    station_2_id = sorted_indexed_windows[idx + 1][0]
                    gap = gaps[idx]
                    if gap <= self.station_gap:
                        print(
                            f"==[\tThe gap between stations {station_1_id} and {station_2_id} is {gap:3.3f} (should be >= {self.station_gap:3.3f})."
                        )
                    else:
                        print(
                            f"==[\tThe windows for stations {station_1_id} and {station_2_id} overlap by {-gap:3.3f} days."
                        )

            print(f"==[ Total fitness: {fitness}")

        # Plotting
        if plotting:
            schedule = {
                s_id: {
                    "opportunities": [],
                    "assignments": [],
                    "window": [],
                }
                for s_id in range(1, self.n_stations + 1)
            }

            for (_, s_id, _), val in self.flat_db.items():
                schedule[s_id]["opportunities"].append(val[0])

            for triplet in triplets:
                if triplet[1] > 0:
                    schedule[triplet[1]]["assignments"].append(self.flat_db[triplet][0])

            for s_id, window in sorted_indexed_windows:
                if s_id > 0:
                    schedule[s_id]["window"] = window

            ax = self._plot(masses, schedule, ax=ax, path=path)

        return (fitness, eq_constraints, ineq_constraints, ax)

    def fitness(
        self,
        x,
    ):
        """
        A wrapper for the fitness function (called for evaluation only).

        #################### IMPORTANT ######################
        - The chromosome has the following format:

            - Start and end times for each station, *in order of Station ID*
            - Asteroid / station assignments with the corresponding arrival times.

            Format:
            [
                Station 1 start time, Station 1 end time,   |
                Station 2 start time, Station 2 end time,   |
                ...                                         | 2 x number of stations
                Station 11 start time, Station 11 end time, |
                Station 12 start time, Station 12 end time, |
                Asteroid ID, Station ID, Opportunity ID, |
                Asteroid ID, Station ID, Opportunity ID, |
                ...                                      | number of asteroids
                Asteroid ID, Station ID, Opportunity ID, |
                Asteroid ID, Station ID, Opportunity ID  |
            ]

        - All IDs (for asteroids, stations and opportunities) are 1-based.
            - This is particularly relevant for the opportunity IDs
            since they are converted to 0-based indices in the fitness
            evaluation function by subtracting 1.

        - Stations must be activated *sequentially* (*not* in parallel) but not necessarily in order of their ID.

        - There must be a minimal gap (called 'station gap') between the end time of one station
        and the start time of the next.

        - Every asteroid must be either asigned to a station or unassigned.

        - The asteroid / station assignments do not have to be  in any particular order,
        but all asteroid IDs must appear in the chromosome, even if some asteroids are unassigned.

            - Assigned asteroids must have corresponding Station IDs between 1 and the number of stations.
            - Unassigned asteroids must have a Station ID 0.

        ######################################################

        Args:
            x: A chromosome in the format specified above.

        Returns:
            A tuple containing the fitness followed by the equality and inequality constraints.

        """

        (fitness, eq_constraints, ineq_constraints, ax) = self._fitness_impl(x)

        return (fitness, *eq_constraints, *ineq_constraints)

    def pretty(
        self,
        x,
    ):
        """
        Fitness evaluation function with pretty printing.

        Args:
            x: A chromosome.
        """

        (_, _, _, ax) = self._fitness_impl(x, logging=True)

    def plot(
        self,
        x,
        ax=None,
        path=None,
    ):
        """
        Plot the total material masses accumulated at each station
        and the asteroid / station assignments.

        Args:
            x: A chromosome.
            ax: Plot axes. Defaults to None.
            path: A file to save the plot to. Defaults to None.

        Returns:
            Plot axes.
        """
        (_, _, _, ax) = self._fitness_impl(
            x,
            logging=False,
            plotting=True,
            ax=ax,
            path=path,
        )

        return ax

    def example(self):
        """
        An example method returning a minimal chromosome that assigns
        a single asteroid to each station.

        Returns:
            A valid chromosome.
        """

        assignments = {}

        # Iterate over a random permutation of asteroid IDs
        for ast_id in list(self.db.keys()):

            if len(assignments) == self.n_stations:
                break

            # Create a set of station IDs.
            stations = set(self.db[ast_id].keys()).difference(set(assignments.keys()))
            while len(stations) > 0:
                stat_id = stations.pop()

                # Check if the asteroid / station pair
                # is already assigned.
                if stat_id in assignments:
                    break

                opps = self.db[ast_id][stat_id]
                # Check if there are any opportunities
                for opp_idx, opp in enumerate(opps, 1):
                    # Check if the arrival time of the opportunity conflicts with
                    # the assigned asteroid / station build times
                    conflict = False
                    for _, vals in assignments.items():
                        if ast_id == vals[0] or (vals[2] - 2 * self.station_gap) <= opp[
                            0
                        ] <= (vals[3] + 2 * self.station_gap):
                            conflict = True
                            break
                    if not conflict:
                        assignments[stat_id] = [
                            ast_id,
                            opp_idx,
                            opp[0] - self.station_gap,
                            opp[0] + self.station_gap,
                        ]
                        stations = set()
                        break

        # Build the chromosome
        windows = []
        schedule = []
        for stat_id in range(1, self.n_stations + 1):
            # Station start time
            windows.append(assignments[stat_id][2])
            # Station end time
            windows.append(assignments[stat_id][3])
            # Asteroid / station assignment
            schedule.extend([assignments[stat_id][0], stat_id, assignments[stat_id][1]])

        # Stitch the windows and schedule together and create
        # a complete chromosome from this partial solution.
        chromosome = self.convert_to_chromosome(
            np.concatenate((np.array(windows), np.array(schedule)))
        )

        return chromosome

    def convert_to_chromosome(
        self,
        x,
    ):
        """
        Creates a valid chromosome from an incomplete one.

        Here, 'incomplete' means that all station windows are provided
        but only some asteroids are assigned. This method completes the
        chromosome by assigning the missing asteroids to station 0, which
        means that those asteroids will not be considered in the fitness evaluation.

        Args:
            x: Incomplete chromosome.

        Returns:
            Complete chromosome.
        """

        if len(x) < 2 * self.n_stations:
            raise ValueError(
                "The chromosome must contain at least the start and end times for the station windows."
            )

        assignments = list(x[24:])

        # Check if we have any asteroids assigned at all
        if len(assignments) > 0:
            assignments = {
                assignment[0]: assignment
                for assignment in zip(
                    assignments[::3], assignments[1::3], assignments[2::3]
                )
            }

        schedule = []
        for ast_id in range(1, self.n_asteroids + 1):
            if ast_id not in assignments:
                schedule.extend([ast_id, 0, 0])
            else:
                schedule.extend(assignments[ast_id])

        return np.concatenate((np.array(x[:24]), np.array(schedule)))

udp = trappist_schedule()
    
class fitness(object):

    def __init__(self):
        self.udp = trappist_schedule() # store original UDP
        db = self.udp.flat_db # original database
        flat = {} # flatten opportunities for all asteroids
        for row in db:
            ast, station = row[0]-1, row[1]-1
            val = db[row]
            if not ast in flat:
                flat[ast] = []
            flat[ast].append([station] + val) 
        db = [] # convert to list    
        for ast in flat:
            db.append(flat[ast])   
        self.sel_i = np.argsort(np.array([len(r) for r in db]))
        self.sel_size = np.array([len(r) for r in db], dtype=int)
        self.ub_selection = self.sel_size-1E-9 # upper bound for selection
        pad = len(max(db, key=len)) # convert to array padding with zeros
        self.db = np.array([r + [[-1,0,0,0,0]]*(pad-len(r)) for r in db])        
        lb = np.array([0] * (12 + len(db)))
        ub = np.array([80]*12 + list(self.ub_selection))
        self.bounds = Bounds(lb, ub)
        self.dim = len(lb)
        self.best = mp.RawValue(ct.c_double, np.inf)
        self.best_min = mp.RawValue(ct.c_double, 0)
        self.best_x = mp.RawArray(ct.c_double, self.dim)
        weights = np.zeros(12)
        w = 1 # decreasing weights for objective
        fac = 0.16 # prioritize stations low on resources
        inc = 0.1 # all stations are valued
        for i in range(len(weights)):
            weights[i] = w+inc; w = w*fac
        print("weights used", [round(w, 6) for w in weights])
        self.weights = -weights

    def __call__(self, x): # fitness function delegating to numba
        val, ps = _fitness(x, self.db, self.weights, self.sel_size)
        if val < self.best.value: # check for best value
            self.best.value = val          
            print([round(p, 3) for p in np.sort(ps)])
        if ps[0] > self.best_min.value: # check for best score
            self.best_min.value = ps[0]   
            self.best_x[:] = x[:]       
            print([round(p, 3) for p in ps], list(x))
        return val
    
    def windows(self, x): # compute station limits
        window_lb = x[:12].copy() # time window for stations lower bound
        window_ub = np.empty(12) # time window for stations upper bound
        wi = np.argsort(window_lb) # sort time windows
        window_lb[wi[0]] = 0 # first window has lower bound 0
        for i in range(0, 11): # set upper bound, leave 1 day between windows
            window_ub[wi[i]] = window_lb[wi[i+1]] - 1 
        window_ub[wi[11]] = 80 # last window has upper bound 80
        self.window_lb, self.window_ub = np.array(window_lb), np.array(window_ub)
        return window_lb, window_ub

    def convert(self, x): # convert argument vector for verification
        x = np.array(x)
        db0 = self.udp.flat_db # original database
        x0 = []
        window_lb, window_ub = self.windows(x) # compute station limits
        for i in range(12): # put station limits
            x0.append(window_lb[i])
            x0.append(window_ub[i])
        selection = x[12:].astype(int) # extract opportunity selection
        for i, s in enumerate(selection):
            ssize = self.sel_size[i] # start trying selection s
            l = len(x0)
            for j in range(ssize):
                r = self.db[i, s] # selected DB entry for ith asteroid
                station = int(r[0]) # to station
                time = r[1] # if time in window add resources
                if time >= window_lb[station] and time <= window_ub[station]:                   
                    for (astId, stationId, oppId) in db0: # find indices in original DB
                        if astId == i+1 and stationId == station+1: # we count from 0
                            arrival_time, mass_A, mass_B, mass_C = db0[(astId, stationId, oppId)]
                            if time == arrival_time and mass_A == r[2] and mass_B == r[3] and mass_C == r[4]:                           
                                x0.extend([astId, stationId, oppId+1]) # indices start with 1
                                break
                    break
                s = (s + 1) % ssize # if unsuccessful try other alternatives
            if len(x0) == l: # no success                
                x0.extend([i+1, 0, 1]) # assign station 0
        return x0
    
    def optimize(self, maxevals=50000000, retries=2000):     
        # BiteOpt algorithm multi threaded
        store = retry.Store(wrapper(self), self.bounds, logger=logger()) 
        x0 = None
        for i in range(1, 5): # improve incrementally           
            print("iteration", i)
            store = retry.Store(wrapper(self), self.bounds, logger=logger()) 
            retry.retry(store, Bite_cpp(maxevals,x0, M=16, popsize=3200, stall_criterion=200).minimize, 
                        num_retries=retries)
            x0 = self.best_x[:]       

def optimize():
    fit = fitness()
    fit.optimize()
    
def check():
    fit = fitness()
    x = [23.12639746143074, 49.210712151727776, 2.025271098480984, 33.70145021261418, 74.74624854353284, 55.7003375615551, 68.01733901576718, 38.34434794546488, 45.94232240067513, 64.18586868864216, 71.77249965330779, 41.20923483385521, 5.374294766228467, 10.807480433911689, 41.45436283227506, 9.120707788163473, 18.963442563558004, 35.1652835323873, 16.36241134052928, 23.083973995593297, 34.4923496005214, 28.3038806248846, 37.7908866800202, 27.285363391873247, 42.35837759628225, 48.80319538636846, 22.228894776189392, 4.861847410836416, 34.05191571768972, 37.4364516309269, 20.894222781961908, 30.36180412927237, 26.09560460330928, 66.41970995157186, 28.08206446343889, 8.077922495780404, 33.24673601479603, 14.07175866832998, 21.81948009667388, 47.646899320113036, 24.429036271236022, 21.400320205986123, 40.09732302178437, 36.205180686492575, 14.043795280271612, 9.332885295938253, 25.781077815312848, 55.0837591139832, 40.229050229008536, 3.2705796993963636, 23.944771162581848, 9.576358618908138, 0.7519414707995067, 2.165446982806674, 0.8629876235393696, 21.29757750113655, 7.896408080250771, 10.707945057879066, 19.815550345157718, 23.41302787935016, 31.79497185413374, 16.094356434749763, 34.020583208389624, 37.03451128849857, 11.655903730566314, 10.791671608206114, 7.482424158908975, 28.614287756529016, 42.73809126166453, 5.850884414166348, 36.915044881784205, 17.154641513304554, 57.52610605331465, 35.26096215030126, 25.389622893890184, 13.355590173125018, 45.463146534385324, 28.944681491785076, 15.93797193295659, 59.81309578439863, 63.6229385274899, 32.19328569599213, 29.759474767696034, 14.954722381046247, 43.24895058350935, 9.136208700797392, 47.636906520021796, 7.960593320719403, 13.293328635549479, 24.557651496572184, 29.664849566838804, 6.603626632762175, 12.342312430847668, 18.357498338611492, 25.8901467635285, 27.019652731967486, 13.949541372919242, 2.826834484452832, 32.0190188353333, 43.41349961218848, 18.10214257604606, 1.8439445952181253, 5.14433577017559, 1.380986698244865, 2.199776336744757, 3.0559899246650573, 1.5384734264023017, 2.214115742919157, 0.36114517576806726, 1.6835484762243083, 4.476955029879836, 0.8220331719324672, 3.424326563116105, 1.0944325339026684, 1.3154831966976743, 1.6394457153653743, 1.4305063655011039, 1.8825705381404303, 4.413090087199976, 0.8273519264482886, 2.6415718054606314, 1.3301528212662441, 0.5014534192509065, 1.6757035545118395, 4.655401703670822, 4.783356594747494, 4.344098919186095, 1.995702923800821, 4.648135296847233, 4.507157645115363, 3.7421118797333093, 2.492091503280891, 0.9447192437446356, 2.5754660034071484, 2.067015896422351, 1.4668100139041718, 1.4542444435562143, 17.920180424671873, 4.704880165733561, 7.7359501759003075, 8.815574182478974, 2.459770930341602, 3.9436834582426195, 1.40214595002838, 11.157803803451282, 12.072294608648924, 3.9374773813377613, 7.137937469173593, 58.934954506499245, 43.40548317889475, 1.1953354110889134, 25.18828342450627, 36.90689855326262, 17.173050000970434, 7.583979388916177, 50.875621555034925, 17.254725034256357, 7.947616367191294, 10.26204702455983, 10.716326937408422, 9.38944061302195, 10.958267911857149, 44.15723151233567, 23.03649766820622, 54.66826002640253, 29.53610944393554, 40.69988337907609, 30.592279074789086, 14.772303425574053, 21.999720735902557, 56.0181540569659, 12.22344317450934, 34.656267184843706, 2.717822194080356, 35.88332923290119, 46.045327517889206, 22.290125985871878, 31.30942614739834, 22.86084228983393, 12.765990661843, 10.013377021858844, 7.901395681474148, 20.172144420440578, 7.548987864768862, 18.821918898435598, 10.015769969653398, 45.505518895693186, 10.009061068072452, 8.375321362147785, 56.52487531358315, 11.012849533914556, 43.83733595631719, 21.638177092923232, 18.15079445504883, 26.011535256786956, 20.62698510168183, 52.389324600347244, 10.501917302649773, 33.091112360163336, 32.43690555579961, 20.007093624256537, 39.87074032707598, 13.645166819288855, 7.890085723329521, 26.89163951227164, 22.215846013217828, 19.238667319001994, 11.550993212617012, 65.39764799695308, 15.792310132095588, 27.995060280077716, 6.4975930814371265, 8.52111517675269, 7.331870231111029, 33.120084420881845, 22.417904545945348, 10.390828112603185, 4.378530398538042, 15.495896962252754, 1.5985287874363858, 19.531033786857122, 35.74698718189253, 14.379068916899715, 16.966069584534594, 17.5220677454396, 20.91210403659696, 12.059319441191414, 5.769824713126678, 12.923817576414422, 32.257457542227066, 40.88025031411448, 24.277606265011695, 29.79021384654935, 0.1461186361811261, 51.94260158217971, 35.774921841951794, 12.540798049361959, 13.109201370956796, 40.02107633973688, 6.90381914501392, 25.095369782701766, 6.749546040537202, 16.908716684486343, 16.34782003912364, 44.826675412101984, 21.401773193830593, 15.69599198839753, 10.506886333478398, 15.391223912867654, 21.71372724287722, 27.349253869930504, 3.776812993503014, 10.571647226703462, 4.677762499142719, 14.137040274950778, 67.3184876225992, 25.376368656459668, 11.252172650331858, 42.801100263406084, 8.747899000950754, 0.3235066708666327, 19.191346103276363, 22.976752925466155, 26.896812787932298, 8.726066423490053, 18.606237669321416, 30.118997676853578, 9.398360202814118, 10.57810916809076, 14.231705146188897, 9.248502654445975, 19.8062876454034, 7.708078393390288, 7.900169465667145, 20.75466717853311, 7.8539637762555365, 26.840724774383624, 53.73020675259066, 12.52903809721433, 12.068847699411107, 19.425863251373308, 19.10181265510097, 24.401174964570185, 4.287875123081391, 20.63927079190342, 42.81335113619154, 8.629860963234394, 30.905279212840753, 9.762475067899304, 20.600867656528994, 21.927031193817317, 21.9062100227658, 13.567942281140088, 35.56708604513299, 13.994307686157903, 6.992667807819466, 10.182608790296152, 24.421480507123814, 27.23231643468027, 16.539906154792526, 10.60596318127324, 39.56949802026564, 46.68185224882708, 16.790267771758163, 6.289885663632137, 9.055617813320588, 17.817164716112657, 16.85540590369484, 18.788303932780202, 3.1713695980656693, 12.652967972137501, 14.45667176992051, 43.91781053038059, 10.854577314457533, 6.54333131018741, 22.434286816576225, 12.901447996009038, 15.291099631900535, 5.93488171107279, 12.259391810855295, 28.376335316675117, 8.733866985582837, 3.855350648083705, 11.909422048612953, 38.783350646107266, 47.791591127829534, 14.249229279889516, 20.258997485815883, 24.797390172729717, 2.3703977826326073, 19.69226085121024, 13.722560731392694, 36.019956088993546, 65.10016899167518, 49.778608661434106, 7.3134346336476135, 24.606311679932684, 51.97523273571923, 7.638676638278817, 37.02110977859669, 19.03577555276804, 22.521818547976796, 36.09170326817167, 12.458651254537406, 30.502970320265828, 1.7020519629127606, 42.33034444525069, 12.407620449485231, 24.398772129987016, 26.246220709699347, 33.20476192777754, 7.079747483780409]
    print(fit(np.array(x)))
    x0 = fit.convert(x)
    print (fit.udp.fitness(x0))
    fit.udp.pretty(x0)
    fit.udp.plot(x0)
    plt.show()
        
from numba import njit, numba
@njit(fastmath=True, cache=True)    
def _fitness(x, db, weights, sel_size):
    window_lb = x[:12].copy() # time window for stations lower bound
    wi = np.argsort(window_lb) # sort time windows
    window_ub = np.empty(12) # time window for stations upper bound
    window_lb[wi[0]] = 0 # first window has lower bound 0
    for i in range(0, 11): # set upper bound, leave 1 day between windows
        window_ub[wi[i]] = window_lb[wi[i+1]] - 1 
    window_ub[wi[11]] = 80 # last window has upper bound 80
    prod = np.zeros((12,3)) # accumulate resources for stations
    selection = x[12:].astype(numba.int32) # @njit requires numba types    
    for i, s in enumerate(selection):
        ssize = sel_size[i] # start trying selection s
        for _ in range(ssize):
            r = db[i, s] # selected DB entry for ith asteroid
            station = int(r[0]) # to station
            time = r[1] # if time in window add resources
            if time >= window_lb[station] and time <= window_ub[station]:
                prod[station] += r[2:] # add resources for station_fitness
                break
            s = (s + 1) % ssize # if unsuccessful try other alternatives
    stat_prod = np.empty(12) # numba hates fromiter, so we loop
    for station in range(12): # minimum resource at station
        stat_prod[station] = min(prod[station])
    ps = np.sort(stat_prod) # sorted weighted sum of minima at station
    return np.dot(ps, weights), ps # poorest station has priority


if __name__ == '__main__':
    optimize()
    check()
    pass