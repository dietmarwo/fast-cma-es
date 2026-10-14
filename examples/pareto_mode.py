"""Track all feasible MODE evaluations without changing its population update."""
import numpy as np
from numpy.random import Generator, PCG64DXSM
from scipy.optimize import Bounds
from fcmaes.modecpp import MODE_C
from fcmaes.pareto import ParetoArchive2D


def run(generations=40, track=True):
    solver = MODE_C(2, 1, Bounds([0.0, 0.0], [1.0, 1.0]),
                    popsize=32, rg=Generator(PCG64DXSM(42)))
    archive = ParetoArchive2D([0.0, 0.0], [2.0, 2.0], payload_dim=2)
    trace = []
    for _ in range(generations):
        xs = solver.ask()
        ys = np.column_stack((np.square(xs).sum(axis=1),
                              np.square(1.0 - xs).sum(axis=1),
                              xs.sum(axis=1) - 1.5))
        trace.extend(zip(xs.copy(), ys.copy()))
        if track:
            keep = np.isfinite(ys).all(axis=1) & (ys[:, 2] <= 0.0)
            archive.add_batch(ys[keep, :2], xs[keep])
        solver.tell(ys)
    return solver.population(), archive, trace


if __name__ == "__main__":
    population, archive, trace = run()
    print(f"evaluations={len(trace)} population={len(population)} "
          f"archive={len(archive)} HV={archive.hypervolume():.12f}")
