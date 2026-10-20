"""moretry: constraint handling of the weighted scalarization and value limits."""
import numpy as np
from scipy.optimize import Bounds

from fcmaes import moretry
from fcmaes.optimizer import Bite_cpp


def test_satisfied_constraints_add_nothing():
    weights = np.array([0.5, 2.0, 10.0, 10.0])
    objectives = np.array([1.5, 0.25])
    plain = moretry.mo_wrapper(lambda x: objectives, weights[:2], 0)
    expected = plain.eval([0.0])
    # However far inside the feasible region, satisfied constraints leave the
    # scalar unchanged (before, their values entered the norm).
    for slack in [0.0, -1e-3, -25.0, -1e6]:
        wrapper = moretry.mo_wrapper(lambda x: np.append(objectives, [slack, -1.0]), weights, 2)
        assert wrapper.eval([0.0]) == expected


def test_violations_enter_the_norm_and_add_their_weight():
    weights = np.array([0.5, 2.0, 10.0, 3.0])
    y = np.array([1.5, 0.25, 0.2, -4.0])
    wrapper = moretry.mo_wrapper(lambda x: y, weights, 2)
    norm = np.sqrt((0.5 * 1.5) ** 2 + (2.0 * 0.25) ** 2 + (10.0 * 0.2) ** 2)
    assert np.isclose(wrapper.eval([0.0]), norm + 10.0)
    # A larger violation is worse, so the optimizer is led to feasibility.
    worse = moretry.mo_wrapper(lambda x: y + [0, 0, 1.0, 0], weights, 2)
    assert worse.eval([0.0]) > wrapper.eval([0.0])


def constrained(x):
    # Two objectives and one constraint x0 + x1 >= 0.5.
    return np.array([x[0] ** 2 + x[1] ** 2, (x[0] - 1) ** 2 + (x[1] - 1) ** 2, 0.5 - x[0] - x[1]])


def test_value_limits_for_the_objectives_only():
    # value_limits with one entry per objective used to raise an IndexError
    # inside the retry loop, which dropped every result of a constrained run.
    xs, ys = moretry.minimize(constrained, Bounds([0, 0], [1, 1]),
                              Bounds([0.1, 0.1, 10], [1, 1, 10]), ncon=1,
                              value_limits=[10.0, 10.0], num_retries=4, workers=1,
                              optimizer=Bite_cpp(300))
    assert len(xs) == 4
    assert (ys[:, 2] <= 1e-6).all()
