"""The Langevin kernels (spec 004 FR-010; research R13, R30): numba over a chunk of individuals [i0, i1).

The SDE is SRK's: dX = (eta t - beta X / (X + kappa)) dt + sqrt(2 epsilon) dW, held at 0 by the wall, killed at
X >= xc. Per individual i, with its own parameters (single- and multi-parameter heterogeneity run the same code):
- **one step h** (dt, or dt/2 on the coupled rung); the age at step k is k h, computed from k, never accumulated, and
  there is no outer/inner split (no nsteps, time_step_multiplier or step_size: register B2);
- **the noise:** one Marsaglia polar pair per dt: the dt rung (split = 1) steps with (xi_a + xi_b) / sqrt 2, the coupled
  dt/2 rung (split = 2) with xi_a then xi_b, so both follow one Brownian path (research R16);
- **the wall** after the step: "clamp" is SRK's max(0, .) (`SR_hetro.py:383, 420, 481`; D-410, the default), "reflect"
  is |.| (the symmetrized Euler; research R30);
- **the kill:** X >= xc at the step's end; with the bridge kernel, also the Brownian-bridge crossing probability
  exp(-2 (xc - x)(xc - x~) / (2 epsilon h)) when both ends are below xc and x > 0, evaluated (and a uniform drawn from
  the "bridge" stream) only under SRK's e^-30 cutoff (`SR_hetro.py:503-504`), after the wall;
- **the death time** is the end of the step at which the kill happens; an external death time t_ext (the competing
  hazard, drawn outside) stops the path at t_ext if it comes first; a survivor at t_max is censored there.
- **the path buffer** (research R24): X[i, j] is the state at the end of step rec_steps[j] (in this rung's steps), read
  with no draw, and NaN iff the individual is dead by then (T <= a: the death times are step ends, the external one is
  exact); age 0 reads the start, 0.
No random number is drawn for anything but the step's noise and the bridge's test.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

from sraging.langevin.rng import normal_pair, seed, uniform

WALLS = {"clamp": 0, "reflect": 1}
KERNELS = ("bridge", "euler")
CENSORED, SR_DEATH, EXTERNAL = 0, 1, 2              # a raw time's kind
CUTOFF = 30.0                                       # SRK's: the bridge is evaluated only while 2 dx1 dx2 < 30 var
_INV_SQRT2 = 1.0 / math.sqrt(2.0)


@njit(inline="always")
def step(x, k, h, eta, beta, eps, kappa, xi, wall):
    """One Euler step of length h from age k h, then the wall."""
    x_new = x + h * (eta * (k * h) - beta * x / (x + kappa)) + math.sqrt(2.0 * eps * h) * xi
    if wall == 0:
        return max(x_new, 0.0)
    return abs(x_new)


@njit(inline="always")
def bridge_p(x, x_new, xc, eps, h):
    """The Brownian-bridge crossing probability of a step that ends below xc, or -1 where it is not evaluated (an end
    at or above xc, x at the wall, or beyond the e^-30 cutoff)."""
    if x < xc and x_new < xc and x > 0.0:
        var = 2.0 * eps * h
        dx1 = xc - x
        dx2 = xc - x_new
        if var > 0.0 and 2.0 * dx1 * dx2 < CUTOFF * var:
            return math.exp(-2.0 * dx1 * dx2 / var)
    return -1.0


@njit
def run_chunk(i0, i1, seeds, bridge_seeds, eta, beta, eps, kappa, xc, t_ext, dt, t_max, wall, bridge, split,
              rec_steps):
    """Individuals i0..i1-1: (raw times, kinds, the damage where each path stopped, the paths X at rec_steps).
    `split` 1 runs the dt rung, 2 the coupled dt/2 rung (h = dt / 2); rec_steps (ascending, in steps of h) may be
    empty."""
    n = i1 - i0
    nrec = rec_steps.size
    times = np.empty(n)
    kinds = np.empty(n, dtype=np.int8)
    x_end = np.empty(n)
    X = np.full((n, nrec), np.nan)
    state = np.zeros(4, dtype=np.uint64)
    bstate = np.zeros(4, dtype=np.uint64)
    h = dt / split
    for j in range(n):
        i = i0 + j
        seed(state, seeds[i])
        seed(bstate, bridge_seeds[i])
        e, b, ep, ka, c, te = eta[i], beta[i], eps[i], kappa[i], xc[i], t_ext[i]
        x = 0.0
        k = 0
        xb = 0.0
        t = t_max
        kind = CENSORED
        ptr = 0
        while ptr < nrec and rec_steps[ptr] == 0:                  # age 0: the start
            X[j, ptr] = 0.0
            ptr += 1
        while True:
            t_next = (k + 1) * h
            if t_next > t_max:                                     # no full step left: a survivor, unless t_ext
                if te <= t_max:
                    t = te
                    kind = EXTERNAL
                break
            if te <= t_next:                                       # the competing death comes first
                t = te
                kind = EXTERNAL
                break
            if split == 1:
                a, bb = normal_pair(state)
                xi = (a + bb) * _INV_SQRT2
            elif k % 2 == 0:
                xi, xb = normal_pair(state)
            else:
                xi = xb
            x_new = step(x, k, h, e, b, ep, ka, xi, wall)
            if x_new >= c:
                x = x_new
                t = t_next
                kind = SR_DEATH
                break
            if bridge:
                p = bridge_p(x, x_new, c, ep, h)
                if p >= 0.0 and uniform(bstate) < p:
                    x = x_new
                    t = t_next
                    kind = SR_DEATH
                    break
            x = x_new
            k += 1
            while ptr < nrec and rec_steps[ptr] == k:              # read the state; nothing is drawn
                X[j, ptr] = x
                ptr += 1
        times[j] = t
        kinds[j] = kind
        x_end[j] = x
    return times, kinds, x_end, X
