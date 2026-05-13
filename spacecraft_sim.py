"""
spacecraft_sim.py
=================
Lightweight rigid-body spacecraft simulator implementing 3-DOF rotational dynamics.

Physics Model
-------------
Euler's rotational equation (body frame):

    J * dω/dt + ω × (J * ω) = τ_ctrl + τ_dist

where:
    J       — inertia matrix (3×3 diagonal), units: kg·m²
    ω       — angular velocity vector [ωx, ωy, ωz], units: rad/s
    τ_ctrl  — control torque from thruster allocation
    τ_dist  — external disturbance torque (Gaussian-Markov process)

State Vector
------------
    [θx, θy, θz, ωx, ωy, ωz]  →  6-dimensional continuous state
    θ: Euler angles (rad), ω: angular rates (rad/s)

Integration
-----------
    scipy.integrate.solve_ivp with RK45 solver for numerical stability.
    Each step() call advances the simulation by dt seconds.

Thruster Allocation
-------------------
    4 thrusters mapped via allocation matrix B (3×4).
    Thruster forces u ∈ [0, F_max] are clipped and converted to torques.

Data Flow
---------
    reset() → initial state
    step(action) → next_state, reward_info, done
    _dynamics(t, y) → dy/dt (used by solve_ivp)
"""

import numpy as np
# pyrefly: ignore [missing-import]
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

# Inertia matrix (kg·m²) — diagonal rigid body assumption
# Values from the research paper: diag(0.82, 0.95, 1.10)
J_INERTIA = np.diag([0.82, 0.95, 1.10])
J_INV     = np.linalg.inv(J_INERTIA)   # Pre-computed inverse for speed

# Thruster allocation matrix B (3×4)
# Each column = torque axis a single thruster contributes when fired at 1 N
# Rows = [x-axis torque, y-axis torque, z-axis torque]
# This is a plausible symmetric 4-thruster configuration
B_ALLOC = np.array([
    [ 0.5, -0.5,  0.5, -0.5],   # x-axis torques
    [ 0.5,  0.5, -0.5, -0.5],   # y-axis torques
    [ 0.3, -0.3, -0.3,  0.3],   # z-axis torques
])

# Thruster limits
F_MIN = 0.0    # N  (thrusters can only push, not pull)
F_MAX = 2.0    # N  maximum force per thruster

# Disturbance model parameters (Gaussian-Markov process)
DIST_SIGMA = 0.02    # noise amplitude (rad/s² units after dividing by J)
DIST_CORR  = 0.95    # temporal correlation coefficient (0 = white noise, 1 = constant)

# Simulation parameters
DT      = 0.05   # time step (seconds)
MAX_STEPS = 500  # episode length

# Angle/rate limits for done condition
ANGLE_LIMIT = np.pi   # rad  — if attitude exceeds ±π, episode terminates
RATE_LIMIT  = 10.0    # rad/s — safety cutoff

# Target attitude (default: stabilize to zero)
TARGET_ATTITUDE = np.zeros(3)


class SpacecraftSim:
    """
    3-DOF rigid-body spacecraft simulator.

    The simulator behaves like a lightweight Gym environment:
        - reset()  → returns initial observation
        - step()   → returns (obs, info_dict, done)

    Parameters
    ----------
    dt : float
        Integration time step in seconds.
    max_steps : int
        Maximum timesteps per episode before forced termination.
    target : np.ndarray, shape (3,)
        Desired attitude angles [θx, θy, θz] in radians.
    dist_sigma : float
        Standard deviation of disturbance torque noise.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(self, dt=DT, max_steps=MAX_STEPS, target=None,
                 dist_sigma=DIST_SIGMA, seed=None):
        self.dt         = dt
        self.max_steps  = max_steps
        self.target     = target if target is not None else TARGET_ATTITUDE.copy()
        self.dist_sigma = dist_sigma
        self.rng        = np.random.default_rng(seed)

        # Inertia / allocation (class-level references for convenience)
        self.J     = J_INERTIA
        self.J_inv = J_INV
        self.B     = B_ALLOC

        # State variables (initialised in reset())
        self.state        = None   # [θx, θy, θz, ωx, ωy, ωz]
        self.disturbance  = None   # current disturbance torque vector (3,)
        self.step_count   = 0
        self.prev_action  = np.zeros(4)  # previous thruster commands (for smoothness metrics)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self, init_state=None):
        """
        Reset the simulator to the beginning of a new episode.

        Parameters
        ----------
        init_state : array-like, shape (6,) or None
            If None, a small random initial attitude/rate is used.

        Returns
        -------
        obs : np.ndarray, shape (6,)
            Initial observation [θx, θy, θz, ωx, ωy, ωz].
        """
        if init_state is not None:
            self.state = np.array(init_state, dtype=float)
        else:
            # Random small initial attitude and angular velocity
            # Keeps the problem solvable: angles within ±0.5 rad, rates ±0.2 rad/s
            angles = self.rng.uniform(-0.5, 0.5, size=3)
            rates  = self.rng.uniform(-0.2, 0.2, size=3)
            self.state = np.concatenate([angles, rates])

        # Initialise disturbance as small random vector
        self.disturbance = self.rng.normal(0, self.dist_sigma, size=3)
        self.step_count  = 0
        self.prev_action = np.zeros(4)

        return self._get_obs()

    def step(self, action):
        """
        Advance the simulation by one time step.

        Parameters
        ----------
        action : array-like, shape (4,)
            Thruster force commands [u1, u2, u3, u4], clipped to [F_MIN, F_MAX].

        Returns
        -------
        obs  : np.ndarray, shape (6,) — new state observation
        info : dict — contains torques, disturbance, error, actuator info
        done : bool — True if episode should terminate
        """
        action = np.clip(np.asarray(action, dtype=float), F_MIN, F_MAX)

        # Compute control torque from thruster allocation: τ = B · u
        tau_ctrl = self.B @ action

        # Integrate dynamics over one dt using RK45
        sol = solve_ivp(
            fun      = self._dynamics,
            t_span   = (0.0, self.dt),
            y0       = self.state.copy(),
            method   = 'RK45',
            args     = (tau_ctrl,),
            max_step = self.dt / 5,   # internal sub-steps for accuracy
            dense_output=False
        )

        # Take the final integrated state
        self.state = sol.y[:, -1]

        # Wrap angles to [-π, π] to prevent unbounded growth
        self.state[:3] = self._wrap_angles(self.state[:3])

        # Update Gaussian-Markov disturbance for next step
        self.disturbance = (DIST_CORR * self.disturbance
                            + (1 - DIST_CORR) * self.rng.normal(0, self.dist_sigma, 3))

        self.step_count  += 1
        self.prev_action  = action.copy()

        # Build info dictionary for metrics logging
        error = self.state[:3] - self.target
        info  = {
            'error'      : error,
            'tau_ctrl'   : tau_ctrl,
            'tau_dist'   : self.disturbance.copy(),
            'action'     : action.copy(),
            'step'       : self.step_count,
        }

        done = self._is_done()
        return self._get_obs(), info, done

    def get_error(self):
        """Return current attitude error vector (θ - θ_target)."""
        return self.state[:3] - self.target

    def get_state(self):
        """Return full state vector [θx, θy, θz, ωx, ωy, ωz]."""
        return self.state.copy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dynamics(self, t, y, tau_ctrl):
        """
        ODE right-hand side: dy/dt = f(t, y).

        State y = [θx, θy, θz, ωx, ωy, ωz]

        Equations:
            dθ/dt = ω                                (kinematic)
            dω/dt = J⁻¹ (τ_ctrl + τ_dist - ω × Jω)  (Euler's equation)

        Parameters
        ----------
        t         : float  — current time (not used explicitly; required by solve_ivp)
        y         : array  — current state [θ, ω]
        tau_ctrl  : array  — control torque (3,)

        Returns
        -------
        dydt : array, shape (6,) — time derivatives
        """
        theta = y[:3]   # Euler angles
        omega = y[3:]   # angular velocity

        # ω × (Jω): gyroscopic / Coriolis coupling term
        Jomega   = self.J @ omega
        gyro     = np.cross(omega, Jomega)

        # Total torque acting on the body
        tau_total = tau_ctrl + self.disturbance - gyro

        # Angular acceleration: dω/dt = J⁻¹ · τ_total
        domega = self.J_inv @ tau_total

        # Angular rate (attitude kinematics): dθ/dt = ω
        dtheta = omega

        return np.concatenate([dtheta, domega])

    def _get_obs(self):
        """Return the current observation (copies state to avoid aliasing)."""
        return self.state.copy()

    def _is_done(self):
        """
        Check termination conditions:
            1. Max steps reached
            2. Attitude angle exceeds safety limit
            3. Angular rate exceeds safety limit
        """
        if self.step_count >= self.max_steps:
            return True
        if np.any(np.abs(self.state[:3]) > ANGLE_LIMIT):
            return True
        if np.any(np.abs(self.state[3:]) > RATE_LIMIT):
            return True
        return False

    @staticmethod
    def _wrap_angles(angles):
        """Wrap angle array to [-π, π]."""
        return (angles + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=== SpacecraftSim Standalone Test ===")
    sim = SpacecraftSim(seed=42)
    obs = sim.reset()
    print(f"Initial state: θ={obs[:3]}, ω={obs[3:]}")

    total_steps = 0
    for _ in range(200):
        action = np.random.uniform(0, 1, 4)
        obs, info, done = sim.step(action)
        total_steps += 1
        if done:
            break

    print(f"Ran {total_steps} steps")
    print(f"Final state:   θ={obs[:3]}, ω={obs[3:]}")
    print(f"Final error:   {info['error']}")
    print("SpacecraftSim OK")