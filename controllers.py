"""
controllers.py
==============
Three control architectures for spacecraft attitude control.

1. PIDController           — classical multi-axis PID baseline
2. StandardRLAgent         — Q-learning, reward = tracking error only
3. ConstraintAwareRLAgent  — Q-learning, reward = tracking error
                             + smoothness penalty + control effort penalty

Key Design Decision
-------------------
The ONLY difference between StandardRLAgent and ConstraintAwareRLAgent
is the reward function. All Q-learning infrastructure (tile coding, 
epsilon-greedy, Q-update) is identical. This ensures a fair comparison.

Action Space
------------
Thrusters are discretised into N_ACTIONS levels per thruster:
    [0.0, F_MAX/2, F_MAX]  → 3 levels → 3^4 = 81 joint actions

To keep memory manageable we use a factored action scheme:
    Each axis independently chooses a torque level → 3 axis actions
    Then we invert B to get plausible thruster commands.

State Space
-----------
    Continuous 6-D → TileCoder → single integer index

Data Flow
---------
    agent.select_action(obs)   → action (4,)
    agent.update(obs, action, reward, next_obs, done)  → updates Q-table
    agent.compute_reward(...)  → float reward
"""

import numpy as np
from tilecoding import TileCoder

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

# State bounds (used by tile coder)
STATE_LOW  = np.array([-np.pi, -np.pi, -np.pi, -10.0, -10.0, -10.0])
STATE_HIGH = np.array([ np.pi,  np.pi,  np.pi,  10.0,  10.0,  10.0])

# Tile coding hyper-parameters
N_TILINGS     = 16      # Increased from 8 to 16 for better generalization
TILES_PER_DIM = 10
MEMORY_SIZE   = 16384   # Increased from 4096 to 16384 to minimize hashing collisions

# Thruster action levels per thruster (0 = off, 1 = half, 2 = full)
FORCE_LEVELS  = np.array([0.0, 1.0, 2.0])   # N

# Build full 4-thruster action set: all combinations of 3 levels × 4 thrusters
# Shape: (3^4, 4) = (81, 4)
_levels = FORCE_LEVELS
ACTION_SET = np.array(
    [[a, b, c, d]
     for a in _levels for b in _levels for c in _levels for d in _levels]
)
N_ACTIONS = len(ACTION_SET)    # 81

# Q-learning hyper-parameters
ALPHA      = 0.1    # learning rate
GAMMA      = 0.99   # discount factor
EPS_START  = 1.0    # initial exploration rate
EPS_END    = 0.05   # minimum exploration rate
EPS_DECAY  = 0.9995 # per-step multiplicative decay

# Reward shaping coefficients (CA-RL only)
K1 = 5.0    # Increased from 0.5 to 5.0 to aggressively penalise jitter
K2 = 1.0    # Increased from 0.1 to 1.0 to penalise excessive thruster use


# ---------------------------------------------------------------------------
# PID Controller
# ---------------------------------------------------------------------------

class PIDController:
    """
    Multi-axis cascaded PID controller.

    Three independent PID loops (one per rotational axis) compute a
    desired torque; then pseudo-inverse of B maps torque → thruster forces.

    Parameters
    ----------
    kp, ki, kd : float or array-like (3,)
        Proportional, integral, derivative gains.
    dt : float
        Time step for discrete integration.
    target : np.ndarray (3,)
        Desired attitude [θx, θy, θz].
    force_min, force_max : float
        Thruster saturation limits.
    """

    def __init__(self, kp=2.0, ki=0.05, kd=0.8, dt=0.05,
                 target=None, force_min=0.0, force_max=2.0):
        self.kp  = np.ones(3) * kp if np.isscalar(kp) else np.asarray(kp)
        self.ki  = np.ones(3) * ki if np.isscalar(ki) else np.asarray(ki)
        self.kd  = np.ones(3) * kd if np.isscalar(kd) else np.asarray(kd)
        self.dt  = dt
        self.target    = target if target is not None else np.zeros(3)
        self.force_min = force_min
        self.force_max = force_max

        # Import B here to avoid circular imports
        from spacecraft_sim import B_ALLOC
        self.B     = B_ALLOC
        # Pseudo-inverse: maps desired torques → minimum-norm thruster forces
        self.B_inv = np.linalg.pinv(B_ALLOC)

        # Internal PID state
        self._integral    = np.zeros(3)
        self._prev_error  = np.zeros(3)

    def reset(self):
        """Clear integrators and previous error."""
        self._integral   = np.zeros(3)
        self._prev_error = np.zeros(3)

    def select_action(self, obs):
        """
        Compute thruster commands from current state.

        Parameters
        ----------
        obs : np.ndarray (6,) — [θx, θy, θz, ωx, ωy, ωz]

        Returns
        -------
        action : np.ndarray (4,) — thruster forces, clipped to [force_min, force_max]
        """
        error = obs[:3] - self.target          # attitude error
        omega = obs[3:]                        # angular rate (for damping)

        # Derivative term uses angular rate directly for noise robustness
        d_error = (error - self._prev_error) / self.dt

        # PID torque command (proportional + integral + derivative)
        tau_desired = (self.kp * error
                       + self.ki * self._integral
                       + self.kd * d_error)

        # Anti-windup: clip integral to prevent runaway accumulation
        self._integral   = np.clip(self._integral + error * self.dt, -5.0, 5.0)
        self._prev_error = error.copy()

        # Map desired torque to thruster forces via pseudo-inverse
        # τ = B·u  →  u = B⁺·τ
        u_raw  = self.B_inv @ (-tau_desired)   # negate: PID gives correction direction

        # Bias and clip to feasible positive range
        u_shifted = u_raw - u_raw.min()        # shift so minimum is 0
        action = np.clip(u_shifted, self.force_min, self.force_max)
        return action

    def update(self, *args, **kwargs):
        """No-op: PID has no learning step."""
        pass

    def compute_reward(self, *args, **kwargs):
        """No-op: PID doesn't use rewards."""
        return 0.0


# ---------------------------------------------------------------------------
# Base RL Agent (shared Q-learning infrastructure)
# ---------------------------------------------------------------------------

class _BaseRLAgent:
    """
    Abstract base for tabular Q-learning agents with tile coding.

    Sub-classes must implement compute_reward().

    Architecture
    ------------
    Q-table : np.ndarray, shape (total_features, N_ACTIONS)
        Q[s, a] = estimated discounted return from state s taking action a.

    Tile coder converts continuous state → integer index → row of Q-table.

    Update rule (Q-learning / TD(0)):
        Q[s, a] ← Q[s, a] + α [r + γ max_a' Q[s', a'] - Q[s, a]]
    """

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

        # Tile coder: continuous state → integer
        self.coder = TileCoder(
            state_low    = STATE_LOW,
            state_high   = STATE_HIGH,
            n_tilings    = N_TILINGS,
            tiles_per_dim= TILES_PER_DIM,
            memory_size  = MEMORY_SIZE
        )

        # Q-table: shape (total_features, N_ACTIONS)
        n_states = self.coder.total_features
        self.Q   = np.zeros((n_states, N_ACTIONS))

        # Exploration
        self.epsilon = EPS_START

        # Previous action (for smoothness reward)
        self.prev_action = np.zeros(4)

        # Training stats
        self.episode_rewards = []
        self._ep_reward_buf  = 0.0

    def reset(self):
        """Reset episode-level state (prev_action, reward buffer)."""
        self.prev_action  = np.zeros(4)
        self._ep_reward_buf = 0.0

    def select_action(self, obs):
        """
        ε-greedy action selection.

        With probability ε: random action (exploration)
        Otherwise         : greedy action argmax_a Q[s, a] (exploitation)

        Parameters
        ----------
        obs : np.ndarray (6,)

        Returns
        -------
        action : np.ndarray (4,) — thruster forces
        """
        s_idx = self.coder.encode(obs)

        if self.rng.random() < self.epsilon:
            a_idx = self.rng.integers(0, N_ACTIONS)
        else:
            a_idx = int(np.argmax(self.Q[s_idx]))

        return ACTION_SET[a_idx].copy()

    def update(self, obs, action, reward, next_obs, done):
        """
        One-step Q-learning update.

        TD target: r + γ max_a' Q[s', a']  (0 if done)
        TD error:  target - Q[s, a]
        Q update:  Q[s, a] += α * TD_error

        Parameters
        ----------
        obs      : current state (6,)
        action   : action taken  (4,)
        reward   : scalar reward
        next_obs : next state    (6,)
        done     : bool
        """
        # Find action index
        diffs = np.linalg.norm(ACTION_SET - action, axis=1)
        a_idx = int(np.argmin(diffs))

        s_idx  = self.coder.encode(obs)
        s1_idx = self.coder.encode(next_obs)

        # Bootstrap target
        if done:
            target = reward
        else:
            target = reward + GAMMA * np.max(self.Q[s1_idx])

        # Temporal difference update
        self.Q[s_idx, a_idx] += ALPHA * (target - self.Q[s_idx, a_idx])

        # Decay exploration rate
        self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)

        # Track reward
        self._ep_reward_buf += reward
        if done:
            self.episode_rewards.append(self._ep_reward_buf)
            self._ep_reward_buf = 0.0

        # Save action for next smoothness computation
        self.prev_action = action.copy()

    def compute_reward(self, obs, action, info):
        """Override in sub-classes."""
        raise NotImplementedError

    def save_qtable(self, path):
        """Persist Q-table to disk."""
        np.save(path, self.Q)
        print(f"  Q-table saved -> {path}")

    def load_qtable(self, path):
        """Load Q-table from disk."""
        self.Q = np.load(path)
        print(f"  Q-table loaded <- {path}")


# ---------------------------------------------------------------------------
# Standard RL Agent
# ---------------------------------------------------------------------------

class StandardRLAgent(_BaseRLAgent):
    """
    Standard Q-learning agent.

    Reward function:
        r = -||error||₂

    This purely penalises attitude tracking error. No penalty on actuator
    behaviour → tends to generate jittery/oscillatory commands.

    Parameters
    ----------
    seed : int or None
    """

    def __init__(self, seed=None):
        super().__init__(seed=seed)

    def compute_reward(self, obs, action, info):
        """
        Standard reward: negative L2 norm of attitude error + dense shaping.
        """
        error  = info['error']
        err_norm = np.linalg.norm(error)
        
        # 1. Performance penalty (negative distance)
        r_perf = -err_norm
        
        # 2. Living penalty (constant cost per step to encourage speed)
        r_living = -0.1
        
        # 3. Success bonus (reward for staying within 0.1 rad deadband)
        r_success = 10.0 if err_norm < 0.1 else 0.0
        
        return float(r_perf + r_living + r_success)


# ---------------------------------------------------------------------------
# Constraint-Aware RL Agent
# ---------------------------------------------------------------------------

class ConstraintAwareRLAgent(_BaseRLAgent):
    """
    Constraint-aware Q-learning agent with smoothness reward shaping.

    Reward function (from paper Eq. 2–3):
        R = R_performance + R_smoothness

    where:
        R_performance = -||error||₂

        R_smoothness  = -k1 * ||a_t - a_{t-1}||²   ← penalise rapid switching
                        -k2 * ||a_t||²              ← penalise control effort

    k1 and k2 are configurable weights:
        - k1 (default 0.5): controls smoothness (anti-switching penalty)
        - k2 (default 0.1): controls energy efficiency (effort penalty)

    This is the ONLY structural difference from StandardRLAgent.
    Everything else (tile coder, Q-table shape, update rule) is identical.

    Parameters
    ----------
    k1   : float — smoothness penalty weight
    k2   : float — control effort penalty weight
    seed : int or None
    """

    def __init__(self, k1=K1, k2=K2, seed=None):
        super().__init__(seed=seed)
        self.k1 = k1
        self.k2 = k2

    def compute_reward(self, obs, action, info):
        """
        Constraint-aware reward: tracking + smoothness + effort + bonuses.
        """
        error = info['error']
        err_norm = np.linalg.norm(error)

        # 1. Performance: negative attitude error magnitude
        r_perf = -err_norm

        # 2. Smoothness: penalty for large change in actuator commands
        delta_a  = action - self.prev_action
        r_smooth = -self.k1 * float(np.dot(delta_a, delta_a))

        # 3. Control effort: penalty for large actuator magnitudes
        r_effort = -self.k2 * float(np.dot(action, action))
        
        # 4. Living penalty (constant cost per step to encourage speed)
        r_living = -0.1
        
        # 5. Success bonus (reward for staying within 0.1 rad deadband)
        r_success = 10.0 if err_norm < 0.1 else 0.0

        return r_perf + r_smooth + r_effort + r_living + r_success


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import numpy as np
    from spacecraft_sim import SpacecraftSim

    print("=== Controller Standalone Test ===")
    sim = SpacecraftSim(seed=0)

    # Test PID
    pid = PIDController()
    obs = sim.reset(init_state=[0.3, -0.2, 0.1, 0, 0, 0])
    for _ in range(10):
        action = pid.select_action(obs)
        obs, info, done = sim.step(action)
    print(f"PID final error: {info['error']}")

    # Test Standard RL
    rl = StandardRLAgent(seed=1)
    obs = sim.reset(init_state=[0.3, -0.2, 0.1, 0, 0, 0])
    rl.reset()
    for _ in range(10):
        action = rl.select_action(obs)
        reward = rl.compute_reward(obs, action, info)
        obs2, info, done = sim.step(action)
        rl.update(obs, action, reward, obs2, done)
        obs = obs2
    print(f"StandardRL final error: {info['error']}")

    # Test CA-RL
    carl = ConstraintAwareRLAgent(seed=2)
    obs = sim.reset(init_state=[0.3, -0.2, 0.1, 0, 0, 0])
    carl.reset()
    for _ in range(10):
        action = carl.select_action(obs)
        reward = carl.compute_reward(obs, action, info)
        obs2, info, done = sim.step(action)
        carl.update(obs, action, reward, obs2, done)
        obs = obs2
    print(f"CA-RL final error: {info['error']}")

    print("All controllers OK")