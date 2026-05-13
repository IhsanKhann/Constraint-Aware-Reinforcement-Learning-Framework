"""
metrics.py
==========
Pure-function metric library for spacecraft controller evaluation.

Purpose
-------
This module provides clean, standalone NumPy functions for every metric
described in the research proposal (Table 1). It is intentionally
decoupled from the simulator and controllers so it can be called from
anywhere (run_experiment.py, notebooks, unit tests).

Connection to Other Files
--------------------------
- Called by run_experiment.py after each evaluation episode batch.
- Accepts raw arrays produced by evaluate.py's EvaluationMetrics class.
- Outputs are passed to plotting.py and exported as CSV via pandas.

Execution Flow
--------------
1. Collect error[], action[] arrays from an evaluation episode.
2. Call individual compute_*() functions on those arrays.
3. Call aggregate_metrics() to merge per-episode scalars → mean±std dict.
4. Call summarize_results() to build a human-readable DataFrame.
5. Call export_metrics_csv() to persist results to disk.

Implementation Choices
-----------------------
- All functions are stateless and accept plain NumPy arrays.
- Edge cases (empty arrays, NaN, zero-length episodes) return safe
  sentinel values (0.0 or np.nan) rather than raising exceptions.
- Units follow the simulator: radians, seconds, Newtons.
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def compute_sse(errors: np.ndarray) -> float:
    """
    Compute Steady-State Error (SSE).

    Definition
    ----------
    Mean tracking error magnitude over the final 10 % of the episode.
    Using the tail of the trajectory isolates "settled" performance from
    transient response, consistent with the 2%-band settling convention.

    Parameters
    ----------
    errors : np.ndarray, shape (T,)
        Scalar tracking error magnitude (L2 norm of attitude error) at
        each timestep.  All values should be >= 0.

    Returns
    -------
    sse : float
        Mean error in the final 10 % window (radians).
        Returns 0.0 for empty or length-1 arrays.

    Examples
    --------
    >>> errors = np.linspace(1.0, 0.01, 500)
    >>> compute_sse(errors)   # ≈ 0.01
    """
    if len(errors) == 0:
        return 0.0
    n_tail = max(1, len(errors) // 10)      # at least 1 sample
    return float(np.mean(errors[-n_tail:]))


def compute_settling_time(errors: np.ndarray,
                          threshold: float = 0.02,
                          duration: int = 10,
                          dt: float = 0.05) -> float:
    """
    Compute Settling Time.

    Definition
    ----------
    The earliest time (seconds) after which the tracking error remains
    continuously below `threshold` for at least `duration` timesteps.
    This matches the "within 2 % of target for N steps" criterion from
    classical control analysis.

    Parameters
    ----------
    errors : np.ndarray, shape (T,)
        Scalar tracking error magnitude at each timestep.
    threshold : float
        Error magnitude below which the system is considered "settled"
        (default 0.02 rad ≈ 1.1°, the 2 % band).
    duration : int
        Number of consecutive timesteps that must stay below threshold
        (default 10 steps = 0.5 s at dt=0.05).
    dt : float
        Simulation timestep in seconds (default 0.05 s).

    Returns
    -------
    settling_time : float
        Time in seconds when the system first settles.
        Returns episode_length * dt if the system never settles.

    Debugging Tip
    -------------
    If all three controllers return episode_length for settling time,
    the threshold is too tight for the current gains/training level.
    Try raising threshold to 0.05 temporarily to sanity-check.
    """
    if len(errors) == 0:
        return 0.0

    below = errors < threshold          # bool array

    # Sliding window: find first index where the next `duration` steps are all True
    T = len(errors)
    for t in range(T - duration + 1):
        if np.all(below[t : t + duration]):
            return float(t * dt)        # return time in seconds

    return float(T * dt)                # never settled → full episode length


def compute_atv(actions: np.ndarray, dt: float = 0.05) -> float:
    """
    Compute Actuator Total Variation (ATV).

    Definition
    ----------
    ATV = Σ_{t=1}^{T-1} ||a_t - a_{t-1}||_1

    The L1 norm sums absolute changes across all thrusters at each step,
    then accumulates over the full episode. ATV directly quantifies
    actuator switching / wear — the key metric CA-RL aims to minimize.

    Parameters
    ----------
    actions : np.ndarray, shape (T, 4)
        Thruster force commands at each timestep (Newtons).
        Each row is [u1, u2, u3, u4].
    dt : float
        Timestep (not used in standard ATV but kept for API consistency).

    Returns
    -------
    atv : float
        Total variation across all thrusters and all timesteps.
        Returns 0.0 for arrays with fewer than 2 timesteps.

    Notes
    -----
    Lower ATV → smoother actuator profile → less mechanical wear.
    CA-RL should achieve significantly lower ATV than Standard RL.

    Debugging Tip
    -------------
    If CA-RL has higher ATV than Standard RL, check that the smoothness
    reward weights k1 and k2 in controllers.py are large enough, and
    that the agent has trained for enough episodes.
    """
    if len(actions) < 2:
        return 0.0
    actions = np.asarray(actions, dtype=float)
    delta   = np.diff(actions, axis=0)          # shape (T-1, 4)
    return float(np.sum(np.abs(delta)))


def compute_control_energy(actions: np.ndarray, dt: float = 0.05) -> float:
    """
    Compute Control Energy.

    Definition
    ----------
    E = Σ_t ||a_t||² × Δt

    Integrates the squared actuator magnitudes over time. This is the
    discrete approximation of the L2-energy integral ∫ ||u||² dt.

    Parameters
    ----------
    actions : np.ndarray, shape (T, 4)
        Thruster force commands (Newtons).
    dt : float
        Simulation timestep in seconds (default 0.05).

    Returns
    -------
    energy : float
        Total control energy in N²·s.
        Returns 0.0 for empty arrays.

    Notes
    -----
    Minimising control energy reduces fuel consumption — relevant for
    real spacecraft missions with limited propellant.
    CA-RL's effort penalty (k2 * ||a||²) directly discourages high energy.
    """
    if len(actions) == 0:
        return 0.0
    actions = np.asarray(actions, dtype=float)
    return float(np.sum(actions ** 2) * dt)


def compute_disturbance_rejection(errors: np.ndarray,
                                  disturbance_start: int = None) -> float:
    """
    Compute Disturbance Rejection capability.

    Definition
    ----------
    Maximum attitude deviation (peak error) after an injected disturbance.
    Lower values indicate better disturbance attenuation.

    If disturbance_start is provided, only the post-disturbance window is
    analysed. Otherwise the full-episode peak is returned as a conservative
    proxy (suitable when separate disturbance episodes are not available).

    Parameters
    ----------
    errors : np.ndarray, shape (T,)
        Scalar tracking error magnitudes.
    disturbance_start : int or None
        Timestep index when disturbance was injected.
        If None, uses the full trajectory (conservative proxy).

    Returns
    -------
    rejection : float
        Peak error after disturbance injection (radians).
        Returns 0.0 for empty arrays.

    Notes
    -----
    The simulator applies a Gaussian-Markov disturbance throughout every
    episode (dist_sigma=0.02). This function therefore reflects the
    controller's general robustness rather than a one-shot injection test
    unless you explicitly record disturbance_start.
    """
    if len(errors) == 0:
        return 0.0
    errors = np.asarray(errors, dtype=float)

    if disturbance_start is not None:
        window = errors[disturbance_start:]
        if len(window) == 0:
            return 0.0
        return float(np.max(window))

    return float(np.max(errors))


# ---------------------------------------------------------------------------
# Aggregation and summary helpers
# ---------------------------------------------------------------------------

def aggregate_metrics(episodes_errors: list,
                      episodes_actions: list,
                      dt: float = 0.05) -> dict:
    """
    Compute all metrics for a batch of evaluation episodes and return
    mean ± std statistics.

    Parameters
    ----------
    episodes_errors : list of np.ndarray
        One error array per episode, each shape (T_i,).
    episodes_actions : list of np.ndarray
        One action array per episode, each shape (T_i, 4).
    dt : float
        Simulation timestep.

    Returns
    -------
    stats : dict
        Keys: '<metric>_mean', '<metric>_std', '<metric>_min', '<metric>_max'
        for each of: sse, settling_time, atv, control_energy,
        disturbance_rejection.

    Debugging Tip
    -------------
    If std >> mean for any metric, the controller behaviour varies
    dramatically across episodes — investigate initial-condition sensitivity.
    """
    n = len(episodes_errors)
    if n == 0:
        return {}

    sse_vals   = np.array([compute_sse(e)                    for e in episodes_errors])
    settle_vals= np.array([compute_settling_time(e, dt=dt)   for e in episodes_errors])
    atv_vals   = np.array([compute_atv(a, dt=dt)             for a in episodes_actions])
    energy_vals= np.array([compute_control_energy(a, dt=dt)  for a in episodes_actions])
    dr_vals    = np.array([compute_disturbance_rejection(e)   for e in episodes_errors])

    def _stats(arr, name):
        return {
            f'{name}_mean': float(np.mean(arr)),
            f'{name}_std' : float(np.std(arr)),
            f'{name}_min' : float(np.min(arr)),
            f'{name}_max' : float(np.max(arr)),
        }

    stats = {}
    stats.update(_stats(sse_vals,    'sse'))
    stats.update(_stats(settle_vals, 'settling_time'))
    stats.update(_stats(atv_vals,    'atv'))
    stats.update(_stats(energy_vals, 'control_energy'))
    stats.update(_stats(dr_vals,     'disturbance_rejection'))
    stats['n_episodes'] = n

    return stats


def summarize_results(results_by_controller: dict) -> pd.DataFrame:
    """
    Build a publication-ready summary DataFrame.

    Parameters
    ----------
    results_by_controller : dict
        {controller_name: aggregate_metrics() output}
        Example keys: 'PID', 'Standard RL', 'CA-RL'

    Returns
    -------
    df : pd.DataFrame
        Rows = metrics, Columns = controllers.
        Cell format: 'mean ± std'

    Example
    -------
    >>> results = {
    ...     'PID':         aggregate_metrics(pid_errors, pid_actions),
    ...     'Standard RL': aggregate_metrics(rl_errors,  rl_actions),
    ...     'CA-RL':       aggregate_metrics(carl_errors, carl_actions),
    ... }
    >>> df = summarize_results(results)
    >>> print(df.to_string())
    """
    metric_labels = {
        'sse':                   'Steady-State Error (rad)',
        'settling_time':         'Settling Time (s)',
        'atv':                   'Actuator Total Variation',
        'control_energy':        'Control Energy (N²·s)',
        'disturbance_rejection': 'Disturbance Rejection (rad)',
    }

    rows = []
    for key, label in metric_labels.items():
        row = {'Metric': label}
        for ctrl_name, stats in results_by_controller.items():
            m = stats.get(f'{key}_mean', np.nan)
            s = stats.get(f'{key}_std',  np.nan)
            row[ctrl_name] = f'{m:.4f} ± {s:.4f}'
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def export_metrics_csv(results_by_controller: dict,
                       output_dir: str | Path,
                       filename: str = 'metrics_summary.csv') -> Path:
    """
    Export aggregated metrics to CSV files.

    Saves two files:
      1. <filename>        — human-readable mean±std table
      2. metrics_raw.csv   — raw per-metric mean/std/min/max for every controller

    Parameters
    ----------
    results_by_controller : dict
        {controller_name: aggregate_metrics() output}
    output_dir : str or Path
        Directory to write CSV files into.
    filename : str
        Name of the formatted summary CSV.

    Returns
    -------
    summary_path : Path
        Absolute path to the written summary CSV.

    Debugging Tip
    -------------
    Open metrics_raw.csv in a spreadsheet to quickly spot which controller
    has the highest variance (std column) for each metric.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Human-readable summary (mean ± std)
    summary_df = summarize_results(results_by_controller)
    summary_path = output_dir / filename
    summary_df.to_csv(summary_path, index=False)
    print(f"  Metrics summary CSV → {summary_path}")

    # 2. Raw numeric table (for downstream plotting / statistics)
    raw_rows = []
    metric_keys = ['sse', 'settling_time', 'atv', 'control_energy', 'disturbance_rejection']
    for ctrl_name, stats in results_by_controller.items():
        for key in metric_keys:
            raw_rows.append({
                'controller': ctrl_name,
                'metric':     key,
                'mean':       stats.get(f'{key}_mean', np.nan),
                'std':        stats.get(f'{key}_std',  np.nan),
                'min':        stats.get(f'{key}_min',  np.nan),
                'max':        stats.get(f'{key}_max',  np.nan),
                'n_episodes': stats.get('n_episodes',  0),
            })

    raw_path = output_dir / 'metrics_raw.csv'
    pd.DataFrame(raw_rows).to_csv(raw_path, index=False)
    print(f"  Metrics raw CSV    → {raw_path}")

    return summary_path