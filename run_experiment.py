"""
run_experiment.py
=================
Master orchestration script for the spacecraft RL experiment.

Purpose
-------
A single entry point that runs the complete research pipeline from scratch:

    python run_experiment.py

Phases
------
    PHASE 1  — Train Standard RL agent (2000 episodes)
    PHASE 2  — Train Constraint-Aware RL agent (2000 episodes)
    PHASE 3  — Save Q-tables to results/qtables/
    PHASE 4  — Evaluate all three controllers (50 episodes each)
    PHASE 5  — Compute research metrics (SSE, ATV, Energy, Settling, DR)
    PHASE 6  — Generate publication-quality plots
    PHASE 7  — Export CSV results
    PHASE 8  — Print final comparison summary table

Connection to Other Files
--------------------------
    spacecraft_sim.py  — simulator instantiation in every phase
    controllers.py     — PID, StandardRLAgent, ConstraintAwareRLAgent
    train.py           — reuses train_agent() helper
    evaluate.py        — reuses evaluate_controller() helper
    metrics.py         — aggregate_metrics(), export_metrics_csv()
    plotting.py        — save_all_plots()

Execution Flow
--------------
    1. create_results_dirs()
    2. PHASE 1: train Standard RL → save Q-table + training metrics JSON
    3. PHASE 2: train CA-RL      → save Q-table + training metrics JSON
    4. PHASE 3: verify Q-tables on disk
    5. PHASE 4: run evaluation episodes for PID / Standard RL / CA-RL
               (epsilon=0 for RL agents → deterministic greedy policy)
    6. PHASE 5: call metrics.aggregate_metrics() on each controller's batch
    7. PHASE 6: call plotting.save_all_plots()
    8. PHASE 7: call metrics.export_metrics_csv()
    9. PHASE 8: print_summary_table()

Command-line Arguments
----------------------
    --episodes    INT   Training episodes per agent        (default 2000)
    --eval-trials INT   Evaluation episodes per controller (default 50)
    --seed        INT   Master random seed                  (default 42)
    --log-every   INT   Print training log every N episodes (default 100)
    --skip-train        Load existing Q-tables, skip training
    --results-dir STR   Output directory                    (default results)

Debugging Tips
--------------
- Use --episodes 100 --eval-trials 5 for a quick sanity-check run.
- If evaluation metrics all look identical, confirm epsilon=0.0 is set
  before evaluate_controller() is called.
- If Q-tables are not found on a --skip-train run, the script will
  fall back to random (untrained) agents and warn you loudly.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Project modules ──────────────────────────────────────────────────────────
from spacecraft_sim import SpacecraftSim
from controllers    import PIDController, StandardRLAgent, ConstraintAwareRLAgent
from train          import train_agent
from evaluate       import evaluate_controller
import metrics  as M
import plotting as P


# ─────────────────────────────────────────────────────────────────────────────
# Directory helpers
# ─────────────────────────────────────────────────────────────────────────────

def create_results_dirs(base: Path) -> dict:
    """
    Create the canonical results directory tree.

    Returns
    -------
    dirs : dict with keys 'plots', 'csv', 'qtables', 'logs'
    """
    dirs = {
        'base':    base,
        'plots':   base / 'plots',
        'csv':     base / 'csv',
        'qtables': base / 'qtables',
        'logs':    base / 'logs',
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    print(f"[setup] Results directory: {base.resolve()}")
    return dirs


# ─────────────────────────────────────────────────────────────────────────────
# Logging helpers
# ─────────────────────────────────────────────────────────────────────────────

def _banner(text: str, width: int = 70) -> None:
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def _phase(n: int, text: str) -> None:
    print(f"\n{'-'*70}")
    print(f"  PHASE {n}: {text}")
    print(f"{'-'*70}")


def _save_json(data: dict, path: Path) -> None:
    """Serialise a dict to JSON, converting numpy scalars to Python floats."""
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=_convert)
    print(f"  JSON saved -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Training phase
# ─────────────────────────────────────────────────────────────────────────────

def run_training(agent, agent_name: str, sim: SpacecraftSim,
                 n_episodes: int, log_every: int,
                 dirs: dict) -> dict:
    """
    Train one RL agent and persist Q-table + training log.

    Parameters
    ----------
    agent       : StandardRLAgent or ConstraintAwareRLAgent
    agent_name  : short identifier used in file names ('standard_rl' / 'ca_rl')
    sim         : SpacecraftSim instance
    n_episodes  : total training episodes
    log_every   : progress logging frequency
    dirs        : dict returned by create_results_dirs()

    Returns
    -------
    train_metrics : dict from train.py::train_agent()
    """
    print(f"\nTraining {agent_name} for {n_episodes} episodes ...")
    t0 = time.time()

    train_metrics = train_agent(
        agent         = agent,
        sim           = sim,
        n_episodes    = n_episodes,
        eval_interval = log_every,
        verbose       = True,
    )

    elapsed = time.time() - t0
    train_metrics['training_time_min'] = elapsed / 60.0
    final_eval_error = train_metrics['eval_errors'][-1] if train_metrics['eval_errors'] else float('nan')
    print(f"  Finished in {elapsed/60:.1f} min  |  "
          f"final eps = {agent.epsilon:.4f}  |  "
          f"final eval error = {final_eval_error:.4f}")
    return train_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation phase
# ─────────────────────────────────────────────────────────────────────────────

def _extract_eval_arrays(eval_metrics_obj):
    """
    Extract raw per-episode error and action arrays from an
    EvaluationMetrics object (defined in evaluate.py).

    Returns (list_of_error_arrays, list_of_action_arrays).
    """
    return eval_metrics_obj.error_history, eval_metrics_obj.action_history


def run_evaluation(controller, ctrl_name: str, sim: SpacecraftSim,
                   n_trials: int) -> tuple:
    """
    Run deterministic evaluation and return both the EvaluationMetrics
    object and the aggregated stats dict from metrics.py.

    Parameters
    ----------
    controller : PID / StandardRLAgent / ConstraintAwareRLAgent
    ctrl_name  : display name ('PID', 'Standard RL', 'CA-RL')
    sim        : SpacecraftSim
    n_trials   : number of evaluation episodes

    Returns
    -------
    (eval_obj, agg_stats)
        eval_obj  : evaluate.EvaluationMetrics — raw trajectories
        agg_stats : dict — from metrics.aggregate_metrics()
    """
    print(f"  Evaluating {ctrl_name} ({n_trials} episodes) ...")
    eval_obj = evaluate_controller(controller, sim, n_trials=n_trials, verbose=False)

    # Build full per-episode arrays from EvaluationMetrics
    errors_list  = [np.array(e) for e in eval_obj.error_history]
    actions_list = [np.array(a) for a in eval_obj.action_history]

    # Fill remaining episodes from the scalar accumulators
    # (error_history only stores the first 5 trajectories to save RAM;
    #  use the scalar lists for the remaining episodes)
    full_sse     = np.array(eval_obj.steady_state_error)
    full_atv     = np.array(eval_obj.actuator_tv)
    full_energy  = np.array(eval_obj.control_energy)
    full_settle  = np.array(eval_obj.settling_time)
    full_dr      = np.array(eval_obj.disturbance_rejection)

    # Build agg_stats directly from the accumulated scalar lists so we
    # use ALL episodes, not just the 5 stored trajectories
    def _s(arr, name):
        return {f'{name}_mean': float(np.mean(arr)),
                f'{name}_std':  float(np.std(arr)),
                f'{name}_min':  float(np.min(arr)),
                f'{name}_max':  float(np.max(arr))}

    agg_stats = {}
    agg_stats.update(_s(full_sse,    'sse'))
    agg_stats.update(_s(full_settle, 'settling_time'))
    agg_stats.update(_s(full_atv,    'atv'))
    agg_stats.update(_s(full_energy, 'control_energy'))
    agg_stats.update(_s(full_dr,     'disturbance_rejection'))
    agg_stats['n_episodes'] = n_trials

    return eval_obj, agg_stats


# ─────────────────────────────────────────────────────────────────────────────
# Summary printing
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_table(results_by_controller: dict) -> None:
    """
    Print a formatted comparison table to stdout.

    Uses the same metric keys as metrics.summarize_results().
    Highlights the best (lowest) value in each row with an asterisk.
    """
    df = M.summarize_results(results_by_controller)

    _banner("FINAL EXPERIMENT SUMMARY")
    print(df.to_string(index=False))

    # Numeric-only table for "best" highlighting
    metric_keys = ['sse', 'settling_time', 'atv', 'control_energy', 'disturbance_rejection']
    metric_labels = {
        'sse':                   'Steady-State Error',
        'settling_time':         'Settling Time',
        'atv':                   'Actuator Total Variation',
        'control_energy':        'Control Energy',
        'disturbance_rejection': 'Disturbance Rejection',
    }
    names = list(results_by_controller.keys())

    print("\n  Best controller per metric (* = best):")
    print(f"  {'Metric':<35} {'Best Controller':<20} {'Value':>10}")
    print(f"  {'-'*35} {'-'*20} {'-'*10}")
    for key in metric_keys:
        means = {n: results_by_controller[n].get(f'{key}_mean', np.inf) for n in names}
        best  = min(means, key=means.get)
        print(f"  {metric_labels[key]:<35} {best:<20} {means[best]:>10.4f}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── CLI ──────────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description='Full spacecraft RL experiment pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--agent',       type=str, default='both',
                        choices=['standard', 'ca-rl', 'both'],
                        help='Which agent to train/evaluate')
    parser.add_argument('--episodes',    type=int, default=100000,
                        help='Training episodes per RL agent')
    parser.add_argument('--eval-trials', type=int, default=50,
                        help='Evaluation episodes per controller')
    parser.add_argument('--seed',        type=int, default=42,
                        help='Master random seed')
    parser.add_argument('--log-every',   type=int, default=1000,
                        help='Training log interval (episodes)')
    parser.add_argument('--skip-train',  action='store_true',
                        help='Skip training; load existing Q-tables')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Root directory for all outputs')
    args = parser.parse_args()

    t_global = time.time()

    # -- Setup -----------------------------------------------------------------
    _banner("SPACECRAFT RL EXPERIMENT - FULL PIPELINE")
    print(f"  Training episodes  : {args.episodes}")
    print(f"  Evaluation trials  : {args.eval_trials}")
    print(f"  Random seed        : {args.seed}")
    print(f"  Skip training      : {args.skip_train}")
    print(f"  Results directory  : {args.results_dir}")

    dirs = create_results_dirs(Path(args.results_dir))

    # Shared simulator (seeded for reproducibility)
    sim = SpacecraftSim(seed=args.seed)

    # Containers populated across phases
    reward_histories      = {}   # {name: [episode rewards]}
    eval_objects          = {}   # {name: EvaluationMetrics}
    results_by_ctrl       = {}   # {name: agg_stats dict}

    # ── PHASES 1 & 2: Training ────────────────────────────────────────────────
    if not args.skip_train:

        # ── PHASE 1: Standard RL ─────────────────────────────────────────────
        if args.agent in ['standard', 'both']:
            _phase(1, "Train Standard RL Agent")
            std_agent = StandardRLAgent(seed=args.seed)
            std_train = run_training(std_agent, 'Standard RL', sim,
                                     args.episodes, args.log_every, dirs)
            reward_histories['Standard RL'] = std_train['episode_rewards']
            _save_json(std_train, dirs['logs'] / 'standard_rl_training.json')
        else:
            std_agent = StandardRLAgent(seed=args.seed) # Initialize for eval phase if needed

        # ── PHASE 2: Constraint-Aware RL ─────────────────────────────────────
        if args.agent in ['ca-rl', 'both']:
            _phase(2, "Train Constraint-Aware RL Agent")
            ca_agent = ConstraintAwareRLAgent(seed=args.seed + 1)
            ca_train = run_training(ca_agent, 'CA-RL', sim,
                                    args.episodes, args.log_every, dirs)
            reward_histories['CA-RL'] = ca_train['episode_rewards']
            _save_json(ca_train, dirs['logs'] / 'ca_rl_training.json')
        else:
            ca_agent = ConstraintAwareRLAgent(seed=args.seed + 1)

        # ── PHASE 3: Save Q-tables ────────────────────────────────────────────
        _phase(3, "Save Q-tables")
        std_qt_path = dirs['qtables'] / 'standard_rl_qtable.npy'
        ca_qt_path  = dirs['qtables'] / 'ca_rl_qtable.npy'
        std_agent.save_qtable(str(std_qt_path))
        ca_agent.save_qtable(str(ca_qt_path))

        # Also save training reward curves
        P.plot_training_rewards(reward_histories, dirs['plots'])

    else:
        # ── PHASE 1-3 (skipped): Load Q-tables ───────────────────────────────
        _phase(1, "Skip Training — Loading Q-tables")
        std_qt_path = dirs['qtables'] / 'standard_rl_qtable.npy'
        ca_qt_path  = dirs['qtables'] / 'ca_rl_qtable.npy'

        std_agent = StandardRLAgent(seed=args.seed)
        ca_agent  = ConstraintAwareRLAgent(seed=args.seed + 1)

        if std_qt_path.exists():
            std_agent.load_qtable(str(std_qt_path))
            print(f"  Loaded Standard RL Q-table <- {std_qt_path}")
        else:
            print(f"  ⚠  WARNING: {std_qt_path} not found — using untrained agent!")

        if ca_qt_path.exists():
            ca_agent.load_qtable(str(ca_qt_path))
            print(f"  Loaded CA-RL Q-table <- {ca_qt_path}")
        else:
            print(f"  ⚠  WARNING: {ca_qt_path} not found — using untrained agent!")

    # ── PHASE 4: Evaluation ───────────────────────────────────────────────────
    _phase(4, f"Evaluate All Controllers ({args.eval_trials} episodes each)")

    # Use a fresh simulator with a different seed for evaluation
    eval_sim = SpacecraftSim(seed=args.seed + 999)

    # PID
    pid = PIDController()
    pid_eval, pid_stats = run_evaluation(pid, 'PID', eval_sim, args.eval_trials)
    eval_objects['PID']    = pid_eval
    results_by_ctrl['PID'] = pid_stats
    print(f"    PID  -> SSE={pid_stats['sse_mean']:.4f}  "
          f"ATV={pid_stats['atv_mean']:.2f}")

    # Standard RL — set epsilon=0 for deterministic greedy policy
    if args.agent in ['standard', 'both']:
        std_agent.epsilon = 0.0
        rl_eval, rl_stats = run_evaluation(std_agent, 'Standard RL', eval_sim, args.eval_trials)
        eval_objects['Standard RL']    = rl_eval
        results_by_ctrl['Standard RL'] = rl_stats
        print(f"    Std RL -> SSE={rl_stats['sse_mean']:.4f}  "
              f"ATV={rl_stats['atv_mean']:.2f}")

    # CA-RL — set epsilon=0
    if args.agent in ['ca-rl', 'both']:
        ca_agent.epsilon = 0.0
        ca_eval, ca_stats = run_evaluation(ca_agent, 'CA-RL', eval_sim, args.eval_trials)
        eval_objects['CA-RL']    = ca_eval
        results_by_ctrl['CA-RL'] = ca_stats
        print(f"    CA-RL -> SSE={ca_stats['sse_mean']:.4f}  "
              f"ATV={ca_stats['atv_mean']:.2f}")

    # ── PHASE 5: Compute additional metrics / build per-episode arrays ────────
    _phase(5, "Assemble Per-Episode Metric Arrays for Plotting")

    atv_by_episode      = {}
    energy_by_episode   = {}
    settling_by_episode = {}
    mc_results          = {}
    error_traj          = {}
    action_traj         = {}

    for name, ev_obj in eval_objects.items():
        atv_by_episode[name]      = ev_obj.actuator_tv
        energy_by_episode[name]   = ev_obj.control_energy
        settling_by_episode[name] = [
            t * eval_sim.dt for t in ev_obj.settling_time  # convert steps -> seconds
        ]
        mc_results[name] = {
            'sse': ev_obj.steady_state_error,
            'atv': ev_obj.actuator_tv,
        }
        # Use stored sample trajectories for line plots
        if len(ev_obj.error_history) > 0:
            error_traj[name]  = ev_obj.error_history[0]
        if len(ev_obj.action_history) > 0:
            action_traj[name] = ev_obj.action_history[0]

    # ── PHASE 6: Generate all plots ───────────────────────────────────────────
    _phase(6, "Generate Publication-Quality Plots")
    P.save_all_plots(
        reward_histories      = reward_histories,
        error_trajectories    = error_traj,
        action_trajectories   = action_traj,
        atv_by_episode        = atv_by_episode,
        energy_by_episode     = energy_by_episode,
        settling_by_episode   = settling_by_episode,
        results_by_controller = results_by_ctrl,
        mc_results            = mc_results,
        save_dir              = dirs['plots'],
    )

    # ── PHASE 7: Export CSV results ───────────────────────────────────────────
    _phase(7, "Export CSV Results")
    M.export_metrics_csv(results_by_ctrl, dirs['csv'])

    # Also save raw evaluation summary JSON
    _save_json(results_by_ctrl, dirs['logs'] / 'evaluation_summary.json')

    # Per-controller raw episode data CSV
    for name, ev_obj in eval_objects.items():
        rows = []
        n = len(ev_obj.steady_state_error)
        for i in range(n):
            rows.append({
                'episode':               i + 1,
                'sse':                   ev_obj.steady_state_error[i],
                'settling_time_steps':   ev_obj.settling_time[i],
                'actuator_tv':           ev_obj.actuator_tv[i],
                'control_energy':        ev_obj.control_energy[i],
                'disturbance_rejection': ev_obj.disturbance_rejection[i],
                'peak_error':            ev_obj.peak_error[i],
                'episode_length':        ev_obj.episode_lengths[i],
            })
        df = pd.DataFrame(rows)
        safe_name = name.lower().replace(' ', '_').replace('-', '_')
        csv_path = dirs['csv'] / f'{safe_name}_episodes.csv'
        df.to_csv(csv_path, index=False)
        print(f"  {name} episodes CSV -> {csv_path}")

    # ── PHASE 8: Print final summary ──────────────────────────────────────────
    _phase(8, "Final Comparison Summary")
    print_summary_table(results_by_ctrl)

    elapsed_total = time.time() - t_global
    _banner(f"EXPERIMENT COMPLETE  ({elapsed_total/60:.1f} min total)")
    print(f"  Plots   -> {dirs['plots']}")
    print(f"  CSVs    -> {dirs['csv']}")
    print(f"  Q-tables-> {dirs['qtables']}")
    print(f"  Logs    -> {dirs['logs']}")
    print()


if __name__ == '__main__':
    main()