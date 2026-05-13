"""
evaluate.py
===========
Comprehensive evaluation of all three controllers:
  - PID Controller
  - Standard RL Agent
  - Constraint-Aware RL Agent

Computes metrics from the research proposal (Table 1):
  1. Steady-State Error     — final tracking accuracy
  2. Settling Time          — time to remain within 2% of target
  3. Actuator Total Variation — sum of absolute control changes
  4. Control Energy         — integral of squared control inputs
  5. Disturbance Rejection  — stability under noise

Usage:
    python evaluate.py --n-trials 20 --results-dir results
"""

import argparse
import numpy as np
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm
import pandas as pd

from spacecraft_sim import SpacecraftSim
from controllers import PIDController, StandardRLAgent, ConstraintAwareRLAgent


class EvaluationMetrics:
    """Container for all evaluation metrics."""
    
    def __init__(self):
        self.steady_state_error = []     # rad (final 10% of episode)
        self.settling_time = []          # timesteps to reach 2% band
        self.actuator_tv = []            # total variation: sum |Δu|
        self.control_energy = []         # sum u²
        self.disturbance_rejection = []  # RMS error under disturbance
        self.episode_lengths = []
        self.peak_error = []             # max error during episode
        
        # Time series for plotting
        self.error_history = []          # trajectory of errors
        self.action_history = []         # trajectory of actions
    
    def add_episode(self, errors, actions, settling_step):
        """
        Add metrics from one episode.
        
        Parameters
        ----------
        errors : list of float
            Tracking error magnitude at each timestep
        actions : list of array (4,)
            Actuator commands at each timestep
        settling_step : int or None
            First timestep when error < 2% threshold (None if never settled)
        """
        errors = np.array(errors)
        actions = np.array(actions)  # shape (T, 4)
        
        # 1. Steady-state error: mean over final 10% of episode
        n_final = max(1, len(errors) // 10)
        self.steady_state_error.append(float(np.mean(errors[-n_final:])))
        
        # 2. Settling time
        if settling_step is not None:
            self.settling_time.append(settling_step)
        else:
            self.settling_time.append(len(errors))  # never settled
        
        # 3. Actuator total variation: sum of |Δu| across all thrusters
        delta_u = np.diff(actions, axis=0)  # shape (T-1, 4)
        tv = float(np.sum(np.abs(delta_u)))
        self.actuator_tv.append(tv)
        
        # 4. Control energy: sum of u²
        energy = float(np.sum(actions ** 2))
        self.control_energy.append(energy)
        
        # 5. Disturbance rejection: RMS error (proxy for stability)
        self.disturbance_rejection.append(float(np.sqrt(np.mean(errors ** 2))))
        
        # Additional metrics
        self.episode_lengths.append(len(errors))
        self.peak_error.append(float(np.max(errors)))
        
        # Store trajectories (for first few episodes only, to save memory)
        if len(self.error_history) < 5:
            self.error_history.append(errors)
            self.action_history.append(actions)
    
    def summary(self):
        """Return dict of mean ± std for all metrics."""
        return {
            'steady_state_error_mean': float(np.mean(self.steady_state_error)),
            'steady_state_error_std': float(np.std(self.steady_state_error)),
            'settling_time_mean': float(np.mean(self.settling_time)),
            'settling_time_std': float(np.std(self.settling_time)),
            'actuator_tv_mean': float(np.mean(self.actuator_tv)),
            'actuator_tv_std': float(np.std(self.actuator_tv)),
            'control_energy_mean': float(np.mean(self.control_energy)),
            'control_energy_std': float(np.std(self.control_energy)),
            'disturbance_rejection_mean': float(np.mean(self.disturbance_rejection)),
            'disturbance_rejection_std': float(np.std(self.disturbance_rejection)),
            'episode_length_mean': float(np.mean(self.episode_lengths)),
            'peak_error_mean': float(np.mean(self.peak_error)),
        }


def evaluate_controller(controller, sim, n_trials=20, verbose=True):
    """
    Evaluate a controller across multiple test episodes.
    
    Parameters
    ----------
    controller : PIDController, StandardRLAgent, or ConstraintAwareRLAgent
    sim : SpacecraftSim
    n_trials : int
        Number of test episodes
    verbose : bool
    
    Returns
    -------
    metrics : EvaluationMetrics
    """
    metrics = EvaluationMetrics()
    
    # Threshold for "settled": within 2% of zero error
    # Since target is [0,0,0], this is 0.02 rad per axis
    settle_threshold = 0.02
    settle_duration = 10  # must stay below threshold for this many steps
    
    iterator = tqdm(range(n_trials), desc="Evaluating", disable=not verbose)
    
    for trial in iterator:
        obs = sim.reset()
        controller.reset()
        
        episode_errors = []
        episode_actions = []
        settling_step = None
        consecutive_settled = 0
        done = False
        
        while not done:
            action = controller.select_action(obs)
            next_obs, info, done = sim.step(action)
            
            error_mag = np.linalg.norm(info['error'])
            episode_errors.append(error_mag)
            episode_actions.append(action.copy())
            
            # Check for settling
            if settling_step is None:
                if error_mag < settle_threshold:
                    consecutive_settled += 1
                    if consecutive_settled >= settle_duration:
                        settling_step = len(episode_errors) - settle_duration
                else:
                    consecutive_settled = 0
            
            obs = next_obs
        
        metrics.add_episode(episode_errors, episode_actions, settling_step)
    
    return metrics


def create_comparison_table(results_dict, save_path):
    """
    Create a formatted comparison table (LaTeX-ready).
    
    Parameters
    ----------
    results_dict : dict
        {controller_name: metrics.summary()}
    save_path : Path
    """
    rows = []
    
    metric_names = {
        'steady_state_error': 'Steady-State Error (rad)',
        'settling_time': 'Settling Time (steps)',
        'actuator_tv': 'Actuator Total Variation',
        'control_energy': 'Control Energy',
        'disturbance_rejection': 'Disturbance Rejection (RMS)'
    }
    
    for metric_key, metric_label in metric_names.items():
        row = {'Metric': metric_label}
        for controller_name, summary in results_dict.items():
            mean = summary[f'{metric_key}_mean']
            std = summary[f'{metric_key}_std']
            row[controller_name] = f'{mean:.4f} ± {std:.4f}'
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Save as CSV
    csv_path = save_path.with_suffix('.csv')
    df.to_csv(csv_path, index=False)
    print(f"  Comparison table → {csv_path}")
    
    # Save as LaTeX
    latex_path = save_path.with_suffix('.tex')
    latex_str = df.to_latex(index=False, escape=False, column_format='l' + 'c'*len(results_dict))
    with open(latex_path, 'w') as f:
        f.write(latex_str)
    print(f"  LaTeX table → {latex_path}")
    
    # Print to console
    print("\n" + "="*70)
    print("COMPARISON TABLE")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70 + "\n")
    
    return df


def plot_comparative_metrics(results_dict, save_path):
    """
    Bar charts comparing all controllers across all metrics.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Controller Performance Comparison', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    metrics_to_plot = [
        ('steady_state_error', 'Steady-State Error (rad)', 'lower is better'),
        ('settling_time', 'Settling Time (steps)', 'lower is better'),
        ('actuator_tv', 'Actuator Total Variation', 'lower is better'),
        ('control_energy', 'Control Energy', 'lower is better'),
        ('disturbance_rejection', 'Disturbance Rejection (RMS)', 'lower is better'),
        ('episode_length', 'Episode Length (steps)', 'context'),
    ]
    
    controllers = list(results_dict.keys())
    colors = {'PID': 'tab:orange', 'Standard RL': 'tab:blue', 'CA-RL': 'tab:green'}
    
    for idx, (metric_key, ylabel, note) in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        means = [results_dict[c][f'{metric_key}_mean'] for c in controllers]
        stds = [results_dict[c][f'{metric_key}_std'] for c in controllers]
        
        bars = ax.bar(controllers, means, yerr=stds, capsize=5, 
                      color=[colors.get(c, 'gray') for c in controllers],
                      alpha=0.7, edgecolor='black')
        
        ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel}')
        ax.grid(axis='y', alpha=0.3)
        
        # Annotate values on bars
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mean:.3f}',
                   ha='center', va='bottom', fontsize=9)
        
        # Add subtitle note
        ax.text(0.5, 0.95, f'({note})', transform=ax.transAxes,
               ha='center', va='top', fontsize=8, style='italic', color='gray')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Comparison plots → {save_path}")
    plt.close()


def plot_trajectory_comparison(metrics_dict, save_path):
    """
    Plot sample trajectories for visual comparison.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    colors = {'PID': 'tab:orange', 'Standard RL': 'tab:blue', 'CA-RL': 'tab:green'}
    
    # Error trajectories
    ax = axes[0]
    for controller_name, metrics in metrics_dict.items():
        if len(metrics.error_history) > 0:
            error_traj = metrics.error_history[0]  # first episode
            timesteps = np.arange(len(error_traj))
            ax.plot(timesteps, error_traj, label=controller_name, 
                   color=colors.get(controller_name, 'gray'), linewidth=2, alpha=0.8)
    
    ax.axhline(y=0.02, color='red', linestyle='--', linewidth=1, alpha=0.5, 
              label='Settling threshold (2%)')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Tracking Error (rad)')
    ax.set_title('Sample Trajectory: Tracking Error')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Actuator commands (thruster 1 as example)
    ax = axes[1]
    for controller_name, metrics in metrics_dict.items():
        if len(metrics.action_history) > 0:
            action_traj = metrics.action_history[0]  # first episode
            thruster_1 = action_traj[:, 0]  # first thruster
            timesteps = np.arange(len(thruster_1))
            ax.plot(timesteps, thruster_1, label=f'{controller_name} (T1)', 
                   color=colors.get(controller_name, 'gray'), linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Thruster Force (N)')
    ax.set_title('Sample Trajectory: Thruster 1 Command')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Trajectory plots → {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate spacecraft controllers')
    parser.add_argument('--n-trials', type=int, default=20,
                       help='Number of test episodes per controller')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory containing trained models')
    parser.add_argument('--seed', type=int, default=123,
                       help='Random seed for evaluation')
    parser.add_argument('--pid-only', action='store_true',
                       help='Evaluate PID only (no RL)')
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("SPACECRAFT CONTROLLER EVALUATION")
    print("=" * 70)
    print(f"Test episodes:  {args.n_trials}")
    print(f"Seed:           {args.seed}")
    print(f"Results dir:    {results_dir}")
    print()
    
    # Create simulator
    sim = SpacecraftSim(seed=args.seed)
    
    all_results = {}
    all_metrics = {}
    
    # -------------------------------------------------------------------------
    # 1. Evaluate PID Controller
    # -------------------------------------------------------------------------
    print("Evaluating PID Controller...")
    print("-" * 70)
    pid = PIDController()
    pid_metrics = evaluate_controller(pid, sim, args.n_trials)
    all_results['PID'] = pid_metrics.summary()
    all_metrics['PID'] = pid_metrics
    print("✓ PID evaluation complete\n")
    
    if args.pid_only:
        print("PID-only mode: skipping RL agents")
    else:
        # ---------------------------------------------------------------------
        # 2. Evaluate Standard RL Agent
        # ---------------------------------------------------------------------
        print("Evaluating Standard RL Agent...")
        print("-" * 70)
        standard_rl = StandardRLAgent(seed=args.seed)
        qtable_path = results_dir / 'standard_rl_qtable.npy'
        
        if qtable_path.exists():
            standard_rl.load_qtable(qtable_path)
            standard_rl.epsilon = 0.0  # greedy policy for evaluation
            rl_metrics = evaluate_controller(standard_rl, sim, args.n_trials)
            all_results['Standard RL'] = rl_metrics.summary()
            all_metrics['Standard RL'] = rl_metrics
            print("✓ Standard RL evaluation complete\n")
        else:
            print(f"⚠ Warning: {qtable_path} not found. Train first with train.py\n")
        
        # ---------------------------------------------------------------------
        # 3. Evaluate Constraint-Aware RL Agent
        # ---------------------------------------------------------------------
        print("Evaluating Constraint-Aware RL Agent...")
        print("-" * 70)
        ca_rl = ConstraintAwareRLAgent(seed=args.seed + 1)
        qtable_path = results_dir / 'ca_rl_qtable.npy'
        
        if qtable_path.exists():
            ca_rl.load_qtable(qtable_path)
            ca_rl.epsilon = 0.0  # greedy policy
            carl_metrics = evaluate_controller(ca_rl, sim, args.n_trials)
            all_results['CA-RL'] = carl_metrics.summary()
            all_metrics['CA-RL'] = carl_metrics
            print("✓ CA-RL evaluation complete\n")
        else:
            print(f"⚠ Warning: {qtable_path} not found. Train first with train.py\n")
    
    # -------------------------------------------------------------------------
    # Save and visualize results
    # -------------------------------------------------------------------------
    if len(all_results) > 0:
        # Save raw metrics
        summary_path = results_dir / 'evaluation_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Evaluation summary → {summary_path}")
        
        # Create comparison table
        table_path = results_dir / 'comparison_table'
        create_comparison_table(all_results, table_path)
        
        # Create comparison plots
        plot_path = results_dir / 'comparison_plots.png'
        plot_comparative_metrics(all_results, plot_path)
        
        # Create trajectory plots
        traj_path = results_dir / 'trajectory_comparison.png'
        plot_trajectory_comparison(all_metrics, traj_path)
    
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()