"""
plotting.py
===========
Publication-quality plot generation for the spacecraft RL experiment.

Purpose
-------
Centralises every matplotlib figure produced by the experiment so that
run_experiment.py can call save_all_plots() in one line without caring
about axis formatting details.

Connection to Other Files
--------------------------
- Receives training reward histories from train.py / run_experiment.py.
- Receives aggregated metrics dicts from metrics.py.
- Receives raw error/action trajectory arrays from evaluate.py.
- All figures are written to  results/plots/  (created automatically).

Execution Flow
--------------
1. run_experiment.py calls individual plot_*() functions after each phase.
2. Alternatively, call save_all_plots() to regenerate everything at once.
3. All functions accept a `save_dir` argument; figures are saved as PNG.

Formatting Conventions
----------------------
- 150 dpi for screen; raise to 300 for journal submission.
- Color palette consistent across all figures:
    PID         → tab:orange
    Standard RL → tab:blue
    CA-RL       → tab:green
- Grid lines at alpha=0.3 (subtle, not distracting).
- Tight layout + bbox_inches='tight' to avoid clipped labels.

Debugging Tips
--------------
- If a plot shows flat lines for RL agents, training likely did not converge.
  Check that Q-tables were loaded before evaluation (epsilon=0 required).
- If bars in bar charts are equal height, ensure results_dict is populated
  with data from all three controllers before calling the function.
"""

import numpy as np
# pyrefly: ignore [missing-import]
import matplotlib
matplotlib.use('Agg')           # headless backend — safe for servers / CI
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
import matplotlib.patches as mpatches
from pathlib import Path


# ---------------------------------------------------------------------------
# Shared style constants
# ---------------------------------------------------------------------------

COLORS = {
    'PID':         'tab:orange',
    'Standard RL': 'tab:blue',
    'CA-RL':       'tab:green',
}

# Fallback color cycle for unknown controller names
_FALLBACK_COLORS = ['tab:red', 'tab:purple', 'tab:brown', 'tab:pink']

DPI          = 150
FONT_SIZE    = 11
TITLE_SIZE   = 13
LEGEND_SIZE  = 10
LINE_WIDTH   = 2.0
GRID_ALPHA   = 0.3
BAR_ALPHA    = 0.75


def _get_color(name, idx=0):
    """Return consistent color for a controller name."""
    return COLORS.get(name, _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)])


def _apply_style(ax, title, xlabel, ylabel, legend=True, grid=True):
    """Apply common formatting to an Axes object."""
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold', pad=8)
    ax.set_xlabel(xlabel, fontsize=FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE)
    if legend:
        ax.legend(fontsize=LEGEND_SIZE, framealpha=0.9)
    if grid:
        ax.grid(alpha=GRID_ALPHA, linestyle='--')
    ax.tick_params(labelsize=FONT_SIZE - 1)


def _savefig(fig, path):
    """Save figure and close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Plot saved → {path}")


# ---------------------------------------------------------------------------
# 1. Training reward curves
# ---------------------------------------------------------------------------

def plot_training_rewards(reward_histories: dict,
                          save_dir: str | Path,
                          smoothing_window: int = 50) -> None:
    """
    Plot smoothed episode reward curves for all trained RL agents.

    Parameters
    ----------
    reward_histories : dict
        {agent_name: list_of_episode_rewards}
        Example: {'Standard RL': [...], 'CA-RL': [...]}
    save_dir : str or Path
        Directory where the figure is saved.
    smoothing_window : int
        Rolling-average window for noise reduction (default 50 episodes).

    Output
    ------
    results/plots/training_rewards.png

    Notes
    -----
    Raw rewards are plotted at 20 % opacity behind the smoothed curve so
    the reader can see variance without it dominating the figure.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for idx, (name, rewards) in enumerate(reward_histories.items()):
        rewards = np.array(rewards, dtype=float)
        color   = _get_color(name, idx)
        episodes = np.arange(1, len(rewards) + 1)

        # Raw rewards (faint)
        ax.plot(episodes, rewards, color=color, alpha=0.15, linewidth=0.8)

        # Smoothed rewards
        if len(rewards) >= smoothing_window:
            kernel   = np.ones(smoothing_window) / smoothing_window
            smoothed = np.convolve(rewards, kernel, mode='valid')
            ep_smooth = np.arange(smoothing_window, len(rewards) + 1)
            ax.plot(ep_smooth, smoothed, color=color, linewidth=LINE_WIDTH,
                    label=f'{name} (smoothed, w={smoothing_window})')
        else:
            ax.plot(episodes, rewards, color=color, linewidth=LINE_WIDTH, label=name)

    _apply_style(ax,
                 title='Training Reward Curves',
                 xlabel='Episode',
                 ylabel='Episode Reward')
    fig.tight_layout()
    _savefig(fig, Path(save_dir) / 'training_rewards.png')


# ---------------------------------------------------------------------------
# 2. Tracking error over time (sample trajectories)
# ---------------------------------------------------------------------------

def plot_tracking_error(error_trajectories: dict,
                        save_dir: str | Path,
                        settle_threshold: float = 0.02) -> None:
    """
    Plot attitude tracking error trajectories for all controllers.

    Parameters
    ----------
    error_trajectories : dict
        {controller_name: np.ndarray shape (T,)}
        One representative (e.g. first evaluation) episode per controller.
    save_dir : str or Path
    settle_threshold : float
        Horizontal dashed line showing the 2 % settling band.

    Output
    ------
    results/plots/tracking_error.png
    """
    fig, ax = plt.subplots(figsize=(11, 5))

    for idx, (name, errors) in enumerate(error_trajectories.items()):
        errors = np.asarray(errors, dtype=float)
        t      = np.arange(len(errors))
        ax.plot(t, errors, color=_get_color(name, idx),
                linewidth=LINE_WIDTH, label=name)

    ax.axhline(settle_threshold, color='red', linestyle='--',
               linewidth=1.2, alpha=0.7, label=f'Settling threshold ({settle_threshold} rad)')

    _apply_style(ax,
                 title='Tracking Error Comparison (Sample Episode)',
                 xlabel='Timestep',
                 ylabel='Attitude Error ||e|| (rad)')
    fig.tight_layout()
    _savefig(fig, Path(save_dir) / 'tracking_error.png')


# ---------------------------------------------------------------------------
# 3. ATV comparison
# ---------------------------------------------------------------------------

def plot_atv_comparison(atv_by_episode: dict,
                        save_dir: str | Path) -> None:
    """
    Box-plot of per-episode Actuator Total Variation for all controllers.

    Parameters
    ----------
    atv_by_episode : dict
        {controller_name: list of ATV values (one per evaluation episode)}
    save_dir : str or Path

    Output
    ------
    results/plots/atv_comparison.png

    Why Box-Plot?
    -------------
    ATV varies across episodes because initial conditions are randomised.
    A box plot shows median + IQR so the reader can judge both typical
    performance and variance simultaneously.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    names  = list(atv_by_episode.keys())
    values = [np.asarray(atv_by_episode[n], dtype=float) for n in names]
    colors = [_get_color(n, i) for i, n in enumerate(names)]

    # Panel A: box plot
    ax = axes[0]
    bp = ax.boxplot(values, patch_artist=True, notch=False,
                    medianprops={'color': 'black', 'linewidth': 2})
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(BAR_ALPHA)
    ax.set_xticklabels(names, fontsize=FONT_SIZE)
    _apply_style(ax, title='ATV Distribution (Box Plot)',
                 xlabel='Controller', ylabel='Actuator Total Variation',
                 legend=False)

    # Panel B: bar chart of means with std error bars
    ax = axes[1]
    means = [np.mean(v) for v in values]
    stds  = [np.std(v)  for v in values]
    bars  = ax.bar(names, means, yerr=stds, capsize=6,
                   color=colors, alpha=BAR_ALPHA, edgecolor='black', linewidth=0.8)

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + max(stds) * 0.05,
                f'{mean:.2f}', ha='center', va='bottom', fontsize=FONT_SIZE - 1)

    _apply_style(ax, title='Mean ATV (lower = smoother)',
                 xlabel='Controller', ylabel='Mean Actuator Total Variation',
                 legend=False)

    fig.suptitle('Actuator Total Variation Comparison', fontsize=TITLE_SIZE + 1,
                 fontweight='bold', y=1.01)
    fig.tight_layout()
    _savefig(fig, Path(save_dir) / 'atv_comparison.png')


# ---------------------------------------------------------------------------
# 4. Control energy comparison
# ---------------------------------------------------------------------------

def plot_energy_comparison(energy_by_episode: dict,
                           save_dir: str | Path) -> None:
    """
    Bar chart comparing mean control energy across controllers.

    Parameters
    ----------
    energy_by_episode : dict
        {controller_name: list of control energy values}
    save_dir : str or Path

    Output
    ------
    results/plots/control_energy.png
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    names  = list(energy_by_episode.keys())
    values = [np.asarray(energy_by_episode[n], dtype=float) for n in names]
    colors = [_get_color(n, i) for i, n in enumerate(names)]

    # Violin plot
    ax = axes[0]
    parts = ax.violinplot(values, showmeans=True, showmedians=True)
    for i, (pc, color) in enumerate(zip(parts['bodies'], colors)):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(names, fontsize=FONT_SIZE)
    _apply_style(ax, title='Control Energy Distribution',
                 xlabel='Controller', ylabel='Control Energy (N²·s)', legend=False)

    # Bar chart with error bars
    ax = axes[1]
    means = [np.mean(v) for v in values]
    stds  = [np.std(v)  for v in values]
    bars  = ax.bar(names, means, yerr=stds, capsize=6,
                   color=colors, alpha=BAR_ALPHA, edgecolor='black', linewidth=0.8)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + max(stds) * 0.05,
                f'{mean:.1f}', ha='center', va='bottom', fontsize=FONT_SIZE - 1)

    _apply_style(ax, title='Mean Control Energy (lower = efficient)',
                 xlabel='Controller', ylabel='Mean Control Energy (N²·s)',
                 legend=False)

    fig.suptitle('Control Energy Comparison', fontsize=TITLE_SIZE + 1,
                 fontweight='bold', y=1.01)
    fig.tight_layout()
    _savefig(fig, Path(save_dir) / 'control_energy.png')


# ---------------------------------------------------------------------------
# 5. Settling time comparison
# ---------------------------------------------------------------------------

def plot_settling_time(settling_by_episode: dict,
                       save_dir: str | Path) -> None:
    """
    Bar + scatter plot of settling time distributions.

    Parameters
    ----------
    settling_by_episode : dict
        {controller_name: list of settling times (seconds)}
    save_dir : str or Path

    Output
    ------
    results/plots/settling_time.png
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    names  = list(settling_by_episode.keys())
    values = [np.asarray(settling_by_episode[n], dtype=float) for n in names]
    colors = [_get_color(n, i) for i, n in enumerate(names)]
    means  = [np.mean(v) for v in values]
    stds   = [np.std(v)  for v in values]

    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, capsize=6,
                  color=colors, alpha=BAR_ALPHA, edgecolor='black', linewidth=0.8,
                  label='Mean ± Std')

    # Overlay individual data points (jitter for visibility)
    rng = np.random.default_rng(0)
    for i, (vals, color) in enumerate(zip(values, colors)):
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   color=color, alpha=0.4, s=18, zorder=3)

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + max(stds) * 0.05,
                f'{mean:.2f}s', ha='center', va='bottom', fontsize=FONT_SIZE - 1)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=FONT_SIZE)
    _apply_style(ax, title='Settling Time Comparison (lower = faster response)',
                 xlabel='Controller', ylabel='Settling Time (s)')
    ax.legend(['Mean ± Std', 'Individual episodes'], fontsize=LEGEND_SIZE)
    fig.tight_layout()
    _savefig(fig, Path(save_dir) / 'settling_time.png')


# ---------------------------------------------------------------------------
# 6. Comprehensive controller summary bar chart
# ---------------------------------------------------------------------------

def plot_controller_summary(results_by_controller: dict,
                             save_dir: str | Path) -> None:
    """
    Multi-panel bar chart comparing all five research metrics side by side.

    Parameters
    ----------
    results_by_controller : dict
        {controller_name: aggregate_metrics() dict from metrics.py}
        Expected keys per controller:
            sse_mean, settling_time_mean, atv_mean,
            control_energy_mean, disturbance_rejection_mean
    save_dir : str or Path

    Output
    ------
    results/plots/controller_summary.png

    Layout
    ------
    2 × 3 grid (5 metrics + 1 legend panel).
    Each sub-panel shows one metric; bars are grouped by controller.
    """
    metrics_cfg = [
        ('sse',                   'Steady-State Error (rad)',     'lower is better'),
        ('settling_time',         'Settling Time (s)',            'lower is better'),
        ('atv',                   'Actuator Total Variation',     'lower is better'),
        ('control_energy',        'Control Energy (N²·s)',        'lower is better'),
        ('disturbance_rejection', 'Disturbance Rejection (rad)',  'lower is better'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    fig.suptitle('Controller Performance — Research Comparison',
                 fontsize=TITLE_SIZE + 2, fontweight='bold')
    axes_flat = axes.flatten()

    names  = list(results_by_controller.keys())
    colors = [_get_color(n, i) for i, n in enumerate(names)]

    for panel_idx, (key, label, note) in enumerate(metrics_cfg):
        ax = axes_flat[panel_idx]
        means = [results_by_controller[n].get(f'{key}_mean', 0.0) for n in names]
        stds  = [results_by_controller[n].get(f'{key}_std',  0.0) for n in names]

        bars = ax.bar(names, means, yerr=stds, capsize=5,
                      color=colors, alpha=BAR_ALPHA,
                      edgecolor='black', linewidth=0.8)

        for bar, mean in zip(bars, means):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0,
                    h + (max(stds) * 0.05 if max(stds) > 0 else h * 0.02),
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=9)

        ax.text(0.5, 0.97, f'({note})', transform=ax.transAxes,
                ha='center', va='top', fontsize=8, style='italic', color='#555555')

        _apply_style(ax, title=label,
                     xlabel='', ylabel=label, legend=False)

    # Final panel: legend / colour key
    ax = axes_flat[5]
    patches = [mpatches.Patch(color=_get_color(n, i), alpha=BAR_ALPHA, label=n)
               for i, n in enumerate(names)]
    ax.legend(handles=patches, loc='center', fontsize=FONT_SIZE + 1,
              framealpha=0.95, title='Controllers', title_fontsize=FONT_SIZE + 1)
    ax.set_axis_off()

    fig.tight_layout()
    _savefig(fig, Path(save_dir) / 'controller_summary.png')


# ---------------------------------------------------------------------------
# 7. Monte Carlo robustness scatter plot
# ---------------------------------------------------------------------------

def plot_monte_carlo_robustness(mc_results: dict,
                                save_dir: str | Path) -> None:
    """
    Scatter plot of SSE vs ATV across Monte Carlo evaluation episodes.

    This is the key research figure: it shows the trade-off between
    tracking accuracy (SSE) and actuator smoothness (ATV).

    Ideal region: bottom-left (low SSE, low ATV).
    CA-RL should cluster there; Standard RL towards high ATV.

    Parameters
    ----------
    mc_results : dict
        {controller_name: {'sse': [...], 'atv': [...]}}
        Each list has one value per MC episode.
    save_dir : str or Path

    Output
    ------
    results/plots/monte_carlo_robustness.png
    """
    fig, ax = plt.subplots(figsize=(9, 7))

    for idx, (name, data) in enumerate(mc_results.items()):
        sse_vals = np.asarray(data['sse'], dtype=float)
        atv_vals = np.asarray(data['atv'], dtype=float)
        color    = _get_color(name, idx)

        ax.scatter(sse_vals, atv_vals,
                   color=color, alpha=0.55, s=35,
                   label=name, edgecolors='none')

        # Plot centroid with crosshairs
        cx, cy = np.mean(sse_vals), np.mean(atv_vals)
        ax.scatter(cx, cy, color=color, s=120, marker='D',
                   edgecolors='black', linewidths=1.2, zorder=5)

    # Annotate ideal quadrant
    ax.annotate('← Ideal region\n(low error, low switching)',
                xy=(0, 0), xytext=(0.05, 0.08),
                textcoords='axes fraction',
                fontsize=9, color='#555555',
                arrowprops=None)

    _apply_style(ax,
                 title='Monte Carlo Robustness: SSE vs ATV Trade-off',
                 xlabel='Steady-State Error (rad)',
                 ylabel='Actuator Total Variation')
    ax.legend(fontsize=LEGEND_SIZE, markerscale=1.5)
    fig.tight_layout()
    _savefig(fig, Path(save_dir) / 'monte_carlo_robustness.png')


# ---------------------------------------------------------------------------
# 8. Actuator command profile (thruster time series)
# ---------------------------------------------------------------------------

def _plot_actuator_profile(action_trajectories: dict,
                           save_dir: str | Path) -> None:
    """
    Plot all 4 thruster command time-series for one representative episode.

    Parameters
    ----------
    action_trajectories : dict
        {controller_name: np.ndarray shape (T, 4)}
    save_dir : str or Path

    Output
    ------
    results/plots/actuator_profile.png
    """
    n_thrusters = 4
    fig, axes = plt.subplots(n_thrusters, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Thruster Command Profiles (Sample Episode)',
                 fontsize=TITLE_SIZE + 1, fontweight='bold')

    for t_idx in range(n_thrusters):
        ax = axes[t_idx]
        for idx, (name, actions) in enumerate(action_trajectories.items()):
            actions = np.asarray(actions, dtype=float)
            ax.plot(actions[:, t_idx], color=_get_color(name, idx),
                    linewidth=1.5, alpha=0.85, label=name if t_idx == 0 else None)
        ax.set_ylabel(f'T{t_idx+1} (N)', fontsize=FONT_SIZE - 1)
        ax.grid(alpha=GRID_ALPHA, linestyle='--')
        ax.tick_params(labelsize=FONT_SIZE - 2)

    axes[0].legend(fontsize=LEGEND_SIZE)
    axes[-1].set_xlabel('Timestep', fontsize=FONT_SIZE)
    fig.tight_layout()
    _savefig(fig, Path(save_dir) / 'actuator_profile.png')


# ---------------------------------------------------------------------------
# Master save function
# ---------------------------------------------------------------------------

def save_all_plots(reward_histories: dict,
                   error_trajectories: dict,
                   action_trajectories: dict,
                   atv_by_episode: dict,
                   energy_by_episode: dict,
                   settling_by_episode: dict,
                   results_by_controller: dict,
                   mc_results: dict,
                   save_dir: str | Path = 'results/plots') -> None:
    """
    Generate and save every figure in one call.

    Calls all plot_* functions in order.  Safe to call even if some
    dicts are empty — each function skips gracefully on empty input.

    Parameters
    ----------
    reward_histories       : {name: [episode_rewards]}
    error_trajectories     : {name: errors array (T,)}
    action_trajectories    : {name: actions array (T, 4)}
    atv_by_episode         : {name: [atv per episode]}
    energy_by_episode      : {name: [energy per episode]}
    settling_by_episode    : {name: [settling_time per episode]}
    results_by_controller  : {name: aggregate_metrics() dict}
    mc_results             : {name: {'sse': [...], 'atv': [...]}}
    save_dir               : output directory
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[plotting] Generating all figures → {save_dir}")

    if reward_histories:
        plot_training_rewards(reward_histories, save_dir)

    if error_trajectories:
        plot_tracking_error(error_trajectories, save_dir)

    if atv_by_episode:
        plot_atv_comparison(atv_by_episode, save_dir)

    if energy_by_episode:
        plot_energy_comparison(energy_by_episode, save_dir)

    if settling_by_episode:
        plot_settling_time(settling_by_episode, save_dir)

    if results_by_controller:
        plot_controller_summary(results_by_controller, save_dir)

    if mc_results:
        plot_monte_carlo_robustness(mc_results, save_dir)

    if action_trajectories:
        _plot_actuator_profile(action_trajectories, save_dir)

    print(f"[plotting] All figures saved to {save_dir}\n")