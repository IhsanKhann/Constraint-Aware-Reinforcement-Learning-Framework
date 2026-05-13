# Constraint-Aware Reinforcement Learning for Spacecraft Attitude Control

> Muhammad Ihsan Khan — Bachelor of Computer Science, IMS, 2026
>
> Research comparison: **PID** vs **Standard RL** vs **Constraint-Aware RL** for multi-thruster spacecraft attitude control.

---

## Project Overview

This project implements and evaluates three control architectures for a simulated 3-DOF rigid-body spacecraft equipped with four thrusters. The central research question is:

> *Can a lightweight, non-deep reinforcement learning controller achieve PID-level smoothness while retaining adaptive advantages, by shaping its reward function with actuator constraints?*

The answer is demonstrated through quantitative metrics (SSE, ATV, Control Energy, Settling Time, Disturbance Rejection) and publication-quality figures — all reproducible with a single command.

---

## Research Objective

Standard RL controllers learn to minimize tracking error but often generate **jittery, oscillatory actuator signals** — harmful to real hardware. This project introduces **Constraint-Aware RL (CA-RL)**, which adds a smoothness reward penalty:

```
R = R_performance + R_smoothness
R_smoothness = −k1 ‖aₜ − aₜ₋₁‖² − k2 ‖aₜ‖²
```

The hypothesis: CA-RL achieves comparable tracking accuracy to Standard RL while significantly reducing actuator total variation (ATV) and control energy.

---

## Architecture

```
spacecraft_sim.py   ← 3-DOF RK45 rigid-body simulator
    │
    ├── controllers.py   ← PIDController, StandardRLAgent, ConstraintAwareRLAgent
    │       └── tilecoding.py   ← continuous state → Q-table index
    │
    ├── train.py         ← episodic Q-learning training loop
    ├── evaluate.py      ← deterministic evaluation + EvaluationMetrics
    ├── metrics.py       ← compute_sse, compute_atv, compute_energy, …
    ├── plotting.py      ← all matplotlib figure generation
    └── run_experiment.py ← MASTER SCRIPT (runs everything)
```

**Physics model** (Euler's equation):

```
J·ω̇ + ω × (J·ω) = τ_ctrl + τ_dist
```

**State space**: `[θx, θy, θz, ωx, ωy, ωz]` — 6D continuous, tile-coded to integer index.

**Action space**: 81 discrete joint thruster commands (3 levels × 4 thrusters).

---

## Installation

```bash
# 1. Clone or copy the project
git clone <your-repo-url>
cd spacecraft_rl

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

**Minimum Python version**: 3.10 (uses `str | Path` union type hints).

---

## How to Run Experiments

### Full experiment (train + evaluate + plots + CSVs)

```bash
python run_experiment.py
```

This runs the complete 8-phase pipeline automatically.

### Quick sanity-check run (fast, ~2 min)

```bash
python run_experiment.py --episodes 100 --eval-trials 5
```

### Custom training length

```bash
python run_experiment.py --episodes 3000 --eval-trials 50 --seed 7
```

### Skip training (use existing Q-tables)

```bash
python run_experiment.py --skip-train --eval-trials 50
```

### Train agents separately

```bash
python train.py --agent standard --episodes 2000 --seed 42
python train.py --agent ca-rl    --episodes 2000 --seed 43
```

### Evaluate only

```bash
python evaluate.py --n-trials 50 --results-dir results
```

### All CLI options

| Argument | Default | Description |
|---|---|---|
| `--episodes` | 2000 | Training episodes per RL agent |
| `--eval-trials` | 50 | Evaluation episodes per controller |
| `--seed` | 42 | Master random seed |
| `--log-every` | 100 | Training progress log interval |
| `--skip-train` | False | Load existing Q-tables, skip training |
| `--results-dir` | results | Root output directory |

---

## Folder Structure

```
spacecraft_rl/
│
├── spacecraft_sim.py       ← Simulator (DO NOT MODIFY)
├── controllers.py          ← Control architectures
├── tilecoding.py           ← Tile coding for Q-learning
├── train.py                ← Training pipeline
├── evaluate.py             ← Evaluation pipeline
├── metrics.py              ← Metric computation functions
├── plotting.py             ← Figure generation
├── run_experiment.py       ← MASTER SCRIPT
├── requirements.txt
├── README.md
│
└── results/
    ├── plots/              ← All PNG figures
    │   ├── training_rewards.png
    │   ├── tracking_error.png
    │   ├── atv_comparison.png
    │   ├── control_energy.png
    │   ├── settling_time.png
    │   ├── controller_summary.png
    │   ├── monte_carlo_robustness.png
    │   └── actuator_profile.png
    │
    ├── csv/                ← Metric data tables
    │   ├── metrics_summary.csv
    │   ├── metrics_raw.csv
    │   ├── pid_episodes.csv
    │   ├── standard_rl_episodes.csv
    │   └── ca_rl_episodes.csv
    │
    ├── qtables/            ← Trained Q-tables
    │   ├── standard_rl_qtable.npy
    │   └── ca_rl_qtable.npy
    │
    └── logs/               ← Training + evaluation JSON logs
        ├── standard_rl_training.json
        ├── ca_rl_training.json
        └── evaluation_summary.json
```

---

## Metrics Explained

| Metric | Definition | Unit | Goal |
|---|---|---|---|
| **Steady-State Error (SSE)** | Mean error in final 10% of episode | rad | Lower |
| **Settling Time** | First time error stays below 0.02 rad for 10 steps | s | Lower |
| **Actuator Total Variation (ATV)** | Σ ‖aₜ − aₜ₋₁‖₁ over episode | N | Lower |
| **Control Energy** | Σ ‖aₜ‖² × Δt over episode | N²·s | Lower |
| **Disturbance Rejection** | Peak error during Gaussian-Markov disturbance | rad | Lower |

**ATV** is the primary metric distinguishing CA-RL from Standard RL. A lower ATV means smoother actuator signals → less mechanical wear → longer hardware lifespan.

---

## Expected Outputs

After a full run you should see results approximately like:

```
Metric                              PID          Standard RL    CA-RL
─────────────────────────────────────────────────────────────────────
Steady-State Error (rad)       0.05 ± 0.03    0.04 ± 0.02    0.05 ± 0.02
Settling Time (s)             12.0 ± 4.0      9.0 ± 3.5     11.0 ± 3.8
Actuator Total Variation     180.0 ± 30.0   220.0 ± 40.0   140.0 ± 25.0  ← CA-RL wins
Control Energy (N²·s)        420.0 ± 50.0   510.0 ± 60.0   380.0 ± 45.0  ← CA-RL wins
Disturbance Rejection (rad)    0.45 ± 0.10    0.38 ± 0.08    0.40 ± 0.09
```

*(Exact values depend on seed, episode count, and hardware timing.)*

---

## Generated Graphs

| Figure | Description |
|---|---|
| `training_rewards.png` | Smoothed episode reward curves showing learning progress |
| `tracking_error.png` | Attitude error over time for a representative episode |
| `atv_comparison.png` | Box plot + bar chart of ATV distribution across 50 episodes |
| `control_energy.png` | Violin + bar chart of control energy |
| `settling_time.png` | Settling time distribution with individual episode scatter |
| `controller_summary.png` | 5-panel bar chart of all research metrics side by side |
| `monte_carlo_robustness.png` | Scatter plot: SSE vs ATV trade-off across all MC episodes |
| `actuator_profile.png` | All 4 thruster command time-series for one episode |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'spacecraft_sim'`**
Run all scripts from the project root directory, not from a subdirectory.

**Evaluation metrics look identical for all controllers**
Check that `agent.epsilon = 0.0` is set before evaluation. Without this, the RL agents act randomly.

**Q-table files not found with `--skip-train`**
Run without `--skip-train` first to generate the `.npy` files in `results/qtables/`.

**Training is very slow**
Use `--episodes 100 --eval-trials 5` for a quick test run. Full 2000-episode training on a modern laptop takes approximately 20–40 minutes.

**All settling times equal the episode length (25 s)**
The threshold of 0.02 rad may be too tight for the current training level. Try more episodes, or check that the simulator target is `[0, 0, 0]`.

**Plots not generated**
Ensure `matplotlib` is installed and the `results/plots/` directory is writable. The backend is set to `Agg` (headless) so no display is required.

---

## Reproducibility

All experiments are seeded via `--seed` (default 42). The simulator, agents, and evaluation all accept the same seed, ensuring identical results across runs with the same arguments.

To reproduce exactly:

```bash
python run_experiment.py --episodes 2000 --eval-trials 50 --seed 42
```

---

## Citation

If you use this code in academic work, please cite the research proposal:

```
Khan, M.I. (2026). Constraint-Aware Reinforcement Learning for Smooth Multi-Thruster
Maneuver Control in Aerospace Systems. IMS Research Proposal.
```