"""
train.py
========
Training script for RL-based spacecraft controllers.

Trains both StandardRLAgent and ConstraintAwareRLAgent using episodic
Q-learning. Saves Q-tables and training curves for later evaluation.

Usage:
    python train.py --agent standard --episodes 500 --seed 42
    python train.py --agent ca-rl --episodes 500 --seed 42
    python train.py --agent both --episodes 500 --seed 42  (train both)
"""

import argparse
import numpy as np
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
from tqdm import tqdm

from spacecraft_sim import SpacecraftSim
from controllers import StandardRLAgent, ConstraintAwareRLAgent


def train_agent(agent, sim, n_episodes, eval_interval=50, verbose=True):
    """
    Train an RL agent through episodic interaction.
    
    Parameters
    ----------
    agent : StandardRLAgent or ConstraintAwareRLAgent
    sim : SpacecraftSim
    n_episodes : int
        Number of training episodes
    eval_interval : int
        Episodes between progress evaluations
    verbose : bool
        Whether to print progress
    
    Returns
    -------
    metrics : dict
        Training history: episode_rewards, episode_lengths, eval_errors
    """
    episode_rewards = []
    episode_lengths = []
    eval_errors = []       # tracking error at eval checkpoints
    eval_episodes = []     # which episodes had evals
    
    start_time = time.time()
    
    pbar = tqdm(range(n_episodes), desc="Training", disable=not verbose)
    
    for episode in pbar:
        # Reset environment with random initial conditions
        obs = sim.reset()
        agent.reset()
        
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        while not done:
            # Select action
            action = agent.select_action(obs)
            
            # Step environment
            next_obs, info, done = sim.step(action)
            
            # Compute reward
            reward = agent.compute_reward(obs, action, info)
            
            # Update Q-table
            agent.update(obs, action, reward, next_obs, done)
            
            episode_reward += reward
            episode_length += 1
            obs = next_obs
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Periodic evaluation (deterministic, greedy policy)
        if (episode + 1) % eval_interval == 0:
            eval_error = evaluate_tracking(agent, sim, n_trials=5)
            eval_errors.append(eval_error)
            eval_episodes.append(episode + 1)
            
            if verbose:
                pbar.set_postfix({
                    'eps': f'{agent.epsilon:.3f}',
                    'eval_err': f'{eval_error:.4f}',
                    'ep_len': episode_length
                })
    
    elapsed = time.time() - start_time
    
    if verbose:
        print(f"\nTraining complete in {elapsed/60:.2f} min")
        print(f"  Final epsilon: {agent.epsilon:.4f}")
        if eval_errors:
            print(f"  Final eval error: {eval_errors[-1]:.4f}")
        else:
            print(f"  Final eval error: N/A")
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'eval_errors': eval_errors,
        'eval_episodes': eval_episodes,
        'training_time': elapsed,
        'final_epsilon': agent.epsilon
    }


def evaluate_tracking(agent, sim, n_trials=5):
    """
    Evaluate tracking performance with greedy (deterministic) policy.
    
    Returns average RMS tracking error across multiple test episodes.
    """
    errors = []
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0  # greedy policy
    
    for _ in range(n_trials):
        obs = sim.reset()
        agent.reset()
        episode_errors = []
        done = False
        
        while not done:
            action = agent.select_action(obs)
            obs, info, done = sim.step(action)
            episode_errors.append(np.linalg.norm(info['error']))
        
        errors.append(np.mean(episode_errors))
    
    agent.epsilon = old_epsilon
    return float(np.mean(errors))


def plot_training_curves(metrics_dict, save_path):
    """
    Plot training curves for all trained agents.
    
    Parameters
    ----------
    metrics_dict : dict
        {agent_name: metrics} where metrics is from train_agent()
    save_path : Path or str
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
    
    colors = {'standard': 'tab:blue', 'ca-rl': 'tab:green'}
    
    for agent_name, metrics in metrics_dict.items():
        color = colors.get(agent_name, 'tab:gray')
        label = agent_name.upper()
        
        # Episode rewards (smoothed)
        rewards = np.array(metrics['episode_rewards'])
        window = 50
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        axes[0, 0].plot(smoothed, label=label, color=color, alpha=0.8)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Episode Reward (smoothed)')
        axes[0, 0].set_title('Reward Progression')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Episode lengths
        lengths = np.array(metrics['episode_lengths'])
        smoothed_len = np.convolve(lengths, np.ones(window)/window, mode='valid')
        axes[0, 1].plot(smoothed_len, label=label, color=color, alpha=0.8)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Episode Length (steps)')
        axes[0, 1].set_title('Episode Duration')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # Evaluation error (lower is better)
        eval_eps = metrics['eval_episodes']
        eval_err = metrics['eval_errors']
        axes[1, 0].plot(eval_eps, eval_err, 'o-', label=label, color=color, 
                       markersize=4, alpha=0.8)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('RMS Tracking Error (rad)')
        axes[1, 0].set_title('Evaluation Performance')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # Q-table statistics
        if hasattr(metrics, 'q_stats'):
            axes[1, 1].plot(eval_eps, metrics['q_stats'], label=label, color=color)
    
    # Final panel: learning rate decay visualization
    for agent_name, metrics in metrics_dict.items():
        color = colors.get(agent_name, 'tab:gray')
        n_eps = len(metrics['episode_rewards'])
        # Reconstruct epsilon decay
        from controllers import EPS_START, EPS_END, EPS_DECAY
        eps_curve = [max(EPS_END, EPS_START * (EPS_DECAY ** i)) for i in range(n_eps)]
        axes[1, 1].plot(eps_curve, label=f'{agent_name.upper()} ε', color=color, alpha=0.8)
    
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Exploration Rate (ε)')
    axes[1, 1].set_title('Epsilon Decay')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Training curves saved -> {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train RL spacecraft controllers')
    parser.add_argument('--agent', type=str, default='both',
                       choices=['standard', 'ca-rl', 'both'],
                       help='Which agent to train')
    parser.add_argument('--episodes', type=int, default=100000,
                       help='Number of training episodes')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Directory for outputs')
    parser.add_argument('--eval-interval', type=int, default=1000,
                       help='Episodes between evaluations')
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("SPACECRAFT RL TRAINING")
    print("=" * 70)
    print(f"Episodes:       {args.episodes}")
    print(f"Seed:           {args.seed}")
    print(f"Output dir:     {output_dir}")
    print()
    
    # Create simulator
    sim = SpacecraftSim(seed=args.seed)
    
    # Training results
    all_metrics = {}
    
    # Train Standard RL
    if args.agent in ['standard', 'both']:
        print("Training Standard RL Agent...")
        print("-" * 70)
        agent = StandardRLAgent(seed=args.seed)
        metrics = train_agent(agent, sim, args.episodes, args.eval_interval)
        
        # Save Q-table
        qtable_path = output_dir / 'standard_rl_qtable.npy'
        agent.save_qtable(qtable_path)
        
        # Save metrics
        metrics_path = output_dir / 'standard_rl_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump({k: v if isinstance(v, (int, float, str)) else 
                      [float(x) for x in v] 
                      for k, v in metrics.items()}, f, indent=2)
        print(f"  Metrics saved -> {metrics_path}")
        
        all_metrics['standard'] = metrics
        print()
    
    # Train Constraint-Aware RL
    if args.agent in ['ca-rl', 'both']:
        print("Training Constraint-Aware RL Agent...")
        print("-" * 70)
        agent = ConstraintAwareRLAgent(seed=args.seed + 1)
        metrics = train_agent(agent, sim, args.episodes, args.eval_interval)
        
        # Save Q-table
        qtable_path = output_dir / 'ca_rl_qtable.npy'
        agent.save_qtable(qtable_path)
        
        # Save metrics
        metrics_path = output_dir / 'ca_rl_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump({k: v if isinstance(v, (int, float, str)) else 
                      [float(x) for x in v] 
                      for k, v in metrics.items()}, f, indent=2)
        print(f"  Metrics saved -> {metrics_path}")
        
        all_metrics['ca-rl'] = metrics
        print()
    
    # Plot training curves
    if all_metrics:
        plot_path = output_dir / 'training_curves.png'
        plot_training_curves(all_metrics, plot_path)
    
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()