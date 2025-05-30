"""RL performance tracking and monitoring"""

from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
import time
import json
from pathlib import Path
import numpy as np
from datetime import datetime


class RLTracker:
    """Track RL agent performance metrics"""
    
    def __init__(self, 
                 project_name: str,
                 window_size: int = 100,
                 save_interval: int = 100):
        """
        Initialize tracker
        
        Args:
            project_name: Name of the project being tracked
            window_size: Size of rolling window for metrics
            save_interval: How often to save metrics to disk
        """
        self.project_name = project_name
        self.window_size = window_size
        self.save_interval = save_interval
        
        # Metrics storage
        self.metrics = defaultdict(lambda: deque(maxlen=window_size))
        self.episode_rewards = deque(maxlen=window_size)
        self.episode_lengths = deque(maxlen=window_size)
        
        # Timing
        self.start_time = time.time()
        self.episode_start_time = None
        self.total_episodes = 0
        self.total_steps = 0
        
        # Current episode tracking
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
    def start_episode(self) -> None:
        """Mark the start of a new episode"""
        self.episode_start_time = time.time()
        self.current_episode_reward = 0
        self.current_episode_length = 0
    
    def log_step(self, 
                 reward: float,
                 metrics: Optional[Dict[str, float]] = None) -> None:
        """
        Log a single step
        
        Args:
            reward: Reward received
            metrics: Additional metrics to track
        """
        self.current_episode_reward += reward
        self.current_episode_length += 1
        self.total_steps += 1
        
        # Log custom metrics
        if metrics:
            for key, value in metrics.items():
                self.metrics[key].append(value)
    
    def end_episode(self) -> Dict[str, float]:
        """
        Mark the end of an episode
        
        Returns:
            Episode summary statistics
        """
        if self.episode_start_time is None:
            raise ValueError("end_episode called without start_episode")
        
        episode_time = time.time() - self.episode_start_time
        self.total_episodes += 1
        
        # Store episode metrics
        self.episode_rewards.append(self.current_episode_reward)
        self.episode_lengths.append(self.current_episode_length)
        
        # Calculate statistics
        stats = {
            "episode": self.total_episodes,
            "reward": self.current_episode_reward,
            "length": self.current_episode_length,
            "time": episode_time,
            "avg_reward_100": np.mean(self.episode_rewards) if self.episode_rewards else 0,
            "avg_length_100": np.mean(self.episode_lengths) if self.episode_lengths else 0,
        }
        
        # Save periodically
        if self.total_episodes % self.save_interval == 0:
            self.save_metrics()
        
        return stats
    
    def log_training_metrics(self, metrics: Dict[str, float]) -> None:
        """Log training-specific metrics (loss, learning rate, etc.)"""
        for key, value in metrics.items():
            self.metrics[f"training/{key}"].append(value)
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        total_time = time.time() - self.start_time
        
        stats = {
            "project": self.project_name,
            "total_episodes": self.total_episodes,
            "total_steps": self.total_steps,
            "total_time": total_time,
            "episodes_per_hour": self.total_episodes / (total_time / 3600) if total_time > 0 else 0,
            "steps_per_second": self.total_steps / total_time if total_time > 0 else 0,
        }
        
        # Add rolling averages
        if self.episode_rewards:
            stats.update({
                "avg_reward_100": np.mean(self.episode_rewards),
                "std_reward_100": np.std(self.episode_rewards),
                "max_reward_100": np.max(self.episode_rewards),
                "min_reward_100": np.min(self.episode_rewards),
            })
        
        if self.episode_lengths:
            stats.update({
                "avg_length_100": np.mean(self.episode_lengths),
                "std_length_100": np.std(self.episode_lengths),
            })
        
        # Add custom metrics averages
        for key, values in self.metrics.items():
            if values:
                stats[f"{key}_avg"] = np.mean(values)
                stats[f"{key}_std"] = np.std(values)
        
        return stats
    
    def save_metrics(self, path: Optional[Path] = None) -> None:
        """Save metrics to file"""
        if path is None:
            path = Path(f"rl_metrics_{self.project_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        metrics_data = {
            "project": self.project_name,
            "timestamp": datetime.now().isoformat(),
            "stats": self.get_current_stats(),
            "episode_rewards": list(self.episode_rewards),
            "episode_lengths": list(self.episode_lengths),
            "custom_metrics": {key: list(values) for key, values in self.metrics.items()}
        }
        
        path.write_text(json.dumps(metrics_data, indent=2))
    
    def load_metrics(self, path: Path) -> None:
        """Load metrics from file"""
        data = json.loads(path.read_text())
        
        self.project_name = data["project"]
        self.episode_rewards = deque(data["episode_rewards"], maxlen=self.window_size)
        self.episode_lengths = deque(data["episode_lengths"], maxlen=self.window_size)
        
        for key, values in data["custom_metrics"].items():
            self.metrics[key] = deque(values, maxlen=self.window_size)
    
    def plot_summary(self) -> None:
        """Generate a summary plot (requires matplotlib)"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(f"RL Training Summary - {self.project_name}")
            
            # Episode rewards
            if self.episode_rewards:
                axes[0, 0].plot(self.episode_rewards)
                axes[0, 0].set_title("Episode Rewards")
                axes[0, 0].set_xlabel("Episode")
                axes[0, 0].set_ylabel("Reward")
            
            # Episode lengths
            if self.episode_lengths:
                axes[0, 1].plot(self.episode_lengths)
                axes[0, 1].set_title("Episode Lengths")
                axes[0, 1].set_xlabel("Episode")
                axes[0, 1].set_ylabel("Steps")
            
            # Training loss (if available)
            if "training/loss" in self.metrics:
                axes[1, 0].plot(self.metrics["training/loss"])
                axes[1, 0].set_title("Training Loss")
                axes[1, 0].set_xlabel("Update")
                axes[1, 0].set_ylabel("Loss")
            
            # Custom metric
            custom_keys = [k for k in self.metrics.keys() if not k.startswith("training/")]
            if custom_keys:
                key = custom_keys[0]
                axes[1, 1].plot(self.metrics[key])
                axes[1, 1].set_title(key)
                axes[1, 1].set_xlabel("Step")
                axes[1, 1].set_ylabel("Value")
            
            plt.tight_layout()
            plt.savefig(f"rl_summary_{self.project_name}.png")
            plt.close()
            
        except ImportError:
            print("matplotlib not available, skipping plot generation")
    
    def __repr__(self) -> str:
        return f"RLTracker(project={self.project_name}, episodes={self.total_episodes}, steps={self.total_steps})"
