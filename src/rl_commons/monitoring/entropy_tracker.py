"""
Module: entropy_tracker.py
Purpose: Track and monitor policy entropy in RL agents to detect and prevent entropy collapse

External Dependencies:
- numpy: https://numpy.org/doc/stable/
- torch: https://pytorch.org/docs/stable/
- matplotlib: https://matplotlib.org/stable/

Example Usage:
>>> from rl_commons.monitoring.entropy_tracker import EntropyTracker
>>> tracker = EntropyTracker(window_size=100, collapse_threshold=0.5)
>>> entropy_value = 2.0  # High initial entropy
>>> tracker.update(entropy_value)
>>> if tracker.detect_collapse():
...     print("Warning: Entropy collapse detected!")
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import deque
from dataclasses import dataclass
# Matplotlib disabled for headless testing
# import matplotlib.pyplot as plt
from datetime import datetime
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class EntropyMetrics:
    """Container for entropy-related metrics"""
    current: float
    mean: float
    std: float
    min: float
    max: float
    trend: float  # Positive = increasing, negative = decreasing
    collapse_risk: float  # 0-1 probability of collapse
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization"""
        return {
            'current': self.current,
            'mean': self.mean,
            'std': self.std,
            'min': self.min,
            'max': self.max,
            'trend': self.trend,
            'collapse_risk': self.collapse_risk
        }


class EntropyTracker:
    """
    Track and analyze policy entropy over time to detect collapse patterns.
    
    Based on research showing entropy collapse happens rapidly in early training,
    with entropy dropping from ~2.0 to ~0.5 in just a few hundred steps.
    """
    
    def __init__(self, 
                 window_size: int = 100,
                 collapse_threshold: float = 0.5,
                 trend_window: int = 20,
                 min_healthy_entropy: float = 0.5):
        """
        Initialize entropy tracker.
        
        Args:
            window_size: Number of recent values to track
            collapse_threshold: Entropy value below which collapse is detected
            trend_window: Window for calculating trend
            min_healthy_entropy: Minimum entropy for healthy exploration
        """
        self.window_size = window_size
        self.collapse_threshold = collapse_threshold
        self.trend_window = trend_window
        self.min_healthy_entropy = min_healthy_entropy
        
        # Storage
        self.history = deque(maxlen=window_size)
        self.full_history = []
        self.timestamps = []
        
        # Collapse detection
        self.collapse_detected = False
        self.collapse_step = None
        self.initial_entropy = None
        
        logger.info(f"EntropyTracker initialized with collapse_threshold={collapse_threshold}")
    
    def update(self, entropy: float, step: Optional[int] = None) -> EntropyMetrics:
        """
        Update entropy tracking with new value.
        
        Args:
            entropy: Current policy entropy
            step: Optional training step number
            
        Returns:
            Current entropy metrics
        """
        # Store initial entropy
        if self.initial_entropy is None:
            self.initial_entropy = entropy
            logger.info(f"Initial entropy: {entropy:.4f}")
        
        # Update history
        self.history.append(entropy)
        self.full_history.append(entropy)
        self.timestamps.append(step if step is not None else len(self.full_history))
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        
        # Check for collapse
        if not self.collapse_detected and self._check_collapse(metrics):
            self.collapse_detected = True
            self.collapse_step = self.timestamps[-1]
            logger.warning(f"Entropy collapse detected at step {self.collapse_step}! "
                         f"Entropy dropped from {self.initial_entropy:.4f} to {entropy:.4f}")
        
        return metrics
    
    def _calculate_metrics(self) -> EntropyMetrics:
        """Calculate current entropy metrics"""
        if not self.history:
            return EntropyMetrics(0, 0, 0, 0, 0, 0, 0)
        
        history_array = np.array(self.history)
        current = self.history[-1]
        
        # Basic stats
        mean = np.mean(history_array)
        std = np.std(history_array) if len(history_array) > 1 else 0
        min_val = np.min(history_array)
        max_val = np.max(history_array)
        
        # Calculate trend
        trend = self._calculate_trend()
        
        # Calculate collapse risk
        collapse_risk = self._calculate_collapse_risk(current, trend)
        
        return EntropyMetrics(
            current=current,
            mean=mean,
            std=std,
            min=min_val,
            max=max_val,
            trend=trend,
            collapse_risk=collapse_risk
        )
    
    def _calculate_trend(self) -> float:
        """Calculate entropy trend (rate of change)"""
        if len(self.history) < 2:
            return 0.0
        
        # Use recent window for trend
        window = min(self.trend_window, len(self.history))
        recent = list(self.history)[-window:]
        
        # Simple linear regression
        x = np.arange(len(recent))
        y = np.array(recent)
        
        if len(x) < 2:
            return 0.0
        
        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)
    
    def _calculate_collapse_risk(self, current: float, trend: float) -> float:
        """
        Calculate probability of entropy collapse.
        
        High risk when:
        - Current entropy is low
        - Trend is strongly negative
        - Entropy dropped significantly from initial
        """
        risk_factors = []
        
        # Factor 1: Low current entropy
        if current < self.min_healthy_entropy:
            risk_factors.append(1.0)
        else:
            risk_factors.append(max(0, 1 - current / self.min_healthy_entropy))
        
        # Factor 2: Negative trend
        if trend < 0:
            trend_risk = min(1.0, abs(trend) * 10)  # Scale trend impact
            risk_factors.append(trend_risk)
        else:
            risk_factors.append(0.0)
        
        # Factor 3: Drop from initial
        if self.initial_entropy is not None and self.initial_entropy > 0:
            drop_ratio = 1 - (current / self.initial_entropy)
            risk_factors.append(min(1.0, drop_ratio))
        
        # Combine factors
        if risk_factors:
            return np.mean(risk_factors)
        return 0.0
    
    def _check_collapse(self, metrics: EntropyMetrics) -> bool:
        """Check if entropy collapse has occurred"""
        # Collapse if below threshold
        if metrics.current < self.collapse_threshold:
            return True
        
        # Collapse if dropped >90% from initial
        if self.initial_entropy is not None:
            drop_percent = (self.initial_entropy - metrics.current) / self.initial_entropy
            if drop_percent > 0.9:
                return True
        
        return False
    
    def detect_collapse(self) -> bool:
        """Check if entropy collapse has been detected"""
        return self.collapse_detected
    
    def get_recovery_recommendation(self) -> Optional[str]:
        """Get recommendation for entropy recovery if needed"""
        if not self.history:
            return None
        
        metrics = self._calculate_metrics()
        
        if metrics.collapse_risk > 0.7:
            return "High collapse risk! Consider: 1) Reduce learning rate, 2) Add entropy bonus, 3) Increase exploration"
        elif metrics.current < self.min_healthy_entropy:
            return f"Entropy below healthy threshold ({self.min_healthy_entropy}). Consider adding entropy regularization."
        elif metrics.trend < -0.01:
            return "Rapid entropy decrease detected. Monitor closely and consider intervention."
        
        return None
    
    def visualize(self, save_path: Optional[str] = None) -> None:
        """
        Create visualization of entropy dynamics.
        
        Args:
            save_path: Optional path to save figure
        """
        if not self.full_history:
            logger.warning("No data to visualize")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot 1: Entropy over time
        ax1.plot(self.timestamps, self.full_history, 'b-', label='Entropy')
        ax1.axhline(y=self.collapse_threshold, color='r', linestyle='--', label='Collapse Threshold')
        ax1.axhline(y=self.min_healthy_entropy, color='orange', linestyle='--', label='Healthy Minimum')
        
        if self.collapse_step is not None:
            ax1.axvline(x=self.collapse_step, color='r', linestyle=':', label='Collapse Detected')
        
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Entropy')
        ax1.set_title('Policy Entropy Dynamics')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Entropy trend and risk
        if len(self.full_history) > self.trend_window:
            # Calculate rolling trend
            trends = []
            risks = []
            for i in range(self.trend_window, len(self.full_history)):
                window_data = self.full_history[i-self.trend_window:i]
                x = np.arange(len(window_data))
                y = np.array(window_data)
                trend = np.polyfit(x, y, 1)[0]
                trends.append(trend)
                
                # Simple risk calculation
                current = self.full_history[i]
                risk = self._calculate_collapse_risk(current, trend)
                risks.append(risk)
            
            ax2_twin = ax2.twinx()
            
            trend_line = ax2.plot(self.timestamps[self.trend_window:], trends, 'g-', label='Trend')
            risk_line = ax2_twin.plot(self.timestamps[self.trend_window:], risks, 'r-', label='Collapse Risk')
            
            ax2.set_xlabel('Training Step')
            ax2.set_ylabel('Entropy Trend', color='g')
            ax2_twin.set_ylabel('Collapse Risk', color='r')
            ax2.set_title('Entropy Trend and Collapse Risk')
            
            # Combine legends
            lines = trend_line + risk_line
            labels = [l.get_label() for l in lines]
            ax2.legend(lines, labels, loc='upper right')
            
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Entropy visualization saved to {save_path}")
        else:
            plt.show()
    
    def save_history(self, path: str) -> None:
        """Save entropy history to file"""
        data = {
            'initial_entropy': self.initial_entropy,
            'collapse_detected': self.collapse_detected,
            'collapse_step': self.collapse_step,
            'collapse_threshold': self.collapse_threshold,
            'min_healthy_entropy': self.min_healthy_entropy,
            'history': list(self.full_history),
            'timestamps': list(self.timestamps),
            'metrics': self._calculate_metrics().to_dict() if self.history else None
        }
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Entropy history saved to {path}")
    
    def load_history(self, path: str) -> None:
        """Load entropy history from file"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.initial_entropy = data['initial_entropy']
        self.collapse_detected = data['collapse_detected']
        self.collapse_step = data['collapse_step']
        self.full_history = data['history']
        self.timestamps = data['timestamps']
        
        # Rebuild recent history
        if self.full_history:
            recent = self.full_history[-self.window_size:]
            self.history = deque(recent, maxlen=self.window_size)
        
        logger.info(f"Entropy history loaded from {path}")


if __name__ == "__main__":
    # Validation and example usage
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Running EntropyTracker validation...")
    
    # Simulate entropy collapse scenario
    tracker = EntropyTracker(window_size=50, collapse_threshold=0.5)
    
    # Simulate training with entropy collapse
    np.random.seed(42)
    steps = 200
    
    logger.info("Simulating entropy collapse during training...")
    
    for step in range(steps):
        # Simulate entropy that collapses rapidly
        if step < 50:
            # High initial entropy with small noise
            entropy = 2.0 + np.random.normal(0, 0.1)
        elif step < 100:
            # Rapid collapse phase
            progress = (step - 50) / 50
            entropy = 2.0 * (1 - progress) + 0.3 * progress + np.random.normal(0, 0.05)
        else:
            # Collapsed state
            entropy = 0.3 + np.random.normal(0, 0.02)
        
        metrics = tracker.update(entropy, step)
        
        if step % 20 == 0:
            logger.info(f"Step {step}: Entropy={metrics.current:.3f}, "
                      f"Trend={metrics.trend:.3f}, Risk={metrics.collapse_risk:.2f}")
        
        # Check for recommendations
        if step % 10 == 0:
            rec = tracker.get_recovery_recommendation()
            if rec:
                logger.warning(f"Step {step}: {rec}")
    
    # Create visualization
    logger.info("Creating entropy dynamics visualization...")
    tracker.visualize(save_path="entropy_collapse_simulation.png")
    
    # Save history
    tracker.save_history("entropy_history.json")
    
    logger.info(" EntropyTracker validation complete!")
    logger.info(f"Collapse detected: {tracker.detect_collapse()} at step {tracker.collapse_step}")