"""
Module: covariance_analyzer.py
Purpose: Analyze covariance between log probabilities and advantages to identify high-covariance tokens

External Dependencies:
- numpy: https://numpy.org/doc/stable/
- torch: https://pytorch.org/docs/stable/
- scipy: https://docs.scipy.org/doc/scipy/

Example Usage:
>>> from rl_commons.core.covariance_analyzer import CovarianceAnalyzer
>>> analyzer = CovarianceAnalyzer(percentile_threshold=99.8)
>>> high_cov_mask = analyzer.identify_outliers(log_probs, advantages)
>>> print(f"Found {high_cov_mask.sum()} high-covariance tokens")
"""

import numpy as np
import torch
from typing import Union, Tuple, Dict, Optional, List
from dataclasses import dataclass
import logging
from collections import deque
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


@dataclass
class CovarianceMetrics:
    """Container for covariance analysis metrics"""
    mean_covariance: float
    std_covariance: float
    percentile_98: float
    percentile_99: float
    percentile_99_8: float
    outlier_ratio: float
    outlier_indices: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization"""
        result = {
            'mean_covariance': self.mean_covariance,
            'std_covariance': self.std_covariance,
            'percentile_98': self.percentile_98,
            'percentile_99': self.percentile_99,
            'percentile_99_8': self.percentile_99_8,
            'outlier_ratio': self.outlier_ratio,
        }
        if self.outlier_indices is not None:
            result['num_outliers'] = len(self.outlier_indices)
        return result


class CovarianceAnalyzer:
    """
    Analyze covariance between log probabilities and advantages.
    
    Based on research showing 0.2% of tokens exhibit extreme high covariance
    that triggers entropy collapse in RL training.
    """
    
    def __init__(self, 
                 percentile_threshold: float = 99.8,
                 history_size: int = 1000,
                 device: str = 'cpu'):
        """
        Initialize covariance analyzer.
        
        Args:
            percentile_threshold: Percentile for identifying high-covariance tokens (default 99.8)
            history_size: Number of recent covariances to track
            device: Device for computation ('cpu' or 'cuda')
        """
        self.percentile_threshold = percentile_threshold
        self.history_size = history_size
        self.device = device
        
        # History tracking
        self.covariance_history = deque(maxlen=history_size)
        self.outlier_history = deque(maxlen=100)
        
        # Statistics
        self.total_tokens_analyzed = 0
        self.total_outliers_found = 0
        
        logger.info(f"CovarianceAnalyzer initialized with {percentile_threshold}th percentile threshold")
    
    def calculate_token_covariances(self,
                                  log_probs: Union[torch.Tensor, np.ndarray],
                                  advantages: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Calculate token-wise centered cross-product between log_probs and advantages.
        
        This is the key metric from the research:
        cov_i = (log_prob_i - mean(log_prob)) * (advantage_i - mean(advantage))
        
        Args:
            log_probs: Log probabilities of actions [batch_size] or [batch_size, seq_len]
            advantages: Advantage values [batch_size] or [batch_size, seq_len]
            
        Returns:
            Token-wise covariances as numpy array
        """
        # Convert to numpy if needed
        if torch.is_tensor(log_probs):
            log_probs = log_probs.detach().cpu().numpy()
        if torch.is_tensor(advantages):
            advantages = advantages.detach().cpu().numpy()
        
        # Ensure same shape
        if log_probs.shape != advantages.shape:
            raise ValueError(f"Shape mismatch: log_probs {log_probs.shape} vs advantages {advantages.shape}")
        
        # Flatten if needed
        log_probs = log_probs.flatten()
        advantages = advantages.flatten()
        
        # Center the values
        log_probs_centered = log_probs - np.mean(log_probs)
        advantages_centered = advantages - np.mean(advantages)
        
        # Calculate token-wise covariances
        covariances = log_probs_centered * advantages_centered
        
        # Update history
        self.covariance_history.extend(covariances.tolist())
        self.total_tokens_analyzed += len(covariances)
        
        return covariances
    
    def identify_outliers(self,
                         log_probs: Union[torch.Tensor, np.ndarray],
                         advantages: Union[torch.Tensor, np.ndarray],
                         return_metrics: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, CovarianceMetrics]]:
        """
        Identify high-covariance tokens that may trigger entropy collapse.
        
        Args:
            log_probs: Log probabilities of actions
            advantages: Advantage values
            return_metrics: Whether to return detailed metrics
            
        Returns:
            Boolean mask of high-covariance tokens (and optionally metrics)
        """
        # Calculate covariances
        covariances = self.calculate_token_covariances(log_probs, advantages)
        
        # Calculate threshold
        threshold = np.percentile(np.abs(covariances), self.percentile_threshold)
        
        # Identify outliers
        outlier_mask = np.abs(covariances) > threshold
        outlier_indices = np.where(outlier_mask)[0]
        
        # Update statistics
        num_outliers = np.sum(outlier_mask)
        self.total_outliers_found += num_outliers
        self.outlier_history.append(num_outliers / len(covariances))
        
        if num_outliers > 0:
            logger.info(f"Found {num_outliers}/{len(covariances)} high-covariance tokens "
                       f"({num_outliers/len(covariances)*100:.1f}%)")
        
        if return_metrics:
            # Calculate detailed metrics
            metrics = CovarianceMetrics(
                mean_covariance=np.mean(covariances),
                std_covariance=np.std(covariances),
                percentile_98=np.percentile(np.abs(covariances), 98),
                percentile_99=np.percentile(np.abs(covariances), 99),
                percentile_99_8=np.percentile(np.abs(covariances), 99.8),
                outlier_ratio=num_outliers / len(covariances),
                outlier_indices=outlier_indices
            )
            return outlier_mask, metrics
        
        return outlier_mask
    
    def get_adaptive_threshold(self, 
                             current_entropy: float,
                             target_entropy: float = 0.5) -> float:
        """
        Get adaptive threshold based on current entropy level.
        
        When entropy is low, we may want to be more aggressive in identifying
        and handling high-covariance tokens.
        
        Args:
            current_entropy: Current policy entropy
            target_entropy: Target minimum entropy
            
        Returns:
            Adjusted percentile threshold
        """
        if current_entropy < target_entropy:
            # More aggressive when entropy is low
            # Linear interpolation from 99.8 to 99.0 as entropy drops
            ratio = current_entropy / target_entropy
            return 99.0 + 0.8 * ratio
        else:
            # Standard threshold when entropy is healthy
            return self.percentile_threshold
    
    def analyze_distribution(self) -> Dict[str, float]:
        """
        Analyze the distribution of covariances seen so far.
        
        Returns:
            Dictionary of distribution statistics
        """
        if not self.covariance_history:
            return {}
        
        covariances = np.array(self.covariance_history)
        
        return {
            'mean': float(np.mean(covariances)),
            'std': float(np.std(covariances)),
            'min': float(np.min(covariances)),
            'max': float(np.max(covariances)),
            'median': float(np.median(covariances)),
            'skewness': float(self._calculate_skewness(covariances)),
            'kurtosis': float(self._calculate_kurtosis(covariances)),
            'outlier_rate': float(np.mean(self.outlier_history)) if self.outlier_history else 0.0
        }
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of distribution"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of distribution"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def visualize_distribution(self, 
                             log_probs: Optional[Union[torch.Tensor, np.ndarray]] = None,
                             advantages: Optional[Union[torch.Tensor, np.ndarray]] = None,
                             save_path: Optional[str] = None) -> None:
        """
        Visualize covariance distribution and outliers.
        
        Args:
            log_probs: Optional new log probabilities to analyze
            advantages: Optional new advantages to analyze
            save_path: Optional path to save figure
        """
        # Get covariances to plot
        if log_probs is not None and advantages is not None:
            covariances = self.calculate_token_covariances(log_probs, advantages)
        elif self.covariance_history:
            covariances = np.array(list(self.covariance_history)[-1000:])  # Last 1000
        else:
            logger.warning("No covariance data to visualize")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Distribution plot
        ax1.hist(covariances, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        # Add percentile lines
        for p, color in [(98, 'yellow'), (99, 'orange'), (99.8, 'red')]:
            threshold = np.percentile(np.abs(covariances), p)
            ax1.axvline(x=threshold, color=color, linestyle='--', 
                       label=f'{p}th percentile')
            ax1.axvline(x=-threshold, color=color, linestyle='--')
        
        ax1.set_xlabel('Covariance')
        ax1.set_ylabel('Density')
        ax1.set_title('Covariance Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Outlier rate over time
        if self.outlier_history:
            outlier_rates = list(self.outlier_history)
            ax2.plot(outlier_rates, 'r-', linewidth=2)
            ax2.axhline(y=0.002, color='green', linestyle='--', 
                       label='Expected (0.2%)')
            ax2.set_xlabel('Analysis Step')
            ax2.set_ylabel('Outlier Rate')
            ax2.set_title('High-Covariance Token Rate')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, max(0.01, max(outlier_rates) * 1.1))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Covariance visualization saved to {save_path}")
        else:
            plt.show()
    
    def reset_history(self) -> None:
        """Reset history and statistics"""
        self.covariance_history.clear()
        self.outlier_history.clear()
        self.total_tokens_analyzed = 0
        self.total_outliers_found = 0
        logger.info("Covariance analyzer history reset")


def create_batch_analyzer(percentile_threshold: float = 99.8) -> CovarianceAnalyzer:
    """
    Create analyzer optimized for batch processing.
    
    Args:
        percentile_threshold: Percentile threshold for outliers
        
    Returns:
        Configured CovarianceAnalyzer instance
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return CovarianceAnalyzer(
        percentile_threshold=percentile_threshold,
        history_size=10000,  # Larger history for batch processing
        device=device
    )


if __name__ == "__main__":
    # Validation and example usage
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Running CovarianceAnalyzer validation...")
    
    # Create analyzer
    analyzer = CovarianceAnalyzer(percentile_threshold=99.8)
    
    # Simulate realistic RL data
    np.random.seed(42)
    batch_size = 1000
    
    # Most tokens have normal covariance
    log_probs = np.random.normal(-1.5, 0.5, batch_size)
    advantages = np.random.normal(0, 1, batch_size)
    
    # But 0.2% have extreme covariance (outliers)
    num_outliers = int(batch_size * 0.002)
    outlier_indices = np.random.choice(batch_size, num_outliers, replace=False)
    
    # Make outliers have high correlation
    for idx in outlier_indices:
        # High positive or negative correlation
        if np.random.random() > 0.5:
            log_probs[idx] = 2.0
            advantages[idx] = 3.0
        else:
            log_probs[idx] = -3.0
            advantages[idx] = 2.5
    
    # Analyze
    outlier_mask, metrics = analyzer.identify_outliers(log_probs, advantages, return_metrics=True)
    
    logger.info(f"Analysis results:")
    logger.info(f"  Mean covariance: {metrics.mean_covariance:.4f}")
    logger.info(f"  Std covariance: {metrics.std_covariance:.4f}")
    logger.info(f"  99.8th percentile: {metrics.percentile_99_8:.4f}")
    logger.info(f"  Outlier ratio: {metrics.outlier_ratio:.4%}")
    logger.info(f"  Expected outliers: {num_outliers}, Found: {outlier_mask.sum()}")
    
    # Test adaptive threshold
    low_entropy = 0.3
    high_entropy = 1.5
    
    low_threshold = analyzer.get_adaptive_threshold(low_entropy)
    high_threshold = analyzer.get_adaptive_threshold(high_entropy)
    
    logger.info(f"\nAdaptive thresholds:")
    logger.info(f"  Low entropy ({low_entropy}): {low_threshold:.1f}th percentile")
    logger.info(f"  High entropy ({high_entropy}): {high_threshold:.1f}th percentile")
    
    # Visualize
    logger.info("\nCreating covariance distribution visualization...")
    analyzer.visualize_distribution(save_path="covariance_distribution.png")
    
    # Distribution analysis
    dist_stats = analyzer.analyze_distribution()
    logger.info("\nDistribution statistics:")
    for key, value in dist_stats.items():
        logger.info(f"  {key}: {value:.4f}")
    
    logger.info("\n CovarianceAnalyzer validation complete!")