"""
Module: entropy_visualizer.py
Purpose: Advanced visualization tools for entropy dynamics and collapse patterns

External Dependencies:
- matplotlib: https://matplotlib.org/stable/
- seaborn: https://seaborn.pydata.org/
- plotly: https://plotly.com/python/

Example Usage:
>>> from rl_commons.monitoring.entropy_visualizer import EntropyVisualizer
>>> visualizer = EntropyVisualizer()
>>> visualizer.plot_entropy_dynamics(tracker, save_path="entropy_report.png")
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Optional imports
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    logger.warning("Seaborn not available. Using matplotlib defaults.")

# Optional imports for advanced features
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not available. Interactive plots disabled.")


class EntropyVisualizer:
    """Advanced visualization for entropy dynamics and analysis"""
    
    def __init__(self, style: str = "whitegrid"):
        """
        Initialize visualizer with style settings.
        
        Args:
            style: Seaborn style to use (if available)
        """
        if SEABORN_AVAILABLE:
            sns.set_style(style)
        else:
            # Use matplotlib defaults
            plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        
        self.colors = {
            'entropy': '#2E86AB',
            'risk': '#A23B72',
            'threshold': '#F18F01',
            'collapse': '#C73E1D',
            'healthy': '#6A994E'
        }
    
    def plot_entropy_dynamics(self,
                            tracker,
                            save_path: Optional[str] = None,
                            figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Create comprehensive entropy dynamics visualization.
        
        Args:
            tracker: EntropyTracker instance
            save_path: Optional path to save figure
            figsize: Figure size
        """
        if not tracker.full_history:
            logger.warning("No entropy data to visualize")
            return
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)
        
        # Main entropy plot
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_entropy_timeline(ax1, tracker)
        
        # Risk and trend
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_risk_analysis(ax2, tracker)
        
        # Distribution
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_entropy_distribution(ax3, tracker)
        
        # Phase diagram
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_phase_diagram(ax4, tracker)
        
        # Recovery zones
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_recovery_zones(ax5, tracker)
        
        plt.suptitle('Entropy Dynamics Analysis', fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Entropy visualization saved to {save_path}")
        else:
            plt.show()
    
    def _plot_entropy_timeline(self, ax, tracker) -> None:
        """Plot main entropy timeline with annotations"""
        steps = tracker.timestamps
        entropy = tracker.full_history
        
        # Main entropy line
        ax.plot(steps, entropy, color=self.colors['entropy'], linewidth=2, label='Policy Entropy')
        
        # Fill regions
        ax.fill_between(steps, 0, tracker.min_healthy_entropy,
                       color=self.colors['collapse'], alpha=0.2, label='Collapse Zone')
        ax.fill_between(steps, tracker.min_healthy_entropy, tracker.collapse_threshold,
                       color=self.colors['risk'], alpha=0.2, label='Risk Zone')
        ax.fill_between(steps, tracker.collapse_threshold, max(entropy) * 1.1,
                       color=self.colors['healthy'], alpha=0.2, label='Healthy Zone')
        
        # Mark collapse point
        if tracker.collapse_step is not None:
            ax.axvline(x=tracker.collapse_step, color=self.colors['collapse'],
                      linestyle='--', linewidth=2, label='Collapse Detected')
            ax.annotate('Entropy Collapse', xy=(tracker.collapse_step, entropy[tracker.collapse_step]),
                       xytext=(tracker.collapse_step + len(steps) * 0.1, max(entropy) * 0.8),
                       arrowprops=dict(arrowstyle='->', color=self.colors['collapse']),
                       fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('Entropy', fontsize=12)
        ax.set_title('Policy Entropy Evolution', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    def _plot_risk_analysis(self, ax, tracker) -> None:
        """Plot collapse risk over time"""
        if len(tracker.full_history) < tracker.trend_window:
            ax.text(0.5, 0.5, 'Insufficient data for risk analysis',
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Calculate risks
        risks = []
        for i in range(tracker.trend_window, len(tracker.full_history)):
            window_data = tracker.full_history[i-tracker.trend_window:i]
            x = np.arange(len(window_data))
            y = np.array(window_data)
            trend = np.polyfit(x, y, 1)[0] if len(x) > 1 else 0
            
            current = tracker.full_history[i]
            risk = tracker._calculate_collapse_risk(current, trend)
            risks.append(risk)
        
        steps = tracker.timestamps[tracker.trend_window:]
        
        # Plot risk
        ax.fill_between(steps, 0, risks, color=self.colors['risk'], alpha=0.6)
        ax.plot(steps, risks, color=self.colors['risk'], linewidth=2)
        
        # Danger threshold
        ax.axhline(y=0.7, color='red', linestyle='--', label='High Risk Threshold')
        
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Collapse Risk')
        ax.set_title('Entropy Collapse Risk', fontweight='bold')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_entropy_distribution(self, ax, tracker) -> None:
        """Plot entropy value distribution"""
        entropy_values = np.array(tracker.full_history)
        
        # Histogram with KDE
        ax.hist(entropy_values, bins=30, density=True, alpha=0.6,
               color=self.colors['entropy'], edgecolor='black')
        
        # Add vertical lines for key values
        ax.axvline(tracker.collapse_threshold, color=self.colors['threshold'],
                  linestyle='--', label=f'Collapse: {tracker.collapse_threshold:.2f}')
        ax.axvline(tracker.min_healthy_entropy, color=self.colors['healthy'],
                  linestyle='--', label=f'Healthy Min: {tracker.min_healthy_entropy:.2f}')
        
        if tracker.initial_entropy:
            ax.axvline(tracker.initial_entropy, color='green',
                      linestyle=':', label=f'Initial: {tracker.initial_entropy:.2f}')
        
        ax.set_xlabel('Entropy Value')
        ax.set_ylabel('Density')
        ax.set_title('Entropy Distribution', fontweight='bold')
        ax.legend()
    
    def _plot_phase_diagram(self, ax, tracker) -> None:
        """Plot entropy vs trend phase diagram"""
        if len(tracker.full_history) < tracker.trend_window + 1:
            ax.text(0.5, 0.5, 'Insufficient data for phase diagram',
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Calculate entropy and trends
        entropies = []
        trends = []
        
        for i in range(tracker.trend_window, len(tracker.full_history)):
            window_data = tracker.full_history[i-tracker.trend_window:i]
            x = np.arange(len(window_data))
            y = np.array(window_data)
            trend = np.polyfit(x, y, 1)[0] if len(x) > 1 else 0
            
            entropies.append(tracker.full_history[i])
            trends.append(trend)
        
        # Color by time
        colors = np.arange(len(entropies))
        scatter = ax.scatter(entropies, trends, c=colors, cmap='viridis',
                           s=20, alpha=0.6, edgecolors='none')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Training Progress', rotation=270, labelpad=15)
        
        # Mark regions
        ax.axvline(tracker.collapse_threshold, color='red', linestyle='--', alpha=0.5)
        ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
        
        ax.set_xlabel('Entropy')
        ax.set_ylabel('Entropy Trend')
        ax.set_title('Entropy Phase Space', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_recovery_zones(self, ax, tracker) -> None:
        """Plot recovery strategy zones"""
        # Define zones
        zones = [
            (0, 0.3, 'Critical', self.colors['collapse']),
            (0.3, 0.5, 'Recovery Needed', self.colors['risk']),
            (0.5, 1.0, 'Stable', self.colors['healthy']),
            (1.0, 2.5, 'Optimal', self.colors['entropy'])
        ]
        
        for i, (low, high, label, color) in enumerate(zones):
            ax.barh(i, high - low, left=low, color=color, alpha=0.6, label=label)
        
        # Mark current entropy
        if tracker.history:
            current = tracker.history[-1]
            ax.axvline(current, color='black', linewidth=2, label=f'Current: {current:.2f}')
        
        ax.set_xlim(0, 2.5)
        ax.set_ylim(-0.5, len(zones) - 0.5)
        ax.set_xlabel('Entropy Value')
        ax.set_yticks(range(len(zones)))
        ax.set_yticklabels([z[2] for z in zones])
        ax.set_title('Entropy Recovery Zones', fontweight='bold')
        ax.legend(loc='upper right')
    
    def create_interactive_plot(self, tracker, save_path: Optional[str] = None) -> None:
        """
        Create interactive Plotly visualization.
        
        Args:
            tracker: EntropyTracker instance
            save_path: Optional HTML file path
        """
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly not available. Install with: pip install plotly")
            return
        
        if not tracker.full_history:
            logger.warning("No entropy data to visualize")
            return
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Entropy Timeline', 'Collapse Risk', 
                          'Entropy Distribution', 'Phase Diagram'),
            specs=[[{"colspan": 2}, None],
                   [{}, {}]]
        )
        
        # Timeline
        fig.add_trace(
            go.Scatter(x=tracker.timestamps, y=tracker.full_history,
                      mode='lines', name='Entropy',
                      line=dict(color=self.colors['entropy'], width=2)),
            row=1, col=1
        )
        
        # Add threshold lines
        fig.add_hline(y=tracker.collapse_threshold, line_dash="dash",
                     line_color=self.colors['threshold'],
                     annotation_text="Collapse Threshold",
                     row=1, col=1)
        
        # Risk calculation
        if len(tracker.full_history) >= tracker.trend_window:
            risks = []
            for i in range(tracker.trend_window, len(tracker.full_history)):
                window_data = tracker.full_history[i-tracker.trend_window:i]
                x = np.arange(len(window_data))
                y = np.array(window_data)
                trend = np.polyfit(x, y, 1)[0] if len(x) > 1 else 0
                
                current = tracker.full_history[i]
                risk = tracker._calculate_collapse_risk(current, trend)
                risks.append(risk)
            
            fig.add_trace(
                go.Scatter(x=tracker.timestamps[tracker.trend_window:], y=risks,
                          mode='lines', name='Collapse Risk',
                          line=dict(color=self.colors['risk'], width=2)),
                row=2, col=1
            )
        
        # Distribution
        fig.add_trace(
            go.Histogram(x=tracker.full_history, name='Entropy Distribution',
                        marker_color=self.colors['entropy']),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Interactive Entropy Analysis",
            showlegend=True,
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Interactive plot saved to {save_path}")
        else:
            fig.show()
    
    def generate_report(self, tracker, save_dir: str) -> None:
        """
        Generate comprehensive entropy analysis report.
        
        Args:
            tracker: EntropyTracker instance
            save_dir: Directory to save report files
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamp for report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate plots
        self.plot_entropy_dynamics(tracker, 
                                 save_path=save_dir / f"entropy_dynamics_{timestamp}.png")
        
        if PLOTLY_AVAILABLE:
            self.create_interactive_plot(tracker,
                                       save_path=save_dir / f"entropy_interactive_{timestamp}.html")
        
        # Save entropy history
        tracker.save_history(save_dir / f"entropy_history_{timestamp}.json")
        
        # Generate text report
        report = self._generate_text_report(tracker)
        with open(save_dir / f"entropy_report_{timestamp}.txt", 'w') as f:
            f.write(report)
        
        logger.info(f"Entropy analysis report saved to {save_dir}")
    
    def _generate_text_report(self, tracker) -> str:
        """Generate text summary report"""
        if not tracker.history:
            return "No entropy data available for report."
        
        metrics = tracker._calculate_metrics()
        
        report = f"""
ENTROPY ANALYSIS REPORT
======================
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

SUMMARY STATISTICS
------------------
Initial Entropy: {tracker.initial_entropy:.4f if tracker.initial_entropy else 'N/A'}
Current Entropy: {metrics.current:.4f}
Mean Entropy: {metrics.mean:.4f} (±{metrics.std:.4f})
Min/Max: {metrics.min:.4f} / {metrics.max:.4f}
Current Trend: {metrics.trend:.4f}
Collapse Risk: {metrics.collapse_risk:.2%}

COLLAPSE DETECTION
------------------
Collapse Detected: {'Yes' if tracker.collapse_detected else 'No'}
Collapse Step: {tracker.collapse_step if tracker.collapse_step else 'N/A'}
Collapse Threshold: {tracker.collapse_threshold:.4f}
Healthy Minimum: {tracker.min_healthy_entropy:.4f}

RECOMMENDATIONS
---------------
{tracker.get_recovery_recommendation() or 'System operating normally. No intervention needed.'}

ENTROPY DYNAMICS
----------------
Total Steps Tracked: {len(tracker.full_history)}
Episodes Completed: {tracker.timestamps[-1] if tracker.timestamps else 0}
"""
        
        # Add entropy drop analysis
        if tracker.initial_entropy and tracker.initial_entropy > 0:
            drop_percent = (tracker.initial_entropy - metrics.current) / tracker.initial_entropy * 100
            report += f"\nEntropy Drop: {drop_percent:.1f}% from initial value"
        
        return report


if __name__ == "__main__":
    # Validation example
    from entropy_tracker import EntropyTracker
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Running EntropyVisualizer validation...")
    
    # Create sample data
    tracker = EntropyTracker()
    np.random.seed(42)
    
    # Simulate entropy collapse
    for step in range(300):
        if step < 50:
            entropy = 2.0 + np.random.normal(0, 0.1)
        elif step < 150:
            progress = (step - 50) / 100
            entropy = 2.0 * (1 - progress) + 0.3 * progress + np.random.normal(0, 0.05)
        else:
            entropy = 0.3 + np.random.normal(0, 0.02)
        
        tracker.update(entropy, step)
    
    # Create visualizations
    visualizer = EntropyVisualizer()
    
    logger.info("Creating static visualization...")
    visualizer.plot_entropy_dynamics(tracker, save_path="entropy_validation.png")
    
    logger.info("Generating full report...")
    visualizer.generate_report(tracker, save_dir="entropy_reports")
    
    logger.info(" EntropyVisualizer validation complete!")