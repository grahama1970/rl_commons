"""
Module: entropy_aware_selector.py
Purpose: Algorithm selector with entropy health awareness

External Dependencies:
- numpy: https://numpy.org/doc/stable/

Example Usage:
>>> from rl_commons.core.entropy_aware_selector import EntropyAwareSelector
>>> selector = EntropyAwareSelector()
>>> agent = selector.select_algorithm(task_properties, current_entropy=0.3)
>>> print(f"Selected: {agent.__class__.__name__}")
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Type
from dataclasses import dataclass, field
import logging
from pathlib import Path
import json

from .algorithm_selector import AlgorithmSelector, TaskProperties, AlgorithmPerformance
from ..core.base import RLAgent
from ..monitoring.entropy_tracker import EntropyTracker, EntropyMetrics
from ..algorithms.ppo import EntropyAwarePPO, KLCovPPO

logger = logging.getLogger(__name__)


@dataclass
class EntropyHealth:
    """Current entropy health status"""
    current_entropy: float
    is_healthy: bool
    collapse_risk: float
    recommendation: Optional[str] = None
    entropy_history: List[float] = field(default_factory=list)
    
    @property
    def needs_intervention(self) -> bool:
        """Check if entropy intervention is needed"""
        return not self.is_healthy or self.collapse_risk > 0.7


class EntropyAwareSelector(AlgorithmSelector):
    """
    Enhanced algorithm selector that considers entropy health in selection decisions.
    
    Prefers entropy-aware algorithms when collapse risk is detected.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Entropy-specific configuration
        self.entropy_weight = self.config.get("entropy_weight", 0.3)
        self.entropy_threshold = self.config.get("entropy_threshold", 0.5)
        self.collapse_risk_threshold = self.config.get("collapse_risk_threshold", 0.7)
        
        # Register entropy-aware algorithms
        self._register_entropy_aware_algorithms()
        
        # Track entropy health by agent
        self.entropy_health_history: Dict[str, EntropyHealth] = {}
        
        logger.info("EntropyAwareSelector initialized with entropy weight %.2f", self.entropy_weight)
    
    def _register_entropy_aware_algorithms(self):
        """Register entropy-aware algorithm variants"""
        # Import enums from base class
        from .algorithm_selector import ActionSpace, TaskType
        
        # EntropyAwarePPO (Clip-Cov)
        self.register_algorithm(
            name="EntropyAwarePPO",
            agent_class=EntropyAwarePPO,
            suitable_for={
                "action_space": [ActionSpace.DISCRETE, ActionSpace.CONTINUOUS],
                "task_type": [TaskType.EPISODIC, TaskType.CONTINUOUS],
                "requires_stability": True,
                "requires_exploration": True
            },
            performance_profile={
                "sample_efficiency": 0.7,
                "stability": 0.85,
                "exploration": 0.9,  # High exploration maintenance
                "entropy_preservation": 0.95
            }
        )
        
        # KLCovPPO
        self.register_algorithm(
            name="KLCovPPO",
            agent_class=KLCovPPO,
            suitable_for={
                "action_space": [ActionSpace.DISCRETE, ActionSpace.CONTINUOUS],
                "task_type": [TaskType.EPISODIC, TaskType.CONTINUOUS],
                "requires_stability": True,
                "has_sparse_rewards": True  # Good for sparse rewards
            },
            performance_profile={
                "sample_efficiency": 0.75,
                "stability": 0.9,
                "exploration": 0.85,
                "entropy_preservation": 0.9
            }
        )
    
    def update_entropy_health(self,
                            agent: RLAgent,
                            current_entropy: float,
                            entropy_metrics: Optional[EntropyMetrics] = None) -> EntropyHealth:
        """
        Update entropy health tracking for an agent.
        
        Args:
            agent: The RL agent
            current_entropy: Current policy entropy
            entropy_metrics: Optional detailed entropy metrics
            
        Returns:
            Updated entropy health status
        """
        agent_name = agent.name
        
        # Get or create health tracking
        if agent_name not in self.entropy_health_history:
            self.entropy_health_history[agent_name] = EntropyHealth(
                current_entropy=current_entropy,
                is_healthy=current_entropy > self.entropy_threshold,
                collapse_risk=0.0
            )
        
        health = self.entropy_health_history[agent_name]
        health.current_entropy = current_entropy
        health.entropy_history.append(current_entropy)
        
        # Update health status
        health.is_healthy = current_entropy > self.entropy_threshold
        
        # Calculate collapse risk
        if entropy_metrics:
            health.collapse_risk = entropy_metrics.collapse_risk
        else:
            # Simple heuristic if no metrics
            if len(health.entropy_history) > 1:
                entropy_drop = (health.entropy_history[0] - current_entropy) / max(health.entropy_history[0], 0.1)
                health.collapse_risk = min(1.0, entropy_drop)
        
        # Generate recommendation
        if health.collapse_risk > self.collapse_risk_threshold:
            health.recommendation = "High collapse risk - switch to entropy-aware algorithm"
        elif not health.is_healthy:
            health.recommendation = "Low entropy - consider entropy-aware variant"
        else:
            health.recommendation = None
        
        logger.debug(f"Entropy health for {agent_name}: entropy={current_entropy:.3f}, "
                    f"risk={health.collapse_risk:.2f}, healthy={health.is_healthy}")
        
        return health
    
    def _calculate_algorithm_scores(self, 
                                  task_properties: TaskProperties,
                                  performance_weight: float = 0.7,
                                  entropy_health: Optional[EntropyHealth] = None) -> List[Tuple[str, float]]:
        """
        Calculate algorithm scores with entropy awareness.
        
        Extends base scoring to consider entropy health.
        """
        # Get base scores
        base_scores = super()._calculate_algorithm_scores(task_properties, performance_weight)
        
        if not entropy_health:
            return base_scores
        
        # Adjust scores based on entropy health
        adjusted_scores = []
        
        for name, base_score in base_scores:
            score = base_score
            
            # Boost entropy-aware algorithms if entropy is unhealthy
            if entropy_health.needs_intervention:
                algo_info = self.algorithm_registry.get(name, {})
                profile = algo_info.get("performance_profile", {})
                
                # Check if algorithm has good entropy preservation
                entropy_preservation = profile.get("entropy_preservation", 0.5)
                
                if entropy_preservation > 0.8:
                    # Significant boost for entropy-aware algorithms
                    boost = self.entropy_weight * (1 - entropy_health.current_entropy / self.entropy_threshold)
                    score = score * (1 + boost)
                    logger.debug(f"Boosted {name} score by {boost:.2f} due to entropy concerns")
                elif name in ["DQN", "DoubleDQN"]:
                    # Penalize algorithms known for entropy collapse
                    penalty = self.entropy_weight * entropy_health.collapse_risk * 0.5
                    score = score * (1 - penalty)
                    logger.debug(f"Penalized {name} score by {penalty:.2f} due to entropy risk")
            
            adjusted_scores.append((name, score))
        
        # Re-sort by adjusted scores
        adjusted_scores.sort(key=lambda x: x[1], reverse=True)
        return adjusted_scores
    
    def select_algorithm(self, 
                        task_properties: TaskProperties,
                        force_algorithm: Optional[str] = None,
                        current_entropy: Optional[float] = None,
                        entropy_metrics: Optional[EntropyMetrics] = None) -> RLAgent:
        """
        Select algorithm considering entropy health.
        
        Args:
            task_properties: Task characteristics
            force_algorithm: Force specific algorithm selection
            current_entropy: Current policy entropy (if switching)
            entropy_metrics: Detailed entropy metrics
            
        Returns:
            Selected RL agent
        """
        # Check if we're switching due to entropy concerns
        entropy_health = None
        if current_entropy is not None:
            entropy_health = EntropyHealth(
                current_entropy=current_entropy,
                is_healthy=current_entropy > self.entropy_threshold,
                collapse_risk=entropy_metrics.collapse_risk if entropy_metrics else 0.0
            )
            
            if entropy_health.needs_intervention:
                logger.warning(f"Entropy intervention needed: entropy={current_entropy:.3f}, "
                             f"risk={entropy_health.collapse_risk:.2f}")
        
        # Force entropy-aware algorithm if specified
        if force_algorithm:
            return super().select_algorithm(task_properties, force_algorithm)
        
        # Calculate scores with entropy awareness
        scores = self._calculate_algorithm_scores(
            task_properties, 
            entropy_health=entropy_health
        )
        
        # Select based on scores
        if np.random.random() < self.exploration_rate:
            # Exploration with entropy bias
            k = min(3, len(scores))
            top_k = scores[:k]
            
            # If entropy is unhealthy, bias toward entropy-aware algorithms
            if entropy_health and entropy_health.needs_intervention:
                entropy_aware_algos = ["EntropyAwarePPO", "KLCovPPO", "PPO", "A3C"]
                entropy_scores = [(n, s) for n, s in top_k if n in entropy_aware_algos]
                
                if entropy_scores:
                    top_k = entropy_scores
                    logger.info("Biasing exploration toward entropy-aware algorithms")
            
            weights = np.array([s[1] for s in top_k])
            weights = weights / weights.sum()
            idx = np.random.choice(len(top_k), p=weights)
            selected = top_k[idx][0]
            logger.info(f"Exploration with entropy awareness: selected {selected}")
        else:
            # Exploit best option
            selected = scores[0][0]
            logger.info(f"Exploitation: selected {selected} (score: {scores[0][1]:.3f})")
        
        # Track selection
        self.selection_count[selected] += 1
        
        # Create agent with entropy tracking enabled
        agent = self._create_agent(selected, task_properties, enable_entropy_tracking=True)
        
        return agent
    
    def _create_agent(self, 
                     algorithm_name: str, 
                     task_properties: TaskProperties,
                     enable_entropy_tracking: bool = True) -> RLAgent:
        """Create agent with entropy tracking enabled"""
        algo_info = self.algorithm_registry[algorithm_name]
        agent_class = algo_info["class"]
        
        # Import time for naming
        import time
        
        # Build configuration
        config = {
            "name": f"{algorithm_name}_{int(time.time())}",
            "track_entropy": enable_entropy_tracking,
            "entropy_tracker_config": {
                "window_size": 100,
                "collapse_threshold": self.entropy_threshold,
                "min_healthy_entropy": self.entropy_threshold
            }
        }
        
        # Add dimensions
        if task_properties.state_dim is not None:
            config["state_dim"] = task_properties.state_dim
        if task_properties.action_dim is not None:
            config["action_dim"] = task_properties.action_dim
        
        # Algorithm-specific configuration
        if algorithm_name == "EntropyAwarePPO":
            config.update({
                "clip_cov_percentile": 99.8,
                "asymmetric_clipping": True,
                "adaptive_entropy_coef": True,
                "entropy_target": max(0.5, self.entropy_threshold)
            })
        elif algorithm_name == "KLCovPPO":
            config.update({
                "kl_coef": 0.1,
                "kl_cov_percentile": 99.8,
                "dynamic_kl_coef": True,
                "kl_target": 0.01
            })
        elif algorithm_name in ["PPO", "A3C"]:
            config["continuous_actions"] = task_properties.has_continuous_actions
        
        # Create agent - handle different initialization patterns
        if algorithm_name in ["PPO", "A3C", "DQN", "DoubleDQN"]:
            # These agents require positional arguments
            agent = agent_class(
                name=config["name"],
                state_dim=config.get("state_dim", 4),
                action_dim=config.get("action_dim", 2),
                continuous=config.get("continuous_actions", True) if algorithm_name in ["PPO", "A3C"] else None,
                config=config
            )
        elif algorithm_name in ["EntropyAwarePPO", "KLCovPPO"]:
            # These use named arguments - filter out internal config
            filtered_config = {k: v for k, v in config.items() 
                             if k not in ["name", "state_dim", "action_dim", "track_entropy", "entropy_tracker_config"]}
            agent = agent_class(
                name=config["name"],
                state_dim=config.get("state_dim", 4),
                action_dim=config.get("action_dim", 2),
                continuous=True,
                **filtered_config
            )
        else:
            # Generic fallback
            agent = agent_class(config=config)
        
        # Store reference
        self.active_agents[agent.name] = agent
        
        logger.info(f"Created {algorithm_name} agent with entropy tracking: {agent.name}")
        return agent
    
    def should_switch_algorithm(self,
                              current_agent: RLAgent,
                              task_properties: TaskProperties,
                              recent_performance: List[float],
                              current_entropy: Optional[float] = None) -> bool:
        """
        Determine if algorithm should be switched, considering entropy.
        
        Extends base method to check entropy health.
        """
        # Check base conditions
        should_switch = super().should_switch_algorithm(
            current_agent, task_properties, recent_performance
        )
        
        if should_switch:
            return True
        
        # Check entropy health
        if current_entropy is not None:
            agent_name = current_agent.name
            
            # Update health tracking
            health = self.update_entropy_health(current_agent, current_entropy)
            
            # Switch if entropy is critically low and current algo isn't entropy-aware
            current_algo = current_agent.__class__.__name__
            if health.collapse_risk > self.collapse_risk_threshold:
                if current_algo not in ["EntropyAwarePPO", "KLCovPPO"]:
                    logger.warning(f"Switching from {current_algo} due to entropy collapse risk: {health.collapse_risk:.2f}")
                    return True
            
            # Switch if entropy has been unhealthy for too long
            if len(health.entropy_history) > 20:
                recent_entropy = health.entropy_history[-20:]
                unhealthy_count = sum(1 for e in recent_entropy if e < self.entropy_threshold)
                if unhealthy_count > 15:  # 75% unhealthy
                    logger.warning(f"Switching from {current_algo} due to persistent low entropy")
                    return True
        
        return False
    
    def get_entropy_statistics(self) -> Dict[str, Any]:
        """Get entropy health statistics across all agents"""
        stats = {
            "total_agents_tracked": len(self.entropy_health_history),
            "healthy_agents": 0,
            "at_risk_agents": 0,
            "collapsed_agents": 0,
            "average_entropy": 0.0,
            "average_collapse_risk": 0.0,
            "agents_needing_intervention": []
        }
        
        total_entropy = 0.0
        total_risk = 0.0
        
        for agent_name, health in self.entropy_health_history.items():
            total_entropy += health.current_entropy
            total_risk += health.collapse_risk
            
            if health.is_healthy and health.collapse_risk < 0.3:
                stats["healthy_agents"] += 1
            elif health.collapse_risk > self.collapse_risk_threshold:
                stats["collapsed_agents"] += 1
                stats["agents_needing_intervention"].append(agent_name)
            else:
                stats["at_risk_agents"] += 1
        
        if stats["total_agents_tracked"] > 0:
            stats["average_entropy"] = total_entropy / stats["total_agents_tracked"]
            stats["average_collapse_risk"] = total_risk / stats["total_agents_tracked"]
        
        return stats
    
    def recommend_entropy_intervention(self, 
                                     agent: RLAgent,
                                     current_entropy: float) -> Dict[str, Any]:
        """
        Recommend specific intervention for low entropy.
        
        Returns:
            Dictionary with intervention recommendations
        """
        health = self.update_entropy_health(agent, current_entropy)
        current_algo = agent.__class__.__name__
        
        intervention = {
            "needs_intervention": health.needs_intervention,
            "current_algorithm": current_algo,
            "current_entropy": current_entropy,
            "collapse_risk": health.collapse_risk,
            "recommendations": []
        }
        
        if not health.needs_intervention:
            intervention["recommendations"].append("No intervention needed - entropy is healthy")
            return intervention
        
        # Specific recommendations based on severity
        if health.collapse_risk > 0.9:
            # Critical - immediate switch
            intervention["recommendations"].append("CRITICAL: Immediate algorithm switch recommended")
            intervention["recommended_algorithm"] = "EntropyAwarePPO"
            intervention["urgency"] = "immediate"
        elif health.collapse_risk > self.collapse_risk_threshold:
            # High risk - switch soon
            intervention["recommendations"].append("HIGH RISK: Consider switching to entropy-aware algorithm")
            intervention["recommended_algorithm"] = "KLCovPPO" if current_algo == "PPO" else "EntropyAwarePPO"
            intervention["urgency"] = "high"
        elif not health.is_healthy:
            # Low entropy but not critical
            intervention["recommendations"].append("Monitor closely and consider adjusting entropy coefficient")
            if hasattr(agent, 'entropy_coef'):
                intervention["recommendations"].append(f"Increase entropy coefficient (current: {agent.entropy_coef})")
            intervention["urgency"] = "medium"
        
        # Add general recommendations
        if health.collapse_risk > 0.5:
            intervention["recommendations"].extend([
                "Reduce learning rate to slow down convergence",
                "Add exploration noise or increase temperature",
                "Consider curriculum adjustment to easier tasks temporarily"
            ])
        
        return intervention


if __name__ == "__main__":
    import time
    
    # Test entropy-aware selection
    print("Testing EntropyAwareSelector...")
    
    selector = EntropyAwareSelector({
        "entropy_weight": 0.3,
        "entropy_threshold": 0.5
    })
    
    # Import needed classes
    from .algorithm_selector import TaskType, ActionSpace
    
    # Test 1: Normal selection
    print("\nTest 1: Normal selection with healthy entropy")
    task_props = TaskProperties(
        task_type=TaskType.EPISODIC,
        action_space=ActionSpace.CONTINUOUS,
        state_dim=4,
        action_dim=2,
        requires_stability=True,
        requires_exploration=True
    )
    
    agent1 = selector.select_algorithm(task_props)
    print(f"Selected: {agent1.__class__.__name__}")
    
    # Test 2: Selection with low entropy
    print("\nTest 2: Selection with low entropy")
    agent2 = selector.select_algorithm(
        task_props,
        current_entropy=0.3,  # Low entropy
        entropy_metrics=EntropyMetrics(
            current=0.3,
            mean=0.4,
            std=0.1,
            min=0.2,
            max=0.8,
            trend=-0.02,
            collapse_risk=0.8
        )
    )
    print(f"Selected with low entropy: {agent2.__class__.__name__}")
    
    # Test 3: Check switching recommendation
    print("\nTest 3: Switching recommendation")
    
    # Simulate standard PPO with collapsing entropy
    from ..algorithms.ppo import PPOAgent
    standard_ppo = PPOAgent(
        name="test_ppo",
        state_dim=4,
        action_dim=2,
        continuous=True
    )
    
    should_switch = selector.should_switch_algorithm(
        standard_ppo,
        task_props,
        recent_performance=[100, 95, 90, 85, 80, 75, 70, 65, 60, 55],
        current_entropy=0.2
    )
    print(f"Should switch from PPO: {should_switch}")
    
    # Test 4: Get intervention recommendation
    print("\nTest 4: Intervention recommendation")
    intervention = selector.recommend_entropy_intervention(standard_ppo, current_entropy=0.2)
    print(f"Intervention needed: {intervention['needs_intervention']}")
    print(f"Urgency: {intervention.get('urgency', 'N/A')}")
    for rec in intervention['recommendations']:
        print(f"  - {rec}")
    
    # Test 5: Entropy statistics
    print("\nTest 5: Entropy statistics")
    
    # Update some entropy health data
    selector.update_entropy_health(agent1, 1.5)  # Healthy
    selector.update_entropy_health(agent2, 0.3)  # Unhealthy
    selector.update_entropy_health(standard_ppo, 0.1)  # Critical
    
    stats = selector.get_entropy_statistics()
    print(f"Entropy statistics: {json.dumps(stats, indent=2)}")
    
    print("\n✅ EntropyAwareSelector validation complete!")