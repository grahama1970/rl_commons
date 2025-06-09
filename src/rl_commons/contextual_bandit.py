from typing import Dict, Any, List, Optional
"""
Module: contextual_bandit.py
Description: Contextual bandit implementation for decision making

External Dependencies:
- None (standard library only)

Sample Input:
>>> bandit = ContextualBandit(actions=["a", "b", "c"], context_features=["f1"], exploration_rate=0.1)
>>> action = bandit.select_action({"f1": 0.5})

Expected Output:
>>> print(action)
"b"
"""

import random
import math
from typing import List, Dict, Any, Optional
from collections import defaultdict


class ContextualBandit:
    """Contextual multi-armed bandit for decision making"""
    
    def __init__(self, actions: List[str] = None, context_features: List[str] = None, 
                 exploration_rate: float = 0.1, name: str = "bandit", 
                 n_arms: int = 3, n_features: int = 5):
        """
        Initialize contextual bandit
        
        Args:
            actions: List of possible actions
            context_features: List of context feature names
            exploration_rate: Epsilon for epsilon-greedy exploration
            name: Name of the bandit
            n_arms: Number of arms (for compatibility)
            n_features: Number of features (for compatibility)
        """
        # Handle default values
        if actions is None:
            actions = [f"option_{i}" for i in range(n_arms)]
        if context_features is None:
            context_features = [f"feature_{i}" for i in range(n_features)]
        
        if not actions:
            raise ValueError("Actions list cannot be empty")
        
        if exploration_rate < 0 or exploration_rate > 1:
            raise ValueError(f"Exploration rate must be between 0 and 1, got {exploration_rate}")
        
        self.actions = actions
        self.context_features = context_features
        self.exploration_rate = exploration_rate
        self.name = name
        self.n_arms = len(actions)
        self.n_features = len(context_features)
        
        # Track rewards for each action
        self.action_values = defaultdict(float)
        self.action_counts = defaultdict(int)
        
        # Track context-specific rewards
        self.context_values = defaultdict(lambda: defaultdict(float))
        self.context_counts = defaultdict(lambda: defaultdict(int))
        
        # Track last decision info
        self.last_was_exploration = False
        self.total_decisions = 0
    
    def select_action(self, context: Dict[str, Any]) -> str:
        """
        Select an action given the context
        
        Args:
            context: Dictionary of context features
            
        Returns:
            Selected action
        """
        self.total_decisions += 1
        
        # Epsilon-greedy exploration
        if random.random() < self.exploration_rate:
            self.last_was_exploration = True
            return random.choice(self.actions)
        
        self.last_was_exploration = False
        
        # Exploitation: choose best action based on context
        context_key = self._get_context_key(context)
        
        # If we have context-specific data, use it
        if context_key in self.context_counts:
            best_action = max(
                self.actions,
                key=lambda a: self._get_action_value(a, context_key)
            )
        else:
            # Fall back to global action values
            best_action = max(
                self.actions,
                key=lambda a: self.action_values[a] if self.action_counts[a] > 0 else 0
            )
        
        return best_action
    

    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process optimization request - standard API method"""
        action = request.get("action", "")
        
        if action == "select_provider":
            context = request.get("context", {})
            # Convert context to features
            features = []
            for feature in self.context_features:
                features.append(context.get(feature, 0.0))
            
            # Select action using bandit algorithm
            selected_idx = self.select_action(features)
            selected = self.actions[selected_idx] if selected_idx < len(self.actions) else self.actions[0]
            
            return {
                "selected": selected,
                "confidence": 1.0 - self.exploration_rate,
                "decision_id": f"decision_{self.n_selections}"
            }
        elif action == "update_reward":
            # Update the bandit with reward information
            decision_id = request.get("decision_id")
            reward = request.get("reward", 0.0)
            return {"status": "reward_updated", "decision_id": decision_id}
        
        return {"status": "unknown_action", "action": action}

    def update(self, action: str, reward: float, context: Optional[Dict[str, Any]] = None):
        """
        Update action values based on reward
        
        Args:
            action: Action that was taken
            reward: Reward received
            context: Context in which action was taken
        """
        if action not in self.actions:
            return
        
        # Update global action values
        self.action_counts[action] += 1
        count = self.action_counts[action]
        current_value = self.action_values[action]
        
        # Incremental average update
        self.action_values[action] = current_value + (reward - current_value) / count
        
        # Update context-specific values if context provided
        if context:
            context_key = self._get_context_key(context)
            self.context_counts[context_key][action] += 1
            
            ctx_count = self.context_counts[context_key][action]
            ctx_value = self.context_values[context_key][action]
            
            self.context_values[context_key][action] = ctx_value + (reward - ctx_value) / ctx_count
    
    def _get_context_key(self, context: Dict[str, Any]) -> str:
        """Create a hashable key from context"""
        # Simple discretization of context features
        key_parts = []
        for feature in self.context_features:
            if feature in context:
                value = context[feature]
                if isinstance(value, (int, float)):
                    # Discretize numeric values
                    discretized = int(value * 10) / 10
                    key_parts.append(f"{feature}={discretized}")
                else:
                    key_parts.append(f"{feature}={value}")
        
        return "|".join(key_parts)
    
    def _get_action_value(self, action: str, context_key: str) -> float:
        """Get the value of an action in a specific context"""
        if self.context_counts[context_key][action] > 0:
            return self.context_values[context_key][action]
        
        # Fall back to global value
        if self.action_counts[action] > 0:
            return self.action_values[action]
        
        # No data, return 0
        return 0.0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get bandit statistics"""
        return {
            "total_decisions": self.total_decisions,
            "exploration_rate": self.exploration_rate,
            "action_counts": dict(self.action_counts),
            "action_values": dict(self.action_values),
            "unique_contexts": len(self.context_counts)
        }
    
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process optimization request - standard API method"""
        action = request.get("action", "")
        
        if action == "select_provider":
            context = request.get("context", {})
            # Convert context dict to list of feature values
            features = []
            for feature in self.context_features:
                features.append(context.get(feature, 0.0))
            
            # Select action using bandit algorithm
            selected = self.select_action(context)
            
            return {
                "selected": selected,
                "confidence": 1.0 - self.exploration_rate,
                "decision_id": f"decision_{self.total_decisions}"
            }
        elif action == "update_reward":
            # Update the bandit with reward information
            decision_id = request.get("decision_id")
            reward = request.get("reward", 0.0)
            action_taken = request.get("action_taken")
            context = request.get("context", {})
            
            if action_taken:
                self.update(action_taken, reward, context)
            
            return {"status": "reward_updated", "decision_id": decision_id}
        
        return {"status": "unknown_action", "action": action}


if __name__ == "__main__":
    # Test the bandit
    bandit = ContextualBandit(
        actions=["action_a", "action_b", "action_c"],
        context_features=["time_of_day", "load"],
        exploration_rate=0.1
    )
    
    # Simulate some decisions
    for i in range(10):
        context = {
            "time_of_day": i / 24,
            "load": random.random()
        }
        
        action = bandit.select_action(context)
        reward = random.random()
        bandit.update(action, reward, context)
        
        print(f"Decision {i+1}: {action} -> reward {reward:.2f}")
    
    # Print statistics
    stats = bandit.get_statistics()
    print(f"\n✅ Made {stats['total_decisions']} decisions")
    print(f"   Action counts: {stats['action_counts']}")
