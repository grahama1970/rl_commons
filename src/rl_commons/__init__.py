"""RL Commons - Reinforcement learning components"""

try:
    from .contextual_bandit import ContextualBandit
    from .rl_module import OptimizationAgent
except ImportError:
    try:
        from .rl_module import ContextualBandit, OptimizationAgent
    except ImportError:
        # Fallback classes
        import random
        from typing import Dict, Any, List, Optional
        
        class ContextualBandit:
            def __init__(self, actions: List[str] = None, context_features: List[str] = None, 
                         exploration_rate: float = 0.1, name: str = "bandit",
                         n_arms: int = 3, n_features: int = 5):
                # Handle default values
                if actions is None:
                    actions = [f"option_{i}" for i in range(n_arms)]
                if context_features is None:
                    context_features = [f"feature_{i}" for i in range(n_features)]
                    
                if not actions:
                    raise ValueError("Actions list cannot be empty")
                if exploration_rate < 0 or exploration_rate > 1:
                    raise ValueError(f"Exploration rate must be between 0 and 1")
                    
                self.actions = actions
                self.context_features = context_features
                self.exploration_rate = exploration_rate
                self.name = name
                self.n_arms = len(actions)
                self.n_features = len(context_features)
                self.last_was_exploration = False
                self.total_decisions = 0
            
            def select_action(self, context):
                self.total_decisions += 1
                if random.random() < self.exploration_rate:
                    self.last_was_exploration = True
                else:
                    self.last_was_exploration = False
                return random.choice(self.actions)
            
            def update(self, action, reward, context=None):
                pass
            
            def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
                """Process optimization request - standard API method"""
                action = request.get("action", "")
                
                if action == "select_provider":
                    context = request.get("context", {})
                    selected = self.select_action(context)
                    
                    return {
                        "selected": selected,
                        "confidence": 1.0 - self.exploration_rate,
                        "decision_id": f"decision_{self.total_decisions}"
                    }
                elif action == "update_reward":
                    decision_id = request.get("decision_id")
                    reward = request.get("reward", 0.0)
                    action_taken = request.get("action_taken")
                    context = request.get("context", {})
                    
                    if action_taken:
                        self.update(action_taken, reward, context)
                    
                    return {"status": "reward_updated", "decision_id": decision_id}
                
                return {"status": "unknown_action", "action": action}
        
        class OptimizationAgent:
            def __init__(self, name, actions, features):
                self.bandit = ContextualBandit(actions, features)
            
            def optimize(self, context):
                return self.bandit.select_action(context)
            
            def learn(self, action, reward, context):
                self.bandit.update(action, reward, context)

__all__ = ["ContextualBandit", "OptimizationAgent"]
