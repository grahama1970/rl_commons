"""
Module: maml.py
Purpose: Model-Agnostic Meta-Learning implementation for RL

External Dependencies:
- torch: https://pytorch.org/docs/stable/
- numpy: https://numpy.org/doc/stable/

Example Usage:
>>> from rl_commons.algorithms.meta import MAML
>>> maml = MAML(model, inner_lr=0.01, meta_lr=0.001)
>>> adapted_model = maml.adapt(task_data, num_steps=5)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Any, Optional, Tuple, Callable
import numpy as np
from copy import deepcopy
from collections import OrderedDict

from ...core.base import RLState, RLAction, RLReward, RLAgent


class MAML:
    """Model-Agnostic Meta-Learning for fast adaptation."""
    
    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        meta_lr: float = 0.001,
        num_inner_steps: int = 5,
        first_order: bool = False,
        device: str = None
    ):
        """Initialize MAML.
        
        Args:
            model: Base model to meta-train
            inner_lr: Learning rate for inner loop adaptation
            meta_lr: Learning rate for meta-optimization
            num_inner_steps: Number of gradient steps in inner loop
            first_order: Whether to use first-order approximation
            device: Device to use for computation
        """
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.num_inner_steps = num_inner_steps
        self.first_order = first_order
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model.to(self.device)
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=meta_lr)
        
        # Track meta-training progress
        self.meta_losses = []
        self.adaptation_losses = []
        
    def inner_loop(
        self,
        support_data: Dict[str, torch.Tensor],
        num_steps: Optional[int] = None
    ) -> nn.Module:
        """Perform inner loop adaptation on support data.
        
        Args:
            support_data: Data for adaptation containing 'states', 'actions', 'rewards'
            num_steps: Number of adaptation steps (default: self.num_inner_steps)
            
        Returns:
            Adapted model
        """
        num_steps = num_steps or self.num_inner_steps
        
        # Clone model for adaptation
        adapted_model = deepcopy(self.model)
        adapted_model.to(self.device)
        adapted_model.train()
        
        # Inner loop optimizer
        inner_optimizer = optim.SGD(adapted_model.parameters(), lr=self.inner_lr)
        
        for step in range(num_steps):
            # Compute loss on support data
            loss = self.compute_loss(adapted_model, support_data)
            
            # Update adapted model
            inner_optimizer.zero_grad()
            loss.backward(create_graph=not self.first_order)
            inner_optimizer.step()
            
            self.adaptation_losses.append(loss.item())
        
        return adapted_model
    
    def compute_loss(
        self,
        model: nn.Module,
        data: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute loss for given model and data.
        
        Args:
            model: Model to evaluate
            data: Data dictionary with 'states', 'actions', 'rewards', etc.
            
        Returns:
            Loss tensor
        """
        states = data['states'].to(self.device)
        
        # Forward pass
        predictions = model(states)
        
        # Compute loss based on task type
        if 'targets' in data:
            targets = data['targets'].to(self.device)
            if len(predictions.shape) > 1 and predictions.shape[1] > 1 and len(targets.shape) == 1:
                # Classification task
                loss = nn.CrossEntropyLoss()(predictions, targets.long())
            else:
                # Regression task
                loss = nn.MSELoss()(predictions.squeeze(), targets)
        elif 'actions' in data:
            actions = data['actions'].to(self.device)
            # Cross-entropy for discrete actions
            loss = nn.CrossEntropyLoss()(predictions, actions.long())
        elif 'rewards' in data:
            rewards = data['rewards'].to(self.device)
            loss = nn.MSELoss()(predictions.squeeze(), rewards)
        else:
            raise ValueError("Data must contain 'targets', 'actions', or 'rewards'")
        
        return loss
    
    def meta_update(
        self,
        tasks: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]
    ) -> float:
        """Perform meta-update across multiple tasks.
        
        Args:
            tasks: List of (support_data, query_data) tuples
            
        Returns:
            Average meta-loss
        """
        meta_loss = 0.0
        
        for support_data, query_data in tasks:
            # Adapt to task using support data
            adapted_model = self.inner_loop(support_data)
            
            # Evaluate on query data
            task_loss = self.compute_loss(adapted_model, query_data)
            meta_loss += task_loss
        
        # Meta-optimization step
        meta_loss = meta_loss / len(tasks)
        
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.meta_optimizer.step()
        
        self.meta_losses.append(meta_loss.item())
        
        return meta_loss.item()
    
    def adapt(
        self,
        task_data: Dict[str, torch.Tensor],
        num_steps: Optional[int] = None
    ) -> nn.Module:
        """Adapt to a new task.
        
        Args:
            task_data: Data from new task
            num_steps: Number of adaptation steps
            
        Returns:
            Model adapted to new task
        """
        return self.inner_loop(task_data, num_steps)
    
    def evaluate(
        self,
        tasks: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]],
        num_steps: Optional[int] = None
    ) -> Dict[str, float]:
        """Evaluate meta-learned model on tasks.
        
        Args:
            tasks: List of (support_data, query_data) tuples
            num_steps: Number of adaptation steps
            
        Returns:
            Evaluation metrics
        """
        num_steps = num_steps or self.num_inner_steps
        
        support_losses = []
        query_losses = []
        improvements = []
        
        for support_data, query_data in tasks:
            # Initial performance
            initial_loss = self.compute_loss(self.model, support_data).item()
            support_losses.append(initial_loss)
            
            # Adapt to task
            adapted_model = self.inner_loop(support_data, num_steps)
            
            # Performance after adaptation
            adapted_loss = self.compute_loss(adapted_model, query_data).item()
            query_losses.append(adapted_loss)
            
            # Improvement
            improvement = (initial_loss - adapted_loss) / (initial_loss + 1e-8)
            improvements.append(improvement)
        
        return {
            'avg_support_loss': np.mean(support_losses),
            'avg_query_loss': np.mean(query_losses),
            'avg_improvement': np.mean(improvements),
            'std_improvement': np.std(improvements)
        }


class MAMLAgent(RLAgent):
    """RL Agent that uses MAML for fast adaptation."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        inner_lr: float = 0.01,
        meta_lr: float = 0.001,
        num_inner_steps: int = 5
    ):
        """Initialize MAML agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Hidden layer dimension
            inner_lr: Inner loop learning rate
            meta_lr: Meta learning rate
            num_inner_steps: Adaptation steps
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Base network
        self.base_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # MAML wrapper
        self.maml = MAML(
            self.base_network,
            inner_lr=inner_lr,
            meta_lr=meta_lr,
            num_inner_steps=num_inner_steps
        )
        
        # Current adapted model
        self.adapted_model = None
        self.adaptation_data = {
            'states': [],
            'actions': [],
            'rewards': []
        }
        
    def select_action(self, state: RLState) -> RLAction:
        """Select action using current model."""
        state_tensor = torch.FloatTensor(state.features).unsqueeze(0).to(self.maml.device)
        
        # Use adapted model if available
        model = self.adapted_model if self.adapted_model else self.base_network
        
        with torch.no_grad():
            action_logits = model(state_tensor)
            
            # Softmax for discrete actions
            action_probs = torch.softmax(action_logits, dim=-1)
            action = torch.multinomial(action_probs, 1).item()
        
        return RLAction(
            action_type='discrete',
            action_id=action,
            parameters={'logits': action_logits.cpu().numpy()}
        )
    
    def update_policy(
        self,
        state: RLState,
        action: RLAction,
        reward: RLReward,
        next_state: RLState,
        done: bool
    ):
        """Collect experience for adaptation."""
        self.adaptation_data['states'].append(state.features)
        self.adaptation_data['actions'].append(action.action_id)
        self.adaptation_data['rewards'].append(reward.value)
    
    def adapt_to_task(self, num_steps: Optional[int] = None):
        """Adapt to current task using collected data."""
        if len(self.adaptation_data['states']) == 0:
            return
        
        # Convert to tensors
        task_data = {
            'states': torch.FloatTensor(self.adaptation_data['states']),
            'actions': torch.LongTensor(self.adaptation_data['actions']),
            'rewards': torch.FloatTensor(self.adaptation_data['rewards'])
        }
        
        # Adapt model
        self.adapted_model = self.maml.adapt(task_data, num_steps)
    
    def meta_update(self, task_batch: List[Dict[str, Any]]):
        """Perform meta-update with task batch."""
        # Convert task batch to MAML format
        tasks = []
        for task in task_batch:
            support_data = {
                'states': torch.FloatTensor(task['support']['states']),
                'actions': torch.LongTensor(task['support']['actions']),
                'rewards': torch.FloatTensor(task['support']['rewards'])
            }
            query_data = {
                'states': torch.FloatTensor(task['query']['states']),
                'actions': torch.LongTensor(task['query']['actions']),
                'rewards': torch.FloatTensor(task['query']['rewards'])
            }
            tasks.append((support_data, query_data))
        
        # Meta-update
        meta_loss = self.maml.meta_update(tasks)
        return meta_loss
    
    def reset_adaptation(self):
        """Reset adaptation for new task."""
        self.adapted_model = None
        self.adaptation_data = {
            'states': [],
            'actions': [],
            'rewards': []
        }
    
    def update(self, *args, **kwargs):
        """Update agent (calls update_policy for compatibility)."""
        return self.update_policy(*args, **kwargs)
    
    def save(self, path: str):
        """Save agent state."""
        torch.save({
            'base_network': self.base_network.state_dict(),
            'maml_state': {
                'meta_losses': self.maml.meta_losses,
                'adaptation_losses': self.maml.adaptation_losses
            },
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }, path)
    
    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path)
        self.base_network.load_state_dict(checkpoint['base_network'])
        self.maml.meta_losses = checkpoint['maml_state']['meta_losses']
        self.maml.adaptation_losses = checkpoint['maml_state']['adaptation_losses']


if __name__ == "__main__":
    # Validation test
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(4, 32),
        nn.ReLU(),
        nn.Linear(32, 2)
    )
    
    # Initialize MAML
    maml = MAML(model, inner_lr=0.01, meta_lr=0.001)
    
    # Create dummy task data
    support_data = {
        'states': torch.randn(16, 4),
        'actions': torch.randint(0, 2, (16,)),
        'rewards': torch.randn(16)
    }
    query_data = {
        'states': torch.randn(16, 4),
        'actions': torch.randint(0, 2, (16,)),
        'rewards': torch.randn(16)
    }
    
    # Test adaptation
    adapted_model = maml.adapt(support_data, num_steps=3)
    assert adapted_model is not None, "Adaptation failed"
    
    # Test meta-update
    tasks = [(support_data, query_data) for _ in range(4)]
    meta_loss = maml.meta_update(tasks)
    assert isinstance(meta_loss, float), f"Invalid meta loss: {meta_loss}"
    assert meta_loss > 0, "Meta loss should be positive"
    
    # Test MAML agent
    agent = MAMLAgent(state_dim=4, action_dim=2)
    state = RLState(features=np.random.randn(4))
    action = agent.select_action(state)
    
    assert isinstance(action, RLAction), "Invalid action type"
    assert 0 <= action.action_id < 2, f"Invalid action: {action.action_id}"
    
    print("✅ MAML validation passed")