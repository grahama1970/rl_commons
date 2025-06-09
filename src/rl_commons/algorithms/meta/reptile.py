"""
Module: reptile.py
Purpose: Reptile meta-learning algorithm (simpler alternative to MAML)

External Dependencies:
- torch: https://pytorch.org/docs/stable/
- numpy: https://numpy.org/doc/stable/

Example Usage:
>>> from rl_commons.algorithms.meta import Reptile
>>> reptile = Reptile(model, inner_lr=0.01, meta_lr=0.1)
>>> reptile.train_on_task(task_data, num_steps=100)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Any, Optional
import numpy as np
from copy import deepcopy


class Reptile:
    """Reptile meta-learning algorithm for fast adaptation."""
    
    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        meta_lr: float = 0.1,
        device: str = None
    ):
        """Initialize Reptile.
        
        Args:
            model: Base model to meta-train
            inner_lr: Learning rate for task-specific training
            meta_lr: Meta learning rate (interpolation factor)
            device: Device for computation
        """
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model.to(self.device)
        self.initial_params = deepcopy(self.model.state_dict())
        
        # Track training progress
        self.task_losses = []
        self.meta_iterations = 0
        
    def train_on_task(
        self,
        task_data: Dict[str, torch.Tensor],
        num_steps: int = 100
    ) -> nn.Module:
        """Train on a single task.
        
        Args:
            task_data: Task-specific training data
            num_steps: Number of gradient steps
            
        Returns:
            Task-adapted model
        """
        # Clone model for task-specific training
        task_model = deepcopy(self.model)
        task_model.train()
        
        # Task-specific optimizer
        optimizer = optim.Adam(task_model.parameters(), lr=self.inner_lr)
        
        # Training loop
        losses = []
        for step in range(num_steps):
            # Sample batch from task data
            batch_size = min(32, len(task_data['states']))
            indices = np.random.choice(len(task_data['states']), batch_size, replace=False)
            
            states = task_data['states'][indices].to(self.device)
            targets = task_data['targets'][indices].to(self.device)
            
            # Forward pass
            predictions = task_model(states)
            
            # Compute loss
            if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                # Classification
                loss = nn.CrossEntropyLoss()(predictions, targets.long())
            else:
                # Regression
                loss = nn.MSELoss()(predictions.squeeze(), targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        self.task_losses.extend(losses)
        return task_model
    
    def meta_update(
        self,
        tasks: List[Dict[str, torch.Tensor]],
        num_steps_per_task: int = 100
    ):
        """Perform Reptile meta-update.
        
        Args:
            tasks: List of task data dictionaries
            num_steps_per_task: Training steps per task
        """
        # Store initial parameters
        initial_params = deepcopy(self.model.state_dict())
        
        # Accumulate parameter updates
        accumulated_grads = {}
        for name, param in self.model.named_parameters():
            accumulated_grads[name] = torch.zeros_like(param.data)
        
        # Train on each task
        for task_data in tasks:
            # Train on task
            task_model = self.train_on_task(task_data, num_steps_per_task)
            
            # Compute parameter difference
            task_params = task_model.state_dict()
            for name, param in self.model.named_parameters():
                if name in task_params:
                    # Reptile gradient: (θ_task - θ_initial)
                    accumulated_grads[name] += (task_params[name] - initial_params[name])
        
        # Average gradients
        for name in accumulated_grads:
            accumulated_grads[name] /= len(tasks)
        
        # Reptile meta-update: θ = θ + ε * (θ_task - θ)
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in accumulated_grads:
                    param.data += self.meta_lr * accumulated_grads[name]
        
        self.meta_iterations += 1
    
    def adapt_to_task(
        self,
        task_data: Dict[str, torch.Tensor],
        num_steps: int = 50
    ) -> nn.Module:
        """Quickly adapt to a new task.
        
        Args:
            task_data: New task data
            num_steps: Number of adaptation steps
            
        Returns:
            Adapted model
        """
        return self.train_on_task(task_data, num_steps)
    
    def evaluate_adaptation(
        self,
        tasks: List[Dict[str, torch.Tensor]],
        num_adaptation_steps: int = 50
    ) -> Dict[str, float]:
        """Evaluate adaptation performance.
        
        Args:
            tasks: List of evaluation tasks
            num_adaptation_steps: Steps for adaptation
            
        Returns:
            Evaluation metrics
        """
        initial_losses = []
        adapted_losses = []
        
        for task_data in tasks:
            # Split into support and query
            n = len(task_data['states'])
            split_idx = n // 2
            
            support_data = {
                'states': task_data['states'][:split_idx],
                'targets': task_data['targets'][:split_idx]
            }
            query_data = {
                'states': task_data['states'][split_idx:],
                'targets': task_data['targets'][split_idx:]
            }
            
            # Initial performance
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(query_data['states'].to(self.device))
                if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                    initial_loss = nn.CrossEntropyLoss()(
                        predictions,
                        query_data['targets'].long().to(self.device)
                    ).item()
                else:
                    initial_loss = nn.MSELoss()(
                        predictions.squeeze(),
                        query_data['targets'].to(self.device)
                    ).item()
            initial_losses.append(initial_loss)
            
            # Adapt to task
            adapted_model = self.adapt_to_task(support_data, num_adaptation_steps)
            
            # Adapted performance
            adapted_model.eval()
            with torch.no_grad():
                predictions = adapted_model(query_data['states'].to(self.device))
                if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                    adapted_loss = nn.CrossEntropyLoss()(
                        predictions,
                        query_data['targets'].long().to(self.device)
                    ).item()
                else:
                    adapted_loss = nn.MSELoss()(
                        predictions.squeeze(),
                        query_data['targets'].to(self.device)
                    ).item()
            adapted_losses.append(adapted_loss)
        
        return {
            'avg_initial_loss': np.mean(initial_losses),
            'avg_adapted_loss': np.mean(adapted_losses),
            'avg_improvement': np.mean([
                (init - adapt) / (init + 1e-8)
                for init, adapt in zip(initial_losses, adapted_losses)
            ]),
            'adaptation_steps': num_adaptation_steps
        }
    
    def save(self, path: str):
        """Save model and meta-learning state."""
        torch.save({
            'model_state': self.model.state_dict(),
            'meta_iterations': self.meta_iterations,
            'inner_lr': self.inner_lr,
            'meta_lr': self.meta_lr
        }, path)
    
    def load(self, path: str):
        """Load model and meta-learning state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.meta_iterations = checkpoint['meta_iterations']
        self.inner_lr = checkpoint['inner_lr']
        self.meta_lr = checkpoint['meta_lr']


if __name__ == "__main__":
    # Validation test
    # Simple classification model
    model = nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 4)  # 4-class classification
    )
    
    # Initialize Reptile
    reptile = Reptile(model, inner_lr=0.01, meta_lr=0.1)
    
    # Create dummy tasks
    tasks = []
    for _ in range(3):
        task_data = {
            'states': torch.randn(100, 8),
            'targets': torch.randint(0, 4, (100,))
        }
        tasks.append(task_data)
    
    # Test meta-update
    initial_params = deepcopy(model.state_dict())
    reptile.meta_update(tasks, num_steps_per_task=50)
    updated_params = model.state_dict()
    
    # Check that parameters changed
    params_changed = False
    for name in initial_params:
        if not torch.equal(initial_params[name], updated_params[name]):
            params_changed = True
            break
    
    assert params_changed, "Parameters didn't change after meta-update"
    
    # Test adaptation
    new_task = {
        'states': torch.randn(50, 8),
        'targets': torch.randint(0, 4, (50,))
    }
    adapted_model = reptile.adapt_to_task(new_task, num_steps=10)
    assert adapted_model is not None, "Adaptation failed"
    
    # Test evaluation
    eval_metrics = reptile.evaluate_adaptation([new_task], num_adaptation_steps=10)
    assert 'avg_improvement' in eval_metrics, "Missing evaluation metrics"
    
    print("✅ Reptile validation passed")