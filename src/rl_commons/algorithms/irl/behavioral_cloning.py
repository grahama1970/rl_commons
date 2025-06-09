"""
Module: behavioral_cloning.py
Purpose: Behavioral Cloning for imitation learning

External Dependencies:
- torch: https://pytorch.org/docs/stable/
- numpy: https://numpy.org/doc/stable/

Example Usage:
>>> from rl_commons.algorithms.irl import BehavioralCloning
>>> bc = BehavioralCloning(state_dim=8, action_dim=4)
>>> bc.train(demonstrations)
>>> action = bc.predict(state)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Any, Optional, Callable, Union
import time

from .base_irl import Demonstration
from ...core.base import RLState, RLAction


class BehavioralCloning:
    """Behavioral Cloning for learning policies from demonstrations."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        discrete_actions: bool = True,
        hidden_dim: int = 128,
        num_layers: int = 3,
        learning_rate: float = 0.001,
        dropout_rate: float = 0.1,
        device: str = None
    ):
        """Initialize Behavioral Cloning.
        
        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            discrete_actions: Whether actions are discrete
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            learning_rate: Learning rate
            dropout_rate: Dropout rate for regularization
            device: Device for computation
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete_actions = discrete_actions
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Build policy network
        layers = []
        input_dim = state_dim
        
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, action_dim))
        
        self.policy_network = nn.Sequential(*layers).to(self.device)
        
        # Loss function
        if discrete_actions:
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = nn.MSELoss()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        
        # Training history
        self.training_history = {
            'losses': [],
            'accuracies': [],
            'val_losses': [],
            'val_accuracies': []
        }
    
    def train(
        self,
        demonstrations: List[Demonstration],
        num_epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """Train policy using behavioral cloning.
        
        Args:
            demonstrations: List of expert demonstrations
            num_epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Fraction of data for validation
            
        Returns:
            Training results
        """
        # Combine all demonstrations
        all_states = []
        all_actions = []
        
        for demo in demonstrations:
            all_states.append(demo.states)
            all_actions.append(demo.actions)
        
        all_states = torch.cat(all_states, dim=0)
        all_actions = torch.cat(all_actions, dim=0)
        
        # Train/validation split
        n_samples = len(all_states)
        n_train = int(n_samples * (1 - validation_split))
        
        indices = torch.randperm(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        train_states = all_states[train_indices].to(self.device)
        train_actions = all_actions[train_indices].to(self.device)
        val_states = all_states[val_indices].to(self.device)
        val_actions = all_actions[val_indices].to(self.device)
        
        print(f"Training on {len(train_states)} samples, validating on {len(val_states)} samples")
        
        # Training loop
        for epoch in range(num_epochs):
            # Training
            self.policy_network.train()
            train_losses = []
            train_correct = 0
            
            # Mini-batch training
            for i in range(0, len(train_states), batch_size):
                batch_states = train_states[i:i+batch_size]
                batch_actions = train_actions[i:i+batch_size]
                
                # Forward pass
                predictions = self.policy_network(batch_states)
                
                # Compute loss
                if self.discrete_actions:
                    loss = self.loss_fn(predictions, batch_actions.long())
                    
                    # Compute accuracy
                    pred_actions = predictions.argmax(dim=-1)
                    train_correct += (pred_actions == batch_actions).sum().item()
                else:
                    loss = self.loss_fn(predictions, batch_actions)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
                
                self.optimizer.step()
                
                train_losses.append(loss.item())
            
            # Validation
            self.policy_network.eval()
            val_losses = []
            val_correct = 0
            
            with torch.no_grad():
                for i in range(0, len(val_states), batch_size):
                    batch_states = val_states[i:i+batch_size]
                    batch_actions = val_actions[i:i+batch_size]
                    
                    predictions = self.policy_network(batch_states)
                    
                    if self.discrete_actions:
                        loss = self.loss_fn(predictions, batch_actions.long())
                        pred_actions = predictions.argmax(dim=-1)
                        val_correct += (pred_actions == batch_actions).sum().item()
                    else:
                        loss = self.loss_fn(predictions, batch_actions)
                    
                    val_losses.append(loss.item())
            
            # Record metrics
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            
            self.training_history['losses'].append(avg_train_loss)
            self.training_history['val_losses'].append(avg_val_loss)
            
            if self.discrete_actions:
                train_acc = train_correct / len(train_states)
                val_acc = val_correct / len(val_states)
                self.training_history['accuracies'].append(train_acc)
                self.training_history['val_accuracies'].append(val_acc)
                
                if epoch % 20 == 0:
                    print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, "
                          f"Train Acc: {train_acc:.4f}, Val Loss: {avg_val_loss:.4f}, "
                          f"Val Acc: {val_acc:.4f}")
            else:
                if epoch % 20 == 0:
                    print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, "
                          f"Val Loss: {avg_val_loss:.4f}")
        
        return {
            'final_train_loss': self.training_history['losses'][-1],
            'final_val_loss': self.training_history['val_losses'][-1],
            'final_train_acc': self.training_history['accuracies'][-1] if self.discrete_actions else None,
            'final_val_acc': self.training_history['val_accuracies'][-1] if self.discrete_actions else None
        }
    
    def predict(self, state: Union[RLState, torch.Tensor]) -> RLAction:
        """Predict action for given state.
        
        Args:
            state: State to predict action for
            
        Returns:
            Predicted action
        """
        self.policy_network.eval()
        
        # Convert state to tensor
        if isinstance(state, RLState):
            state_tensor = torch.FloatTensor(state.features).unsqueeze(0).to(self.device)
        else:
            state_tensor = state.unsqueeze(0).to(self.device) if state.dim() == 1 else state.to(self.device)
        
        with torch.no_grad():
            prediction = self.policy_network(state_tensor)
            
            if self.discrete_actions:
                # Get action probabilities
                probs = nn.functional.softmax(prediction, dim=-1)
                action_id = prediction.argmax(dim=-1).item()
                
                return RLAction(
                    action_type='discrete',
                    action_id=action_id,
                    parameters={'probs': probs.cpu().numpy()}
                )
            else:
                # Continuous action
                action_values = prediction.squeeze().cpu().numpy()
                
                return RLAction(
                    action_type='continuous',
                    action_id=0,
                    parameters={'values': action_values}
                )
    
    def evaluate(self, demonstrations: List[Demonstration]) -> Dict[str, float]:
        """Evaluate policy on demonstrations.
        
        Args:
            demonstrations: Test demonstrations
            
        Returns:
            Evaluation metrics
        """
        self.policy_network.eval()
        
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for demo in demonstrations:
                demo = demo.to_device(self.device)
                
                predictions = self.policy_network(demo.states)
                
                if self.discrete_actions:
                    loss = self.loss_fn(predictions, demo.actions.long())
                    pred_actions = predictions.argmax(dim=-1)
                    total_correct += (pred_actions == demo.actions).sum().item()
                else:
                    loss = self.loss_fn(predictions, demo.actions)
                
                total_loss += loss.item() * len(demo.states)
                total_samples += len(demo.states)
        
        metrics = {
            'avg_loss': total_loss / total_samples,
            'accuracy': total_correct / total_samples if self.discrete_actions else None
        }
        
        return metrics
    
    def save(self, path: str):
        """Save model."""
        torch.save({
            'policy_network': self.policy_network.state_dict(),
            'training_history': self.training_history,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'discrete_actions': self.discrete_actions
        }, path)
    
    def load(self, path: str):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_network.load_state_dict(checkpoint['policy_network'])
        self.training_history = checkpoint['training_history']


if __name__ == "__main__":
    # Validation test
    # Create expert demonstrations
    demonstrations = []
    for _ in range(10):
        states = torch.randn(100, 8)
        actions = torch.randint(0, 4, (100,))
        demo = Demonstration(states, actions)
        demonstrations.append(demo)
    
    # Initialize BC
    bc = BehavioralCloning(state_dim=8, action_dim=4, discrete_actions=True)
    
    # Train
    print("Training behavioral cloning...")
    results = bc.train(demonstrations, num_epochs=20, batch_size=32)
    print(f"Training results: {results}")
    
    # Test prediction
    test_state = RLState(features=np.random.randn(8))
    action = bc.predict(test_state)
    
    assert isinstance(action, RLAction), "Invalid action type"
    assert 0 <= action.action_id < 4, f"Invalid action: {action.action_id}"
    assert 'probs' in action.parameters, "Missing action probabilities"
    
    # Evaluate
    test_demos = [Demonstration(torch.randn(20, 8), torch.randint(0, 4, (20,)))]
    metrics = bc.evaluate(test_demos)
    print(f"Evaluation metrics: {metrics}")
    
    assert 'accuracy' in metrics, "Missing accuracy metric"
    assert metrics['accuracy'] is not None, "Accuracy should be computed for discrete actions"
    
    print(" Behavioral Cloning validation passed")