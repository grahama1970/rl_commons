"""
Module: gnn_integration.py
Purpose: Integration of GNN with existing RL algorithms

External Dependencies:
- torch: https://pytorch.org/docs/stable/
- numpy: https://numpy.org/doc/stable/

Example Usage:
>>> from rl_commons.algorithms.gnn import GNNEnhancedDQN
>>> gnn_dqn = GNNEnhancedDQN(node_dim=64, action_dim=4)
>>> action = gnn_dqn.select_action(graph_state)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from dataclasses import dataclass

from .graph_networks import GraphNetwork, GraphData
from ...core.base import RLState, RLAction, RLReward
from ..dqn.vanilla_dqn import DQNAgent
from ..ppo.ppo import PPOAgent


class GraphState(RLState):
    """State representation for graph-structured environments."""
    def __init__(self, features: np.ndarray, graph_data: GraphData, 
                 global_features: Optional[torch.Tensor] = None, 
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(features, context)
        self.graph_data = graph_data
        self.global_features = global_features
    
    def to_tensor(self, device='cpu') -> Dict[str, torch.Tensor]:
        """Convert to tensor representation."""
        result = {
            'node_features': self.graph_data.node_features.to(device),
            'edge_index': self.graph_data.edge_index.to(device)
        }
        if self.graph_data.edge_features is not None:
            result['edge_features'] = self.graph_data.edge_features.to(device)
        if self.global_features is not None:
            result['global_features'] = self.global_features.to(device)
        return result


class GraphStateEncoder(nn.Module):
    """Encodes graph states for RL algorithms."""
    
    def __init__(self, node_dim: int, edge_dim: Optional[int] = None,
                 hidden_dim: int = 128, num_gnn_layers: int = 3,
                 output_dim: int = 256, use_attention: bool = False):
        """Initialize graph state encoder.
        
        Args:
            node_dim: Node feature dimension
            edge_dim: Edge feature dimension
            hidden_dim: GNN hidden dimension
            num_gnn_layers: Number of GNN layers
            output_dim: Output encoding dimension
            use_attention: Whether to use attention
        """
        super().__init__()
        
        # GNN for processing graph structure
        self.gnn = GraphNetwork(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gnn_layers,
            use_attention=use_attention,
            global_pooling='mean'
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        
    def forward(self, graph_state: GraphState) -> torch.Tensor:
        """Encode graph state to vector representation.
        
        Args:
            graph_state: Graph-structured state
            
        Returns:
            State encoding vector
        """
        # Process through GNN
        gnn_output = self.gnn(graph_state.graph_data)
        graph_embedding = gnn_output['graph_embedding']
        
        # Add global features if available
        if graph_state.global_features is not None:
            graph_embedding = torch.cat([
                graph_embedding,
                graph_state.global_features.unsqueeze(0) 
                if graph_embedding.dim() == 1 else graph_state.global_features
            ], dim=-1)
        
        # Project to output dimension
        encoding = self.output_proj(graph_embedding)
        
        return encoding.squeeze(0) if encoding.size(0) == 1 else encoding


class GNNQNetwork(nn.Module):
    """Q-Network using GNN for state processing."""
    
    def __init__(self, state_encoder: GraphStateEncoder, action_dim: int,
                 hidden_dim: int = 256):
        """Initialize GNN Q-Network.
        
        Args:
            state_encoder: Graph state encoder
            action_dim: Number of actions
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.state_encoder = state_encoder
        
        # Q-value head
        self.q_head = nn.Sequential(
            nn.Linear(state_encoder.output_proj[-1].out_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, graph_state: GraphState) -> torch.Tensor:
        """Compute Q-values for graph state.
        
        Args:
            graph_state: Current graph state
            
        Returns:
            Q-values for each action
        """
        state_encoding = self.state_encoder(graph_state)
        q_values = self.q_head(state_encoding)
        return q_values


class GNNEnhancedDQN:
    """DQN agent enhanced with Graph Neural Networks."""
    
    def __init__(self, node_dim: int, action_dim: int,
                 edge_dim: Optional[int] = None,
                 hidden_dim: int = 128,
                 num_gnn_layers: int = 3,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 buffer_size: int = 10000,
                 batch_size: int = 32,
                 update_frequency: int = 4,
                 use_attention: bool = False):
        """Initialize GNN-enhanced DQN.
        
        Args:
            node_dim: Node feature dimension
            action_dim: Number of actions
            edge_dim: Edge feature dimension
            hidden_dim: Hidden layer size
            num_gnn_layers: Number of GNN layers
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay rate
            epsilon_min: Minimum epsilon
            buffer_size: Replay buffer size
            batch_size: Training batch size
            update_frequency: Target network update frequency
            use_attention: Whether to use graph attention
        """
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create state encoder
        self.state_encoder = GraphStateEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_gnn_layers=num_gnn_layers,
            output_dim=256,
            use_attention=use_attention
        ).to(self.device)
        
        # Q-networks
        self.q_network = GNNQNetwork(
            self.state_encoder,
            action_dim,
            hidden_dim=256
        ).to(self.device)
        
        self.target_network = GNNQNetwork(
            GraphStateEncoder(
                node_dim=node_dim,
                edge_dim=edge_dim,
                hidden_dim=hidden_dim,
                num_gnn_layers=num_gnn_layers,
                output_dim=256,
                use_attention=use_attention
            ),
            action_dim,
            hidden_dim=256
        ).to(self.device)
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer for graph states
        self.replay_buffer = []
        self.buffer_size = buffer_size
        
        # Training metrics
        self.steps = 0
        self.update_count = 0
        self.losses = []
        
    def select_action(self, graph_state: GraphState, explore: bool = True) -> RLAction:
        """Select action using epsilon-greedy policy.
        
        Args:
            graph_state: Current graph state
            explore: Whether to use exploration
            
        Returns:
            Selected action
        """
        if explore and np.random.random() < self.epsilon:
            action_id = np.random.randint(0, self.action_dim)
        else:
            with torch.no_grad():
                q_values = self.q_network(graph_state)
                action_id = q_values.argmax().item()
        
        return RLAction(
            action_type='discrete',
            action_id=action_id,
            parameters={'q_values': q_values.cpu().numpy() if 'q_values' in locals() else None}
        )
    
    def update(self, state: GraphState, action: RLAction, reward: float,
               next_state: GraphState, done: bool):
        """Update Q-network with experience.
        
        Args:
            state: Current graph state
            action: Action taken
            reward: Reward received
            next_state: Next graph state
            done: Whether episode ended
        """
        # Store experience
        if len(self.replay_buffer) >= self.buffer_size:
            self.replay_buffer.pop(0)
            
        self.replay_buffer.append({
            'state': state,
            'action': action.action_id,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })
        
        # Update network
        if len(self.replay_buffer) >= self.batch_size:
            if self.steps % self.update_frequency == 0:
                self._update_network()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.steps += 1
        
    def _update_network(self):
        """Perform network update using replay buffer."""
        # Sample batch
        indices = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in indices]
        
        # Prepare batch tensors
        states = [exp['state'] for exp in batch]
        actions = torch.LongTensor([exp['action'] for exp in batch]).to(self.device)
        rewards = torch.FloatTensor([exp['reward'] for exp in batch]).to(self.device)
        next_states = [exp['next_state'] for exp in batch]
        dones = torch.FloatTensor([exp['done'] for exp in batch]).to(self.device)
        
        # Current Q values
        current_q_values = []
        for state in states:
            q_vals = self.q_network(state)
            current_q_values.append(q_vals)
        current_q_values = torch.stack(current_q_values)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1))
        
        # Target Q values
        next_q_values = []
        with torch.no_grad():
            for next_state in next_states:
                next_q = self.target_network(next_state)
                next_q_values.append(next_q)
            next_q_values = torch.stack(next_q_values)
            max_next_q = next_q_values.max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q
        
        # Loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        self.losses.append(loss.item())
        self.update_count += 1
        
        # Update target network
        if self.update_count % 100 == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())


class GNNEnhancedPPO:
    """PPO agent enhanced with Graph Neural Networks."""
    
    def __init__(self, node_dim: int, action_dim: int,
                 edge_dim: Optional[int] = None,
                 hidden_dim: int = 128,
                 num_gnn_layers: int = 3,
                 continuous: bool = False,
                 use_attention: bool = False):
        """Initialize GNN-enhanced PPO.
        
        Args:
            node_dim: Node feature dimension
            action_dim: Action dimension
            edge_dim: Edge feature dimension
            hidden_dim: Hidden layer size
            num_gnn_layers: Number of GNN layers
            continuous: Whether action space is continuous
            use_attention: Whether to use graph attention
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # State encoder
        self.state_encoder = GraphStateEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_gnn_layers=num_gnn_layers,
            output_dim=256,
            use_attention=use_attention
        ).to(self.device)
        
        # Actor-Critic networks
        self.actor = self._build_actor(256, action_dim, continuous)
        self.critic = self._build_critic(256)
        
        # Note: Full PPO implementation would include:
        # - Rollout buffer
        # - GAE computation
        # - PPO loss with clipping
        # - Multiple update epochs
        
    def _build_actor(self, input_dim: int, action_dim: int, 
                     continuous: bool) -> nn.Module:
        """Build actor network."""
        if continuous:
            return nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, action_dim * 2)  # mean and log_std
            ).to(self.device)
        else:
            return nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, action_dim),
                nn.Softmax(dim=-1)
            ).to(self.device)
    
    def _build_critic(self, input_dim: int) -> nn.Module:
        """Build critic network."""
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(self.device)


class GNNIntegration:
    """Main integration class for GNN with RL algorithms."""
    
    def __init__(self, node_features: int, edge_features: int = 0, hidden_dim: int = 64):
        """Initialize GNN integration.
        
        Args:
            node_features: Number of node features
            edge_features: Number of edge features
            hidden_dim: Hidden dimension for GNN
        """
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        
    def create_gnn_actor(self, node_features: int, edge_features: int, 
                        action_dim: int, use_attention: bool = True) -> nn.Module:
        """Create a GNN-based actor network.
        
        Args:
            node_features: Number of node features
            edge_features: Number of edge features
            action_dim: Action dimension
            use_attention: Whether to use attention mechanism
            
        Returns:
            GNN actor network
        """
        return GraphStateEncoder(
            node_dim=node_features,
            edge_dim=edge_features,
            hidden_dim=self.hidden_dim,
            num_gnn_layers=3,
            output_dim=action_dim,
            use_attention=use_attention
        )
    
    def create_gnn_critic(self, node_features: int, edge_features: int,
                         use_attention: bool = True) -> nn.Module:
        """Create a GNN-based critic network.
        
        Args:
            node_features: Number of node features
            edge_features: Number of edge features
            use_attention: Whether to use attention mechanism
            
        Returns:
            GNN critic network
        """
        return GraphStateEncoder(
            node_dim=node_features,
            edge_dim=edge_features,
            hidden_dim=self.hidden_dim,
            num_gnn_layers=3,
            output_dim=1,  # Value function output
            use_attention=use_attention
        )
    
    def process_graph(self, nodes: Any, edges: Any, 
                     node_features: torch.Tensor, 
                     edge_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Process graph data through GNN.
        
        Args:
            nodes: Node identifiers
            edges: Edge connections
            node_features: Node feature tensor
            edge_features: Optional edge feature tensor
            
        Returns:
            Processed graph features
        """
        # Create graph data
        if isinstance(edges, list):
            edge_index = torch.tensor(edges, dtype=torch.long).t()
        else:
            edge_index = edges
            
        graph_data = GraphData(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features
        )
        
        # Process through a simple GNN
        gnn = GraphNetwork(
            node_dim=node_features.shape[1],
            edge_dim=edge_features.shape[1] if edge_features is not None else 0,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim
        )
        
        return gnn(graph_data)


if __name__ == "__main__":
    # Validation test
    # Create a module graph: 5 modules with connections
    num_modules = 5
    node_features = torch.randn(num_modules, 32)  # 32-dim features per module
    
    # Module connections (directed graph)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4],  # source modules
        [1, 2, 3, 3, 4, 4, 1, 0]   # target modules
    ], dtype=torch.long)
    
    graph_data = GraphData(node_features=node_features, edge_index=edge_index)
    graph_state = GraphState(features=node_features.flatten().numpy(), graph_data=graph_data)
    
    # Test GNN-enhanced DQN
    gnn_dqn = GNNEnhancedDQN(
        node_dim=32,
        action_dim=5,  # 5 possible routing decisions
        num_gnn_layers=2
    )
    
    # Select action
    action = gnn_dqn.select_action(graph_state)
    assert isinstance(action, RLAction), "Invalid action type"
    assert 0 <= action.action_id < 5, f"Invalid action: {action.action_id}"
    
    # Simulate experience
    next_graph_state = GraphState(
        features=torch.randn(num_modules * 32).numpy(),
        graph_data=GraphData(
            node_features=torch.randn(num_modules, 32),
            edge_index=edge_index
        )
    )
    
    gnn_dqn.update(graph_state, action, 0.5, next_graph_state, False)
    
    assert len(gnn_dqn.replay_buffer) == 1, "Experience not stored"
    
    print("✅ GNN integration validation passed")