"""
Module: graph_networks.py
Purpose: Core Graph Neural Network implementations for RL

External Dependencies:
- torch: https://pytorch.org/docs/stable/
- torch_geometric: https://pytorch-geometric.readthedocs.io/
- numpy: https://numpy.org/doc/stable/

Example Usage:
>>> from rl_commons.algorithms.gnn import GraphNetwork
>>> gnn = GraphNetwork(node_dim=64, edge_dim=32, hidden_dim=128)
>>> node_embeddings = gnn(node_features, edge_index, edge_features)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
from dataclasses import dataclass


@dataclass
class GraphData:
    """Container for graph data."""
    node_features: torch.Tensor  # [num_nodes, node_dim]
    edge_index: torch.Tensor     # [2, num_edges]
    edge_features: Optional[torch.Tensor] = None  # [num_edges, edge_dim]
    batch: Optional[torch.Tensor] = None  # [num_nodes] for batched graphs
    
    def to(self, device):
        """Move graph data to device."""
        return GraphData(
            node_features=self.node_features.to(device),
            edge_index=self.edge_index.to(device),
            edge_features=self.edge_features.to(device) if self.edge_features is not None else None,
            batch=self.batch.to(device) if self.batch is not None else None
        )


class MessagePassingLayer(nn.Module):
    """Basic message passing layer for GNNs."""
    
    def __init__(self, node_dim: int, edge_dim: Optional[int] = None, 
                 hidden_dim: int = 128, aggregation: str = 'mean'):
        """Initialize message passing layer.
        
        Args:
            node_dim: Dimension of node features
            edge_dim: Dimension of edge features (optional)
            hidden_dim: Hidden dimension for message computation
            aggregation: Aggregation method ('mean', 'sum', 'max')
        """
        super().__init__()
        self.aggregation = aggregation
        
        # Message computation
        message_input_dim = node_dim * 2  # source and target nodes
        if edge_dim is not None:
            message_input_dim += edge_dim
            
        self.message_mlp = nn.Sequential(
            nn.Linear(message_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )
        
        # Node update
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim * 2, hidden_dim),  # old features + aggregated messages
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )
        
    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor,
                edge_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of message passing.
        
        Args:
            node_features: Node feature tensor [num_nodes, node_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_features: Optional edge features [num_edges, edge_dim]
            
        Returns:
            Updated node features [num_nodes, node_dim]
        """
        num_nodes = node_features.size(0)
        row, col = edge_index[0], edge_index[1]  # row: source, col: target
        
        # Compute messages
        source_features = node_features[row]
        target_features = node_features[col]
        
        if edge_features is not None:
            message_input = torch.cat([source_features, target_features, edge_features], dim=-1)
        else:
            message_input = torch.cat([source_features, target_features], dim=-1)
            
        messages = self.message_mlp(message_input)
        
        # Aggregate messages
        aggregated = self._aggregate(messages, col, num_nodes)
        
        # Update nodes
        node_input = torch.cat([node_features, aggregated], dim=-1)
        updated_features = self.node_mlp(node_input)
        
        # Residual connection
        return node_features + updated_features
    
    def _aggregate(self, messages: torch.Tensor, indices: torch.Tensor, 
                   num_nodes: int) -> torch.Tensor:
        """Aggregate messages for each node.
        
        Args:
            messages: Messages to aggregate [num_edges, node_dim]
            indices: Target node indices for each message
            num_nodes: Total number of nodes
            
        Returns:
            Aggregated messages [num_nodes, node_dim]
        """
        node_dim = messages.size(-1)
        aggregated = torch.zeros(num_nodes, node_dim, device=messages.device)
        
        if self.aggregation == 'sum':
            aggregated.scatter_add_(0, indices.unsqueeze(-1).expand_as(messages), messages)
        elif self.aggregation == 'mean':
            aggregated.scatter_add_(0, indices.unsqueeze(-1).expand_as(messages), messages)
            # Count messages per node for mean
            count = torch.zeros(num_nodes, device=messages.device)
            count.scatter_add_(0, indices, torch.ones_like(indices, dtype=torch.float))
            count = count.clamp(min=1).unsqueeze(-1)
            aggregated = aggregated / count
        elif self.aggregation == 'max':
            aggregated, _ = torch.zeros_like(aggregated).scatter_reduce_(
                0, indices.unsqueeze(-1).expand_as(messages), messages, reduce='amax'
            )
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")
            
        return aggregated


class GraphAttentionLayer(nn.Module):
    """Graph Attention Network (GAT) layer."""
    
    def __init__(self, node_dim: int, hidden_dim: int = 128, 
                 num_heads: int = 4, dropout: float = 0.1):
        """Initialize GAT layer.
        
        Args:
            node_dim: Dimension of node features
            hidden_dim: Hidden dimension per attention head
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Multi-head attention
        self.W = nn.Linear(node_dim, hidden_dim * num_heads, bias=False)
        self.a_src = nn.Parameter(torch.zeros(num_heads, hidden_dim))
        self.a_dst = nn.Parameter(torch.zeros(num_heads, hidden_dim))
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim * num_heads, node_dim)
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        
    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention.
        
        Args:
            node_features: Node features [num_nodes, node_dim]
            edge_index: Edge connectivity [2, num_edges]
            
        Returns:
            Updated node features [num_nodes, node_dim]
        """
        num_nodes = node_features.size(0)
        
        # Linear transformation
        h = self.W(node_features)  # [num_nodes, hidden_dim * num_heads]
        h = h.view(num_nodes, self.num_heads, self.hidden_dim)
        
        # Compute attention scores
        row, col = edge_index[0], edge_index[1]
        
        # Source and target features for edges
        h_src = h[row]  # [num_edges, num_heads, hidden_dim]
        h_dst = h[col]
        
        # Attention scores
        e_src = (h_src * self.a_src).sum(dim=-1)  # [num_edges, num_heads]
        e_dst = (h_dst * self.a_dst).sum(dim=-1)
        e = self.leaky_relu(e_src + e_dst)
        
        # Softmax per node
        e_exp = e.exp()
        e_sum = torch.zeros(num_nodes, self.num_heads, device=e.device)
        e_sum.scatter_add_(0, col.unsqueeze(-1).expand_as(e_exp), e_exp)
        
        # Normalize
        alpha = e_exp / (e_sum[col] + 1e-8)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Aggregate with attention weights
        out = torch.zeros(num_nodes, self.num_heads, self.hidden_dim, device=h.device)
        alpha_expanded = alpha.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        out.scatter_add_(0, col.view(-1, 1, 1).expand(-1, self.num_heads, self.hidden_dim),
                         h_src * alpha_expanded)
        
        # Concatenate heads and project
        out = out.view(num_nodes, -1)
        out = self.out_proj(out)
        
        # Residual connection
        return node_features + F.dropout(out, p=self.dropout, training=self.training)


class GraphNetwork(nn.Module):
    """Complete Graph Neural Network for RL."""
    
    def __init__(self, node_dim: int, edge_dim: Optional[int] = None,
                 hidden_dim: int = 128, num_layers: int = 3,
                 use_attention: bool = False, global_pooling: str = 'mean'):
        """Initialize Graph Network.
        
        Args:
            node_dim: Input node feature dimension
            edge_dim: Edge feature dimension (optional)
            hidden_dim: Hidden dimension
            num_layers: Number of message passing layers
            use_attention: Whether to use attention mechanism
            global_pooling: Global pooling method for graph-level output
        """
        super().__init__()
        self.num_layers = num_layers
        self.global_pooling = global_pooling
        
        # Node embedding
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Message passing layers
        self.mp_layers = nn.ModuleList()
        for i in range(num_layers):
            if use_attention:
                layer = GraphAttentionLayer(hidden_dim, hidden_dim)
            else:
                layer = MessagePassingLayer(hidden_dim, edge_dim, hidden_dim)
            self.mp_layers.append(layer)
            
        # Output head
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, graph_data: GraphData, 
                return_node_embeddings: bool = True) -> Dict[str, torch.Tensor]:
        """Forward pass through GNN.
        
        Args:
            graph_data: Graph data container
            return_node_embeddings: Whether to return node-level embeddings
            
        Returns:
            Dictionary containing:
                - 'node_embeddings': Node-level embeddings [num_nodes, hidden_dim]
                - 'graph_embedding': Graph-level embedding [batch_size, hidden_dim]
        """
        # Encode nodes
        h = self.node_encoder(graph_data.node_features)
        
        # Message passing
        for layer in self.mp_layers:
            if isinstance(layer, GraphAttentionLayer):
                h = layer(h, graph_data.edge_index)
            else:
                h = layer(h, graph_data.edge_index, graph_data.edge_features)
                
        # Output transformation
        h = self.output_mlp(h)
        
        result = {}
        if return_node_embeddings:
            result['node_embeddings'] = h
            
        # Global pooling for graph-level representation
        if graph_data.batch is not None:
            # Batched graphs
            batch_size = graph_data.batch.max().item() + 1
            graph_embedding = torch.zeros(batch_size, h.size(-1), device=h.device)
            
            for i in range(batch_size):
                mask = graph_data.batch == i
                if self.global_pooling == 'mean':
                    graph_embedding[i] = h[mask].mean(dim=0)
                elif self.global_pooling == 'sum':
                    graph_embedding[i] = h[mask].sum(dim=0)
                elif self.global_pooling == 'max':
                    graph_embedding[i] = h[mask].max(dim=0)[0]
        else:
            # Single graph
            if self.global_pooling == 'mean':
                graph_embedding = h.mean(dim=0, keepdim=True)
            elif self.global_pooling == 'sum':
                graph_embedding = h.sum(dim=0, keepdim=True)
            elif self.global_pooling == 'max':
                graph_embedding = h.max(dim=0, keepdim=True)[0]
                
        result['graph_embedding'] = graph_embedding
        
        return result


class MessagePassingNN(GraphNetwork):
    """Alias for GraphNetwork with message passing."""
    pass


class GraphAttentionNetwork(GraphNetwork):
    """Alias for GraphNetwork with attention."""
    
    def __init__(self, *args, **kwargs):
        kwargs['use_attention'] = True
        super().__init__(*args, **kwargs)


if __name__ == "__main__":
    # Validation test
    # Create a simple graph: 4 nodes in a square
    node_features = torch.randn(4, 16)  # 4 nodes, 16-dim features
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 0],  # source nodes
        [1, 0, 2, 1, 3, 2, 0, 3]   # target nodes
    ], dtype=torch.long)
    
    graph_data = GraphData(node_features=node_features, edge_index=edge_index)
    
    # Test basic GNN
    gnn = GraphNetwork(node_dim=16, hidden_dim=32, num_layers=2)
    output = gnn(graph_data)
    
    assert 'node_embeddings' in output, "Missing node embeddings"
    assert 'graph_embedding' in output, "Missing graph embedding"
    assert output['node_embeddings'].shape == (4, 32), f"Wrong node embedding shape: {output['node_embeddings'].shape}"
    assert output['graph_embedding'].shape == (1, 32), f"Wrong graph embedding shape: {output['graph_embedding'].shape}"
    
    # Test attention GNN
    gat = GraphAttentionNetwork(node_dim=16, hidden_dim=32, num_layers=2)
    output_att = gat(graph_data)
    
    assert output_att['node_embeddings'].shape == (4, 32), "Wrong attention output shape"
    
    print(" Graph networks validation passed")