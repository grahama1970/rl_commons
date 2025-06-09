"""
Module: dynamic_graphs.py
Purpose: Handle dynamic graph structures for module networks

External Dependencies:
- torch: https://pytorch.org/docs/stable/
- numpy: https://numpy.org/doc/stable/

Example Usage:
>>> from rl_commons.algorithms.gnn import DynamicGraphHandler
>>> handler = DynamicGraphHandler()
>>> handler.add_node(node_id=5, features=torch.randn(32))
>>> handler.add_edge(source=1, target=5)
"""

import torch
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
import time

from .graph_networks import GraphData


@dataclass
class GraphUpdateEvent:
    """Event representing a change in graph structure."""
    event_type: str  # 'add_node', 'remove_node', 'add_edge', 'remove_edge', 'update_features'
    timestamp: float
    node_ids: Optional[List[int]] = None
    edge_ids: Optional[List[Tuple[int, int]]] = None
    features: Optional[torch.Tensor] = None
    metadata: Optional[Dict] = None


class DynamicGraphHandler:
    """Handles dynamic graph updates for module networks."""
    
    def __init__(self, initial_node_dim: int = 32):
        """Initialize dynamic graph handler.
        
        Args:
            initial_node_dim: Dimension of node features
        """
        self.node_dim = initial_node_dim
        self.nodes = {}  # node_id -> features
        self.edges = set()  # set of (source, target) tuples
        self.node_id_map = {}  # external_id -> internal_index
        self.reverse_id_map = {}  # internal_index -> external_id
        self.update_history = []
        self._next_internal_id = 0
        
    def add_node(self, node_id: int, features: torch.Tensor) -> bool:
        """Add a new node to the graph.
        
        Args:
            node_id: External node identifier
            features: Node feature tensor
            
        Returns:
            Whether node was added successfully
        """
        if node_id in self.node_id_map:
            return False  # Node already exists
            
        # Assign internal index
        internal_idx = self._next_internal_id
        self._next_internal_id += 1
        
        self.nodes[node_id] = features
        self.node_id_map[node_id] = internal_idx
        self.reverse_id_map[internal_idx] = node_id
        
        # Record event
        event = GraphUpdateEvent(
            event_type='add_node',
            timestamp=time.time(),
            node_ids=[node_id],
            features=features
        )
        self.update_history.append(event)
        
        return True
    
    def remove_node(self, node_id: int) -> bool:
        """Remove a node from the graph.
        
        Args:
            node_id: Node to remove
            
        Returns:
            Whether node was removed successfully
        """
        if node_id not in self.node_id_map:
            return False
            
        # Remove associated edges
        edges_to_remove = [
            (s, t) for s, t in self.edges 
            if s == node_id or t == node_id
        ]
        for edge in edges_to_remove:
            self.edges.remove(edge)
            
        # Remove node
        del self.nodes[node_id]
        internal_idx = self.node_id_map[node_id]
        del self.node_id_map[node_id]
        del self.reverse_id_map[internal_idx]
        
        # Record event
        event = GraphUpdateEvent(
            event_type='remove_node',
            timestamp=time.time(),
            node_ids=[node_id],
            edge_ids=edges_to_remove
        )
        self.update_history.append(event)
        
        # Note: In practice, we'd need to reindex remaining nodes
        return True
    
    def add_edge(self, source: int, target: int, bidirectional: bool = True) -> bool:
        """Add an edge between nodes.
        
        Args:
            source: Source node ID
            target: Target node ID
            bidirectional: Whether to add edge in both directions
            
        Returns:
            Whether edge was added successfully
        """
        if source not in self.node_id_map or target not in self.node_id_map:
            return False
            
        edges_added = []
        if (source, target) not in self.edges:
            self.edges.add((source, target))
            edges_added.append((source, target))
            
        if bidirectional and (target, source) not in self.edges:
            self.edges.add((target, source))
            edges_added.append((target, source))
            
        if edges_added:
            event = GraphUpdateEvent(
                event_type='add_edge',
                timestamp=time.time(),
                edge_ids=edges_added
            )
            self.update_history.append(event)
            
        return len(edges_added) > 0
    
    def remove_edge(self, source: int, target: int, bidirectional: bool = True) -> bool:
        """Remove an edge between nodes.
        
        Args:
            source: Source node ID
            target: Target node ID
            bidirectional: Whether to remove edge in both directions
            
        Returns:
            Whether edge was removed successfully
        """
        edges_removed = []
        
        if (source, target) in self.edges:
            self.edges.remove((source, target))
            edges_removed.append((source, target))
            
        if bidirectional and (target, source) in self.edges:
            self.edges.remove((target, source))
            edges_removed.append((target, source))
            
        if edges_removed:
            event = GraphUpdateEvent(
                event_type='remove_edge',
                timestamp=time.time(),
                edge_ids=edges_removed
            )
            self.update_history.append(event)
            
        return len(edges_removed) > 0
    
    def update_node_features(self, node_id: int, features: torch.Tensor) -> bool:
        """Update features for a node.
        
        Args:
            node_id: Node to update
            features: New feature tensor
            
        Returns:
            Whether update was successful
        """
        if node_id not in self.node_id_map:
            return False
            
        self.nodes[node_id] = features
        
        event = GraphUpdateEvent(
            event_type='update_features',
            timestamp=time.time(),
            node_ids=[node_id],
            features=features
        )
        self.update_history.append(event)
        
        return True
    
    def get_graph_data(self) -> GraphData:
        """Get current graph as GraphData object.
        
        Returns:
            GraphData object representing current graph
        """
        if not self.nodes:
            # Empty graph
            return GraphData(
                node_features=torch.zeros(0, self.node_dim),
                edge_index=torch.zeros(2, 0, dtype=torch.long)
            )
            
        # Sort nodes by internal index for consistent ordering
        sorted_nodes = sorted(self.node_id_map.items(), key=lambda x: x[1])
        node_features = torch.stack([self.nodes[node_id] for node_id, _ in sorted_nodes])
        
        # Build edge index using internal indices
        edge_list = []
        for source, target in self.edges:
            source_idx = self.node_id_map[source]
            target_idx = self.node_id_map[target]
            edge_list.append([source_idx, target_idx])
            
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            
        return GraphData(
            node_features=node_features,
            edge_index=edge_index
        )
    
    def get_node_degree(self, node_id: int) -> Dict[str, int]:
        """Get degree information for a node.
        
        Args:
            node_id: Node to query
            
        Returns:
            Dictionary with in_degree, out_degree, total_degree
        """
        if node_id not in self.node_id_map:
            return {'in_degree': 0, 'out_degree': 0, 'total_degree': 0}
            
        in_degree = sum(1 for _, t in self.edges if t == node_id)
        out_degree = sum(1 for s, _ in self.edges if s == node_id)
        
        return {
            'in_degree': in_degree,
            'out_degree': out_degree,
            'total_degree': in_degree + out_degree
        }
    
    def get_neighbors(self, node_id: int) -> Dict[str, List[int]]:
        """Get neighbors of a node.
        
        Args:
            node_id: Node to query
            
        Returns:
            Dictionary with incoming and outgoing neighbors
        """
        if node_id not in self.node_id_map:
            return {'incoming': [], 'outgoing': []}
            
        incoming = [s for s, t in self.edges if t == node_id]
        outgoing = [t for s, t in self.edges if s == node_id]
        
        return {
            'incoming': incoming,
            'outgoing': outgoing
        }
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the current graph.
        
        Returns:
            Dictionary of graph statistics
        """
        num_nodes = len(self.nodes)
        num_edges = len(self.edges)
        
        if num_nodes == 0:
            return {
                'num_nodes': 0,
                'num_edges': 0,
                'density': 0,
                'avg_degree': 0,
                'max_degree': 0,
                'num_components': 0
            }
            
        # Calculate degree statistics
        degrees = []
        for node_id in self.nodes:
            degree_info = self.get_node_degree(node_id)
            degrees.append(degree_info['total_degree'])
            
        max_possible_edges = num_nodes * (num_nodes - 1)
        density = num_edges / max_possible_edges if max_possible_edges > 0 else 0
        
        return {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'density': density,
            'avg_degree': np.mean(degrees),
            'max_degree': max(degrees),
            'min_degree': min(degrees),
            'num_updates': len(self.update_history)
        }
    
    def reset(self):
        """Reset the graph to empty state."""
        self.nodes.clear()
        self.edges.clear()
        self.node_id_map.clear()
        self.reverse_id_map.clear()
        self.update_history.clear()
        self._next_internal_id = 0


if __name__ == "__main__":
    # Validation test
    handler = DynamicGraphHandler(initial_node_dim=16)
    
    # Add some modules
    for i in range(4):
        success = handler.add_node(i, torch.randn(16))
        assert success, f"Failed to add node {i}"
    
    # Add connections
    handler.add_edge(0, 1)
    handler.add_edge(1, 2)
    handler.add_edge(2, 3)
    handler.add_edge(3, 0)  # Create a cycle
    
    # Get graph data
    graph_data = handler.get_graph_data()
    assert graph_data.node_features.shape == (4, 16), f"Wrong node shape: {graph_data.node_features.shape}"
    assert graph_data.edge_index.shape[1] == 8, f"Wrong edge count: {graph_data.edge_index.shape}"  # Bidirectional
    
    # Test node removal
    handler.remove_node(1)
    graph_data2 = handler.get_graph_data()
    assert graph_data2.node_features.shape[0] == 3, "Node not removed"
    
    # Test statistics
    stats = handler.get_graph_stats()
    assert stats['num_nodes'] == 3, f"Wrong node count: {stats['num_nodes']}"
    assert stats['num_edges'] == 4, f"Wrong edge count: {stats['num_edges']}"  # Some edges removed with node
    
    print(" Dynamic graph handler validation passed")