"""
Graph Neural Network (GNN) integration for RL algorithms.
# Module: __init__.py
# Description: Package initialization and exports

This module provides GNN-enhanced RL algorithms for topology-aware decision making
in environments with graph-structured state spaces.
"""

from .graph_networks import GraphNetwork, MessagePassingNN, GraphAttentionNetwork, GraphData
from .gnn_integration import GNNEnhancedDQN, GNNEnhancedPPO, GraphStateEncoder, GraphState, GNNIntegration
from .dynamic_graphs import DynamicGraphHandler, GraphUpdateEvent

__all__ = [
    'GraphNetwork',
    'MessagePassingNN',
    'GraphAttentionNetwork',
    'GraphData',
    'GNNEnhancedDQN',
    'GNNEnhancedPPO',
    'GraphStateEncoder',
    'GraphState',
    'GNNIntegration',
    'DynamicGraphHandler',
    'GraphUpdateEvent'
]