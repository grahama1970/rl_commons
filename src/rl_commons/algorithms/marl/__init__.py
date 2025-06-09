"""
Multi-Agent Reinforcement Learning (MARL) module.
# Module: __init__.py
# Description: Package initialization and exports

This module provides decentralized and centralized multi-agent RL algorithms
for coordinating multiple agents/modules in complex environments.
"""

from .base_marl import MARLAgent, MARLEnvironment, AgentObservation
from .independent_q import IndependentQLearning
from .communication import CommunicationProtocol, Message, MessageType

__all__ = [
    'MARLAgent',
    'MARLEnvironment', 
    'AgentObservation',
    'IndependentQLearning',
    'CommunicationProtocol',
    'Message',
    'MessageType'
]