"""
Module: communication.py
Purpose: Communication protocols for multi-agent coordination

External Dependencies:
- numpy: https://numpy.org/doc/stable/
- torch: https://pytorch.org/docs/stable/

Example Usage:
>>> from rl_commons.algorithms.marl import CommunicationProtocol, Message
>>> protocol = CommunicationProtocol(num_agents=3)
>>> message = Message(sender=0, recipient=1, content={'action': 'coordinate'})
>>> protocol.send(message)
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable, Tuple
from collections import defaultdict, deque
import numpy as np
import torch
import torch.nn as nn
import time
from enum import Enum


class MessageType(Enum):
    """Types of messages agents can send."""
    INTENTION = "intention"
    OBSERVATION = "observation"
    REQUEST = "request"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    COORDINATION = "coordination"


@dataclass
class Message:
    """Message structure for agent communication."""
    sender: int
    recipient: int  # -1 for broadcast
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: float = None
    priority: int = 0  # Higher priority messages processed first
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = 0.0  # Start from 0 for testing


class CommunicationChannel:
    """Communication channel between agents."""
    
    def __init__(self, capacity: int = 100, latency: float = 0.0):
        """Initialize communication channel.
        
        Args:
            capacity: Maximum messages in channel
            latency: Simulated communication latency
        """
        self.capacity = capacity
        self.latency = latency
        self.messages = deque(maxlen=capacity)
        self.message_count = 0
        self.dropped_messages = 0
    
    def send(self, message: Message) -> bool:
        """Send message through channel.
        
        Returns:
            Whether message was successfully sent
        """
        if len(self.messages) >= self.capacity:
            self.dropped_messages += 1
            return False
        
        # Add latency to message
        if self.latency > 0:
            message.timestamp += self.latency
        
        self.messages.append(message)
        self.message_count += 1
        return True
    
    def receive(self, current_time: float) -> List[Message]:
        """Receive messages that have arrived.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            List of messages that have arrived
        """
        arrived = []
        while self.messages and self.messages[0].timestamp <= current_time:
            arrived.append(self.messages.popleft())
        return arrived
    
    def clear(self):
        """Clear all messages."""
        self.messages.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get channel statistics."""
        return {
            'total_messages': self.message_count,
            'dropped_messages': self.dropped_messages,
            'current_queue_size': len(self.messages),
            'drop_rate': self.dropped_messages / max(1, self.message_count)
        }


class CommunicationProtocol:
    """Protocol for managing multi-agent communication."""
    
    def __init__(
        self,
        num_agents: int,
        channel_capacity: int = 100,
        default_latency: float = 0.0,
        enable_broadcast: bool = True
    ):
        """Initialize communication protocol.
        
        Args:
            num_agents: Number of agents in system
            channel_capacity: Capacity of each channel
            default_latency: Default communication latency
            enable_broadcast: Whether to allow broadcast messages
        """
        self.num_agents = num_agents
        self.enable_broadcast = enable_broadcast
        
        # Create channels between all agent pairs
        self.channels = {}
        for i in range(num_agents):
            for j in range(num_agents):
                if i != j:
                    self.channels[(i, j)] = CommunicationChannel(
                        capacity=channel_capacity,
                        latency=default_latency
                    )
        
        # Broadcast channel
        if enable_broadcast:
            self.broadcast_channel = CommunicationChannel(
                capacity=channel_capacity * 2,
                latency=default_latency
            )
        
        # Message handlers
        self.handlers = defaultdict(list)
        self.current_time = 0.0
    
    def register_handler(
        self,
        agent_id: int,
        message_type: MessageType,
        handler: Callable[[Message], Any]
    ):
        """Register message handler for an agent.
        
        Args:
            agent_id: Agent to register handler for
            message_type: Type of message to handle
            handler: Function to handle message
        """
        self.handlers[(agent_id, message_type)].append(handler)
    
    def send(self, message: Message) -> bool:
        """Send message according to protocol.
        
        Returns:
            Whether message was successfully sent
        """
        if message.recipient == -1:  # Broadcast
            if not self.enable_broadcast:
                return False
            return self.broadcast_channel.send(message)
        else:
            # Point-to-point
            channel_key = (message.sender, message.recipient)
            if channel_key in self.channels:
                return self.channels[channel_key].send(message)
            return False
    
    def receive(self, agent_id: int) -> List[Message]:
        """Receive messages for an agent.
        
        Args:
            agent_id: Agent to receive messages for
            
        Returns:
            List of received messages
        """
        messages = []
        
        # Check point-to-point channels
        for (sender, recipient), channel in self.channels.items():
            if recipient == agent_id:
                messages.extend(channel.receive(self.current_time))
        
        # Check broadcast channel
        if self.enable_broadcast:
            broadcast_messages = self.broadcast_channel.receive(self.current_time)
            # Filter out own broadcasts
            messages.extend([
                msg for msg in broadcast_messages 
                if msg.sender != agent_id
            ])
        
        # Sort by priority and timestamp
        messages.sort(key=lambda m: (-m.priority, m.timestamp))
        
        # Process through handlers
        for message in messages:
            handlers = self.handlers.get((agent_id, message.message_type), [])
            for handler in handlers:
                handler(message)
        
        return messages
    
    def broadcast(self, sender: int, message_type: MessageType, 
                  content: Dict[str, Any], priority: int = 0) -> bool:
        """Broadcast message to all agents.
        
        Returns:
            Whether broadcast was successful
        """
        if not self.enable_broadcast:
            return False
        
        message = Message(
            sender=sender,
            recipient=-1,
            message_type=message_type,
            content=content,
            priority=priority
        )
        return self.send(message)
    
    def update_time(self, delta: float):
        """Update protocol time."""
        self.current_time += delta
    
    def get_stats(self) -> Dict[str, Any]:
        """Get communication statistics."""
        stats = {
            'num_agents': self.num_agents,
            'total_messages': 0,
            'total_dropped': 0,
            'channels': {}
        }
        
        # Aggregate channel stats
        for key, channel in self.channels.items():
            channel_stats = channel.get_stats()
            stats['total_messages'] += channel_stats['total_messages']
            stats['total_dropped'] += channel_stats['dropped_messages']
            stats['channels'][f"{key[0]}->{key[1]}"] = channel_stats
        
        if self.enable_broadcast:
            broadcast_stats = self.broadcast_channel.get_stats()
            stats['broadcast'] = broadcast_stats
            stats['total_messages'] += broadcast_stats['total_messages']
            stats['total_dropped'] += broadcast_stats['dropped_messages']
        
        return stats
    
    def clear_all(self):
        """Clear all communication channels."""
        for channel in self.channels.values():
            channel.clear()
        if self.enable_broadcast:
            self.broadcast_channel.clear()


class LearnedCommunication(nn.Module):
    """Learned communication protocol using neural networks."""
    
    def __init__(
        self,
        obs_dim: int,
        message_dim: int,
        hidden_dim: int = 64,
        num_agents: int = 3
    ):
        """Initialize learned communication.
        
        Args:
            obs_dim: Dimension of observations
            message_dim: Dimension of messages
            hidden_dim: Hidden layer dimension
            num_agents: Number of agents
        """
        super().__init__()
        
        # Message encoder
        self.message_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, message_dim),
            nn.Tanh()  # Bounded messages
        )
        
        # Message decoder/processor
        self.message_processor = nn.Sequential(
            nn.Linear(message_dim * (num_agents - 1) + obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.num_agents = num_agents
        self.message_dim = message_dim
    
    def encode_message(self, observation: torch.Tensor) -> torch.Tensor:
        """Encode observation into message.
        
        Args:
            observation: Agent's observation'
            
        Returns:
            Encoded message
        """
        return self.message_encoder(observation)
    
    def process_messages(
        self,
        own_obs: torch.Tensor,
        messages: List[torch.Tensor]
    ) -> torch.Tensor:
        """Process received messages with own observation.
        
        Args:
            own_obs: Agent's own observation'
            messages: List of messages from other agents
            
        Returns:
            Processed communication embedding
        """
        # Pad messages if needed
        while len(messages) < self.num_agents - 1:
            messages.append(torch.zeros(self.message_dim))
        
        # Concatenate all information
        if messages:
            all_messages = torch.cat(messages, dim=-1)
            combined = torch.cat([own_obs, all_messages], dim=-1)
        else:
            # No messages case
            zero_messages = torch.zeros(self.message_dim * (self.num_agents - 1))
            combined = torch.cat([own_obs, zero_messages], dim=-1)
        
        return self.message_processor(combined)


if __name__ == "__main__":
    # Validation test
    protocol = CommunicationProtocol(num_agents=3)
    
    # Test message sending
    msg = Message(
        sender=0,
        recipient=1,
        message_type=MessageType.INTENTION,
        content={'action': 'move_left'}
    )
    
    success = protocol.send(msg)
    assert success, "Failed to send message"
    
    # Test message receiving
    protocol.update_time(0.1)  # Advance time
    received = protocol.receive(agent_id=1)
    assert len(received) == 1, f"Expected 1 message, got {len(received)}"
    assert received[0].content['action'] == 'move_left', "Message content mismatch"
    
    # Test broadcast
    success = protocol.broadcast(
        sender=2,
        message_type=MessageType.COORDINATION,
        content={'plan': 'converge_center'}
    )
    assert success, "Failed to broadcast"
    
    protocol.update_time(0.1)
    for agent_id in [0, 1]:
        received = protocol.receive(agent_id)
        assert len(received) == 1, f"Agent {agent_id} should receive broadcast"
    
    # Test learned communication
    learned_comm = LearnedCommunication(obs_dim=4, message_dim=8, num_agents=3)
    obs = torch.randn(1, 4)
    message = learned_comm.encode_message(obs)
    assert message.shape == (1, 8), f"Invalid message shape: {message.shape}"
    
    processed = learned_comm.process_messages(obs, [message, message])
    assert processed.shape == (1, 64), f"Invalid processed shape: {processed.shape}"
    
    print(" Communication protocol validation passed")