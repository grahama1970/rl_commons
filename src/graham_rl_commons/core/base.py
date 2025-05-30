"""Base classes for RL components"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class RLState:
    """Base state representation for RL agents"""
    features: np.ndarray
    context: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization"""
        return {
            "features": self.features.tolist(),
            "context": self.context
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RLState":
        """Create state from dictionary"""
        return cls(
            features=np.array(data["features"]),
            context=data.get("context")
        )
    
    def __len__(self) -> int:
        """Return the dimensionality of the state"""
        return len(self.features)


@dataclass
class RLAction:
    """Base action representation"""
    action_type: str
    action_id: int
    parameters: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert action to dictionary"""
        return {
            "action_type": self.action_type,
            "action_id": self.action_id,
            "parameters": self.parameters
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RLAction":
        """Create action from dictionary"""
        return cls(
            action_type=data["action_type"],
            action_id=data["action_id"],
            parameters=data.get("parameters")
        )


@dataclass
class RLReward:
    """Reward with multi-objective support"""
    value: float
    components: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        if self.components is None:
            self.components = {"total": self.value}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert reward to dictionary"""
        return {
            "value": self.value,
            "components": self.components
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RLReward":
        """Create reward from dictionary"""
        return cls(
            value=data["value"],
            components=data.get("components")
        )


class RLAgent(ABC):
    """Base RL agent interface"""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.training_steps = 0
        self.episodes = 0
        self.training = True
        
        # Initialize logging
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
        
    @abstractmethod
    def select_action(self, state: RLState, explore: bool = True) -> RLAction:
        """
        Select an action given a state
        
        Args:
            state: Current state
            explore: Whether to explore (training) or exploit (evaluation)
            
        Returns:
            Selected action
        """
        pass
        
    @abstractmethod
    def update(self, 
               state: RLState, 
               action: RLAction, 
               reward: RLReward, 
               next_state: RLState, 
               done: bool = False) -> Dict[str, float]:
        """
        Update the agent based on experience
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Whether episode ended
            
        Returns:
            Dictionary of training metrics
        """
        pass
        
    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """
        Save the agent's state
        
        Args:
            path: Path to save location
        """
        pass
        
    @abstractmethod
    def load(self, path: Union[str, Path]) -> None:
        """
        Load the agent's state
        
        Args:
            path: Path to saved model
        """
        pass
    
    def train_mode(self) -> None:
        """Set agent to training mode"""
        self.training = True
        self.logger.info(f"Agent {self.name} set to training mode")
        
    def eval_mode(self) -> None:
        """Set agent to evaluation mode"""
        self.training = False
        self.logger.info(f"Agent {self.name} set to evaluation mode")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current agent metrics"""
        return {
            "name": self.name,
            "training_steps": self.training_steps,
            "episodes": self.episodes,
            "training": self.training,
        }
    
    def reset_episode(self) -> None:
        """Reset any episode-specific state"""
        self.episodes += 1
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, steps={self.training_steps})"


class BatchRLAgent(RLAgent):
    """Base class for agents that can process batches"""
    
    @abstractmethod
    def select_actions(self, states: List[RLState], explore: bool = True) -> List[RLAction]:
        """Select actions for a batch of states"""
        pass
    
    @abstractmethod
    def update_batch(self, 
                     states: List[RLState],
                     actions: List[RLAction],
                     rewards: List[RLReward],
                     next_states: List[RLState],
                     dones: List[bool]) -> Dict[str, float]:
        """Update from a batch of experiences"""
        pass
