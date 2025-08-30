"""
RL Agent for MASA Framework
Implements the reinforcement learning agent using PSformer and SAM optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any
try:
    from .base_neural import (
        BaseNeuralLayer, PSformerBlock, SAMOptimizedLinear, 
        MASAOptimizer, initialize_weights, entropy_regularization
    )
except ImportError:
    from base_neural import (
        BaseNeuralLayer, PSformerBlock, SAMOptimizedLinear, 
        MASAOptimizer, initialize_weights, entropy_regularization
    )


class RLAgent(nn.Module):
    """
    RL Agent that optimizes portfolio returns using PSformer analysis and SAM optimization.
    
    The agent uses PSformer blocks for environment analysis and makes decisions
    through a lightweight perceptron with SAM optimization.
    """
    
    def __init__(
        self,
        window: int,        # Input feature dimension
        units_count: int,   # Sequence length
        segments: int,      # Number of segments for PSformer
        rho: float,         # Blurring coefficient for PSformer
        layers: int,        # Number of PSformer layers
        n_actions: int,     # Action space size (portfolio weights)
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        sam_rho: float = 0.05
    ):
        super().__init__()
        
        self.window = window
        self.units_count = units_count
        self.segments = segments
        self.rho = rho
        self.layers = layers
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.sam_rho = sam_rho
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Build the RL Agent architecture
        self._build_architecture()
        
        # Initialize weights
        self.apply(initialize_weights)
        
        # Initialize SAM optimizer
        self.optimizer = MASAOptimizer(
            self.parameters(),
            base_optimizer=torch.optim.Adam,
            lr=learning_rate,
            rho=sam_rho
        )
        
        # Experience replay buffer
        self.experience_buffer = []
        self.buffer_size = 10000
        
    def _build_architecture(self):
        """Build the RL Agent neural network architecture."""
        
        # PSformer layers for state observation
        self.psformer_layers = nn.ModuleList()
        for _ in range(self.layers):
            psformer = PSformerBlock(
                d_model=self.window,
                units_count=self.units_count,
                segments=self.segments,
                rho=self.rho,
                batch_size=self.batch_size
            )
            self.psformer_layers.append(psformer)
            
        # Decision making layers
        self.decision_conv = nn.Conv1d(
            in_channels=self.window,
            out_channels=self.n_actions,
            kernel_size=1,
            padding=0
        )
        
        self.decision_linear = SAMOptimizedLinear(
            input_size=self.n_actions * self.units_count,
            output_size=self.n_actions,
            activation='sigmoid',
            batch_size=self.batch_size
        )
        
        # Value function for TD3-like training
        self.value_network = nn.Sequential(
            nn.Linear(self.window * self.units_count, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Target networks for stable training
        self.target_decision_conv = nn.Conv1d(
            in_channels=self.window,
            out_channels=self.n_actions,
            kernel_size=1,
            padding=0
        )
        
        self.target_decision_linear = SAMOptimizedLinear(
            input_size=self.n_actions * self.units_count,
            output_size=self.n_actions,
            activation='sigmoid',
            batch_size=self.batch_size
        )
        
        # Copy weights to target networks
        self._update_target_networks(tau=1.0)
        
    def forward(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through RL Agent.
        
        Args:
            state: Input state tensor
            deterministic: Whether to use deterministic policy
            
        Returns:
            Tuple of (actions, state_value)
        """
        batch_size = state.shape[0]
        
        # Process state through PSformer layers
        psformer_output = state
        for psformer_layer in self.psformer_layers:
            psformer_output = psformer_layer(psformer_output)
            
        # Reshape for convolutional layer
        conv_input = psformer_output.view(batch_size, self.window, self.units_count)
        
        # Apply decision convolution
        conv_output = self.decision_conv(conv_input)
        conv_output = F.gelu(conv_output)
        
        # Flatten for linear layer
        linear_input = conv_output.view(batch_size, -1)
        
        # Generate actions
        actions = self.decision_linear(linear_input)
        
        # Add exploration noise if not deterministic
        if not deterministic and self.training:
            noise = torch.randn_like(actions) * 0.1
            actions = torch.clamp(actions + noise, 0, 1)
            
        # Normalize actions to sum to 1 (portfolio weights)
        actions = F.softmax(actions, dim=-1)
        
        # Compute state value
        state_value = self.value_network(state)
        
        return actions, state_value
    
    def get_target_actions(self, state: torch.Tensor) -> torch.Tensor:
        """Get actions from target network for stable training."""
        with torch.no_grad():
            batch_size = state.shape[0]
            
            # Process through PSformer layers (use main network)
            psformer_output = state
            for psformer_layer in self.psformer_layers:
                psformer_output = psformer_layer(psformer_output)
                
            # Reshape for target convolutional layer
            conv_input = psformer_output.view(batch_size, self.window, self.units_count)
            
            # Apply target decision layers
            conv_output = self.target_decision_conv(conv_input)
            conv_output = F.gelu(conv_output)
            
            linear_input = conv_output.view(batch_size, -1)
            target_actions = self.target_decision_linear(linear_input)
            
            # Normalize to portfolio weights
            target_actions = F.softmax(target_actions, dim=-1)
            
            return target_actions
    
    def compute_td3_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        gamma: float = 0.99
    ) -> Dict[str, torch.Tensor]:
        """
        Compute TD3-style loss for training.
        
        Args:
            states: Current states
            actions: Taken actions
            rewards: Received rewards
            next_states: Next states
            dones: Episode termination flags
            gamma: Discount factor
            
        Returns:
            Dictionary containing loss components
        """
        batch_size = states.shape[0]
        
        # Current state-action values
        current_actions, current_values = self.forward(states, deterministic=True)
        
        # Target actions and values
        with torch.no_grad():
            target_actions = self.get_target_actions(next_states)
            _, target_values = self.forward(next_states, deterministic=True)
            
            # Add noise to target actions (TD3 policy smoothing)
            noise = torch.randn_like(target_actions) * 0.1
            noise = torch.clamp(noise, -0.2, 0.2)
            target_actions = torch.clamp(target_actions + noise, 0, 1)
            target_actions = F.softmax(target_actions, dim=-1)
            
            # Compute target Q-values
            target_q = rewards + gamma * (1 - dones) * target_values.squeeze()
            
        # Actor loss (policy gradient)
        actor_loss = -current_values.mean()
        
        # Add entropy regularization for exploration
        entropy_bonus = entropy_regularization(current_actions)
        actor_loss = actor_loss - 0.01 * entropy_bonus
        
        # Critic loss (value function)
        critic_loss = F.mse_loss(current_values.squeeze(), target_q)
        
        # Action consistency loss (difference from taken actions)
        action_loss = F.mse_loss(current_actions, actions)
        
        # Total loss
        total_loss = actor_loss + critic_loss + 0.1 * action_loss
        
        return {
            'total_loss': total_loss,
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'action_loss': action_loss,
            'entropy_bonus': entropy_bonus
        }
    
    def update_parameters(self, loss: torch.Tensor):
        """Update parameters using SAM optimization."""
        def closure():
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            return loss
            
        self.optimizer.step(closure)
        
    def _update_target_networks(self, tau: float = 0.005):
        """Update target networks with soft update."""
        # Update target decision conv
        for target_param, param in zip(self.target_decision_conv.parameters(), 
                                     self.decision_conv.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
        # Update target decision linear
        for target_param, param in zip(self.target_decision_linear.parameters(), 
                                     self.decision_linear.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def add_experience(self, state, action, reward, next_state, done):
        """Add experience to replay buffer."""
        experience = (state, action, reward, next_state, done)
        
        if len(self.experience_buffer) >= self.buffer_size:
            self.experience_buffer.pop(0)
            
        self.experience_buffer.append(experience)
        
    def sample_experience(self, batch_size: int) -> Optional[Tuple]:
        """Sample batch of experiences from replay buffer."""
        if len(self.experience_buffer) < batch_size:
            return None
            
        indices = np.random.choice(len(self.experience_buffer), batch_size, replace=False)
        batch = [self.experience_buffer[i] for i in indices]
        
        # Unpack batch
        states = torch.stack([exp[0] for exp in batch])
        actions = torch.stack([exp[1] for exp in batch])
        rewards = torch.stack([exp[2] for exp in batch])
        next_states = torch.stack([exp[3] for exp in batch])
        dones = torch.stack([exp[4] for exp in batch])
        
        return states, actions, rewards, next_states, dones
    
    def train_step(self, batch_size: int = 64) -> Optional[Dict[str, float]]:
        """Perform one training step using experience replay."""
        experience_batch = self.sample_experience(batch_size)
        if experience_batch is None:
            return None
            
        states, actions, rewards, next_states, dones = experience_batch
        
        # Compute loss
        loss_dict = self.compute_td3_loss(states, actions, rewards, next_states, dones)
        
        # Update parameters
        self.update_parameters(loss_dict['total_loss'])
        
        # Update target networks
        self._update_target_networks()
        
        # Convert to float for logging
        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
    
    def get_portfolio_weights(self, state: torch.Tensor) -> torch.Tensor:
        """Get portfolio weights for given market state."""
        with torch.no_grad():
            actions, _ = self.forward(state, deterministic=True)
            return actions
    
    def save_checkpoint(self, filepath: str):
        """Save agent checkpoint."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.base_optimizer.state_dict(),
            'config': {
                'window': self.window,
                'units_count': self.units_count,
                'segments': self.segments,
                'rho': self.rho,
                'layers': self.layers,
                'n_actions': self.n_actions,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'sam_rho': self.sam_rho
            }
        }
        torch.save(checkpoint, filepath)
        
    def load_checkpoint(self, filepath: str):
        """Load agent checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.base_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['config']
    
    def to(self, device):
        """Move model to specified device."""
        super().to(device)
        self.device = device
        return self


class TD3RLAgent(RLAgent):
    """
    Enhanced RL Agent implementing TD3 (Twin Delayed Deep Deterministic) algorithm.
    
    Includes twin critics for reduced overestimation bias and delayed policy updates.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Twin critics for TD3
        self.critic1 = nn.Sequential(
            nn.Linear(self.window * self.units_count + self.n_actions, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.critic2 = nn.Sequential(
            nn.Linear(self.window * self.units_count + self.n_actions, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Target critics
        self.target_critic1 = nn.Sequential(
            nn.Linear(self.window * self.units_count + self.n_actions, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.target_critic2 = nn.Sequential(
            nn.Linear(self.window * self.units_count + self.n_actions, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Initialize target networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # Separate optimizers for actor and critics
        self.actor_optimizer = MASAOptimizer(
            list(self.psformer_layers.parameters()) + 
            list(self.decision_conv.parameters()) + 
            list(self.decision_linear.parameters()),
            base_optimizer=torch.optim.Adam,
            lr=self.learning_rate,
            rho=self.sam_rho
        )
        
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=self.learning_rate
        )
        
        # Training counters
        self.update_counter = 0
        self.policy_update_freq = 2
        
    def compute_td3_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        gamma: float = 0.99
    ) -> Dict[str, torch.Tensor]:
        """Compute TD3 loss with twin critics and delayed policy updates."""
        
        # Current Q-values
        state_action = torch.cat([states, actions], dim=-1)
        q1_current = self.critic1(state_action)
        q2_current = self.critic2(state_action)
        
        # Target Q-values
        with torch.no_grad():
            next_actions, _ = self.forward(next_states, deterministic=True)
            
            # Add noise to target actions (policy smoothing)
            noise = torch.randn_like(next_actions) * 0.1
            noise = torch.clamp(noise, -0.2, 0.2)
            next_actions = torch.clamp(next_actions + noise, 0, 1)
            next_actions = F.softmax(next_actions, dim=-1)
            
            next_state_action = torch.cat([next_states, next_actions], dim=-1)
            target_q1 = self.target_critic1(next_state_action)
            target_q2 = self.target_critic2(next_state_action)
            
            # Take minimum of twin critics (reduces overestimation)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards.unsqueeze(-1) + gamma * (1 - dones.unsqueeze(-1)) * target_q
            
        # Critic losses
        critic1_loss = F.mse_loss(q1_current, target_q)
        critic2_loss = F.mse_loss(q2_current, target_q)
        critic_loss = critic1_loss + critic2_loss
        
        # Actor loss (delayed update)
        actor_loss = torch.tensor(0.0, device=self.device)
        if self.update_counter % self.policy_update_freq == 0:
            current_actions, _ = self.forward(states, deterministic=True)
            state_action_current = torch.cat([states, current_actions], dim=-1)
            actor_loss = -self.critic1(state_action_current).mean()
            
            # Add entropy regularization
            entropy_bonus = entropy_regularization(current_actions)
            actor_loss = actor_loss - 0.01 * entropy_bonus
        
        self.update_counter += 1
        
        return {
            'critic_loss': critic_loss,
            'critic1_loss': critic1_loss,
            'critic2_loss': critic2_loss,
            'actor_loss': actor_loss,
            'total_loss': critic_loss + actor_loss
        }
    
    def train_step(self, batch_size: int = 64) -> Optional[Dict[str, float]]:
        """Perform one TD3 training step."""
        experience_batch = self.sample_experience(batch_size)
        if experience_batch is None:
            return None
            
        states, actions, rewards, next_states, dones = experience_batch
        
        # Compute losses
        loss_dict = self.compute_td3_loss(states, actions, rewards, next_states, dones)
        
        # Update critics
        self.critic_optimizer.zero_grad()
        loss_dict['critic_loss'].backward(retain_graph=True)
        self.critic_optimizer.step()
        
        # Update actor (delayed)
        if self.update_counter % self.policy_update_freq == 0:
            def actor_closure():
                self.actor_optimizer.zero_grad()
                loss_dict['actor_loss'].backward(retain_graph=True)
                return loss_dict['actor_loss']
                
            self.actor_optimizer.step(actor_closure)
            
            # Update target networks
            self._update_target_networks()
            
        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
    
    def _update_target_networks(self, tau: float = 0.005):
        """Update target networks with soft update."""
        # Update target critics
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
        # Update target decision networks
        super()._update_target_networks(tau)


# Factory function for creating RL Agent instances
def create_rl_agent(config: dict, use_td3: bool = True) -> RLAgent:
    """
    Factory function to create RL Agent instances.
    
    Args:
        config: Configuration dictionary with required parameters
        use_td3: Whether to use TD3 enhancement
        
    Returns:
        RLAgent instance
    """
    required_keys = ['window', 'units_count', 'segments', 'rho', 'layers', 'n_actions']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    
    agent_class = TD3RLAgent if use_td3 else RLAgent
    
    return agent_class(
        window=config['window'],
        units_count=config['units_count'],
        segments=config['segments'],
        rho=config['rho'],
        layers=config['layers'],
        n_actions=config['n_actions'],
        batch_size=config.get('batch_size', 32),
        learning_rate=config.get('learning_rate', 1e-4),
        sam_rho=config.get('sam_rho', 0.05)
    )