"""
MASA (Multi-Agent Self-Adaptive) Framework
Integrates Market Observer, RL Agent, and Controller Agent for dynamic portfolio management.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass
import logging
from pathlib import Path

try:
    from .market_observer import MarketObserver, EnhancedMarketObserver, create_market_observer
    from .rl_agent import RLAgent, TD3RLAgent, create_rl_agent
    from .controller_agent import ControllerAgent, AdvancedControllerAgent, create_controller_agent
except ImportError:
    from market_observer import MarketObserver, EnhancedMarketObserver, create_market_observer
    from rl_agent import RLAgent, TD3RLAgent, create_rl_agent
    from controller_agent import ControllerAgent, AdvancedControllerAgent, create_controller_agent


@dataclass
class MASAConfig:
    """Configuration for MASA framework."""
    
    # Market Observer config
    mo_window: int = 5              # OHLCV
    mo_window_key: int = 64         # Attention dimension
    mo_units_count: int = 50        # Historical depth
    mo_heads: int = 8               # Attention heads
    mo_layers: int = 3              # Attention layers
    mo_forecast: int = 10           # Forecast horizon
    
    # RL Agent config
    rl_window: int = 5              # Feature dimension
    rl_units_count: int = 50        # Sequence length
    rl_segments: int = 10           # PSformer segments
    rl_rho: float = 0.5            # PSformer blurring
    rl_layers: int = 3              # PSformer layers
    rl_n_actions: int = 10          # Portfolio assets
    rl_learning_rate: float = 1e-4
    rl_sam_rho: float = 0.05
    
    # Controller config
    ctrl_window: int = 10           # Action dimension
    ctrl_window_key: int = 64       # Attention dimension
    ctrl_units_count: int = 1       # Action sequence length
    ctrl_heads: int = 8             # Attention heads
    ctrl_window_kv: int = 5         # Market data dimension
    ctrl_units_kv: int = 50         # Market sequence length
    ctrl_layers: int = 2            # Decoder layers
    
    # Training config
    batch_size: int = 32
    gamma: float = 0.99             # Discount factor
    tau: float = 0.005              # Target network update rate
    
    # Risk management
    risk_tolerance: float = 0.5     # User risk tolerance
    max_position_size: float = 0.3  # Maximum position per asset
    rebalance_threshold: float = 0.05  # Rebalancing threshold


class MASAFramework:
    """
    Main MASA framework integrating all three agents.
    
    Implements the multi-agent self-adaptive approach for dynamic portfolio management
    with balanced risk-return optimization.
    """
    
    def __init__(
        self,
        config: MASAConfig,
        enhanced_agents: bool = True,
        device: Optional[str] = None
    ):
        self.config = config
        self.enhanced_agents = enhanced_agents
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize logging
        self.logger = self._setup_logging()
        
        # Initialize agents
        self._initialize_agents()
        
        # Training state
        self.training_step = 0
        self.episode_rewards = []
        self.portfolio_values = []
        
        # Performance tracking
        self.performance_metrics = {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'volatility': 0.0,
            'win_rate': 0.0
        }
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the MASA framework."""
        logger = logging.getLogger('MASA')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    def _initialize_agents(self):
        """Initialize all three agents with proper configurations."""
        
        # Market Observer
        mo_config = {
            'window': self.config.mo_window,
            'window_key': self.config.mo_window_key,
            'units_count': self.config.mo_units_count,
            'heads': self.config.mo_heads,
            'layers': self.config.mo_layers,
            'forecast': self.config.mo_forecast,
            'batch_size': self.config.batch_size
        }
        self.market_observer = create_market_observer(mo_config, self.enhanced_agents)
        self.market_observer.to(self.device)
        
        # RL Agent
        rl_config = {
            'window': self.config.rl_window,
            'units_count': self.config.rl_units_count,
            'segments': self.config.rl_segments,
            'rho': self.config.rl_rho,
            'layers': self.config.rl_layers,
            'n_actions': self.config.rl_n_actions,
            'batch_size': self.config.batch_size,
            'learning_rate': self.config.rl_learning_rate,
            'sam_rho': self.config.rl_sam_rho
        }
        self.rl_agent = create_rl_agent(rl_config, use_td3=self.enhanced_agents)
        self.rl_agent.to(self.device)
        
        # Controller Agent
        ctrl_config = {
            'window': self.config.ctrl_window,
            'window_key': self.config.ctrl_window_key,
            'units_count': self.config.ctrl_units_count,
            'heads': self.config.ctrl_heads,
            'window_kv': self.config.ctrl_window_kv,
            'units_kv': self.config.ctrl_units_kv,
            'layers': self.config.ctrl_layers,
            'batch_size': self.config.batch_size
        }
        self.controller = create_controller_agent(ctrl_config, self.enhanced_agents)
        self.controller.to(self.device)
        
        # Controller optimizer
        self.controller_optimizer = optim.Adam(self.controller.parameters(), lr=1e-4)
        
        self.logger.info("MASA Framework initialized successfully")
        
    def forward(
        self,
        market_data: torch.Tensor,
        training: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete MASA system.
        
        Args:
            market_data: Input market data tensor
            training: Whether in training mode
            
        Returns:
            Dictionary containing all agent outputs and final portfolio weights
        """
        batch_size = market_data.shape[0]
        
        # 1. Market Observer analysis
        market_forecast, risk_boundary = self.market_observer(market_data)
        market_vector = self.market_observer.get_market_vector(market_data)
        
        # 2. RL Agent decision making
        rl_actions, state_value = self.rl_agent(market_data, deterministic=not training)
        
        # 3. Controller risk adjustment
        final_actions, risk_assessment = self.controller(
            rl_actions, market_forecast, risk_boundary
        )
        
        # 4. Apply position size constraints
        final_actions = self._apply_position_constraints(final_actions)
        
        return {
            'market_forecast': market_forecast,
            'risk_boundary': risk_boundary,
            'market_vector': market_vector,
            'rl_actions': rl_actions,
            'state_value': state_value,
            'final_actions': final_actions,
            'risk_assessment': risk_assessment
        }
    
    def _apply_position_constraints(self, actions: torch.Tensor) -> torch.Tensor:
        """Apply position size and other portfolio constraints."""
        # Limit maximum position size
        max_pos = self.config.max_position_size
        actions = torch.clamp(actions, 0, max_pos)
        
        # Renormalize to ensure weights sum to 1
        actions = actions / actions.sum(dim=-1, keepdim=True)
        
        return actions
    
    def train_episode(
        self,
        market_data: torch.Tensor,
        returns: torch.Tensor,
        episode_length: int = 100
    ) -> Dict[str, float]:
        """
        Train the MASA system for one episode.
        
        Args:
            market_data: Market data for the episode
            returns: Actual returns for the episode
            episode_length: Length of the episode
            
        Returns:
            Dictionary with training metrics
        """
        self.market_observer.train()
        self.rl_agent.train()
        self.controller.train()
        
        episode_rewards = []
        episode_losses = {}
        
        for step in range(episode_length):
            if step + 1 >= market_data.shape[0]:
                break
                
            # Current state
            current_state = market_data[step:step+1]
            next_state = market_data[step+1:step+2] if step+1 < market_data.shape[0] else current_state
            
            # Forward pass
            outputs = self.forward(current_state, training=True)
            
            # Compute reward based on portfolio performance
            portfolio_weights = outputs['final_actions']
            step_return = returns[step:step+1]
            portfolio_return = torch.sum(portfolio_weights * step_return.unsqueeze(0), dim=-1)
            
            # Risk-adjusted reward
            risk_penalty = outputs['risk_assessment'].squeeze() * 0.1
            reward = portfolio_return - risk_penalty
            episode_rewards.append(reward.item())
            
            # Add experience to RL agent
            done = torch.tensor([step == episode_length - 1], dtype=torch.float32, device=self.device)
            self.rl_agent.add_experience(
                current_state.squeeze(),
                outputs['rl_actions'].squeeze(),
                reward.squeeze(),
                next_state.squeeze(),
                done.squeeze()
            )
            
            # Train RL agent
            if len(self.rl_agent.experience_buffer) >= self.config.batch_size:
                rl_losses = self.rl_agent.train_step(self.config.batch_size)
                if rl_losses:
                    for key, value in rl_losses.items():
                        if key not in episode_losses:
                            episode_losses[key] = []
                        episode_losses[key].append(value)
            
            # Train Controller
            if step > 0:
                controller_losses = self.controller.compute_controller_loss(
                    outputs['rl_actions'],
                    outputs['market_forecast'],
                    outputs['risk_boundary'],
                    portfolio_return,
                    portfolio_return  # Simplified: using same value
                )
                
                self.controller_optimizer.zero_grad()
                controller_losses['total_loss'].backward()
                self.controller_optimizer.step()
                
                for key, value in controller_losses.items():
                    ctrl_key = f'controller_{key}'
                    if ctrl_key not in episode_losses:
                        episode_losses[ctrl_key] = []
                    episode_losses[ctrl_key].append(value.item())
        
        # Compute episode metrics
        episode_return = sum(episode_rewards)
        episode_metrics = {
            'episode_return': episode_return,
            'average_reward': np.mean(episode_rewards),
            'reward_std': np.std(episode_rewards)
        }
        
        # Add average losses
        for key, values in episode_losses.items():
            episode_metrics[f'avg_{key}'] = np.mean(values)
            
        self.episode_rewards.append(episode_return)
        self.training_step += 1
        
        return episode_metrics
    
    def evaluate(
        self,
        market_data: torch.Tensor,
        returns: torch.Tensor,
        initial_portfolio_value: float = 100000.0
    ) -> Dict[str, Any]:
        """
        Evaluate the MASA system performance.
        
        Args:
            market_data: Market data for evaluation
            returns: Actual returns
            initial_portfolio_value: Starting portfolio value
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.market_observer.eval()
        self.rl_agent.eval()
        self.controller.eval()
        
        portfolio_values = [initial_portfolio_value]
        portfolio_weights_history = []
        risk_assessments = []
        
        with torch.no_grad():
            for step in range(len(market_data)):
                current_state = market_data[step:step+1]
                
                # Get portfolio weights
                outputs = self.forward(current_state, training=False)
                weights = outputs['final_actions'].squeeze().cpu().numpy()
                
                portfolio_weights_history.append(weights)
                risk_assessments.append(outputs['risk_assessment'].item())
                
                # Compute portfolio return
                if step < len(returns):
                    step_returns = returns[step].cpu().numpy()
                    portfolio_return = np.sum(weights * step_returns)
                    new_value = portfolio_values[-1] * (1 + portfolio_return)
                    portfolio_values.append(new_value)
        
        # Compute performance metrics
        portfolio_values = np.array(portfolio_values[1:])  # Remove initial value
        portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        metrics = self._compute_performance_metrics(
            portfolio_returns, portfolio_values, initial_portfolio_value
        )
        
        return {
            'metrics': metrics,
            'portfolio_values': portfolio_values,
            'portfolio_weights': np.array(portfolio_weights_history),
            'risk_assessments': np.array(risk_assessments),
            'portfolio_returns': portfolio_returns
        }
    
    def _compute_performance_metrics(
        self,
        returns: np.ndarray,
        values: np.ndarray,
        initial_value: float
    ) -> Dict[str, float]:
        """Compute comprehensive performance metrics."""
        
        # Total return
        total_return = (values[-1] - initial_value) / initial_value
        
        # Annualized return (assuming daily data)
        trading_days = 252
        annualized_return = (1 + total_return) ** (trading_days / len(returns)) - 1
        
        # Volatility
        volatility = np.std(returns) * np.sqrt(trading_days)
        
        # Sharpe ratio
        risk_free_rate = 0.02  # Assume 2% risk-free rate
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / peak
        max_drawdown = np.max(drawdown)
        
        # Win rate
        win_rate = np.mean(returns > 0)
        
        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(trading_days) if len(downside_returns) > 0 else 0
        sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'calmar_ratio': calmar_ratio,
            'sortino_ratio': sortino_ratio
        }
    
    def backtest(
        self,
        market_data: torch.Tensor,
        returns: torch.Tensor,
        train_ratio: float = 0.7,
        initial_value: float = 100000.0
    ) -> Dict[str, Any]:
        """
        Perform backtesting of the MASA system.
        
        Args:
            market_data: Historical market data
            returns: Historical returns
            train_ratio: Ratio of data to use for training
            initial_value: Initial portfolio value
            
        Returns:
            Dictionary with backtesting results
        """
        data_length = len(market_data)
        train_length = int(data_length * train_ratio)
        
        # Split data
        train_data = market_data[:train_length]
        train_returns = returns[:train_length]
        test_data = market_data[train_length:]
        test_returns = returns[train_length:]
        
        self.logger.info(f"Starting backtest: {train_length} training samples, {len(test_data)} test samples")
        
        # Training phase
        training_metrics = []
        for episode in range(10):  # Train for 10 episodes
            episode_metrics = self.train_episode(train_data, train_returns)
            training_metrics.append(episode_metrics)
            
            if episode % 2 == 0:
                self.logger.info(f"Episode {episode}: Return = {episode_metrics['episode_return']:.4f}")
        
        # Evaluation phase
        test_results = self.evaluate(test_data, test_returns, initial_value)
        
        return {
            'training_metrics': training_metrics,
            'test_results': test_results,
            'config': self.config
        }
    
    def get_portfolio_allocation(
        self,
        market_data: torch.Tensor,
        asset_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get current portfolio allocation recommendation.
        
        Args:
            market_data: Current market data
            asset_names: Names of assets (optional)
            
        Returns:
            Dictionary with allocation details
        """
        self.market_observer.eval()
        self.rl_agent.eval()
        self.controller.eval()
        
        with torch.no_grad():
            outputs = self.forward(market_data, training=False)
            
            weights = outputs['final_actions'].squeeze().cpu().numpy()
            risk_score = outputs['risk_assessment'].item()
            
            # Create allocation dictionary
            if asset_names is None:
                asset_names = [f'Asset_{i+1}' for i in range(len(weights))]
                
            allocation = {
                name: float(weight) for name, weight in zip(asset_names, weights)
            }
            
            # Market analysis
            if hasattr(self.market_observer, 'analyze_trends'):
                trend_analysis = self.market_observer.analyze_trends(market_data)
                market_regime = trend_analysis['market_regime'].item()
                regime_name = ['Sideways', 'Bull', 'Bear'][int(market_regime) + 1]
            else:
                regime_name = 'Unknown'
            
            return {
                'allocation': allocation,
                'risk_score': risk_score,
                'market_regime': regime_name,
                'total_allocated': sum(weights),
                'max_position': max(weights),
                'diversification_ratio': 1.0 / np.sum(weights ** 2),  # Inverse HHI
                'recommendation': self._generate_recommendation(weights, risk_score)
            }
    
    def _generate_recommendation(self, weights: np.ndarray, risk_score: float) -> str:
        """Generate human-readable recommendation."""
        max_weight = np.max(weights)
        diversification = 1.0 / np.sum(weights ** 2)
        
        if risk_score > 0.7:
            risk_level = "High"
        elif risk_score > 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"
            
        if max_weight > 0.4:
            concentration = "Concentrated"
        elif diversification > 5:
            concentration = "Well-diversified"
        else:
            concentration = "Moderately diversified"
            
        return f"{risk_level} risk, {concentration} portfolio"
    
    def save_system(self, filepath: str):
        """Save the complete MASA system."""
        filepath = Path(filepath)
        filepath.mkdir(parents=True, exist_ok=True)
        
        # Save individual agents
        torch.save(self.market_observer.state_dict(), filepath / 'market_observer.pth')
        self.rl_agent.save_checkpoint(str(filepath / 'rl_agent.pth'))
        torch.save(self.controller.state_dict(), filepath / 'controller.pth')
        torch.save(self.controller_optimizer.state_dict(), filepath / 'controller_optimizer.pth')
        
        # Save configuration and metrics
        system_state = {
            'config': self.config,
            'training_step': self.training_step,
            'episode_rewards': self.episode_rewards,
            'performance_metrics': self.performance_metrics,
            'enhanced_agents': self.enhanced_agents
        }
        torch.save(system_state, filepath / 'system_state.pth')
        
        self.logger.info(f"MASA system saved to {filepath}")
        
    def load_system(self, filepath: str):
        """Load the complete MASA system."""
        filepath = Path(filepath)
        
        # Load individual agents
        self.market_observer.load_state_dict(torch.load(filepath / 'market_observer.pth', map_location=self.device))
        self.rl_agent.load_checkpoint(str(filepath / 'rl_agent.pth'))
        self.controller.load_state_dict(torch.load(filepath / 'controller.pth', map_location=self.device))
        self.controller_optimizer.load_state_dict(torch.load(filepath / 'controller_optimizer.pth', map_location=self.device))
        
        # Load system state
        system_state = torch.load(filepath / 'system_state.pth', map_location=self.device)
        self.training_step = system_state['training_step']
        self.episode_rewards = system_state['episode_rewards']
        self.performance_metrics = system_state['performance_metrics']
        
        self.logger.info(f"MASA system loaded from {filepath}")
        
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and health metrics."""
        return {
            'training_step': self.training_step,
            'total_episodes': len(self.episode_rewards),
            'average_episode_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'performance_metrics': self.performance_metrics,
            'device': str(self.device),
            'config': self.config,
            'agents_status': {
                'market_observer': 'Enhanced' if self.enhanced_agents else 'Standard',
                'rl_agent': 'TD3' if isinstance(self.rl_agent, TD3RLAgent) else 'Standard',
                'controller': 'Advanced' if isinstance(self.controller, AdvancedControllerAgent) else 'Standard'
            }
        }


class MASATradingEnvironment:
    """
    Trading environment for MASA framework with realistic market simulation.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        lookback_window: int = 50,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005
    ):
        self.data = data
        self.lookback_window = lookback_window
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        
        self.current_step = 0
        self.portfolio_value = 100000.0
        self.positions = None
        
        # Preprocess data
        self._preprocess_data()
        
    def _preprocess_data(self):
        """Preprocess market data for MASA input."""
        # Normalize features
        self.normalized_data = self.data.copy()
        for col in self.data.columns:
            if col not in ['Date', 'Symbol']:
                self.normalized_data[col] = (self.data[col] - self.data[col].mean()) / self.data[col].std()
        
        # Compute returns
        price_cols = [col for col in self.data.columns if 'Close' in col or 'Price' in col]
        self.returns = self.data[price_cols].pct_change().fillna(0)
        
    def reset(self) -> torch.Tensor:
        """Reset environment to initial state."""
        self.current_step = self.lookback_window
        self.portfolio_value = 100000.0
        self.positions = None
        
        return self._get_state()
    
    def _get_state(self) -> torch.Tensor:
        """Get current market state for MASA input."""
        start_idx = max(0, self.current_step - self.lookback_window)
        end_idx = self.current_step
        
        state_data = self.normalized_data.iloc[start_idx:end_idx].values
        
        # Pad if necessary
        if state_data.shape[0] < self.lookback_window:
            padding = np.zeros((self.lookback_window - state_data.shape[0], state_data.shape[1]))
            state_data = np.vstack([padding, state_data])
        
        return torch.tensor(state_data, dtype=torch.float32).flatten().unsqueeze(0)
    
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, Dict]:
        """
        Execute one step in the trading environment.
        
        Args:
            action: Portfolio weights from MASA
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        action_np = action.squeeze().cpu().numpy()
        
        # Apply transaction costs and slippage
        if self.positions is not None:
            weight_changes = np.abs(action_np - self.positions)
            transaction_costs = np.sum(weight_changes) * self.transaction_cost
        else:
            transaction_costs = np.sum(action_np) * self.transaction_cost
            
        # Compute portfolio return
        if self.current_step < len(self.returns):
            step_returns = self.returns.iloc[self.current_step].values
            portfolio_return = np.sum(action_np * step_returns)
            
            # Apply slippage
            portfolio_return -= self.slippage * np.sum(np.abs(action_np))
            
            # Apply transaction costs
            portfolio_return -= transaction_costs
            
            # Update portfolio value
            self.portfolio_value *= (1 + portfolio_return)
        else:
            portfolio_return = 0
            
        # Update positions
        self.positions = action_np.copy()
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(self.data) - 1
        
        # Get next state
        next_state = self._get_state() if not done else self._get_state()
        
        # Compute reward (risk-adjusted return)
        reward = portfolio_return
        
        # Info dictionary
        info = {
            'portfolio_value': self.portfolio_value,
            'step_return': portfolio_return,
            'transaction_costs': transaction_costs,
            'positions': self.positions.copy()
        }
        
        return next_state, reward, done, info


# Utility functions for MASA framework
def create_masa_system(
    config: Optional[MASAConfig] = None,
    enhanced: bool = True,
    device: Optional[str] = None
) -> MASAFramework:
    """
    Create a complete MASA system with default or custom configuration.
    
    Args:
        config: Custom configuration (uses default if None)
        enhanced: Whether to use enhanced agents
        device: Device to run on
        
    Returns:
        Initialized MASAFramework
    """
    if config is None:
        config = MASAConfig()
        
    return MASAFramework(config, enhanced, device)


def load_market_data(filepath: str) -> pd.DataFrame:
    """
    Load market data from CSV file.
    
    Expected format: Date, Open, High, Low, Close, Volume for each asset
    """
    data = pd.read_csv(filepath)
    data['Date'] = pd.to_datetime(data['Date'])
    return data


def prepare_masa_input(data: pd.DataFrame, lookback: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare market data for MASA input format.
    
    Args:
        data: Market data DataFrame
        lookback: Lookback window size
        
    Returns:
        Tuple of (market_data_tensor, returns_tensor)
    """
    # Select numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    # Normalize data
    normalized_data = (data[numeric_cols] - data[numeric_cols].mean()) / data[numeric_cols].std()
    
    # Create sequences
    sequences = []
    returns_sequences = []
    
    for i in range(lookback, len(normalized_data)):
        sequence = normalized_data.iloc[i-lookback:i].values.flatten()
        sequences.append(sequence)
        
        # Compute returns for this step
        if i < len(normalized_data):
            current_prices = data[numeric_cols].iloc[i].values
            prev_prices = data[numeric_cols].iloc[i-1].values
            step_returns = (current_prices - prev_prices) / (prev_prices + 1e-8)
            returns_sequences.append(step_returns)
    
    market_tensor = torch.tensor(np.array(sequences), dtype=torch.float32)
    returns_tensor = torch.tensor(np.array(returns_sequences), dtype=torch.float32)
    
    return market_tensor, returns_tensor


# Example stress test scenarios
STRESS_SCENARIOS = {
    'market_crash': torch.tensor([0.7, 0.7, 0.7, 0.7, 0.7]),  # 30% decline across all assets
    'volatility_spike': torch.tensor([1.5, 0.8, 1.2, 0.9, 1.3]),  # Mixed volatility
    'sector_rotation': torch.tensor([0.8, 1.2, 0.9, 1.1, 0.7]),  # Sector-specific impacts
    'interest_rate_shock': torch.tensor([0.9, 0.8, 1.1, 0.9, 0.8]),  # Interest rate sensitivity
    'geopolitical_crisis': torch.tensor([0.6, 0.9, 0.8, 1.1, 0.7])  # Flight to quality
}