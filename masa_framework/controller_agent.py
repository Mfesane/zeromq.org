"""
Controller Agent for MASA Framework
Implements risk assessment and action adjustment using Transformer decoder architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any
try:
    from .base_neural import (
        BaseNeuralLayer, TransposeLayer, RelativeSelfAttention, 
        RelativeCrossAttention, ResidualConvBlock, SAMOptimizedLinear,
        initialize_weights
    )
except ImportError:
    from base_neural import (
        BaseNeuralLayer, TransposeLayer, RelativeSelfAttention, 
        RelativeCrossAttention, ResidualConvBlock, SAMOptimizedLinear,
        initialize_weights
    )


class TransformerDecoderLayer(nn.Module):
    """Single Transformer decoder layer with relative attention."""
    
    def __init__(
        self,
        d_model: int,
        d_key: int,
        n_heads: int,
        seq_len_q: int,
        seq_len_kv: int,
        use_self_attention: bool = True
    ):
        super().__init__()
        
        self.d_model = d_model
        self.use_self_attention = use_self_attention
        
        # Self-attention (optional, only if sequence length > 1)
        if use_self_attention and seq_len_q > 1:
            self.self_attention = RelativeSelfAttention(
                d_model=d_model,
                n_heads=n_heads,
                seq_len=seq_len_q
            )
            self.norm1 = nn.LayerNorm(d_model)
        else:
            self.self_attention = None
            self.norm1 = None
            
        # Cross-attention
        self.cross_attention = RelativeCrossAttention(
            d_model=d_model,
            n_heads=n_heads,
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv
        )
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network (using residual conv block)
        self.ffn = ResidualConvBlock(
            in_channels=d_model,
            out_channels=d_model
        )
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """Forward pass through decoder layer."""
        batch_size = query.shape[0]
        seq_len_q = query.shape[1] // self.d_model
        
        # Reshape query for processing
        query_reshaped = query.view(batch_size, seq_len_q, self.d_model)
        
        # Self-attention (if applicable)
        if self.self_attention is not None:
            attn_output = self.self_attention(query)
            attn_output = attn_output.view(batch_size, seq_len_q, self.d_model)
            query_reshaped = self.norm1(query_reshaped + self.dropout(attn_output))
            
        # Cross-attention
        query_flat = query_reshaped.view(batch_size, -1)
        cross_attn_output = self.cross_attention(query_flat, key_value)
        cross_attn_output = cross_attn_output.view(batch_size, seq_len_q, self.d_model)
        query_reshaped = self.norm2(query_reshaped + self.dropout(cross_attn_output))
        
        # Feed-forward network
        ffn_input = query_reshaped.view(batch_size, -1)
        ffn_output = self.ffn(ffn_input)
        ffn_output = ffn_output.view(batch_size, seq_len_q, self.d_model)
        output = self.norm3(query_reshaped + self.dropout(ffn_output))
        
        return output.view(batch_size, -1)


class ControllerAgent(nn.Module):
    """
    Controller Agent that assesses risk and adjusts RL agent actions.
    
    Uses Transformer decoder architecture to process dual input streams:
    1. RL agent actions
    2. Market Observer forecasts
    """
    
    def __init__(
        self,
        window: int,        # Action vector dimension
        window_key: int,    # Attention key dimension
        units_count: int,   # Action sequence length
        heads: int,         # Number of attention heads
        window_kv: int,     # Market forecast dimension
        units_kv: int,      # Market forecast sequence length
        layers: int,        # Number of decoder layers
        batch_size: int = 32
    ):
        super().__init__()
        
        self.window = window
        self.window_key = window_key
        self.units_count = units_count
        self.heads = heads
        self.window_kv = window_kv
        self.units_kv = units_kv
        self.layers = layers
        self.batch_size = batch_size
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Build the Controller architecture
        self._build_architecture()
        
        # Initialize weights
        self.apply(initialize_weights)
        
    def _build_architecture(self):
        """Build the Controller Agent neural network architecture."""
        
        # Secondary input processing (Market Observer data)
        self.secondary_input_linear = nn.Linear(
            self.window_kv * self.units_kv,
            self.window_kv * self.units_kv
        )
        
        self.secondary_transpose = TransposeLayer(
            self.units_kv,
            self.window_kv,
            self.batch_size
        )
        
        # Primary input projection
        self.primary_projection = nn.Linear(
            self.window * self.units_count,
            self.window_key * self.units_count
        )
        
        # Transformer decoder layers
        self.decoder_layers = nn.ModuleList()
        for _ in range(self.layers):
            decoder_layer = TransformerDecoderLayer(
                d_model=self.window_key,
                d_key=self.window_key,
                n_heads=self.heads,
                seq_len_q=self.units_count,
                seq_len_kv=self.units_kv,
                use_self_attention=(self.units_count > 1)
            )
            self.decoder_layers.append(decoder_layer)
            
        # Output layer with activation constraint
        self.output_conv = nn.Conv1d(
            in_channels=self.window_key,
            out_channels=self.window,
            kernel_size=1
        )
        
        # Final output projection
        self.output_projection = nn.Linear(
            self.window * self.units_count,
            self.window * self.units_count
        )
        
        # Risk assessment network
        self.risk_assessor = nn.Sequential(
            nn.Linear(self.window_kv * self.units_kv + self.window * self.units_count, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        rl_actions: torch.Tensor,
        market_forecast: torch.Tensor,
        risk_boundary: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Controller Agent.
        
        Args:
            rl_actions: Actions from RL agent
            market_forecast: Forecast from Market Observer
            risk_boundary: Risk boundary from Market Observer
            
        Returns:
            Tuple of (adjusted_actions, risk_assessment)
        """
        batch_size = rl_actions.shape[0]
        
        # Process secondary input (market forecast)
        secondary_processed = self.secondary_input_linear(market_forecast)
        secondary_transposed = self.secondary_transpose(secondary_processed)
        
        # Project primary input (RL actions)
        primary_projected = self.primary_projection(rl_actions)
        
        # Process through decoder layers
        decoder_output = primary_projected
        for decoder_layer in self.decoder_layers:
            decoder_output = decoder_layer(decoder_output, secondary_transposed)
            
        # Apply output convolution
        conv_input = decoder_output.view(batch_size, self.window_key, self.units_count)
        conv_output = self.output_conv(conv_input)
        conv_output = torch.sigmoid(conv_output)  # Constrain to [0, 1]
        
        # Final output projection
        output_flat = conv_output.view(batch_size, -1)
        adjusted_actions = self.output_projection(output_flat)
        adjusted_actions = torch.sigmoid(adjusted_actions)
        
        # Normalize to portfolio weights
        adjusted_actions = F.softmax(adjusted_actions, dim=-1)
        
        # Assess risk
        risk_input = torch.cat([market_forecast, rl_actions], dim=-1)
        risk_assessment = self.risk_assessor(risk_input)
        
        # Apply risk-based adjustment
        if risk_boundary is not None:
            risk_factor = torch.clamp(risk_boundary, 0.1, 1.0)
            adjusted_actions = self._apply_risk_adjustment(adjusted_actions, risk_factor)
            
        return adjusted_actions, risk_assessment
    
    def _apply_risk_adjustment(self, actions: torch.Tensor, risk_factor: torch.Tensor) -> torch.Tensor:
        """Apply risk-based adjustment to actions."""
        batch_size = actions.shape[0]
        
        # Reduce concentration when risk is high
        risk_adjustment = 1.0 - 0.5 * risk_factor  # Scale factor based on risk
        
        # Apply uniform distribution mixing for high risk
        uniform_dist = torch.ones_like(actions) / actions.shape[-1]
        adjusted_actions = risk_adjustment * actions + (1 - risk_adjustment) * uniform_dist
        
        # Renormalize
        adjusted_actions = F.softmax(adjusted_actions, dim=-1)
        
        return adjusted_actions
    
    def compute_controller_loss(
        self,
        rl_actions: torch.Tensor,
        market_forecast: torch.Tensor,
        risk_boundary: torch.Tensor,
        target_returns: torch.Tensor,
        actual_returns: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss for Controller Agent training.
        
        Args:
            rl_actions: Actions from RL agent
            market_forecast: Market Observer forecast
            risk_boundary: Risk boundary from Market Observer
            target_returns: Expected returns
            actual_returns: Actual realized returns
            
        Returns:
            Dictionary containing loss components
        """
        adjusted_actions, risk_assessment = self.forward(rl_actions, market_forecast, risk_boundary)
        
        # Return prediction loss
        return_loss = F.mse_loss(risk_assessment.squeeze(), actual_returns)
        
        # Risk constraint loss (penalize high risk when boundary is low)
        risk_constraint_loss = torch.relu(risk_assessment.squeeze() - risk_boundary.squeeze()).mean()
        
        # Action smoothness loss (prevent drastic changes)
        action_diff = torch.diff(adjusted_actions, dim=-1)
        smoothness_loss = torch.mean(torch.abs(action_diff))
        
        # Portfolio concentration penalty (encourage diversification)
        concentration_penalty = torch.mean(torch.sum(adjusted_actions ** 2, dim=-1))
        
        # Total loss
        total_loss = (return_loss + 
                     0.5 * risk_constraint_loss + 
                     0.1 * smoothness_loss + 
                     0.1 * concentration_penalty)
        
        return {
            'total_loss': total_loss,
            'return_loss': return_loss,
            'risk_constraint_loss': risk_constraint_loss,
            'smoothness_loss': smoothness_loss,
            'concentration_penalty': concentration_penalty
        }
    
    def evaluate_risk_return_tradeoff(
        self,
        rl_actions: torch.Tensor,
        market_forecast: torch.Tensor,
        risk_boundary: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate the risk-return tradeoff of the adjusted actions.
        
        Returns:
            Dictionary with risk-return metrics
        """
        with torch.no_grad():
            adjusted_actions, risk_assessment = self.forward(rl_actions, market_forecast, risk_boundary)
            
            # Expected return (simplified)
            forecast_reshaped = market_forecast.view(market_forecast.shape[0], -1, self.window_kv)
            expected_returns = torch.sum(adjusted_actions.unsqueeze(-1) * forecast_reshaped.mean(dim=-1), dim=-1)
            
            # Portfolio variance (risk measure)
            action_variance = torch.var(adjusted_actions, dim=-1)
            
            # Sharpe ratio approximation
            sharpe_ratio = expected_returns / (torch.sqrt(action_variance) + 1e-8)
            
            # Maximum drawdown estimate
            cumulative_returns = torch.cumsum(expected_returns, dim=0)
            running_max = torch.cummax(cumulative_returns, dim=0)[0]
            drawdown = (running_max - cumulative_returns) / (running_max + 1e-8)
            max_drawdown = torch.max(drawdown, dim=0)[0]
            
            return {
                'adjusted_actions': adjusted_actions,
                'expected_returns': expected_returns,
                'portfolio_variance': action_variance,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'risk_assessment': risk_assessment.squeeze()
            }
    
    def get_risk_adjusted_weights(
        self,
        rl_actions: torch.Tensor,
        market_forecast: torch.Tensor,
        risk_boundary: torch.Tensor,
        risk_tolerance: float = 0.5
    ) -> torch.Tensor:
        """
        Get risk-adjusted portfolio weights.
        
        Args:
            rl_actions: Original RL agent actions
            market_forecast: Market Observer forecast
            risk_boundary: Risk boundary from Market Observer
            risk_tolerance: User's risk tolerance (0-1)
            
        Returns:
            Risk-adjusted portfolio weights
        """
        with torch.no_grad():
            adjusted_actions, risk_assessment = self.forward(rl_actions, market_forecast, risk_boundary)
            
            # Further adjust based on user risk tolerance
            if risk_tolerance < 0.5:  # Conservative
                # Increase diversification for conservative investors
                uniform_weight = 1.0 / adjusted_actions.shape[-1]
                conservative_factor = 2 * (0.5 - risk_tolerance)  # 0 to 1
                adjusted_actions = (1 - conservative_factor) * adjusted_actions + conservative_factor * uniform_weight
                
            elif risk_tolerance > 0.5:  # Aggressive
                # Allow more concentration for aggressive investors
                aggressive_factor = 2 * (risk_tolerance - 0.5)  # 0 to 1
                # Amplify differences from uniform distribution
                uniform_dist = torch.ones_like(adjusted_actions) / adjusted_actions.shape[-1]
                deviation = adjusted_actions - uniform_dist
                adjusted_actions = uniform_dist + (1 + aggressive_factor) * deviation
                
            # Ensure valid portfolio weights
            adjusted_actions = torch.clamp(adjusted_actions, 0, 1)
            adjusted_actions = adjusted_actions / adjusted_actions.sum(dim=-1, keepdim=True)
            
            return adjusted_actions
    
    def to(self, device):
        """Move model to specified device."""
        super().to(device)
        self.device = device
        return self


class AdvancedControllerAgent(ControllerAgent):
    """
    Advanced Controller Agent with additional risk management features.
    
    Includes:
    - Dynamic risk budgeting
    - Stress testing capabilities
    - Regime-aware adjustments
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Additional risk management components
        self.risk_budgeter = nn.Sequential(
            nn.Linear(self.window_kv * self.units_kv, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.window),  # Risk budget per asset
            nn.Softmax(dim=-1)
        )
        
        self.stress_detector = nn.Sequential(
            nn.Linear(self.window_kv * self.units_kv, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.regime_adapter = nn.Sequential(
            nn.Linear(self.window_kv * self.units_kv, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),  # Bull, Bear, Sideways adjustments
            nn.Softmax(dim=-1)
        )
        
        # Apply initialization to new components
        self.risk_budgeter.apply(initialize_weights)
        self.stress_detector.apply(initialize_weights)
        self.regime_adapter.apply(initialize_weights)
        
    def forward(
        self,
        rl_actions: torch.Tensor,
        market_forecast: torch.Tensor,
        risk_boundary: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Enhanced forward pass with advanced risk management."""
        
        # Get base adjusted actions and risk assessment
        base_actions, base_risk = super().forward(rl_actions, market_forecast, risk_boundary)
        
        # Compute additional risk factors
        risk_budget = self.risk_budgeter(market_forecast)
        stress_level = self.stress_detector(market_forecast)
        regime_weights = self.regime_adapter(market_forecast)
        
        # Apply risk budgeting
        risk_adjusted_actions = base_actions * risk_budget
        
        # Apply stress-based adjustments
        stress_factor = 1.0 - 0.5 * stress_level  # Reduce concentration under stress
        uniform_dist = torch.ones_like(risk_adjusted_actions) / risk_adjusted_actions.shape[-1]
        stress_adjusted_actions = stress_factor * risk_adjusted_actions + (1 - stress_factor) * uniform_dist
        
        # Apply regime-based adjustments
        regime_adjustments = regime_weights.unsqueeze(-1)  # Bull, Bear, Sideways
        
        # Bull market: slight concentration increase
        bull_adjustment = 1.1 * stress_adjusted_actions
        # Bear market: significant diversification
        bear_adjustment = 0.7 * stress_adjusted_actions + 0.3 * uniform_dist
        # Sideways: maintain current allocation
        sideways_adjustment = stress_adjusted_actions
        
        # Weighted combination based on regime probabilities
        final_actions = (regime_adjustments[:, 0:1] * bull_adjustment +
                        regime_adjustments[:, 1:2] * bear_adjustment +
                        regime_adjustments[:, 2:3] * sideways_adjustment)
        
        # Ensure valid portfolio weights
        final_actions = torch.clamp(final_actions, 0, 1)
        final_actions = final_actions / final_actions.sum(dim=-1, keepdim=True)
        
        # Enhanced risk assessment
        enhanced_risk = base_risk + 0.3 * stress_level
        enhanced_risk = torch.clamp(enhanced_risk, 0, 1)
        
        return final_actions, enhanced_risk
    
    def compute_var_risk(self, actions: torch.Tensor, market_forecast: torch.Tensor, confidence: float = 0.05) -> torch.Tensor:
        """
        Compute Value at Risk (VaR) for the portfolio.
        
        Args:
            actions: Portfolio weights
            market_forecast: Market forecast data
            confidence: Confidence level for VaR (default 5%)
            
        Returns:
            VaR estimate
        """
        with torch.no_grad():
            batch_size = actions.shape[0]
            
            # Reshape market forecast
            forecast_reshaped = market_forecast.view(batch_size, -1, self.window_kv)
            
            # Compute portfolio returns for each forecast scenario
            portfolio_returns = torch.sum(actions.unsqueeze(-1) * forecast_reshaped, dim=1)
            
            # Compute VaR (simplified approach)
            var_quantile = torch.quantile(portfolio_returns, confidence, dim=-1)
            
            return var_quantile.unsqueeze(-1)
    
    def compute_cvar_risk(self, actions: torch.Tensor, market_forecast: torch.Tensor, confidence: float = 0.05) -> torch.Tensor:
        """
        Compute Conditional Value at Risk (CVaR) for the portfolio.
        
        Args:
            actions: Portfolio weights
            market_forecast: Market forecast data
            confidence: Confidence level for CVaR
            
        Returns:
            CVaR estimate
        """
        with torch.no_grad():
            batch_size = actions.shape[0]
            
            # Reshape market forecast
            forecast_reshaped = market_forecast.view(batch_size, -1, self.window_kv)
            
            # Compute portfolio returns
            portfolio_returns = torch.sum(actions.unsqueeze(-1) * forecast_reshaped, dim=1)
            
            # Compute VaR threshold
            var_threshold = torch.quantile(portfolio_returns, confidence, dim=-1, keepdim=True)
            
            # Compute CVaR (expected return below VaR)
            below_var_mask = portfolio_returns <= var_threshold
            below_var_returns = torch.where(below_var_mask, portfolio_returns, torch.zeros_like(portfolio_returns))
            below_var_count = below_var_mask.sum(dim=-1, keepdim=True).float()
            
            cvar = below_var_returns.sum(dim=-1, keepdim=True) / (below_var_count + 1e-8)
            
            return cvar
    
    def stress_test_portfolio(
        self,
        actions: torch.Tensor,
        market_forecast: torch.Tensor,
        stress_scenarios: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Perform stress testing on the portfolio.
        
        Args:
            actions: Portfolio weights
            market_forecast: Base market forecast
            stress_scenarios: Dictionary of stress scenario adjustments
            
        Returns:
            Dictionary with stress test results
        """
        results = {}
        
        with torch.no_grad():
            for scenario_name, stress_adjustment in stress_scenarios.items():
                # Apply stress to market forecast
                stressed_forecast = market_forecast * stress_adjustment
                
                # Get adjusted actions under stress
                stressed_actions, stressed_risk = self.forward(actions, stressed_forecast)
                
                # Compute metrics under stress
                batch_size = actions.shape[0]
                forecast_reshaped = stressed_forecast.view(batch_size, -1, self.window_kv)
                portfolio_returns = torch.sum(stressed_actions.unsqueeze(-1) * forecast_reshaped, dim=1)
                
                results[scenario_name] = {
                    'adjusted_actions': stressed_actions,
                    'risk_assessment': stressed_risk,
                    'expected_return': portfolio_returns.mean(dim=-1),
                    'return_volatility': portfolio_returns.std(dim=-1),
                    'var_5': self.compute_var_risk(stressed_actions, stressed_forecast, 0.05),
                    'cvar_5': self.compute_cvar_risk(stressed_actions, stressed_forecast, 0.05)
                }
                
        return results


# Factory function for creating Controller Agent instances
def create_controller_agent(config: dict, advanced: bool = False) -> ControllerAgent:
    """
    Factory function to create Controller Agent instances.
    
    Args:
        config: Configuration dictionary with required parameters
        advanced: Whether to create advanced version with additional features
        
    Returns:
        ControllerAgent instance
    """
    required_keys = ['window', 'window_key', 'units_count', 'heads', 'window_kv', 'units_kv', 'layers']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    
    controller_class = AdvancedControllerAgent if advanced else ControllerAgent
    
    return controller_class(
        window=config['window'],
        window_key=config['window_key'],
        units_count=config['units_count'],
        heads=config['heads'],
        window_kv=config['window_kv'],
        units_kv=config['units_kv'],
        layers=config['layers'],
        batch_size=config.get('batch_size', 32)
    )