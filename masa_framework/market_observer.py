"""
Market Observer Agent for MASA Framework
Implements market trend analysis and forecasting using attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
try:
    from .base_neural import (
        BaseNeuralLayer, TransposeLayer, PiecewiseLinearRepresentation,
        RelativeSelfAttention, ResidualConvBlock, initialize_weights
    )
except ImportError:
    from base_neural import (
        BaseNeuralLayer, TransposeLayer, PiecewiseLinearRepresentation,
        RelativeSelfAttention, ResidualConvBlock, initialize_weights
    )


class MarketObserver(nn.Module):
    """
    Market Observer agent that analyzes market trends and provides forecasts.
    
    Uses a hybrid approach:
    1. Piecewise-linear representation for trend capture
    2. Attention with relative positional encoding for dependency analysis
    3. MLP for forecasting probable market behavior
    """
    
    def __init__(
        self,
        window: int,           # Size of vector describing single sequence element
        window_key: int,       # Dimensionality of attention components (Q, K, V)
        units_count: int,      # Historical depth of data
        heads: int,           # Number of attention heads
        layers: int,          # Number of attention layers
        forecast: int,        # Forecasting horizon
        batch_size: int = 32
    ):
        super().__init__()
        
        self.window = window
        self.window_key = window_key
        self.units_count = units_count
        self.heads = heads
        self.layers = layers
        self.forecast = forecast
        self.batch_size = batch_size
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Build the Market Observer architecture
        self._build_architecture()
        
        # Initialize weights
        self.apply(initialize_weights)
        
    def _build_architecture(self):
        """Build the Market Observer neural network architecture."""
        
        # 1. Transpose input data for proper sequence handling
        self.input_transpose = TransposeLayer(self.units_count, self.window, self.batch_size)
        
        # 2. Piecewise linear representation
        self.plr = PiecewiseLinearRepresentation(self.units_count, self.window, self.batch_size)
        
        # 3. Self-Attention layers for analyzing dependencies
        self.attention_layers = nn.ModuleList()
        for _ in range(self.layers):
            attention = RelativeSelfAttention(
                d_model=self.window_key,
                n_heads=self.heads,
                seq_len=self.window,
                batch_size=self.batch_size
            )
            self.attention_layers.append(attention)
            
        # 4. Projection layer to match attention input requirements
        self.attention_projection = nn.Linear(self.window, self.window_key)
        
        # 5. Forecast mapping using residual convolution
        self.forecast_conv = ResidualConvBlock(
            in_channels=self.window_key,
            out_channels=self.forecast,
            batch_size=self.batch_size
        )
        
        # 6. Output transpose to restore original dimensionality
        self.output_transpose = TransposeLayer(self.window_key, self.forecast, self.batch_size)
        
        # 7. Final projection to output size
        self.output_projection = nn.Linear(self.window_key * self.forecast, self.window * self.forecast)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Market Observer.
        
        Args:
            x: Input tensor of shape (batch_size, window * units_count)
            
        Returns:
            Tuple of (market_forecast, risk_boundary)
        """
        batch_size = x.shape[0]
        
        # 1. Transpose input for sequence processing
        x = self.input_transpose(x)
        
        # 2. Apply piecewise linear representation
        x = self.plr(x)
        
        # 3. Project to attention dimension
        x_reshaped = x.view(batch_size, self.window, self.units_count)
        x_projected = self.attention_projection(x_reshaped)
        x_flat = x_projected.view(batch_size, -1)
        
        # 4. Apply attention layers
        attention_output = x_flat
        for attention_layer in self.attention_layers:
            attention_output = attention_layer(attention_output)
            
        # 5. Apply forecast mapping
        forecast_output = self.forecast_conv(attention_output)
        
        # 6. Transpose output
        transposed_output = self.output_transpose(forecast_output)
        
        # 7. Final projection
        market_forecast = self.output_projection(transposed_output)
        
        # 8. Compute risk boundary (simplified approach)
        risk_boundary = self._compute_risk_boundary(attention_output, market_forecast)
        
        return market_forecast, risk_boundary
    
    def _compute_risk_boundary(self, attention_output: torch.Tensor, forecast: torch.Tensor) -> torch.Tensor:
        """
        Compute risk boundary based on attention output and forecast.
        
        This is a simplified implementation that computes volatility-based risk measures.
        """
        batch_size = attention_output.shape[0]
        
        # Compute volatility from attention patterns
        attention_reshaped = attention_output.view(batch_size, self.window_key, -1)
        volatility = torch.std(attention_reshaped, dim=-1)
        
        # Compute forecast uncertainty
        forecast_reshaped = forecast.view(batch_size, self.window, self.forecast)
        forecast_std = torch.std(forecast_reshaped, dim=-1)
        
        # Combine volatility and forecast uncertainty
        risk_components = torch.cat([
            volatility.mean(dim=-1, keepdim=True),
            forecast_std.mean(dim=-1, keepdim=True)
        ], dim=-1)
        
        # Simple risk boundary computation
        risk_boundary = torch.sigmoid(risk_components.mean(dim=-1, keepdim=True))
        
        return risk_boundary
    
    def get_market_vector(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract market vector representation for other agents.
        
        Args:
            x: Input tensor
            
        Returns:
            Market vector representation
        """
        with torch.no_grad():
            batch_size = x.shape[0]
            
            # Process through initial layers
            x = self.input_transpose(x)
            x = self.plr(x)
            
            # Project and apply attention
            x_reshaped = x.view(batch_size, self.window, self.units_count)
            x_projected = self.attention_projection(x_reshaped)
            x_flat = x_projected.view(batch_size, -1)
            
            # Apply first attention layer for market vector
            if len(self.attention_layers) > 0:
                market_vector = self.attention_layers[0](x_flat)
            else:
                market_vector = x_flat
                
            return market_vector
    
    def analyze_trends(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Analyze market trends and return detailed analysis.
        
        Args:
            x: Input market data
            
        Returns:
            Dictionary containing trend analysis results
        """
        with torch.no_grad():
            forecast, risk_boundary = self.forward(x)
            market_vector = self.get_market_vector(x)
            
            batch_size = x.shape[0]
            forecast_reshaped = forecast.view(batch_size, self.window, self.forecast)
            
            # Compute trend directions
            trend_direction = torch.sign(torch.diff(forecast_reshaped, dim=-1)).mean(dim=-1)
            
            # Compute trend strength
            trend_strength = torch.abs(torch.diff(forecast_reshaped, dim=-1)).mean(dim=-1)
            
            # Compute market regime (bull/bear/sideways)
            overall_trend = trend_direction.mean(dim=-1)
            market_regime = torch.where(
                overall_trend > 0.1, 
                torch.ones_like(overall_trend),  # Bull market
                torch.where(
                    overall_trend < -0.1,
                    -torch.ones_like(overall_trend),  # Bear market
                    torch.zeros_like(overall_trend)   # Sideways market
                )
            )
            
            return {
                'forecast': forecast,
                'risk_boundary': risk_boundary,
                'market_vector': market_vector,
                'trend_direction': trend_direction,
                'trend_strength': trend_strength,
                'market_regime': market_regime
            }
    
    def to(self, device):
        """Move model to specified device."""
        super().to(device)
        self.device = device
        return self


class EnhancedMarketObserver(MarketObserver):
    """
    Enhanced Market Observer with additional market analysis capabilities.
    
    Includes:
    - Volatility clustering detection
    - Regime change detection
    - Market stress indicators
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Additional components for enhanced analysis
        self.volatility_detector = nn.Sequential(
            nn.Linear(self.window * self.forecast, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.regime_detector = nn.Sequential(
            nn.Linear(self.window * self.forecast, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),  # Bull, Bear, Sideways
            nn.Softmax(dim=-1)
        )
        
        # Apply weight initialization to new components
        self.volatility_detector.apply(initialize_weights)
        self.regime_detector.apply(initialize_weights)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Enhanced forward pass with additional market analysis."""
        # Get base forecast and risk boundary
        market_forecast, risk_boundary = super().forward(x)
        
        # Compute enhanced risk boundary
        volatility_score = self.volatility_detector(market_forecast)
        regime_probs = self.regime_detector(market_forecast)
        
        # Enhance risk boundary with volatility and regime information
        enhanced_risk = risk_boundary + 0.3 * volatility_score + 0.2 * regime_probs[:, 1:2]  # Bear market probability
        enhanced_risk = torch.clamp(enhanced_risk, 0, 1)
        
        return market_forecast, enhanced_risk
    
    def get_market_stress_indicator(self, x: torch.Tensor) -> torch.Tensor:
        """Compute market stress indicator based on multiple factors."""
        with torch.no_grad():
            forecast, risk_boundary = self.forward(x)
            
            # Volatility component
            volatility = self.volatility_detector(forecast)
            
            # Regime uncertainty (entropy of regime probabilities)
            regime_probs = self.regime_detector(forecast)
            regime_entropy = -(regime_probs * torch.log(regime_probs + 1e-8)).sum(dim=-1, keepdim=True)
            regime_uncertainty = regime_entropy / np.log(3)  # Normalize by max entropy
            
            # Combine components
            stress_indicator = 0.4 * volatility + 0.3 * risk_boundary + 0.3 * regime_uncertainty
            
            return torch.clamp(stress_indicator, 0, 1)


# Factory function for creating Market Observer instances
def create_market_observer(
    config: dict,
    enhanced: bool = False
) -> MarketObserver:
    """
    Factory function to create Market Observer instances.
    
    Args:
        config: Configuration dictionary with required parameters
        enhanced: Whether to create enhanced version with additional analysis
        
    Returns:
        MarketObserver instance
    """
    required_keys = ['window', 'window_key', 'units_count', 'heads', 'layers', 'forecast']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    
    observer_class = EnhancedMarketObserver if enhanced else MarketObserver
    
    return observer_class(
        window=config['window'],
        window_key=config['window_key'],
        units_count=config['units_count'],
        heads=config['heads'],
        layers=config['layers'],
        forecast=config['forecast'],
        batch_size=config.get('batch_size', 32)
    )