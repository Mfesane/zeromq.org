"""
MASA (Multi-Agent Self-Adaptive) Framework for Neural Networks in Trading

A comprehensive implementation of the multi-agent reinforcement learning framework
for dynamic portfolio risk management under volatile market conditions.

Main Components:
- MarketObserver: Analyzes market trends and provides forecasts
- RLAgent: Optimizes portfolio returns using reinforcement learning
- ControllerAgent: Manages risk and adjusts portfolio weights
- MASAFramework: Integrates all agents into a complete trading system
"""

from .base_neural import (
    BaseNeuralLayer,
    TransposeLayer,
    PiecewiseLinearRepresentation,
    RelativePositionalEncoding,
    RelativeSelfAttention,
    RelativeCrossAttention,
    ResidualConvBlock,
    SAMOptimizedLinear,
    PSformerBlock,
    MASAOptimizer,
    initialize_weights,
    entropy_regularization
)

from .market_observer import (
    MarketObserver,
    EnhancedMarketObserver,
    create_market_observer
)

from .rl_agent import (
    RLAgent,
    TD3RLAgent,
    create_rl_agent
)

from .controller_agent import (
    ControllerAgent,
    AdvancedControllerAgent,
    create_controller_agent
)

from .masa_system import (
    MASAConfig,
    MASAFramework,
    MASATradingEnvironment,
    create_masa_system,
    load_market_data,
    prepare_masa_input,
    STRESS_SCENARIOS
)

__version__ = "1.0.0"
__author__ = "MASA Framework Implementation"

__all__ = [
    # Base components
    'BaseNeuralLayer',
    'TransposeLayer', 
    'PiecewiseLinearRepresentation',
    'RelativePositionalEncoding',
    'RelativeSelfAttention',
    'RelativeCrossAttention',
    'ResidualConvBlock',
    'SAMOptimizedLinear',
    'PSformerBlock',
    'MASAOptimizer',
    'initialize_weights',
    'entropy_regularization',
    
    # Agents
    'MarketObserver',
    'EnhancedMarketObserver',
    'RLAgent',
    'TD3RLAgent',
    'ControllerAgent',
    'AdvancedControllerAgent',
    
    # Main system
    'MASAConfig',
    'MASAFramework',
    'MASATradingEnvironment',
    
    # Factory functions
    'create_market_observer',
    'create_rl_agent',
    'create_controller_agent',
    'create_masa_system',
    
    # Utilities
    'load_market_data',
    'prepare_masa_input',
    'STRESS_SCENARIOS'
]