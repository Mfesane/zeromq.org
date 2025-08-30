# MASA Framework: Multi-Agent Self-Adaptive Neural Networks for Trading

A comprehensive PyTorch implementation of the Multi-Agent Self-Adaptive (MASA) framework for dynamic portfolio risk management, based on the research paper "Developing A Multi-Agent and Self-Adaptive Framework with Deep Reinforcement Learning for Dynamic Portfolio Risk Management".

## 🎯 Overview

The MASA framework addresses the limitations of traditional reinforcement learning approaches in portfolio management by integrating three intelligent agents that work together to balance return optimization and risk management:

1. **Market Observer Agent**: Analyzes market trends using attention mechanisms with relative positional encoding
2. **RL Agent**: Optimizes portfolio returns using PSformer architecture and SAM optimization  
3. **Controller Agent**: Manages risk and adjusts portfolio weights using Transformer decoder architecture

## 🏗️ Architecture

### Market Observer Agent
- **Input**: Multimodal time series (OHLCV data)
- **Processing**: Piecewise Linear Representation → Attention Analysis → MLP Forecasting
- **Output**: Market forecasts and risk boundaries
- **Key Features**: 
  - Trend identification and forecasting
  - Risk boundary computation
  - Market regime detection

### RL Agent  
- **Algorithm**: TD3 (Twin Delayed Deep Deterministic) with PSformer
- **Input**: Market state data
- **Processing**: PSformer blocks → Convolutional layers → Decision network
- **Output**: Portfolio weight recommendations
- **Key Features**:
  - SAM (Sharpness-Aware Minimization) optimization
  - Experience replay buffer
  - Target networks for stable training

### Controller Agent
- **Architecture**: Transformer decoder with dual input streams
- **Input**: RL agent actions + Market Observer forecasts
- **Processing**: Self-attention → Cross-attention → Feed-forward
- **Output**: Risk-adjusted portfolio weights
- **Key Features**:
  - Risk assessment and adjustment
  - Position size constraints
  - Stress testing capabilities

## 🚀 Installation

```bash
# Clone or download the framework
cd masa_framework

# Install dependencies
pip install -r requirements.txt
```

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy 1.21+
- Pandas 1.3+
- Matplotlib 3.5+
- Seaborn 0.11+
- Scikit-learn 1.0+

## 🎮 Quick Start

```python
from masa_framework import MASAConfig, create_masa_system, prepare_masa_input
import torch
import pandas as pd

# 1. Create configuration
config = MASAConfig(
    mo_window=5,        # OHLCV features
    mo_units_count=30,  # Historical depth
    rl_n_actions=5,     # Number of assets
    batch_size=32
)

# 2. Initialize MASA system
masa_system = create_masa_system(config, enhanced=True)

# 3. Prepare market data
# market_data should be a DataFrame with OHLCV columns
market_tensor, returns_tensor = prepare_masa_input(market_data)

# 4. Get portfolio allocation
allocation = masa_system.get_portfolio_allocation(
    market_tensor[-1:], 
    asset_names=['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
)

print("Portfolio Allocation:", allocation['allocation'])
print("Risk Score:", allocation['risk_score'])
print("Market Regime:", allocation['market_regime'])
```

## 📊 Training Example

```python
# Backtest and train the system
results = masa_system.backtest(
    market_tensor,
    returns_tensor,
    train_ratio=0.7,
    initial_value=100000.0
)

# View performance metrics
metrics = results['test_results']['metrics']
print(f"Total Return: {metrics['total_return']:.2%}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
```

## 🔧 Configuration Options

### MASAConfig Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `mo_window` | Market features per time step | 5 |
| `mo_units_count` | Historical lookback period | 50 |
| `mo_forecast` | Forecast horizon | 10 |
| `rl_n_actions` | Number of portfolio assets | 10 |
| `rl_layers` | PSformer layers in RL agent | 3 |
| `ctrl_layers` | Decoder layers in controller | 2 |
| `batch_size` | Training batch size | 32 |
| `risk_tolerance` | User risk tolerance (0-1) | 0.5 |

## 🧪 Advanced Features

### Stress Testing
```python
# Perform stress testing
if hasattr(masa_system.controller, 'stress_test_portfolio'):
    stress_results = masa_system.controller.stress_test_portfolio(
        rl_actions, market_forecast, STRESS_SCENARIOS
    )
```

### Model Persistence
```python
# Save trained model
masa_system.save_system('./saved_models/masa_trained')

# Load model
new_system = create_masa_system(config)
new_system.load_system('./saved_models/masa_trained')
```

### Custom Risk Management
```python
# Get risk-adjusted weights with custom tolerance
weights = masa_system.controller.get_risk_adjusted_weights(
    rl_actions, market_forecast, risk_boundary, 
    risk_tolerance=0.3  # Conservative
)
```

## 📈 Performance Metrics

The framework provides comprehensive performance analysis:

- **Return Metrics**: Total return, annualized return, Sharpe ratio
- **Risk Metrics**: Maximum drawdown, volatility, VaR, CVaR  
- **Efficiency Metrics**: Calmar ratio, Sortino ratio, win rate
- **Portfolio Metrics**: Diversification ratio, turnover, concentration

## 🎛️ Customization

### Custom Market Observer
```python
from masa_framework.market_observer import MarketObserver

class CustomMarketObserver(MarketObserver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add custom components
        self.sentiment_analyzer = SentimentAnalyzer()
        
    def forward(self, x):
        forecast, risk = super().forward(x)
        # Add sentiment analysis
        sentiment_score = self.sentiment_analyzer(x)
        enhanced_risk = risk + 0.1 * sentiment_score
        return forecast, enhanced_risk
```

### Custom RL Agent
```python
from masa_framework.rl_agent import RLAgent

class CustomRLAgent(RLAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add custom reward function
        
    def compute_custom_reward(self, returns, actions):
        # Implement custom reward logic
        return reward
```

## 🔬 Research Applications

The MASA framework is designed for research and can be extended for:

- **Alternative Data Integration**: News sentiment, social media, economic indicators
- **Multi-Asset Classes**: Stocks, bonds, commodities, cryptocurrencies
- **High-Frequency Trading**: Intraday portfolio rebalancing
- **ESG Integration**: Environmental, social, governance factors
- **Regime-Aware Strategies**: Bull/bear/sideways market adaptations

## ⚠️ Important Disclaimers

1. **Research Purpose**: This implementation is for research and educational purposes
2. **Not Financial Advice**: Results do not constitute investment recommendations
3. **Backtesting Limitations**: Past performance does not guarantee future results
4. **Risk Management**: Always implement proper risk controls in live trading
5. **Regulatory Compliance**: Ensure compliance with applicable financial regulations

## 📚 References

- Original MASA paper: "Developing A Multi-Agent and Self-Adaptive Framework with Deep Reinforcement Learning for Dynamic Portfolio Risk Management"
- TD3 Algorithm: "Addressing Function Approximation Error in Actor-Critic Methods"
- PSformer: "PSformer: Point-wise 3D Scene Parsing Transformer"
- SAM Optimization: "Sharpness-Aware Minimization for Efficiently Improving Generalization"

## 🤝 Contributing

Contributions are welcome! Please consider:

1. **Code Quality**: Follow PEP 8 style guidelines
2. **Documentation**: Add docstrings and comments
3. **Testing**: Include unit tests for new features
4. **Performance**: Optimize for computational efficiency
5. **Research**: Cite relevant academic sources

## 📄 License

This implementation is provided under the MIT License. See LICENSE file for details.

## 🆘 Support

For questions, issues, or contributions:

1. Check the demo notebook for usage examples
2. Review the code documentation and comments
3. Test with synthetic data before real market data
4. Start with conservative risk tolerance settings

---

**Note**: This is a research implementation. Always validate thoroughly before any real-world application and consider consulting with financial professionals for investment decisions.