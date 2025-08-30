"""
Example Usage of MASA Framework
Demonstrates basic usage of the Multi-Agent Self-Adaptive framework for portfolio management.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Import MASA framework
from masa_framework import (
    MASAConfig, create_masa_system, prepare_masa_input, STRESS_SCENARIOS
)


def generate_sample_market_data(n_days=100, n_assets=5):
    """Generate sample market data for demonstration."""
    np.random.seed(42)
    
    # Create date range
    dates = pd.date_range(start='2024-01-01', periods=n_days, freq='D')
    
    # Generate price data
    initial_prices = np.array([100, 150, 80, 200, 120])  # Different starting prices
    prices = [initial_prices]
    
    for day in range(1, n_days):
        # Random walk with drift
        returns = np.random.normal(0.001, 0.02, n_assets)  # Small positive drift, 2% daily vol
        new_prices = prices[-1] * (1 + returns)
        prices.append(new_prices)
    
    prices = np.array(prices)
    
    # Create OHLCV data
    data = []
    for day in range(n_days):
        day_data = {'Date': dates[day]}
        
        for i in range(n_assets):
            asset_name = f'Asset_{i+1}'
            
            # Generate OHLC from closing price
            close = prices[day, i]
            open_price = prices[day-1, i] if day > 0 else close
            
            # Random intraday movements
            daily_range = close * 0.02  # 2% daily range
            high = close + np.random.uniform(0, daily_range)
            low = close - np.random.uniform(0, daily_range)
            
            # Ensure OHLC consistency
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            volume = np.random.uniform(100000, 1000000)
            
            day_data.update({
                f'{asset_name}_Open': open_price,
                f'{asset_name}_High': high,
                f'{asset_name}_Low': low,
                f'{asset_name}_Close': close,
                f'{asset_name}_Volume': volume
            })
        
        data.append(day_data)
    
    return pd.DataFrame(data)


def main():
    """Main demonstration function."""
    print("MASA Framework Example Usage")
    print("=" * 40)
    
    # 1. Generate sample data
    print("\n1. Generating sample market data...")
    market_data = generate_sample_market_data(n_days=100, n_assets=5)
    print(f"Generated {len(market_data)} days of data for 5 assets")
    
    # 2. Create MASA configuration
    print("\n2. Creating MASA configuration...")
    config = MASAConfig(
        # Market Observer
        mo_window=5,
        mo_units_count=20,
        mo_forecast=5,
        
        # RL Agent  
        rl_window=5,
        rl_units_count=20,
        rl_n_actions=5,
        
        # Controller
        ctrl_window=5,
        ctrl_units_count=1,
        ctrl_window_kv=5,
        ctrl_units_kv=20,
        
        # Training
        batch_size=16,
        risk_tolerance=0.6
    )
    print("Configuration created successfully")
    
    # 3. Initialize MASA system
    print("\n3. Initializing MASA system...")
    masa_system = create_masa_system(config, enhanced=True)
    print("MASA system initialized")
    
    # 4. Prepare data
    print("\n4. Preparing market data...")
    market_tensor, returns_tensor = prepare_masa_input(market_data, lookback=20)
    print(f"Market tensor shape: {market_tensor.shape}")
    print(f"Returns tensor shape: {returns_tensor.shape}")
    
    # 5. Test individual agents
    print("\n5. Testing individual agents...")
    sample_input = market_tensor[:1]
    
    # Market Observer
    market_forecast, risk_boundary = masa_system.market_observer(sample_input)
    print(f"Market Observer - Forecast shape: {market_forecast.shape}, Risk: {risk_boundary.item():.3f}")
    
    # RL Agent
    rl_actions, state_value = masa_system.rl_agent(sample_input)
    print(f"RL Agent - Actions: {rl_actions.squeeze().cpu().numpy()}")
    print(f"RL Agent - State value: {state_value.item():.3f}")
    
    # Controller
    final_actions, risk_assessment = masa_system.controller(rl_actions, market_forecast, risk_boundary)
    print(f"Controller - Final actions: {final_actions.squeeze().cpu().numpy()}")
    print(f"Controller - Risk assessment: {risk_assessment.item():.3f}")
    
    # 6. Complete system test
    print("\n6. Testing complete MASA system...")
    outputs = masa_system.forward(sample_input, training=False)
    
    asset_names = [f'Asset_{i+1}' for i in range(5)]
    allocation = masa_system.get_portfolio_allocation(sample_input, asset_names)
    
    print("\nPortfolio Allocation Recommendation:")
    print(f"Risk Score: {allocation['risk_score']:.3f}")
    print(f"Market Regime: {allocation['market_regime']}")
    print(f"Recommendation: {allocation['recommendation']}")
    print("\nAsset Weights:")
    for asset, weight in allocation['allocation'].items():
        print(f"  {asset}: {weight:.3f} ({weight*100:.1f}%)")
    
    # 7. Training demonstration
    print("\n7. Training demonstration...")
    
    # Split data for training
    train_size = int(len(market_tensor) * 0.8)
    train_data = market_tensor[:train_size]
    train_returns = returns_tensor[:train_size]
    
    print(f"Training on {len(train_data)} samples...")
    
    # Train for a few episodes
    for episode in range(3):
        episode_metrics = masa_system.train_episode(train_data, train_returns, episode_length=20)
        print(f"Episode {episode+1}: Return = {episode_metrics['episode_return']:.4f}")
    
    # 8. Evaluation
    print("\n8. Evaluating performance...")
    test_data = market_tensor[train_size:]
    test_returns = returns_tensor[train_size:]
    
    if len(test_data) > 0:
        test_results = masa_system.evaluate(test_data, test_returns)
        metrics = test_results['metrics']
        
        print("\nPerformance Metrics:")
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Win Rate: {metrics['win_rate']:.1%}")
    
    # 9. Save system
    print("\n9. Saving trained system...")
    masa_system.save_system('./saved_models/example_masa')
    print("System saved successfully")
    
    print("\n✅ MASA Framework example completed successfully!")
    print("\nNext steps:")
    print("- Try with real market data")
    print("- Experiment with different configurations") 
    print("- Implement custom agents or strategies")
    print("- Integrate with trading platforms")


if __name__ == "__main__":
    main()