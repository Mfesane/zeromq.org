"""
Test script for MASA Framework
Verifies that all components work correctly.
"""

import torch
import numpy as np
import pandas as pd
import sys
import traceback

def test_imports():
    """Test that all imports work correctly."""
    try:
        from masa_framework import (
            MASAConfig, MASAFramework, create_masa_system,
            MarketObserver, RLAgent, ControllerAgent
        )
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic functionality of MASA components."""
    try:
        from masa_framework import MASAConfig, create_masa_system
        
        # Create minimal config
        config = MASAConfig(
            mo_window=5,
            mo_units_count=10,
            mo_forecast=3,
            rl_window=5,
            rl_units_count=10,
            rl_n_actions=3,
            ctrl_window=3,
            ctrl_units_count=1,
            ctrl_window_kv=5,
            ctrl_units_kv=10,
            batch_size=2
        )
        
        # Create system
        masa_system = create_masa_system(config, enhanced=False)
        
        # Test with dummy data
        dummy_input = torch.randn(1, 5 * 10)  # batch_size=1, features=5*10
        
        # Test forward pass
        outputs = masa_system.forward(dummy_input, training=False)
        
        print("✅ Basic functionality test passed")
        print(f"   Output keys: {list(outputs.keys())}")
        print(f"   Final actions shape: {outputs['final_actions'].shape}")
        print(f"   Final actions sum: {outputs['final_actions'].sum().item():.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_individual_agents():
    """Test each agent individually."""
    try:
        from masa_framework.market_observer import MarketObserver
        from masa_framework.rl_agent import RLAgent  
        from masa_framework.controller_agent import ControllerAgent
        
        # Test Market Observer
        mo = MarketObserver(
            window=5, window_key=16, units_count=10, 
            heads=2, layers=1, forecast=3, batch_size=2
        )
        mo_input = torch.randn(2, 5 * 10)
        mo_forecast, mo_risk = mo(mo_input)
        print(f"✅ Market Observer: forecast {mo_forecast.shape}, risk {mo_risk.shape}")
        
        # Test RL Agent
        rl = RLAgent(
            window=5, units_count=10, segments=5, rho=0.5,
            layers=1, n_actions=3, batch_size=2
        )
        rl_actions, rl_value = rl(mo_input)
        print(f"✅ RL Agent: actions {rl_actions.shape}, value {rl_value.shape}")
        
        # Test Controller
        ctrl = ControllerAgent(
            window=3, window_key=16, units_count=1, heads=2,
            window_kv=5, units_kv=10, layers=1, batch_size=2
        )
        ctrl_actions, ctrl_risk = ctrl(rl_actions, mo_forecast, mo_risk)
        print(f"✅ Controller: actions {ctrl_actions.shape}, risk {ctrl_risk.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Individual agents test failed: {e}")
        traceback.print_exc()
        return False

def test_training():
    """Test training functionality."""
    try:
        from masa_framework import MASAConfig, create_masa_system
        
        config = MASAConfig(
            mo_window=5, mo_units_count=10, mo_forecast=3,
            rl_window=5, rl_units_count=10, rl_n_actions=3,
            ctrl_window=3, ctrl_units_count=1, ctrl_window_kv=5, ctrl_units_kv=10,
            batch_size=4
        )
        
        masa_system = create_masa_system(config, enhanced=False)
        
        # Generate dummy training data
        n_samples = 20
        dummy_market_data = torch.randn(n_samples, 5 * 10)
        dummy_returns = torch.randn(n_samples, 3) * 0.01  # Small returns
        
        # Test one training episode
        episode_metrics = masa_system.train_episode(
            dummy_market_data, dummy_returns, episode_length=10
        )
        
        print("✅ Training test passed")
        print(f"   Episode return: {episode_metrics['episode_return']:.4f}")
        print(f"   Average reward: {episode_metrics['average_reward']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests and report results."""
    print("MASA Framework Test Suite")
    print("=" * 40)
    
    tests = [
        ("Import Test", test_imports),
        ("Basic Functionality Test", test_basic_functionality),
        ("Individual Agents Test", test_individual_agents),
        ("Training Test", test_training)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\nRunning {test_name}...")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 40)
    print("Test Results Summary:")
    passed = sum(results)
    total = len(results)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASSED" if results[i] else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! MASA framework is working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the error messages above.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)