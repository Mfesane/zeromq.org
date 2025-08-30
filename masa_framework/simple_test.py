"""
Simple test for MASA Framework structure
Tests the basic class definitions and imports without requiring external dependencies.
"""

def test_structure():
    """Test that the basic structure is correct."""
    print("Testing MASA Framework Structure...")
    
    # Test file existence
    import os
    required_files = [
        'base_neural.py',
        'market_observer.py', 
        'rl_agent.py',
        'controller_agent.py',
        'masa_system.py',
        '__init__.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files present")
    
    # Test basic syntax
    try:
        import ast
        for file in required_files:
            with open(file, 'r') as f:
                content = f.read()
                ast.parse(content)
        print("✅ All files have valid Python syntax")
    except SyntaxError as e:
        print(f"❌ Syntax error in {file}: {e}")
        return False
    
    # Test class definitions
    try:
        exec(open('base_neural.py').read())
        print("✅ Base neural classes defined correctly")
    except Exception as e:
        print(f"❌ Error in base_neural.py: {e}")
        return False
    
    return True

def test_documentation():
    """Test that documentation is complete."""
    print("\nTesting Documentation...")
    
    # Check README exists
    if os.path.exists('README.md'):
        print("✅ README.md exists")
    else:
        print("❌ README.md missing")
        return False
    
    # Check requirements.txt
    if os.path.exists('requirements.txt'):
        print("✅ requirements.txt exists")
    else:
        print("❌ requirements.txt missing")
        return False
    
    # Check example files
    if os.path.exists('example_usage.py'):
        print("✅ example_usage.py exists")
    else:
        print("❌ example_usage.py missing")
        return False
    
    return True

def main():
    """Run all tests."""
    print("MASA Framework Structure Test")
    print("=" * 40)
    
    tests = [
        test_structure,
        test_documentation
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test crashed: {e}")
            results.append(False)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\n{'='*40}")
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All structure tests passed!")
        print("\nMASA Framework is properly structured.")
        print("To run full tests, install dependencies with:")
        print("  pip install -r requirements.txt")
        print("  python test_masa.py")
    else:
        print("⚠️ Some tests failed.")
    
    return passed == total

if __name__ == "__main__":
    import os
    os.chdir('/workspace/masa_framework')
    success = main()
    exit(0 if success else 1)