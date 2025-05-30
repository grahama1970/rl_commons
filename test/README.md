# RL Commons Test Suite

This directory contains all tests for the Graham RL Commons library. The test structure mirrors the source code structure for easy navigation.

## 📁 Test Structure

```
test/
├── algorithms/         # Algorithm-specific tests
│   ├── a3c/           # A3C algorithm tests
│   ├── bandits/       # Contextual bandit tests
│   ├── dqn/           # DQN algorithm tests
│   ├── hierarchical/  # Hierarchical RL tests
│   └── ppo/           # PPO algorithm tests
├── core/              # Core component tests
├── integrations/      # Integration tests
├── monitoring/        # Monitoring and tracking tests
├── utils/             # Utility function tests
└── cli/               # CLI tests
```

## 🚀 Running Tests

### Run All Tests
```bash
# From project root
python -m pytest test/

# With coverage
python -m pytest test/ --cov=graham_rl_commons --cov-report=html

# Verbose output
python -m pytest test/ -v
```

### Run Specific Test Categories

#### Algorithm Tests
```bash
# All algorithm tests
python -m pytest test/algorithms/

# Specific algorithm
python -m pytest test/algorithms/ppo/
python -m pytest test/algorithms/a3c/
```

#### Integration Tests
```bash
python -m pytest test/integrations/
```

### Run Individual Test Files
```bash
# PPO tests
python -m pytest test/algorithms/ppo/test_ppo.py

# A3C tests
python -m pytest test/algorithms/a3c/test_a3c.py

# ArangoDB integration
python -m pytest test/integrations/test_arangodb_optimizer.py
```

## 🧪 Quick Validation Scripts

For quick validation without pytest, run the standalone test scripts:

```bash
# Validate PPO implementation
python test/algorithms/ppo/test_ppo.py

# Validate A3C implementation (simple test without multiprocessing)
python test/algorithms/a3c/test_a3c_simple.py

# Run comprehensive validation
python scripts/run_comprehensive_tests.py
```

## 📊 Test Coverage

To generate a detailed coverage report:

```bash
# Generate HTML coverage report
python -m pytest test/ --cov=graham_rl_commons --cov-report=html

# View coverage in terminal
python -m pytest test/ --cov=graham_rl_commons --cov-report=term-missing
```

Coverage reports will be generated in `htmlcov/` directory.

## ✅ Pre-Push Checklist

Before pushing changes, ensure:

1. **All tests pass**:
   ```bash
   python -m pytest test/
   ```

2. **No import errors**:
   ```bash
   python -c "import graham_rl_commons; print('✅ Import successful')"
   ```

3. **Examples still work**:
   ```bash
   python examples/arangodb_integration.py
   python examples/sparta_a3c_integration.py
   ```

4. **Validation passes**:
   ```bash
   python scripts/run_final_validation.py
   ```

## 🔧 Writing New Tests

When adding new features, ensure:

1. **Test location mirrors source location**:
   - Source: `src/graham_rl_commons/algorithms/new_algo/`
   - Tests: `test/algorithms/new_algo/`

2. **Test naming convention**:
   - Test files: `test_*.py`
   - Test functions: `test_*`
   - Test classes: `Test*`

3. **Include both unit and integration tests**:
   - Unit tests: Test individual components
   - Integration tests: Test component interactions

Example test structure:
```python
import pytest
from graham_rl_commons.algorithms.new_algo import NewAlgorithm

class TestNewAlgorithm:
    def test_initialization(self):
        """Test algorithm initialization"""
        algo = NewAlgorithm()
        assert algo is not None
    
    def test_action_selection(self):
        """Test action selection"""
        # Test implementation
        pass

def test_integration():
    """Test algorithm in realistic scenario"""
    # Integration test
    pass
```

## 🐛 Debugging Failed Tests

If tests fail:

1. **Run with verbose output**:
   ```bash
   python -m pytest test/path/to/test.py -v -s
   ```

2. **Run specific test**:
   ```bash
   python -m pytest test/path/to/test.py::TestClass::test_method
   ```

3. **Check logs**:
   ```bash
   tail -f logs/*.log
   ```

## 📝 Test Maintenance

- Keep tests up to date with source changes
- Remove obsolete tests
- Archive old tests in `archive/` if needed for reference
- Update this README when test structure changes