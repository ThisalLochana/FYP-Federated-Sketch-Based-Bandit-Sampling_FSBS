## 2. Test Cases README

Create `D:\IIT\4th Year\FYP\Implementation_1\fsbs-demo\fsbs-platform\sidecar\tests\README.md`:

```markdown
# FSBS Test Suite

52 unit, integration, and performance tests covering every component
of the FSBS sidecar architecture.

## Running Tests

```powershell
cd D:\IIT\4th Year\FYP\Implementation_1\fsbs-demo\fsbs-platform\sidecar

# Create and activate virtual environment (first time only)
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# Run all tests with verbose output
python -m pytest tests/test_components.py -v

# Run all tests with performance numbers printed
python -m pytest tests/test_components.py -v -s

# Run a specific test class
python -m pytest tests/test_components.py::TestCountMinSketch -v

# Run a single test
python -m pytest tests/test_components.py::TestLinUCB::test_ucb_score_positive -v

# Expected result: 52 passed, 0 warnings