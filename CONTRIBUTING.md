# Contributing to X-FuzzyScore Credit Risk Analysis Framework

Thank you for your interest in contributing to the Explainable Fuzzy Credit-Risk Prediction (X-FuzzyScore) project! This document provides guidelines for contributing to this research project.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Project Structure](#project-structure)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Submission Process](#submission-process)

## Code of Conduct

This project follows standard academic and open-source collaboration principles:

- Be respectful and inclusive
- Provide constructive feedback
- Credit others' work appropriately
- Follow ethical guidelines for AI/ML research

## Getting Started

### Prerequisites

- Python 3.8+
- Docker (for dev container)
- Git and GitHub account
- Familiarity with ML, fuzzy logic, or explainable AI

### Development Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/Credit-Risk-Analysis-and-Prediction-Framework.git
   cd Credit-Risk-Analysis-and-Prediction-Framework
   ```

2. **Open in dev container** (recommended)

   - Use VS Code with Remote-Containers extension
   - The container includes all required tools and dependencies

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   pytest tests/
   ```

## How to Contribute

### Types of Contributions

We welcome contributions in the following areas:

#### 1. **Data Engineering**

- Dataset preprocessing and cleaning
- Feature engineering scripts
- Data integration pipelines (German Credit, Taiwan Credit, LendingClub)

#### 2. **Model Development**

- Fuzzy logic implementation using `scikit-fuzzy`
- ML ensemble models (XGBoost, LightGBM)
- Hyperparameter tuning

#### 3. **Explainability Features**

- SHAP integration and analysis
- LIME implementations
- Custom explanation visualizations

#### 4. **Frontend/Visualization**

- Streamlit/Dash dashboard components
- Interactive plots with Plotly
- UI/UX improvements

#### 5. **Documentation**

- Code documentation and docstrings
- Tutorial notebooks
- Research paper contributions

#### 6. **Testing & Evaluation**

- Unit tests for core functionality
- Performance benchmarks
- Interpretability assessments

## Project Structure

```
Credit-Risk-Analysis-and-Prediction-Framework/
├── data/                  # Raw and processed datasets
│   ├── raw/
│   ├── processed/
│   └── scripts/          # Data preprocessing scripts
├── models/               # Trained models and checkpoints
├── src/                  # Source code
│   ├── fuzzy/           # Fuzzy logic components
│   ├── ml/              # Machine learning models
│   ├── explainability/  # SHAP/LIME implementations
│   └── visualization/   # Dashboard and plotting
├── notebooks/           # Jupyter notebooks for exploration
├── tests/               # Unit and integration tests
├── docs/                # Documentation and paper drafts
└── requirements.txt     # Python dependencies
```

## Coding Standards

### Python Style Guide

- Follow **PEP 8** style guidelines
- Use **type hints** for function signatures
- Maximum line length: **88 characters** (Black formatter)
- Use meaningful variable names (avoid single letters except for loops)

### Documentation

- Add docstrings to all functions, classes, and modules
- Use **Google-style** docstrings format:

```python
def calculate_fuzzy_risk(income: float, debt: float) -> dict:
    """Calculate fuzzy risk score based on income and debt.

    Args:
        income: Monthly income in normalized range [0, 1]
        debt: Debt ratio in normalized range [0, 1]

    Returns:
        Dictionary containing risk score and activated rules

    Example:
        >>> calculate_fuzzy_risk(0.8, 0.2)
        {'risk_score': 0.15, 'rules': [...]}
    """
    # ...existing code...
```

### Git Commit Messages

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit first line to 72 characters
- Reference issues and pull requests when applicable

**Example:**

```
Add SHAP waterfall plot to dashboard

- Implement waterfall visualization for individual predictions
- Add interactive feature to select samples
- Update dashboard layout

Closes #42
```

## Testing Guidelines

### Writing Tests

- Place tests in the `tests/` directory
- Use `pytest` framework
- Aim for >80% code coverage
- Test edge cases and error handling

**Example test:**

```python
# tests/test_fuzzy_logic.py
def test_fuzzy_membership_bounds():
    """Test that fuzzy membership values are in [0, 1]."""
    result = fuzzy_membership(income=0.5)
    assert 0 <= result <= 1
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_fuzzy_logic.py
```

## Submission Process

### Before Submitting

1. **Update your branch**

   ```bash
   git checkout main
   git pull origin main
   git checkout your-branch
   git rebase main
   ```

2. **Run tests and linting**

   ```bash
   pytest
   black src/ tests/
   flake8 src/ tests/
   ```

3. **Update documentation** if needed

### Pull Request Process

1. **Create a feature branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the guidelines above

3. **Commit your changes**

   ```bash
   git add .
   git commit -m "Descriptive commit message"
   ```

4. **Push to your fork**

   ```bash
   git push origin feature/your-feature-name
   ```

5. **Open a Pull Request** on GitHub

   - Fill out the PR template
   - Link related issues
   - Request review from relevant team members

6. **Address review feedback**
   - Make requested changes
   - Push updates to the same branch
   - Re-request review when ready

### PR Review Criteria

- [ ] Code follows style guidelines
- [ ] Tests pass and coverage is maintained
- [ ] Documentation is updated
- [ ] No merge conflicts
- [ ] Commits are clean and well-described

## Team Roles & Contact

For specific questions, reach out to:

- **Data Engineering**: [Contact/Channel]
- **ML Development**: [Contact/Channel]
- **Explainability**: [Contact/Channel]
- **Frontend/Visualization**: [Contact/Channel]
- **Research/Paper**: [Contact/Channel]

## Resources

- **Project README**: See [README.md](README.md) for project overview
- **Documentation**: See [docs/](docs/) for detailed documentation
- **Research Paper**: See [docs/paper/](docs/paper/) for paper drafts
- **Issues**: Report bugs and request features on GitHub Issues
- **Discussions**: Join project discussions on GitHub Discussions

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (see [LICENSE](LICENSE) file).

---

**Thank you for contributing to X-FuzzyScore! 🎉**
