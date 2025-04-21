# Code Quality Standards

## Overview

Maintaining high code quality is a priority for the Planwise project. This document outlines our standards and practices for ensuring code readability, maintainability, and reliability.

## Style Guides

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with a few project-specific adjustments:

- Maximum line length: 88 characters (Black default)
- Use double quotes for strings except when single quotes avoid backslashes
- Import order: standard library, third-party, local applications
- Class names: `CamelCase`
- Function and variable names: `snake_case`
- Constants: `UPPER_CASE_WITH_UNDERSCORES`

### Documentation Style

- Use [Google-style docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) for Python code
- Markdown for standalone documentation files
- Keep API endpoint documentation consistent with Swagger annotations

## Automated Tools

### Code Formatting

We use automatic formatters to maintain consistent style:

- **Black**: Python code formatter
  ```bash
  black .
  ```

- **isort**: Import sorter
  ```bash
  isort .
  ```

### Static Analysis

We employ multiple static analysis tools:

- **Flake8**: Linter for style and logical errors
  ```bash
  flake8 .
  ```

- **mypy**: Type checking
  ```bash
  mypy .
  ```

- **Bandit**: Security vulnerability scanner
  ```bash
  bandit -r .
  ```

### Pre-commit Hooks

We use pre-commit hooks to automate quality checks before committing:

```yaml
# .pre-commit-config.yaml
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
    -   id: trailing-whitespace
    -   id: end-of-file-fixer
    -   id: check-yaml
    -   id: check-added-large-files

-   repo: https://github.com/psf/black
    rev: 23.1.0
    hooks:
    -   id: black

-   repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
    -   id: isort

-   repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
    -   id: flake8
        additional_dependencies: [flake8-docstrings]

-   repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.0.1
    hooks:
    -   id: mypy
        additional_dependencies: [types-requests, types-PyYAML]
```

To install pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

## Code Documentation

### Code Comments

- Use comments to explain **why**, not **what**
- Keep comments up-to-date when changing code
- Comment complex algorithms and business logic

### Function and Class Documentation

All public APIs should have comprehensive docstrings:

```python
def calculate_recommendation_score(
    preference_score: float, 
    distance: float, 
    popularity: int
) -> float:
    """
    Calculate the final recommendation score for a place.
    
    Args:
        preference_score: User preference score (0-5)
        distance: Distance in meters from user location
        popularity: Number of user ratings
        
    Returns:
        Final recommendation score (0-1)
        
    Raises:
        ValueError: If preference_score is outside the valid range
    """
    # Implementation...
```

### Module Documentation

Each module should have a top-level docstring explaining its purpose:

```python
"""
Recommendation Engine Core
==========================

This module implements the core recommendation algorithms,
including autoencoder-based and SVD-based recommenders.
"""
```

## Testing Standards

### Unit Tests

- Every function should have at least one test
- Tests should cover normal cases, edge cases, and error cases
- Use `pytest` for test implementation
- Organize tests to mirror the source code structure

### Test Coverage

We aim for at least 80% code coverage:

```bash
pytest --cov=src --cov-report=term-missing
```

### Integration Tests

- Test API endpoints with realistic workflows
- Verify database interactions
- Test the full recommendation pipeline

## Code Review Checklist

During code reviews, check for:

1. **Functionality**: Does the code do what it claims?
2. **Security**: Are there any security vulnerabilities?
3. **Performance**: Are there any performance concerns?
4. **Error Handling**: Is error handling appropriate?
5. **Documentation**: Are docstrings and comments clear and complete?
6. **Test Coverage**: Are new changes adequately tested?
7. **Style Compliance**: Does the code follow our style guidelines?
8. **Compatibility**: Are there any backward compatibility issues?
9. **Dependencies**: Are new dependencies necessary and appropriate?
10. **Complexity**: Is the code unnecessarily complex?

## Continuous Integration

Our CI pipeline runs the following checks on every pull request:

- Run all tests
- Code coverage report
- Style checks (Black, isort)
- Linting (Flake8)
- Type checking (mypy)
- Security scanning (Bandit)
- Dependency vulnerability scanning

## Best Practices

### Error Handling

- Use specific exception types
- Provide helpful error messages
- Log exceptions with context
- Don't silence exceptions without good reason

### Performance Considerations

- Profile code for bottlenecks
- Use efficient data structures
- Consider caching expensive operations
- Be mindful of memory usage

### Security Practices

- Never commit secrets or credentials
- Validate all user inputs
- Use parameterized SQL queries
- Sanitize data before rendering
- Follow the principle of least privilege 