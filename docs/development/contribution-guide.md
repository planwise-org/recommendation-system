# Contribution Guide

## Welcome to Planwise!

Thank you for your interest in contributing to Planwise. This guide will help you understand our development process and how you can contribute to the project.

## Code of Conduct

We expect all contributors to follow our Code of Conduct. Please be respectful and constructive in all communications and interactions.

## Getting Started

1. **Fork the Repository**: Start by forking the main repository on GitHub
2. **Clone Your Fork**: `git clone https://github.com/your-username/planwise.git`
3. **Add Upstream Remote**: `git remote add upstream https://github.com/original-owner/planwise.git`
4. **Create a Branch**: `git checkout -b feature/your-feature-name`

## Development Workflow

### Issues

- Check existing issues for tasks that need attention
- If you're working on something new, create an issue to discuss it first
- Assign yourself to an issue when you start working on it
- Use labels to categorize issues (bug, enhancement, etc.)

### Branching Strategy

We follow a feature branch workflow:

- `main`: Stable production code
- `develop`: Integration branch for features
- `feature/*`: Individual feature branches
- `bugfix/*`: Bug fix branches
- `release/*`: Release preparation branches

### Pull Requests

1. **Keep PRs Focused**: Each PR should address a single issue or feature
2. **Write Descriptive Titles**: Clearly indicate what the PR does
3. **Link Related Issues**: Use GitHub keywords (e.g., "Fixes #123")
4. **Provide Details**: Explain what changes you made and why
5. **Include Tests**: Add tests for new features or bug fixes

#### PR Template

```markdown
## Description
Brief description of the changes

## Related Issue
Fixes #123

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Code refactoring
- [ ] Other (please describe):

## Testing
Describe the tests you ran to verify your changes

## Screenshots (if applicable)

## Additional Notes
```

### Code Review Process

1. All PRs require at least one review before merging
2. Address all review comments before requesting re-review
3. Maintainers will merge approved PRs

## Coding Standards

### Python Guidelines

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Use type hints for function signatures
- Write docstrings for modules, classes, and functions
- Use f-strings for string formatting
- Maximum line length: 88 characters (Black default)

### Documentation

- Keep documentation up to date with code changes
- Use Markdown for documentation files
- Add inline comments for complex logic
- Document API endpoints with clear descriptions

### Testing

- Write unit tests for new features
- Ensure all tests pass before submitting PRs
- Maintain or improve code coverage

## Setting Up Your Development Environment

See our [Environment Setup Guide](environment-setup.md) for detailed instructions.

## Making Changes to Models

When working on recommendation models:

1. Document model architecture and parameters
2. Save training metrics and evaluation results
3. Version model artifacts clearly
4. Add validation tests for model performance

## Database Changes

When making database schema changes:

1. Create an Alembic migration
2. Test migrations both up and down
3. Document changes in the migration file

## Frontend Development

For Streamlit app changes:

1. Keep UI components consistent
2. Use session state for managing user state
3. Test on different screen sizes
4. Document new UI features with screenshots

## Release Process

1. **Version Bump**: Update version numbers according to [Semantic Versioning](https://semver.org/)
2. **Changelog**: Update the CHANGELOG.md file
3. **Release Branch**: Create a release branch from develop
4. **Testing**: Perform final testing on release branch
5. **Merge**: Merge release branch to main and develop
6. **Tag**: Create a Git tag for the new version
7. **Release**: Create a GitHub release with release notes

## Where to Get Help

- GitHub Discussions: Ask questions and share ideas
- Issue Comments: Discuss specifics of an issue
- Developer Chat: Join our developer chat room
- Documentation: Refer to our comprehensive documentation

Thank you for contributing to Planwise! Your help makes this project better for everyone. 