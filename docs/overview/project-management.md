# Project Management & Sprint Planning

## Agile Approach

From day one, Planwise embraced an agile, sprint-driven workflow to keep our team focused and adaptable. This approach enabled us to incrementally deliver value while maintaining flexibility to incorporate feedback and new insights.

## Sprint Planning & Execution

### Sprint Backlog & Goals

We kicked off with a sprint planning where the Product Manager organized a group meeting to set goals and distribute tasks. High-level objectives were translated into small achievable incremental goals, allowing us to move from an MVP to the final delivery.

Our process was managed on Trello, where team members could:
- Choose tasks during each sprint
- Mark tasks as completed
- Move on to new tasks as previous ones were completed

Tasks were distributed to play to each team member's strengths and preferences across:
- Data engineering
- Model development
- Front-end development
- Back-end services

This approach enabled simultaneous progress toward our MVP and final product.

### MVP Delivery

By the end of the first sprint, we launched a functional Streamlit interface where users could:
- Adjust preference sliders
- Receive a ranked list of Madrid's top spots
- View recommendations powered by our initial autoencoder model

This early prototype validated our end-to-end pipeline and set the stage for rapid feature expansion in subsequent sprints.

## Development Workflow

### Version Control

We established a branch-based workflow with:
- Main branch reserved for stable releases
- Feature branches for all new work
- Pull requests with thorough code reviews
- Issue tracking for bugs and feature requests

### Code Quality Practices

To maintain high standards of code quality:
- Comprehensive docstrings and type hints span the entire codebase
- Pre-commit hooks enforce Black formatting, organize imports with isort, and catch linting errors via flake8
- Unit tests cover data-pipeline logic, model training and inference, and API endpoints
- Coverage reporting ensures code is adequately tested

### Deployment Pipeline

Our CI/CD pipeline automates:
- Linting and style checks
- Type safety verification
- Automated testing 
- Docker builds
- Deployment to staging and production environments

This approach ensures that every change is thoroughly vetted before reaching production, giving the team confidence in each commit. 