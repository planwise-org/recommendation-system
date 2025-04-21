# System Architecture

## Overview

Planwise employs a modular architecture that separates concerns between data processing, recommendation algorithms, API services, and user interfaces. This design enables flexibility, scalability, and maintainability.

## Architecture Components

### User Interfaces

- **Streamlit Web App**: A user-friendly interface where users can adjust preference sliders, view recommendations on a map, and provide feedback
- **FastAPI Chatbot**: Conversational interface that extracts preferences from natural language and returns recommendations

### Core Recommendation Engine

Planwise uses four different recommendation strategies, combined in an ensemble approach:

1. **Autoencoder-Based Recommender**: Learns latent patterns in user preferences
2. **SVD-Based Recommender**: Employs matrix factorization for collaborative filtering
3. **Transfer Learning Recommender**: Adapts pre-trained models to new domains
4. **Madrid Embeddings Recommender**: Uses text embeddings for cold-start recommendations

The **Ensemble Layer** intelligently combines these algorithms with weighted scoring to produce the final recommendations.

### API Layer

Built with FastAPI, our API provides:
- Authentication and user management
- Preference extraction and storage
- Place data retrieval
- Review submission and retrieval
- Recommendation generation
- API documentation with Swagger

### Data Layer

- **PostgreSQL Database**: Stores user data, preferences, reviews, and place metadata
- **Supabase**: Provides managed database services with authentication (in production)
- **Model Storage**: Versioned machine learning models and embeddings

### Deployment Infrastructure

- **Docker Containers**: Package both API and Streamlit app for consistent deployment
- **CI/CD Pipeline**: Automated testing and deployment via GitHub Actions
- **MLOps**: Model versioning, experiment tracking, and automated evaluation

## Data Flow

1. User inputs preferences via sliders or natural language
2. Preferences are processed and stored in the database
3. Recommendation engine retrieves places matching user preferences
4. Multiple recommendation models generate candidate places
5. Ensemble layer combines and ranks recommendations
6. Results are returned to the user interface
7. User feedback is collected and stored for future improvements

## System Diagram

```
┌─────────────┐       ┌─────────────────────┐       ┌───────────────┐
│  Web UI     │◄─────►│                     │◄─────►│ PostgreSQL DB │
│ (Streamlit) │       │                     │       └───────────────┘
└─────────────┘       │                     │
                      │                     │       ┌───────────────┐
┌─────────────┐       │  FastAPI Backend   │◄─────►│ ML Models     │
│  Mobile App │◄─────►│                     │       └───────────────┘
│  (Future)   │       │                     │
└─────────────┘       │                     │       ┌───────────────┐
                      │                     │◄─────►│ Place Data    │
┌─────────────┐       │                     │       └───────────────┘
│ 3rd Party   │◄─────►│                     │
│ Integrations│       └─────────────────────┘
└─────────────┘
``` 