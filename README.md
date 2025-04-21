# Planwise Recommendation System

[![Python](https://img.shields.io/badge/Python-FFD700?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Supabase](https://img.shields.io/badge/Supabase-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white)](https://supabase.io/)
[![Documentation](https://img.shields.io/badge/Documentation-GitHub%20Pages-blue.svg)](https://planwise-org.github.io/recommendation-system/)

## 🌟 Overview

Planwise is an AI-powered recommendation platform that helps users discover personalized plans in Madrid based on their preferences and location. Our system combines multiple advanced machine learning techniques to deliver recommendations that are accurate, diverse, and tailored to each individual user.

![Planwise System](docs/img/demo.gif)

## 🚀 Features

- **Personalized Recommendations**: Get place suggestions that match your unique interests
- **Location Awareness**: Discover places within comfortable travel distance
- **Multi-model Architecture**: Leverages ensemble learning for superior recommendation quality
- **Category Diversity**: Balanced recommendations across different categories
- **Seamless Integration**: API-driven architecture with multiple frontends

## 📋 Project Structure

The project consists of two main components:

```
recommendation-system/
├── api/                # FastAPI backend service
├── reco/               # Core recommendation engine
│   ├── planwise/       # Model implementation and streamlit app
│   └── tests/          # Test suite
└── docs/               # Project documentation
```

## 🛠️ Getting Started

To run Planwise locally, you'll need to start two separate components:

### 1. API Service

The API provides recommendation endpoints and database access.

```bash
# Navigate to the API directory
cd api

# Set up using Docker
docker compose -f docker-compose.dev.yml up -d --build

# API will be available at http://localhost:8080
```

For detailed API setup instructions, see the [API README](api/README.md).

### 2. Streamlit Application

The Streamlit app provides a user-friendly interface to interact with the system.

```bash
# Navigate to the planwise directory
cd reco/planwise

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

# Streamlit interface will be available at http://localhost:8501
```

## 🧠 How It Works

Planwise uses a hybrid ensemble of recommendation models:

1. **Autoencoder Recommender**: Deep learning model that reconstructs user preferences
2. **SVD Recommender**: Collaborative filtering using matrix factorization
3. **Transfer Learning Recommender**: Leverages patterns from movie domain to place recommendations
4. **Madrid Embeddings Recommender**: Location-specific semantic understanding
5. **Ensemble Recommender**: Meta-model that combines all approaches for optimal results

## 📚 Documentation

For comprehensive documentation, please visit our [GitHub Pages documentation site](https://planwise-org.github.io/recommendation-system/).

Key documentation sections:
- [System Architecture](https://planwise-org.github.io/recommendation-system/overview/architecture)
- [Model Documentation](https://planwise-org.github.io/recommendation-system/models/ensemble)
- [API Reference](https://planwise-org.github.io/recommendation-system/api/overview)
- [Development Guide](https://planwise-org.github.io/recommendation-system/development/environment-setup)

## 🤝 Contributing

Contributions are welcome! Please read our [contributing guidelines](docs/development/contribution-guide.md) before submitting pull requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
