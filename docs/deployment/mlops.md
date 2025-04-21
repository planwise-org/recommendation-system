# MLOps

## Overview

Planwise employs MLOps (Machine Learning Operations) practices to streamline the development, deployment, and maintenance of our recommendation models. This document outlines our approach to model versioning, training, evaluation, and deployment.

## Model Lifecycle Management

### Model Development

Our model development process follows these steps:

1. **Data Preparation**: Process and clean data from various sources
2. **Experimentation**: Test different algorithms and hyperparameters
3. **Training**: Train models on prepared datasets
4. **Evaluation**: Assess model performance using metrics
5. **Validation**: Validate models against business requirements

### Model Storage

We version and store all model artifacts in a standardized way:

- **Directory Structure**:
  ```
  models/
  ├── autoencoder/
  │   ├── v1.0.0/
  │   │   ├── model.h5
  │   │   ├── scaler.save
  │   │   └── metadata.json
  │   └── v1.1.0/
  │       └── ...
  ├── svd/
  │   └── ...
  ├── transfer/
  │   └── ...
  └── embeddings/
      └── ...
  ```

- **Metadata File Example**:
  ```json
  {
    "model_name": "autoencoder_recommender",
    "version": "1.0.0",
    "created_at": "2023-05-15T10:30:00Z",
    "framework": "tensorflow",
    "framework_version": "2.10.0",
    "python_version": "3.10.4",
    "input_shape": [29],
    "performance": {
      "validation_loss": 0.142,
      "test_loss": 0.157
    },
    "hyperparameters": {
      "hidden_layers": [16, 8, 16],
      "learning_rate": 0.001,
      "dropout_rate": 0.2,
      "epochs": 100
    },
    "training_dataset": "final_users_over_20_categories.csv",
    "description": "Denoising autoencoder with 3 hidden layers"
  }
  ```

## Model Training Pipeline

Our automated training pipeline consists of:

1. **Data Extraction**: 
   - Fetch data from the database and data files
   - Apply data cleaning and preprocessing

2. **Feature Engineering**:
   - Transform raw data into model-ready features
   - Apply normalization and encoding

3. **Model Training**:
   - Train models with optimized hyperparameters
   - Log training metrics to monitoring system

4. **Evaluation**:
   - Calculate performance metrics
   - Generate evaluation reports

5. **Model Registration**:
   - Register models in model registry
   - Store model artifacts with metadata

### Training Job Example

```python
def train_autoencoder_model(config):
    """Train autoencoder model with the given configuration."""
    # Load and prepare data
    data = load_data(config['data_path'])
    X_train, X_val = preprocess_data(data)
    
    # Build model
    model = build_autoencoder(
        input_dim=X_train.shape[1],
        hidden_layers=config['hidden_layers'],
        dropout_rate=config['dropout_rate']
    )
    
    # Train model
    history = model.fit(
        X_train, X_train,  # Autoencoder reconstructs inputs
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        validation_data=(X_val, X_val),
        callbacks=get_callbacks(config)
    )
    
    # Evaluate model
    metrics = evaluate_model(model, X_val)
    
    # Save model and artifacts
    version = get_next_version('autoencoder')
    save_path = f"models/autoencoder/v{version}"
    os.makedirs(save_path, exist_ok=True)
    
    model.save(f"{save_path}/model.h5")
    joblib.dump(scaler, f"{save_path}/scaler.save")
    
    # Save metadata
    metadata = {
        "model_name": "autoencoder_recommender",
        "version": version,
        "created_at": datetime.now().isoformat(),
        "framework": "tensorflow",
        "framework_version": tf.__version__,
        "python_version": platform.python_version(),
        "input_shape": [X_train.shape[1]],
        "performance": metrics,
        "hyperparameters": config,
        "training_dataset": os.path.basename(config['data_path']),
        "description": config['description']
    }
    
    with open(f"{save_path}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
        
    return model, metrics, save_path
```

## Model Evaluation

We evaluate models using multiple techniques:

### Offline Evaluation

- **Metrics**: RMSE, MAE, precision, recall, F1-score
- **Cross-Validation**: k-fold cross-validation for robust performance estimation
- **Benchmark Comparison**: Compare against baseline and previous versions

### Online Evaluation

- **A/B Testing**: Test new models against current production models
- **User Feedback**: Collect explicit and implicit user feedback
- **Business Metrics**: Track recommendation clicks, conversion rates, etc.

## Model Deployment

Our model deployment process ensures smooth transitions and minimal downtime:

### Deployment Options

1. **Direct Model Loading**:
   - Load models directly in the API service
   - Best for smaller models with lower latency requirements

2. **Containerized Model Serving**:
   - Package models in Docker containers
   - Deploy as separate microservices
   - Use TensorFlow Serving or similar tools

3. **Serverless Inference**:
   - Deploy models to serverless platforms
   - Ideal for variable load patterns

### Deployment Process

1. **Model Selection**: Choose the model version to deploy
2. **Canary Deployment**: Roll out to a small percentage of traffic
3. **Monitoring**: Monitor performance metrics and errors
4. **Progressive Rollout**: Gradually increase traffic to new model
5. **Rollback Plan**: Maintain ability to revert to previous version

## Model Monitoring

Continuous monitoring is essential for maintaining model quality:

### Key Monitoring Areas

1. **Model Performance**: Track prediction quality metrics
2. **Prediction Drift**: Detect shifts in prediction patterns
3. **Data Drift**: Monitor changes in input data distribution
4. **Resource Usage**: Track memory, CPU, and latency
5. **Business Metrics**: Monitor engagement with recommendations

### Monitoring Dashboard

Our monitoring dashboard displays:

- Real-time performance metrics
- Request/response logs
- Error rates and types
- Resource utilization
- Data distribution visualizations

## Continuous Improvement

We maintain a feedback loop for model improvements:

1. **Data Collection**: Continuously gather new data
2. **Hypothesis Testing**: Test ideas for model improvements
3. **Periodic Retraining**: Update models with new data
4. **Algorithm Updates**: Evaluate and implement new algorithms
5. **Feature Engineering**: Develop new features to improve performance

## Tools and Infrastructure

Our MLOps infrastructure uses:

- **Version Control**: Git for code, DVC for model artifacts
- **CI/CD**: GitHub Actions for automated pipelines
- **Model Registry**: Custom model storage with versioning
- **Monitoring**: Prometheus and Grafana
- **Experiment Tracking**: MLflow for experiment management
- **Containerization**: Docker for reproducible environments
- **Infrastructure as Code**: Terraform for infrastructure management

## Best Practices

Our MLOps approach follows these best practices:

1. **Reproducibility**: All experiments and deployments are reproducible
2. **Versioning**: Code, data, and models are versioned
3. **Automation**: Training and deployment processes are automated
4. **Testing**: Models undergo rigorous testing before deployment
5. **Documentation**: Models are well-documented with metadata
6. **Monitoring**: Continuous monitoring for model health
7. **Security**: Model artifacts and data are securely managed 