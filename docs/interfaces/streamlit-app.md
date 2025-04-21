# Streamlit Web Application

## Overview

The Planwise Streamlit web application provides an interactive, user-friendly interface for our recommendation system. Users can adjust preference sliders, chat with the recommender, view recommended places on a map, and provide feedback through ratings and reviews.

## Features

### User Authentication

- **Login/Signup**: Secure user authentication with username and password
- **Session Management**: Persistent user sessions with OAuth2 token-based authentication
- **Profile Management**: User preferences and reviews are saved and associated with user accounts

### Preference Input Methods

#### Interactive Sliders

Users can adjust preference sliders (0-5) across 29 categories including:
- Art galleries
- Museums
- Restaurants
- Parks
- Cafes
- And many more

The interface allows users to:
- Select which categories matter to them
- Specify preference strength for each category
- Save preferences for future sessions

#### Natural Language Chat

Users can express preferences conversationally:
- Type messages like "I love nature and museums but don't care for nightlife"
- The system extracts preferences from natural language
- Extracted preferences automatically update the sliders

### Recommendation Controls

- **Location Setting**: Users can specify their current location coordinates
- **Recommendation Method**: Choose between different algorithms:
  - Autoencoder-Based
  - SVD-Based
  - Transfer-Based
  - Embeddings-Based
  - Ensemble (combines all models)
- **Routing Profile**: Select between "foot-walking" and "driving-car" for route optimization

### Recommendation Visualization

#### List View

Each recommended place shows:
- Name and icon
- Average rating
- User rating count
- Distance from user location
- Recommendation score
- Category tags
- Expandable details section

#### Map View

- Interactive map showing all recommended places
- Optimized route between locations
- Tooltips with place names
- Color-coded paths

#### Route Details

- Step-by-step breakdown of the optimized route
- Distance between consecutive stops
- Total travel distance
- Categorized stops for better planning

### User Feedback

- **Rating System**: Users can rate places on a 1-5 scale
- **Review Comments**: Optional text comments for more detailed feedback
- **Edit Functionality**: Users can update ratings and reviews they've previously submitted
- **Persistent Reviews**: All feedback is saved to the user's profile and used to improve future recommendations

## Technical Implementation

### Application Structure

The Streamlit app is organized into several functional sections:

1. **Authentication Flow**: Handles login, signup, and session management
2. **User Input Collection**: Processes preference input through sliders and chat
3. **Recommendation Engine Interface**: Connects to the underlying models
4. **Results Visualization**: Displays recommendations and routes
5. **Feedback Collection**: Manages user ratings and reviews

### Key Components

- **Session State Management**: Uses Streamlit's session state for persistent user data
- **API Integration**: Communicates with the FastAPI backend for authentication, preferences, and reviews
- **Model Integration**: Direct integration with recommendation models for real-time prediction
- **Map Rendering**: Uses PyDeck for interactive map visualization
- **Route Optimization**: Integrates with OpenRouteService API for route planning

## Usage Guide

### Getting Started

1. **Create an Account**:
   - Navigate to the login tab
   - Click "Sign Up" and enter your credentials
   - Return to login and enter your credentials

2. **Set Your Preferences**:
   - Use the sliders to indicate your interests
   - Or chat with the recommender to express preferences naturally

3. **Generate Recommendations**:
   - Set your current location
   - Choose a recommendation method
   - Click "Generate Recommendations"

4. **Explore Results**:
   - Browse the list of recommended places
   - View them on the interactive map
   - Check the optimized route details

5. **Provide Feedback**:
   - Rate and review places you've visited
   - Edit previous reviews as needed
   - Your feedback improves future recommendations

## Running the Application

To run the Streamlit app locally:

```bash
cd reco/streamlit
streamlit run app.py
```

Or with Docker:

```bash
docker compose -f docker-compose.dev.yml up
```

The application will be available at `http://localhost:8501`. 