# Chatbot Interface

## Overview

The Planwise Chatbot Interface provides a conversational way to interact with our recommendation system. Through natural language processing, users can express their preferences, receive personalized recommendations, and explore places in Madrid without navigating through traditional UI elements.

## Features

### Natural Language Preference Extraction

- **Conversational Input**: Users can express preferences in natural language
  - Example: "I'm interested in museums and parks, but not shopping malls"
  - Example: "I love food and coffee, especially in quiet areas"
- **Contextual Understanding**: The system interprets qualifiers and modifiers
  - "Very interested in art" → higher rating for art-related categories
  - "Not really into nightlife" → lower rating for nightlife categories
- **Preference Refinement**: Users can adjust previously stated preferences
  - "Actually, I prefer outdoor activities more than museums"

### Recommendation Delivery

- **Formatted Results**: Recommendations are presented in a clear, conversational format
- **Explanation**: Each recommendation includes a brief explanation of why it was suggested
- **Grouping**: Related places are grouped together for easier browsing
- **Follow-up Questions**: Users can ask for more details about specific places

### Interactive Exploration

- **Location-Based Queries**: "What's good near Plaza Mayor?"
- **Category Filtering**: "Show me only parks and museums"
- **Sorting Options**: "Sort by distance" or "Show me the highest-rated first"
- **Details On Demand**: "Tell me more about the Prado Museum"

## Technical Implementation

### API Endpoints

The chatbot is powered by our FastAPI backend with these key endpoints:

#### Preference Extraction

```
POST /api/preferences/extract-preferences
```

- **Input**: JSON with text field containing user message
- **Output**: JSON with extracted preference categories and their ratings

#### Recommendation Generation

```
POST /api/recommendations/chat
```

- **Input**: JSON with user ID, location, and preference context
- **Output**: JSON with recommended places and explanation metadata

### Natural Language Processing

Our chatbot uses a multi-stage NLP pipeline:

1. **Tokenization & POS Tagging**: Identify sentence structure and parts of speech
2. **Named Entity Recognition**: Extract location names and place references
3. **Sentiment Analysis**: Determine attitude toward mentioned categories
4. **Category Mapping**: Link extracted concepts to our internal category system
5. **Rating Inference**: Convert qualitative statements to numerical preference ratings

### Recommendation Engine Integration

The chatbot connects to the same recommendation models as the web interface:

- **Model Selection**: Automatically chooses the most appropriate model based on user context
- **Default Ensemble**: Uses the ensemble model by default for most robust recommendations
- **Context Awareness**: Adapts to user's expressed preferences and location

## Integration Options

### Web Widget

Embed the chatbot in your website with our JavaScript widget:

```html
<script src="https://api.planwise.io/widget.js"></script>
<div id="planwise-chat" data-api-key="YOUR_API_KEY"></div>
```

### REST API

Direct integration with our REST API:

```bash
curl -X POST "https://api.planwise.io/api/recommendations/chat" \
  -H "Authorization: Bearer YOUR_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "message": "I love art museums and parks",
    "location": {
      "latitude": 40.4168,
      "longitude": -3.7038
    }
  }'
```

### Mobile SDK

For iOS and Android applications, use our mobile SDK (coming soon).

## Usage Examples

### Basic Conversation Flow

```
User: "Hi, I'm visiting Madrid and love art and good food."

Chatbot: "Welcome to Madrid! I've noted your interest in art and food. 
Would you like recommendations near a specific area?"

User: "Yes, I'm staying near Gran Vía."

Chatbot: "Great! Here are some art and food recommendations near Gran Vía:

🎨 Thyssen-Bornemisza Museum (0.5 km)
   * World-class art collection spanning eight centuries
   * Rating: 4.7/5 (9,800+ reviews)

🍽️ Mercado de San Miguel (1.2 km)
   * Gourmet food market with Spanish delicacies
   * Rating: 4.5/5 (45,000+ reviews)

Would you like to see more options or get details about any of these places?"
```

### Follow-up Interactions

```
User: "Tell me more about the Thyssen museum."

Chatbot: "The Thyssen-Bornemisza Museum houses one of the most important 
private art collections in the world. Highlights include works by 
Monet, Van Gogh, and Picasso. It's less crowded than the Prado 
and offers a more manageable experience. Opening hours are 
10am-7pm, closed Mondays. Tickets cost €13, with free entry 
during the last 2 hours on Mondays."

User: "Are there any good cafes nearby?"

Chatbot: "Yes, here are some highly-rated cafes near the Thyssen museum:

☕ Café del Círculo de Bellas Artes (0.3 km)
   * Rooftop cafe with amazing city views
   * Rating: 4.6/5 (2,300+ reviews)

☕ Mistura Ice Cream & Coffee (0.2 km)
   * Artisanal coffee and homemade ice cream
   * Rating: 4.8/5 (950+ reviews)"
``` 