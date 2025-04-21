# API Endpoints

## Authentication

### Obtain Access Token

```
POST /api/token
```

Authenticates a user and returns a JWT token for accessing protected endpoints.

**Request Body:**
- `username`: User's username
- `password`: User's password

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

### Get Current User

```
GET /api/users/me
```

Returns the authenticated user's information.

**Headers:**
- `Authorization: Bearer {token}`

**Response:**
```json
{
  "id": 123,
  "username": "johndoe",
  "full_name": "John Doe",
  "role": "user",
  "is_active": true,
  "created_at": "2023-01-01T12:00:00"
}
```

## Users

### Create User

```
POST /api/users/
```

Creates a new user account.

**Request Body:**
```json
{
  "username": "johndoe",
  "password": "securepassword",
  "full_name": "John Doe",
  "role": "user"
}
```

**Response:**
```json
{
  "id": 123,
  "username": "johndoe",
  "full_name": "John Doe",
  "role": "user",
  "is_active": true,
  "created_at": "2023-01-01T12:00:00"
}
```

### Check User Exists

```
GET /api/users/{username}/exists
```

Checks if a username already exists.

**Response:**
- `200 OK`: User exists
- `404 Not Found`: User does not exist

## Preferences

### Get User Preferences

```
GET /api/preferences/
```

Returns the authenticated user's category preferences.

**Headers:**
- `Authorization: Bearer {token}`

**Response:**
```json
[
  {
    "id": 1,
    "user_id": 123,
    "category": "museums",
    "rating": 4.5
  },
  {
    "id": 2,
    "user_id": 123,
    "category": "parks",
    "rating": 3.0
  }
]
```

### Create or Update Preference

```
POST /api/preferences/
```

Creates or updates a category preference for the authenticated user.

**Headers:**
- `Authorization: Bearer {token}`

**Request Body:**
```json
{
  "category": "museums",
  "rating": 4.5
}
```

**Response:**
```json
{
  "id": 1,
  "user_id": 123,
  "category": "museums",
  "rating": 4.5
}
```

### Delete Preference

```
DELETE /api/preferences/{category}
```

Deletes a category preference for the authenticated user.

**Headers:**
- `Authorization: Bearer {token}`

**Response:**
- `200 OK`: Preference deleted
- `404 Not Found`: Preference not found

### Extract Preferences from Text

```
POST /api/preferences/extract-preferences
```

Extracts category preferences from natural language text.

**Headers:**
- `Authorization: Bearer {token}`

**Request Body:**
```json
{
  "text": "I love museums and parks, but don't care for shopping."
}
```

**Response:**
```json
{
  "preferences": {
    "museums": 4.5,
    "parks": 4.0,
    "shopping": 1.0
  }
}
```

## Places

### Get Places

```
GET /api/places/
```

Returns a list of places, with optional filtering.

**Query Parameters:**
- `category` (optional): Filter by category
- `limit` (optional): Maximum number of results
- `offset` (optional): Pagination offset

**Response:**
```json
[
  {
    "id": "place123",
    "name": "Museo del Prado",
    "lat": 40.4137,
    "lng": -3.6921,
    "types": ["museum", "tourist_attraction"],
    "rating": 4.8,
    "user_ratings_total": 45000
  },
  // More places...
]
```

### Get Place by ID

```
GET /api/places/{place_id}
```

Returns information about a specific place.

**Response:**
```json
{
  "id": "place123",
  "name": "Museo del Prado",
  "lat": 40.4137,
  "lng": -3.6921,
  "types": ["museum", "tourist_attraction"],
  "rating": 4.8,
  "user_ratings_total": 45000,
  "description": "Spain's main national art museum, located in Madrid"
}
```

## Reviews

### Get User Reviews

```
GET /api/reviews/
```

Returns all reviews submitted by the authenticated user.

**Headers:**
- `Authorization: Bearer {token}`

**Response:**
```json
[
  {
    "id": 1,
    "user_id": 123,
    "place_id": "place123",
    "rating": 4.5,
    "comment": "Wonderful collection of art!",
    "created_at": "2023-01-01T12:00:00"
  },
  // More reviews...
]
```

### Get Review for Specific Place

```
GET /api/reviews/user/{place_id}
```

Returns the authenticated user's review for a specific place.

**Headers:**
- `Authorization: Bearer {token}`

**Response:**
```json
{
  "id": 1,
  "user_id": 123,
  "place_id": "place123",
  "rating": 4.5,
  "comment": "Wonderful collection of art!",
  "created_at": "2023-01-01T12:00:00",
  "submitted": true
}
```

### Submit Review

```
POST /api/reviews/
```

Submits a review for a place.

**Headers:**
- `Authorization: Bearer {token}`

**Request Body:**
```json
{
  "place_id": "place123",
  "rating": 4.5,
  "comment": "Wonderful collection of art!"
}
```

**Response:**
```json
{
  "id": 1,
  "user_id": 123,
  "place_id": "place123",
  "rating": 4.5,
  "comment": "Wonderful collection of art!",
  "created_at": "2023-01-01T12:00:00"
}
```

### Delete Review

```
DELETE /api/reviews/{place_id}
```

Deletes the authenticated user's review for a specific place.

**Headers:**
- `Authorization: Bearer {token}`

**Response:**
- `200 OK`: Review deleted
- `404 Not Found`: Review not found

## Recommendations

### Get Recommendations

```
GET /api/recommendations/
```

Returns personalized place recommendations based on user preferences.

**Headers:**
- `Authorization: Bearer {token}`

**Query Parameters:**
- `lat` (required): User's latitude
- `lng` (required): User's longitude
- `limit` (optional): Maximum number of recommendations
- `method` (optional): Recommendation method ("autoencoder", "svd", "transfer", "embeddings", "ensemble")

**Response:**
```json
[
  {
    "place_id": "place123",
    "name": "Museo del Prado",
    "lat": 40.4137,
    "lng": -3.6921,
    "types": ["museum", "tourist_attraction"],
    "rating": 4.8,
    "user_ratings_total": 45000,
    "score": 0.92,
    "distance": 1200,
    "explanation": "Recommended based on your preference for art and museums."
  },
  // More recommendations...
]
```

### Get Chatbot Recommendations

```
POST /api/recommendations/chat
```

Returns recommendations based on a conversational interaction.

**Headers:**
- `Authorization: Bearer {token}`

**Request Body:**
```json
{
  "message": "I'm looking for art museums near Plaza Mayor",
  "location": {
    "latitude": 40.4168,
    "longitude": -3.7038
  }
}
```

**Response:**
```json
{
  "recommendations": [
    {
      "place_id": "place123",
      "name": "Museo del Prado",
      "lat": 40.4137,
      "lng": -3.6921,
      "types": ["museum", "tourist_attraction"],
      "rating": 4.8,
      "user_ratings_total": 45000,
      "score": 0.92,
      "distance": 1200
    }
  ],
  "response_text": "Here are some art museums near Plaza Mayor. The Prado Museum is one of the world's finest art museums and is just a 15-minute walk away.",
  "extracted_preferences": {
    "museums": 4.5,
    "art": 4.5
  }
}
``` 