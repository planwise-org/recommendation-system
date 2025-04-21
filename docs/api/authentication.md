# Authentication

## Overview

The Planwise API uses OAuth2 with Password Flow and JWT (JSON Web Tokens) for authentication. This secure approach enables:

1. User authentication with username and password
2. Stateless token-based authorization for API access
3. Role-based access control for endpoints
4. Token expiration and refresh mechanisms

## Authentication Flow

1. **User Login**: Client submits username and password to `/api/token`
2. **Token Generation**: Server validates credentials and returns a JWT access token
3. **Request Authorization**: Client includes the token in the `Authorization` header for subsequent requests
4. **Token Validation**: Server verifies the token signature and expiration for each protected request

## Implementation Details

### Token Endpoint

```
POST /api/token
```

This endpoint implements the OAuth2 password flow:

**Request Body:**
```json
{
  "username": "user123",
  "password": "secure_password"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

**Error Responses:**
- `401 Unauthorized`: Invalid credentials
- `400 Bad Request`: Missing or invalid request parameters

### JWT Token Structure

The generated JWT contains:

1. **Header**: Specifies the algorithm used for signing
2. **Payload**:
   - `sub`: Username (subject)
   - `exp`: Expiration timestamp
   - `iat`: Issued at timestamp
   - Additional custom claims as needed

3. **Signature**: Generated using the server's secret key

### Token Validation

For protected endpoints, the API:

1. Extracts the token from the `Authorization: Bearer {token}` header
2. Verifies the token signature using the secret key
3. Checks the token expiration time
4. Retrieves the user information from the database based on the username in the token

## Authorization Logic

The `get_current_user` dependency function:

```python
async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_session)
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        # Decode the JWT token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
        
    # Fetch the user from the database
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user
```

## Role-Based Access Control

Endpoints can implement role-based access restrictions using:

```python
async def get_current_active_admin(
    current_user: User = Depends(get_current_user),
):
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    return current_user
```

## Security Configuration

Key security settings are configured in the API's environment variables:

- `SECRET_KEY`: Used for JWT signing
- `ACCESS_TOKEN_EXPIRE_MINUTES`: Sets token lifetime (default: 30 minutes)
- `ALGORITHM`: JWT signing algorithm (default: HS256)

## Client Implementation

### JavaScript Example

```javascript
// Login and get token
async function login(username, password) {
  const response = await fetch('/api/token', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded'
    },
    body: new URLSearchParams({
      'username': username,
      'password': password
    })
  });
  
  const data = await response.json();
  return data.access_token;
}

// Using the token for authenticated requests
async function fetchUserData(token) {
  const response = await fetch('/api/users/me', {
    headers: {
      'Authorization': `Bearer ${token}`
    }
  });
  
  return await response.json();
}
```

### Python Example

```python
import requests

# Login and get token
def login(username, password):
    response = requests.post(
        "http://localhost:8080/api/token",
        data={"username": username, "password": password}
    )
    return response.json()["access_token"]

# Using the token for authenticated requests
def fetch_user_data(token):
    response = requests.get(
        "http://localhost:8080/api/users/me",
        headers={"Authorization": f"Bearer {token}"}
    )
    return response.json()
```

## Security Best Practices

1. **Always use HTTPS** in production to protect token transmission
2. **Store tokens securely** on the client side (e.g., in HttpOnly cookies)
3. **Keep token expiration short** and implement refresh token patterns for longer sessions
4. **Validate all inputs** to prevent injection attacks
5. **Implement rate limiting** to protect against brute-force attacks 