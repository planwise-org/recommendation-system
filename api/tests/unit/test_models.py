import pytest
from datetime import datetime
from sqlmodel import Session, SQLModel, create_engine
from src.models import (
    User, Place, Review, Recommendation, Preference,
    UserRole, PlaceType, RecommendationAlgorithm
)
from src.database import init_db, engine, drop_all_tables
from pydantic import ValidationError


@pytest.fixture(name="session")
def session_fixture():
    init_db()
    with Session(engine) as session:
        yield session
    drop_all_tables()


# User Model Tests
def test_user_creation():
    user = User(
        username="testuser",
        hashed_password="hashedpass123",
        full_name="Test User",
        role=UserRole.USER
    )
    assert user.username == "testuser"
    assert user.hashed_password == "hashedpass123"
    assert user.full_name == "Test User"
    assert user.role == UserRole.USER
    assert isinstance(user.created_at, datetime)
    assert isinstance(user.updated_at, datetime)

def test_user_role_enum():
    # Test valid roles
    assert UserRole.ADMIN == "admin"
    assert UserRole.USER == "user"

    # Test that valid roles work in the model
    user1 = User(
        username="testuser1",
        hashed_password="hashedpass123",
        full_name="Test User",
        role=UserRole.ADMIN
    )
    assert user1.role == UserRole.ADMIN

    user2 = User(
        username="testuser2",
        hashed_password="hashedpass123",
        full_name="Test User",
        role=UserRole.USER
    )
    assert user2.role == UserRole.USER

    # Test that invalid role raises error
    with pytest.raises((ValueError, ValidationError)):
        invalid_role = UserRole("invalid_role")
        User(
            username="testuser",
            hashed_password="hashedpass123",
            full_name="Test User",
            role=invalid_role
        )

def test_user_relationships(session):
    # Create test user
    user = User(
        username="testuser",
        hashed_password="hashedpass123",
        full_name="Test User"
    )
    session.add(user)
    session.commit()

    # Create related objects
    place = Place(
        name="Test Place",
        description="Test Description",
        address="123 Test St",
        latitude=40.7128,
        longitude=-74.0060,
        place_type=PlaceType.RESTAURANT
    )
    session.add(place)
    session.commit()

    review = Review(
        user_id=user.id,
        place_id=place.id,
        rating=4.5,
        comment="Great place!"
    )
    session.add(review)

    recommendation = Recommendation(
        user_id=user.id,
        place_id=place.id,
        algorithm=RecommendationAlgorithm.AUTOENCODER,
        score=0.95
    )
    session.add(recommendation)

    preference = Preference(
        user_id=user.id,
        category="restaurants",
        rating=4.5
    )
    session.add(preference)
    session.commit()

    # Test relationships
    assert len(user.reviews) == 1
    assert len(user.recommendations) == 1
    assert len(user.preferences) == 1
    assert user.reviews[0].comment == "Great place!"
    assert user.recommendations[0].score == 0.95
    assert user.preferences[0].category == "restaurants"

# Place Model Tests
def test_place_creation():
    place = Place(
        name="Test Place",
        description="Test Description",
        address="123 Test St",
        latitude=40.7128,
        longitude=-74.0060,
        place_type=PlaceType.RESTAURANT
    )
    assert place.name == "Test Place"
    assert place.description == "Test Description"
    assert place.address == "123 Test St"
    assert place.latitude == 40.7128
    assert place.longitude == -74.0060
    assert place.place_type == PlaceType.RESTAURANT
    assert place.rating == 0.0  # Default value
    assert isinstance(place.created_at, datetime)
    assert isinstance(place.updated_at, datetime)

def test_place_type_enum():
    # Test valid place types
    assert PlaceType.RESTAURANT == "restaurant"
    assert PlaceType.CAFE == "cafe"
    assert PlaceType.BAR == "bar"
    assert PlaceType.CLUB == "club"
    assert PlaceType.SHOPPING == "shopping"
    assert PlaceType.ATTRACTION == "attraction"
    assert PlaceType.PARK == "park"
    assert PlaceType.MUSEUM == "museum"
    assert PlaceType.THEATER == "theater"
    assert PlaceType.OTHER == "other"

    # Test that valid place types work in the model
    place1 = Place(
        name="Test Restaurant",
        description="Test Description",
        address="123 Test St",
        latitude=40.7128,
        longitude=-74.0060,
        place_type=PlaceType.RESTAURANT
    )
    assert place1.place_type == PlaceType.RESTAURANT

    place2 = Place(
        name="Test Cafe",
        description="Test Description",
        address="123 Test St",
        latitude=40.7128,
        longitude=-74.0060,
        place_type=PlaceType.CAFE
    )
    assert place2.place_type == PlaceType.CAFE

    # Test that invalid place type raises error
    with pytest.raises((ValueError, ValidationError)):
        invalid_type = PlaceType("invalid_type")
        Place(
            name="Test Place",
            description="Test Description",
            address="123 Test St",
            latitude=40.7128,
            longitude=-74.0060,
            place_type=invalid_type
        )

# Review Model Tests
def test_review_creation():
    review = Review(
        user_id=1,
        place_id=1,
        rating=4.5,
        comment="Great place!"
    )
    assert review.user_id == 1
    assert review.place_id == 1
    assert review.rating == 4.5
    assert review.comment == "Great place!"
    assert isinstance(review.created_at, datetime)
    assert isinstance(review.updated_at, datetime)

def test_review_rating_validation():
    # Test valid rating
    review = Review(
        user_id=1,
        place_id=1,
        rating=4.5,
        comment="Test"
    )
    assert review.rating == 4.5


# Recommendation Model Tests
def test_recommendation_creation():
    recommendation = Recommendation(
        user_id=1,
        place_id=1,
        algorithm=RecommendationAlgorithm.AUTOENCODER,
        score=0.95
    )
    assert recommendation.user_id == 1
    assert recommendation.place_id == 1
    assert recommendation.algorithm == RecommendationAlgorithm.AUTOENCODER
    assert recommendation.score == 0.95
    assert recommendation.visited == False  # Default value
    assert recommendation.reviewed == False  # Default value
    assert isinstance(recommendation.created_at, datetime)
    assert isinstance(recommendation.updated_at, datetime)

def test_recommendation_algorithm_enum():
    assert RecommendationAlgorithm.AUTOENCODER == "autoencoder"
    assert RecommendationAlgorithm.SVD == "svd"
    assert RecommendationAlgorithm.TRANSFER_LEARNING == "transfer_learning"

    # Test invalid algorithm
    with pytest.raises((ValueError, ValidationError)):
        invalid_algorithm = RecommendationAlgorithm("invalid_algorithm")
        Recommendation(
            user_id=1,
            place_id=1,
            algorithm=invalid_algorithm,
            score=0.95
        )


# Preference Model Tests
def test_preference_creation():
    preference = Preference(
        user_id=1,
        category="restaurants",
        rating=4.5
    )
    assert preference.user_id == 1
    assert preference.category == "restaurants"
    assert preference.rating == 4.5
    assert isinstance(preference.created_at, datetime)
    assert isinstance(preference.updated_at, datetime)


def test_cascade_deletion(session):
    # Create test user
    user = User(
        username="testuser",
        hashed_password="hashedpass123",
        full_name="Test User"
    )
    session.add(user)
    session.commit()

    # Create related objects
    place = Place(
        name="Test Place",
        description="Test Description",
        address="123 Test St",
        latitude=40.7128,
        longitude=-74.0060,
        place_type=PlaceType.RESTAURANT
    )
    session.add(place)
    session.commit()

    review = Review(
        user_id=user.id,
        place_id="test_place_id",
        rating=4.5,
        comment="Great place!"
    )
    session.add(review)

    recommendation = Recommendation(
        user_id=user.id,
        place_id=place.id,
        algorithm=RecommendationAlgorithm.AUTOENCODER,
        score=0.95
    )
    session.add(recommendation)

    preference = Preference(
        user_id=user.id,
        category="restaurants",
        rating=4.5
    )
    session.add(preference)
    session.commit()

    # Delete user and verify cascade
    session.delete(user)
    session.commit()

    # Verify related objects are deleted
    assert session.get(Review, review.id) is None
    assert session.get(Recommendation, recommendation.id) is None
    assert session.get(Preference, preference.id) is None

    # Place should still exist (not dependent on user)
    assert session.get(Place, place.id) is not None
