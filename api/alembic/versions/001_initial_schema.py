"""initial schema

Revision ID: 001
Revises: 
Create Date: 2024-04-12 14:47:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create users table
    op.create_table(
        'user',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('email', sa.String(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('hash_password', sa.String(), nullable=False),
        sa.Column('role', postgresql.ENUM('user', 'admin', name='user_role'), nullable=False),
        sa.Column('initial_preferences', postgresql.JSONB(), nullable=True),
        sa.Column('current_preferences', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('last_login', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('is_verified', sa.Boolean(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('email')
    )
    op.create_index(op.f('ix_user_email'), 'user', ['email'], unique=True)
    
    # Create places table
    op.create_table(
        'place',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('description', sa.String(), nullable=True),
        sa.Column('latitude', sa.Float(), nullable=False),
        sa.Column('longitude', sa.Float(), nullable=False),
        sa.Column('address', sa.String(), nullable=True),
        sa.Column('category', sa.String(), nullable=False),
        sa.Column('types', postgresql.JSONB(), nullable=False),
        sa.Column('rating', sa.Float(), nullable=True),
        sa.Column('total_ratings', sa.Integer(), nullable=True),
        sa.Column('price_level', sa.Integer(), nullable=True),
        sa.Column('opening_hours', postgresql.JSONB(), nullable=True),
        sa.Column('photos', postgresql.JSONB(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_place_name'), 'place', ['name'], unique=False)
    op.create_index(op.f('ix_place_category'), 'place', ['category'], unique=False)
    
    # Create reviews table
    op.create_table(
        'review',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('place_id', sa.Integer(), nullable=False),
        sa.Column('rating', sa.Float(), nullable=False),
        sa.Column('comment', sa.String(), nullable=True),
        sa.Column('photos', postgresql.JSONB(), nullable=False),
        sa.Column('visit_date', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['place_id'], ['place.id'], ),
        sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create recommendations table
    op.create_table(
        'recommendation',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('place_id', sa.Integer(), nullable=False),
        sa.Column('algorithm', postgresql.ENUM('autoencoder', 'svd', 'transfer', name='recommendation_algorithm'), nullable=False),
        sa.Column('score', sa.Float(), nullable=False),
        sa.Column('was_visited', sa.Boolean(), nullable=False),
        sa.Column('was_reviewed', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('viewed_at', sa.DateTime(), nullable=True),
        sa.Column('visited_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['place_id'], ['place.id'], ),
        sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
        sa.PrimaryKeyConstraint('id')
    )


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_table('recommendation')
    op.drop_table('review')
    op.drop_table('place')
    op.drop_table('user')
    
    # Drop enum types
    op.execute('DROP TYPE recommendation_algorithm')
    op.execute('DROP TYPE user_role') 