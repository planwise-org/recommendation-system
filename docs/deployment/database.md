# Database Management

## Overview

Planwise uses PostgreSQL as its primary database, managed through Supabase for easier administration, authentication, and hosting. This document covers our database schema, migration strategies, and operational procedures.

## Database Schema

### Core Tables

Our database schema consists of the following main tables:

#### Users

```sql
CREATE TABLE "user" (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    hashed_password VARCHAR(100) NOT NULL,
    full_name VARCHAR(100) NOT NULL,
    role VARCHAR(20) NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

#### Places

```sql
CREATE TABLE "place" (
    id VARCHAR(100) PRIMARY KEY, -- Google Place ID
    name VARCHAR(200) NOT NULL,
    lat DOUBLE PRECISION NOT NULL,
    lng DOUBLE PRECISION NOT NULL,
    types VARCHAR(500),
    category VARCHAR(100),
    rating DOUBLE PRECISION,
    user_ratings_total INTEGER,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

#### Reviews

```sql
CREATE TABLE "review" (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
    place_id VARCHAR(100) NOT NULL REFERENCES "place"(id) ON DELETE CASCADE,
    rating DOUBLE PRECISION NOT NULL,
    comment TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE (user_id, place_id)
);
```

#### Preferences

```sql
CREATE TABLE "preference" (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
    category VARCHAR(100) NOT NULL,
    rating DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE (user_id, category)
);
```

### Indexes

We maintain several indexes to optimize query performance:

```sql
-- User lookup by username
CREATE INDEX idx_user_username ON "user" (username);

-- Place lookup by location (for proximity searches)
CREATE INDEX idx_place_location ON "place" USING gist (
    ST_SetSRID(ST_MakePoint(lng, lat), 4326)
);

-- Place lookup by category
CREATE INDEX idx_place_category ON "place" (category);

-- Review lookup by user
CREATE INDEX idx_review_user_id ON "review" (user_id);

-- Review lookup by place
CREATE INDEX idx_review_place_id ON "review" (place_id);

-- Preference lookup by user
CREATE INDEX idx_preference_user_id ON "preference" (user_id);

-- Preference lookup by category (for recommendation filtering)
CREATE INDEX idx_preference_category ON "preference" (category);
```

## Entity-Relationship Diagram

```
┌─────────────┐     ┌───────────────┐     ┌─────────────┐
│    User     │     │    Review     │     │    Place    │
├─────────────┤     ├───────────────┤     ├─────────────┤
│ id          │─┐   │ id            │     │ id          │
│ username    │ │   │ user_id       │──┘  │ name        │
│ hashed_pwd  │ │   │ place_id      │─────┤ lat         │
│ full_name   │ │   │ rating        │     │ lng         │
│ role        │ │   │ comment       │     │ types       │
│ is_active   │ │   │ created_at    │     │ category    │
│ created_at  │ │   └───────────────┘     │ rating      │
└─────────────┘ │                         │ ratings_total│
                │                         │ description  │
                │   ┌───────────────┐     │ created_at   │
                │   │  Preference   │     │ updated_at   │
                │   ├───────────────┤     └─────────────┘
                └───┤ id            │
                    │ user_id       │
                    │ category      │
                    │ rating        │
                    │ created_at    │
                    │ updated_at    │
                    └───────────────┘
```

## Database Migrations

We use Alembic for database migrations to ensure consistent schema changes across all environments.

### Migration Workflow

1. **Create Migration**:
   ```bash
   cd api
   alembic revision --autogenerate -m "Description of changes"
   ```

2. **Review Migration**:
   Examine the generated Python file in `alembic/versions/` to ensure it captures the intended changes.

3. **Apply Migration**:
   ```bash
   alembic upgrade head
   ```

4. **Rollback Migration** (if needed):
   ```bash
   alembic downgrade -1
   ```

### Migration Example

A sample migration to add a new field to the `place` table:

```python
"""Add opening_hours to place table

Revision ID: 3a4b5c6d7e8f
Revises: 1a2b3c4d5e6f
Create Date: 2023-06-20 14:30:45.123456

"""
from alembic import op
import sqlalchemy as sa
import sqlmodel

# revision identifiers
revision = '3a4b5c6d7e8f'
down_revision = '1a2b3c4d5e6f'
branch_labels = None
depends_on = None

def upgrade():
    # Add opening_hours column to place table
    op.add_column('place', sa.Column('opening_hours', sa.JSON(), nullable=True))
    
def downgrade():
    # Remove opening_hours column from place table
    op.drop_column('place', 'opening_hours')
```

## Database Operations

### Backups

We perform automated database backups:

- **Daily Full Backups**: Complete database dumps every 24 hours
- **Continuous WAL Archiving**: Write-ahead log archiving for point-in-time recovery
- **Retention Policy**: Daily backups kept for 30 days, weekly backups for 6 months, monthly backups for 2 years

### Backup Commands

```bash
# Manual full backup
pg_dump --dbname=postgresql://username:password@host:port/dbname \
    --format=custom \
    --file=backup_$(date +%Y%m%d_%H%M%S).dump

# Restore from backup
pg_restore --dbname=postgresql://username:password@host:port/dbname \
    --clean \
    backup_file.dump
```

### Monitoring

We monitor several database metrics:

- **Connection Pool Usage**: Track active and idle connections
- **Query Performance**: Monitor slow queries and query execution times
- **Database Size**: Monitor database and table growth
- **Index Usage**: Track index usage statistics
- **Cache Hit Ratio**: Monitor buffer cache effectiveness

## Performance Optimization

### Query Optimization

We follow these practices for optimal query performance:

1. **Use Prepared Statements**: Prevent SQL injection and improve query planning
2. **Limit Result Sets**: Use pagination for large result sets
3. **Optimize JOINs**: Minimize cross-joins and use appropriate join types
4. **Use CTEs**: Use Common Table Expressions for complex queries
5. **Avoid N+1 Queries**: Use eager loading to prevent multiple database round trips

### Example Optimized Query

Fetching recommendations with places and their categories in one query:

```sql
WITH user_prefs AS (
    SELECT category, rating
    FROM preference
    WHERE user_id = :user_id
)
SELECT p.id, p.name, p.lat, p.lng, p.rating, p.user_ratings_total,
       p.category, up.rating as user_preference,
       SQRT(POW(p.lat - :user_lat, 2) + POW(p.lng - :user_lng, 2)) * 111.32 AS distance
FROM place p
JOIN user_prefs up ON p.category = up.category
WHERE SQRT(POW(p.lat - :user_lat, 2) + POW(p.lng - :user_lng, 2)) * 111.32 < 5
ORDER BY (
    up.rating * 0.6 +
    COALESCE(p.rating, 0) / 5 * 0.2 +
    LEAST(1.0, LOG(COALESCE(p.user_ratings_total, 0) + 1) / 10) * 0.1 +
    (1 - LEAST(1, SQRT(POW(p.lat - :user_lat, 2) + POW(p.lng - :user_lng, 2)) * 111.32 / 5)) * 0.1
) DESC
LIMIT :limit
```

## Database Access Control

### User Roles

We use a role-based access control system:

1. **supabase_admin**: Full database access (used only for administration)
2. **app_service**: Service account for application access
3. **app_read**: Read-only access for reporting and analytics

### Row-Level Security

We implement row-level security policies to restrict data access:

```sql
-- Enable RLS on tables
ALTER TABLE "user" ENABLE ROW LEVEL SECURITY;
ALTER TABLE "review" ENABLE ROW LEVEL SECURITY;
ALTER TABLE "preference" ENABLE ROW LEVEL SECURITY;

-- Users can only access their own data
CREATE POLICY user_self_access ON "user"
    USING (id = current_setting('app.user_id', true)::integer);

-- Users can only see their own reviews
CREATE POLICY review_user_access ON "review"
    USING (user_id = current_setting('app.user_id', true)::integer);

-- Users can only see their own preferences
CREATE POLICY preference_user_access ON "preference"
    USING (user_id = current_setting('app.user_id', true)::integer);
```

## Connection Management

### Connection Pooling

We use PgBouncer for connection pooling:

- **Pool Size**: 20 connections per application instance
- **Pool Mode**: Transaction pooling (connection released after transaction)
- **Idle Timeout**: 300 seconds

### Environment Configuration

Environment variables for database connections:

```env
DATABASE_URL=postgresql://username:password@host:port/dbname
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=10
DATABASE_POOL_TIMEOUT=30
```

## Supabase Integration

We leverage Supabase for several database-related features:

1. **Authentication**: User management and JWT-based authentication
2. **Row-Level Security**: Automated policy enforcement
3. **Realtime**: Live database events for reactive UIs
4. **Storage**: Blob storage for user-generated content
5. **Edge Functions**: Serverless database triggers and functions

## Database Migration from Development to Production

When migrating database changes from development to production, we follow this process:

1. **Test on Development**: Apply and test migrations in development environment
2. **Test on Staging**: Apply migrations to staging environment for final validation
3. **Backup Production**: Create a full backup of the production database
4. **Apply to Production**: Run migrations on production during low-traffic period
5. **Verify**: Run validation checks to ensure data integrity
6. **Rollback Plan**: Maintain ability to revert if issues are detected 