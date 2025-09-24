# 🔄 AI Review Engine - Database Migrations

## Overview

This document describes the database migration process and structure. We use Alembic for managing database migrations.

## Migration Directory Structure

```
migrations/
├── README.md
├── alembic.ini
├── env.py
└── versions/
    ├── 001_initial.py
    ├── 002_review_system.py
    ├── 003_user_plans.py
    ├── 004_api_auth.py
    └── 005_attachments.py
```

## Migration Workflow

### Creating a New Migration

```bash
# Create a new migration
alembic revision -m "description_of_changes"
```

### Applying Migrations

```bash
# Apply all pending migrations
alembic upgrade head

# Apply specific migration
alembic upgrade <revision_id>

# Rollback one migration
alembic downgrade -1

# Rollback to specific version
alembic downgrade <revision_id>
```

### Checking Migration Status

```bash
# View current migration status
alembic current

# View migration history
alembic history --verbose
```

## Migration Files

### 1. Initial Migration (001_initial.py)

Creates the foundational tables for the application.

```python
"""Initial migration

Revision ID: 001_initial
Revises: None
Create Date: 2025-09-21 10:00:00.000000
"""

def upgrade():
    # Create users table
    op.create_table(
        'users',
        sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('username', sa.String(50), nullable=False),
        sa.Column('email', sa.String(255), nullable=False),
        sa.Column('password_hash', sa.String(255), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('is_active', sa.Boolean(), server_default='true'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('username'),
        sa.UniqueConstraint('email')
    )

    # Create roles table
    op.create_table(
        'roles',
        sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('name', sa.String(50), nullable=False),
        sa.Column('permissions', sa.JSON(), nullable=False, server_default='{}'),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name')
    )

    # Create products table
    op.create_table(
        'products',
        sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text()),
        sa.Column('category', sa.String(100)),
        sa.Column('attributes', sa.JSON(), nullable=False, server_default='{}'),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id')
    )

    # Create indexes
    op.create_index('idx_users_email', 'users', ['email'])
    op.create_index('idx_users_username', 'users', ['username'])
    op.create_index('idx_roles_name', 'roles', ['name'])
    op.create_index('idx_products_name', 'products', ['name'])
    op.create_index('idx_products_category', 'products', ['category'])

def downgrade():
    # Remove indexes
    op.drop_index('idx_products_category')
    op.drop_index('idx_products_name')
    op.drop_index('idx_roles_name')
    op.drop_index('idx_users_username')
    op.drop_index('idx_users_email')

    # Drop tables
    op.drop_table('products')
    op.drop_table('roles')
    op.drop_table('users')
```

### 2. Review System (002_review_system.py)

Adds review-related tables and functionality.

```python
"""Review system

Revision ID: 002_review_system
Revises: 001_initial
Create Date: 2025-09-21 10:30:00.000000
"""

def upgrade():
    # Create reviews table
    op.create_table(
        'reviews',
        sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('product_id', sa.UUID(), nullable=False),
        sa.Column('rating', sa.Integer(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('sentiment', sa.String(20)),
        sa.Column('spam_probability', sa.Float()),
        sa.Column('metadata', sa.JSON(), nullable=False, server_default='{}'),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('is_processed', sa.Boolean(), server_default='false'),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['product_id'], ['products.id'])
    )

    # Create review_analytics table
    op.create_table(
        'review_analytics',
        sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('review_id', sa.UUID(), nullable=False),
        sa.Column('sentiment_scores', sa.JSON(), nullable=False, server_default='{}'),
        sa.Column('topic_analysis', sa.JSON(), nullable=False, server_default='{}'),
        sa.Column('key_phrases', sa.JSON(), nullable=False, server_default='[]'),
        sa.Column('processed_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['review_id'], ['reviews.id'], ondelete='CASCADE')
    )

    # Create indexes
    op.create_index('idx_reviews_user_id', 'reviews', ['user_id'])
    op.create_index('idx_reviews_product_id', 'reviews', ['product_id'])
    op.create_index('idx_reviews_sentiment', 'reviews', ['sentiment'])
    op.create_index('idx_reviews_created_at', 'reviews', ['created_at'])
    op.create_index('idx_review_analytics_review_id', 'review_analytics', ['review_id'])

    # Create trigger for updating review timestamps
    op.execute("""
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $$ language 'plpgsql';
    """)

    op.execute("""
        CREATE TRIGGER update_reviews_updated_at
            BEFORE UPDATE ON reviews
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
    """)

def downgrade():
    # Drop triggers
    op.execute("DROP TRIGGER IF EXISTS update_reviews_updated_at ON reviews;")
    op.execute("DROP FUNCTION IF EXISTS update_updated_at_column;")

    # Drop indexes
    op.drop_index('idx_review_analytics_review_id')
    op.drop_index('idx_reviews_created_at')
    op.drop_index('idx_reviews_sentiment')
    op.drop_index('idx_reviews_product_id')
    op.drop_index('idx_reviews_user_id')

    # Drop tables
    op.drop_table('review_analytics')
    op.drop_table('reviews')
```

## Migration Best Practices

1. **Atomic Changes**
   - Each migration should be self-contained
   - Include both upgrade and downgrade paths
   - Test migrations in isolation

2. **Data Preservation**
   - Always provide data migration paths
   - Back up data before migrations
   - Handle edge cases and errors

3. **Performance**
   - Use batching for large data migrations
   - Consider table locks and timing
   - Monitor system resources

4. **Testing**
   - Test both upgrade and downgrade paths
   - Verify data integrity
   - Test with realistic data volumes

## Running Migrations in Production

1. **Preparation**
   ```bash
   # Create backup
   pg_dump -Fc -f pre_migration_backup.dump ai_review_engine
   ```

2. **Verification**
   ```bash
   # Check current version
   alembic current

   # View pending migrations
   alembic heads
   ```

3. **Execution**
   ```bash
   # Apply migrations with output
   alembic upgrade head --sql > migration.sql
   psql -d ai_review_engine -f migration.sql
   ```

4. **Validation**
   ```bash
   # Verify database state
   alembic check
   ```

## Troubleshooting

### Common Issues

1. **Migration Conflicts**
   ```bash
   # View migration history
   alembic history --verbose

   # Merge heads if needed
   alembic merge heads -m "merge_migration_heads"
   ```

2. **Failed Migrations**
   ```bash
   # Get migration information
   alembic show <revision>

   # Manual cleanup if needed
   alembic downgrade <last_good_revision>
   ```

3. **Data Inconsistencies**
   ```sql
   -- Check for orphaned records
   SELECT * FROM reviews 
   WHERE user_id NOT IN (SELECT id FROM users);

   -- Fix inconsistencies
   DELETE FROM reviews 
   WHERE user_id NOT IN (SELECT id FROM users);
   ```

## Migration Hooks

```python
# env.py hooks for migration customization
def run_migrations_online():
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # Custom settings
            compare_type=True,
            compare_server_default=True,
            include_schemas=True,
            version_table='alembic_version',
            # Callbacks
            on_version_apply=on_version_apply,
        )

def on_version_apply(context, version, heads):
    """Called after each migration is applied."""
    # Log migration
    logger.info(f"Applied migration {version}")
    
    # Update application version
    update_app_version(version)
    
    # Clear caches
    clear_related_caches()
```