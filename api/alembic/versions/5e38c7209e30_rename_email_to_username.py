"""rename email to username

Revision ID: 5e38c7209e30
Revises:
Create Date: 2025-04-21 14:00:36.279415

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = '5e38c7209e30'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column('user', 'email', new_column_name='username')


def downgrade() -> None:
    op.alter_column('user', 'username', new_column_name='email')
