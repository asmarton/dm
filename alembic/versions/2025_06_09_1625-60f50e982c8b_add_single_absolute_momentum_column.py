"""Add single absolute momentum column

Revision ID: 60f50e982c8b
Revises: f21decd75296
Create Date: 2025-06-09 16:25:28.054824

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import func

# revision identifiers, used by Alembic.
revision: str = '60f50e982c8b'
down_revision: Union[str, None] = 'f21decd75296'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('jobs', sa.Column('single_absolute_momentum', sa.String(), nullable=True))
    op.add_column(
        'jobs',
        sa.Column('created_at', sa.DateTime(), server_default=func.now(), nullable=False),
    )
    op.drop_column('jobs', 'createdAt')
    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('jobs', sa.Column('createdAt', sa.DATETIME(), nullable=False))
    op.drop_column('jobs', 'created_at')
    op.drop_column('jobs', 'single_absolute_momentum')
    # ### end Alembic commands ###
