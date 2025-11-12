"""add_performance_indexes

Revision ID: fb9e61f33cbf
Revises: 2ec489a3eb6b
Create Date: 2025-11-12 13:11:52.869695

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'fb9e61f33cbf'
down_revision = '2ec489a3eb6b'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """
    Add performance indexes to all tables.

    Indexes are strategically placed on:
    - Frequently filtered columns (status, domain)
    - Foreign key columns
    - Sorting columns (scores, dates)
    - Composite indexes for common query patterns

    Expected performance improvement: 5-10x on filtered queries
    """

    # ============================================================================
    # EXPERIMENT TABLE INDEXES
    # ============================================================================

    # Single-column indexes
    op.create_index(
        'idx_experiments_status',
        'experiments',
        ['status'],
        unique=False
    )

    op.create_index(
        'idx_experiments_domain',
        'experiments',
        ['domain'],
        unique=False
    )

    op.create_index(
        'idx_experiments_hypothesis_id',
        'experiments',
        ['hypothesis_id'],
        unique=False
    )

    op.create_index(
        'idx_experiments_created_at',
        'experiments',
        ['created_at'],
        unique=False
    )

    # Composite indexes for common query patterns
    op.create_index(
        'idx_experiments_domain_status',
        'experiments',
        ['domain', 'status'],
        unique=False
    )

    op.create_index(
        'idx_experiments_status_created',
        'experiments',
        ['status', 'created_at'],
        unique=False
    )

    # ============================================================================
    # HYPOTHESIS TABLE INDEXES
    # ============================================================================

    # Single-column indexes
    op.create_index(
        'idx_hypotheses_domain',
        'hypotheses',
        ['domain'],
        unique=False
    )

    op.create_index(
        'idx_hypotheses_status',
        'hypotheses',
        ['status'],
        unique=False
    )

    op.create_index(
        'idx_hypotheses_novelty_score',
        'hypotheses',
        ['novelty_score'],
        unique=False
    )

    op.create_index(
        'idx_hypotheses_testability_score',
        'hypotheses',
        ['testability_score'],
        unique=False
    )

    op.create_index(
        'idx_hypotheses_created_at',
        'hypotheses',
        ['created_at'],
        unique=False
    )

    # Composite indexes for common query patterns
    op.create_index(
        'idx_hypotheses_domain_status',
        'hypotheses',
        ['domain', 'status'],
        unique=False
    )

    op.create_index(
        'idx_hypotheses_domain_novelty',
        'hypotheses',
        ['domain', 'novelty_score'],
        unique=False
    )

    # ============================================================================
    # PAPER TABLE INDEXES
    # ============================================================================

    # Single-column indexes
    op.create_index(
        'idx_papers_domain',
        'papers',
        ['domain'],
        unique=False
    )

    op.create_index(
        'idx_papers_source',
        'papers',
        ['source'],
        unique=False
    )

    op.create_index(
        'idx_papers_relevance_score',
        'papers',
        ['relevance_score'],
        unique=False
    )

    op.create_index(
        'idx_papers_publication_date',
        'papers',
        ['publication_date'],
        unique=False
    )

    op.create_index(
        'idx_papers_created_at',
        'papers',
        ['created_at'],
        unique=False
    )

    # Composite indexes for common query patterns
    op.create_index(
        'idx_papers_domain_relevance',
        'papers',
        ['domain', 'relevance_score'],
        unique=False
    )

    op.create_index(
        'idx_papers_domain_pubdate',
        'papers',
        ['domain', 'publication_date'],
        unique=False
    )

    # ============================================================================
    # RESULT TABLE INDEXES
    # ============================================================================

    # Single-column indexes
    op.create_index(
        'idx_results_experiment_id',
        'results',
        ['experiment_id'],
        unique=False
    )

    op.create_index(
        'idx_results_p_value',
        'results',
        ['p_value'],
        unique=False
    )

    op.create_index(
        'idx_results_supports_hypothesis',
        'results',
        ['supports_hypothesis'],
        unique=False
    )

    op.create_index(
        'idx_results_created_at',
        'results',
        ['created_at'],
        unique=False
    )

    # ============================================================================
    # RESEARCH_SESSION TABLE INDEXES
    # ============================================================================

    # Single-column indexes
    op.create_index(
        'idx_research_sessions_status',
        'research_sessions',
        ['status'],
        unique=False
    )

    op.create_index(
        'idx_research_sessions_domain',
        'research_sessions',
        ['domain'],
        unique=False
    )

    op.create_index(
        'idx_research_sessions_created_at',
        'research_sessions',
        ['created_at'],
        unique=False
    )

    # Composite indexes for common query patterns
    op.create_index(
        'idx_research_sessions_domain_status',
        'research_sessions',
        ['domain', 'status'],
        unique=False
    )

    # ============================================================================
    # AGENT_RECORD TABLE INDEXES
    # ============================================================================

    # Single-column indexes
    op.create_index(
        'idx_agents_agent_type',
        'agents',
        ['agent_type'],
        unique=False
    )

    op.create_index(
        'idx_agents_status',
        'agents',
        ['status'],
        unique=False
    )

    op.create_index(
        'idx_agents_created_at',
        'agents',
        ['created_at'],
        unique=False
    )

    # Composite indexes for common query patterns
    op.create_index(
        'idx_agents_type_status',
        'agents',
        ['agent_type', 'status'],
        unique=False
    )


def downgrade() -> None:
    """Remove all performance indexes."""

    # Agent indexes
    op.drop_index('idx_agents_type_status', table_name='agents')
    op.drop_index('idx_agents_created_at', table_name='agents')
    op.drop_index('idx_agents_status', table_name='agents')
    op.drop_index('idx_agents_agent_type', table_name='agents')

    # Research session indexes
    op.drop_index('idx_research_sessions_domain_status', table_name='research_sessions')
    op.drop_index('idx_research_sessions_created_at', table_name='research_sessions')
    op.drop_index('idx_research_sessions_domain', table_name='research_sessions')
    op.drop_index('idx_research_sessions_status', table_name='research_sessions')

    # Result indexes
    op.drop_index('idx_results_created_at', table_name='results')
    op.drop_index('idx_results_supports_hypothesis', table_name='results')
    op.drop_index('idx_results_p_value', table_name='results')
    op.drop_index('idx_results_experiment_id', table_name='results')

    # Paper indexes
    op.drop_index('idx_papers_domain_pubdate', table_name='papers')
    op.drop_index('idx_papers_domain_relevance', table_name='papers')
    op.drop_index('idx_papers_created_at', table_name='papers')
    op.drop_index('idx_papers_publication_date', table_name='papers')
    op.drop_index('idx_papers_relevance_score', table_name='papers')
    op.drop_index('idx_papers_source', table_name='papers')
    op.drop_index('idx_papers_domain', table_name='papers')

    # Hypothesis indexes
    op.drop_index('idx_hypotheses_domain_novelty', table_name='hypotheses')
    op.drop_index('idx_hypotheses_domain_status', table_name='hypotheses')
    op.drop_index('idx_hypotheses_created_at', table_name='hypotheses')
    op.drop_index('idx_hypotheses_testability_score', table_name='hypotheses')
    op.drop_index('idx_hypotheses_novelty_score', table_name='hypotheses')
    op.drop_index('idx_hypotheses_status', table_name='hypotheses')
    op.drop_index('idx_hypotheses_domain', table_name='hypotheses')

    # Experiment indexes
    op.drop_index('idx_experiments_status_created', table_name='experiments')
    op.drop_index('idx_experiments_domain_status', table_name='experiments')
    op.drop_index('idx_experiments_created_at', table_name='experiments')
    op.drop_index('idx_experiments_hypothesis_id', table_name='experiments')
    op.drop_index('idx_experiments_domain', table_name='experiments')
    op.drop_index('idx_experiments_status', table_name='experiments')
