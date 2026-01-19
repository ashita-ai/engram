"""Unit tests for Engram memory models."""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from engram.models import (
    AuditEntry,
    ConfidenceScore,
    Episode,
    ExtractionMethod,
    ProceduralMemory,
    SemanticMemory,
    generate_id,
)


class TestGenerateId:
    """Tests for the generate_id function."""

    def test_generates_unique_ids(self):
        """Each call should produce a unique ID."""
        ids = [generate_id("test") for _ in range(100)]
        assert len(ids) == len(set(ids))

    def test_includes_prefix(self):
        """Generated ID should start with the prefix."""
        assert generate_id("ep").startswith("ep_")
        assert generate_id("fact").startswith("fact_")
        assert generate_id("sem").startswith("sem_")

    def test_consistent_format(self):
        """ID should be prefix_12chars."""
        id_ = generate_id("test")
        prefix, suffix = id_.split("_")
        assert prefix == "test"
        assert len(suffix) == 12


class TestExtractionMethod:
    """Tests for the ExtractionMethod enum."""

    def test_enum_values(self):
        """Verify expected enum values exist."""
        assert ExtractionMethod.VERBATIM.value == "verbatim"
        assert ExtractionMethod.EXTRACTED.value == "extracted"
        assert ExtractionMethod.INFERRED.value == "inferred"

    def test_string_enum(self):
        """ExtractionMethod should be usable as string."""
        method = ExtractionMethod.VERBATIM
        assert str(method) == "ExtractionMethod.VERBATIM"
        assert method.value == "verbatim"


class TestConfidenceScore:
    """Tests for ConfidenceScore model."""

    def test_for_verbatim(self):
        """Verbatim extraction should have confidence 1.0."""
        score = ConfidenceScore.for_verbatim()
        assert score.value == 1.0
        assert score.extraction_method == ExtractionMethod.VERBATIM
        assert score.extraction_base == 1.0
        assert score.supporting_episodes == 1

    def test_for_extracted(self):
        """Pattern extraction should have confidence 0.9."""
        score = ConfidenceScore.for_extracted()
        assert score.value == 0.9
        assert score.extraction_method == ExtractionMethod.EXTRACTED
        assert score.extraction_base == 0.9

    def test_for_extracted_with_sources(self):
        """Multiple supporting sources should be recorded."""
        score = ConfidenceScore.for_extracted(supporting_episodes=3)
        assert score.supporting_episodes == 3

    def test_for_inferred(self):
        """LLM inference should have default confidence 0.6."""
        score = ConfidenceScore.for_inferred()
        assert score.value == 0.6
        assert score.extraction_method == ExtractionMethod.INFERRED
        assert score.extraction_base == 0.6

    def test_for_inferred_custom_confidence(self):
        """Custom confidence can be specified for inferred."""
        score = ConfidenceScore.for_inferred(confidence=0.75)
        assert score.value == 0.75

    def test_value_bounds(self):
        """Confidence value must be between 0 and 1."""
        with pytest.raises(ValidationError):
            ConfidenceScore(
                value=1.5,
                extraction_method=ExtractionMethod.VERBATIM,
                extraction_base=1.0,
            )
        with pytest.raises(ValidationError):
            ConfidenceScore(
                value=-0.1,
                extraction_method=ExtractionMethod.VERBATIM,
                extraction_base=1.0,
            )

    def test_contradictions_non_negative(self):
        """Contradictions count cannot be negative."""
        with pytest.raises(ValidationError):
            ConfidenceScore(
                value=0.9,
                extraction_method=ExtractionMethod.EXTRACTED,
                extraction_base=0.9,
                contradictions=-1,
            )

    def test_explain_basic(self):
        """explain() should return human-readable description."""
        score = ConfidenceScore.for_extracted(supporting_episodes=2)
        explanation = score.explain()
        assert "0.90:" in explanation
        assert "extracted" in explanation
        assert "2 sources" in explanation

    def test_explain_with_contradictions(self):
        """explain() should include contradictions if present."""
        score = ConfidenceScore(
            value=0.8,
            extraction_method=ExtractionMethod.EXTRACTED,
            extraction_base=0.9,
            contradictions=2,
        )
        explanation = score.explain()
        assert "2 contradictions" in explanation

    def test_explain_singular_source(self):
        """explain() should use singular 'source' for 1 source."""
        score = ConfidenceScore.for_extracted(supporting_episodes=1)
        explanation = score.explain()
        assert "1 source" in explanation
        assert "1 sources" not in explanation

    def test_time_ago_just_now(self):
        """Recent timestamps should show 'just now'."""
        score = ConfidenceScore.for_verbatim()
        explanation = score.explain()
        assert "just now" in explanation

    def test_time_ago_days(self):
        """Timestamps days ago should show 'X days ago'."""
        three_days_ago = datetime.now(UTC) - timedelta(days=3)
        score = ConfidenceScore(
            value=0.9,
            extraction_method=ExtractionMethod.EXTRACTED,
            extraction_base=0.9,
            last_confirmed=three_days_ago,
        )
        explanation = score.explain()
        assert "3 days ago" in explanation

    def test_extra_fields_forbidden(self):
        """Extra fields should raise an error."""
        with pytest.raises(ValidationError):
            ConfidenceScore(
                value=0.9,
                extraction_method=ExtractionMethod.EXTRACTED,
                extraction_base=0.9,
                unknown_field="test",
            )

    def test_recompute_basic(self):
        """recompute() should calculate weighted score."""
        score = ConfidenceScore(
            value=0.0,  # Will be recomputed
            extraction_method=ExtractionMethod.EXTRACTED,
            extraction_base=0.9,
            supporting_episodes=1,
            verified=False,
        )
        score.recompute()
        # extraction: 0.9 * 0.5 = 0.45
        # corroboration: 0.5 * 0.25 = 0.125 (1 source = 0.5)
        # recency: ~1.0 * 0.15 = 0.15 (just now)
        # verification: 0.0 * 0.10 = 0.0
        # Total: ~0.725
        assert 0.70 <= score.value <= 0.75

    def test_recompute_with_verification(self):
        """Verification should add to score."""
        score = ConfidenceScore(
            value=0.0,
            extraction_method=ExtractionMethod.EXTRACTED,
            extraction_base=0.9,
            verified=True,
        )
        score.recompute()
        # With verification: adds 0.10
        assert score.value > 0.75

    def test_recompute_multiple_sources(self):
        """Multiple sources should increase corroboration score."""
        score_1 = ConfidenceScore.for_extracted(supporting_episodes=1)
        score_10 = ConfidenceScore.for_extracted(supporting_episodes=10)
        score_1.recompute()
        score_10.recompute()
        # 10 sources should have higher confidence than 1
        assert score_10.value > score_1.value

    def test_recompute_with_contradictions(self):
        """Contradictions should reduce confidence."""
        score = ConfidenceScore(
            value=0.9,
            extraction_method=ExtractionMethod.EXTRACTED,
            extraction_base=0.9,
            contradictions=2,
        )
        score.recompute()
        # 2 contradictions = 0.9^2 = 0.81 penalty
        assert score.value < 0.7

    def test_recompute_recency_decay(self):
        """Old confirmations should have lower recency score."""
        old_date = datetime.now(UTC) - timedelta(days=365)
        score = ConfidenceScore(
            value=0.9,
            extraction_method=ExtractionMethod.EXTRACTED,
            extraction_base=0.9,
            last_confirmed=old_date,
        )
        score.recompute()
        # 365 days = half-life, so recency contribution halved
        fresh_score = ConfidenceScore.for_extracted()
        fresh_score.recompute()
        assert score.value < fresh_score.value

    def test_recompute_verbatim_high(self):
        """Verbatim extraction should have high base score."""
        score = ConfidenceScore.for_verbatim()
        score.verified = True
        score.recompute()
        # extraction_base=1.0, verified=True
        assert score.value >= 0.75

    def test_recompute_returns_self(self):
        """recompute() should return self for chaining."""
        score = ConfidenceScore.for_extracted()
        result = score.recompute()
        assert result is score

    def test_recompute_custom_weights(self):
        """Custom weights should affect score calculation."""
        score = ConfidenceScore(
            value=0.0,
            extraction_method=ExtractionMethod.EXTRACTED,
            extraction_base=0.9,
            verified=True,
        )
        # Heavy weight on verification
        score.recompute(
            extraction_weight=0.1,
            corroboration_weight=0.1,
            recency_weight=0.1,
            verification_weight=0.7,
        )
        # verification=1.0 * 0.7 = 0.7 dominates
        assert score.value >= 0.7

    def test_recompute_with_weights(self):
        """recompute_with_weights should use ConfidenceWeights config."""
        from engram.config import ConfidenceWeights

        weights = ConfidenceWeights(
            extraction=0.6,
            corroboration=0.2,
            recency=0.1,
            verification=0.1,
            decay_half_life_days=180,
            contradiction_penalty=0.20,
        )
        score = ConfidenceScore.for_extracted()
        score.recompute_with_weights(weights)
        # extraction_base=0.9 * 0.6 = 0.54 dominates
        assert 0.5 <= score.value <= 0.8

    def test_recompute_custom_contradiction_penalty(self):
        """Custom contradiction penalty should affect score."""
        score = ConfidenceScore(
            value=0.9,
            extraction_method=ExtractionMethod.EXTRACTED,
            extraction_base=0.9,
            contradictions=1,
        )
        # 20% penalty per contradiction
        score.recompute(contradiction_penalty=0.20)
        # With 1 contradiction: multiplier = 0.8
        assert score.value < 0.65

    def test_explain_shows_verified(self):
        """explain() should show verified status when true."""
        score = ConfidenceScore.for_extracted()
        score.verified = True
        explanation = score.explain()
        assert "verified" in explanation

    def test_explain_hides_verified_when_false(self):
        """explain() should not show verified when false."""
        score = ConfidenceScore.for_extracted()
        score.verified = False
        explanation = score.explain()
        assert "verified" not in explanation


class TestEpisode:
    """Tests for Episode model."""

    def test_create_episode(self):
        """Basic episode creation."""
        ep = Episode(content="Hello world", role="user", user_id="user_123")
        assert ep.content == "Hello world"
        assert ep.role == "user"
        assert ep.user_id == "user_123"
        assert ep.id.startswith("ep_")

    def test_default_timestamp(self):
        """Episodes should get a default timestamp."""
        ep = Episode(content="Test", role="user", user_id="user_123")
        assert ep.timestamp is not None
        assert isinstance(ep.timestamp, datetime)

    def test_default_importance(self):
        """Default importance should be 0.5."""
        ep = Episode(content="Test", role="user", user_id="user_123")
        assert ep.importance == 0.5

    def test_importance_bounds(self):
        """Importance must be between 0 and 1."""
        with pytest.raises(ValidationError):
            Episode(content="Test", role="user", user_id="user_123", importance=1.5)

    def test_str_representation(self):
        """String representation should show role and content preview."""
        ep = Episode(content="Short message", role="user", user_id="user_123")
        assert "user" in str(ep)
        assert "Short message" in str(ep)

    def test_str_long_content_truncated(self):
        """Long content should be truncated in string representation."""
        long_content = "x" * 100
        ep = Episode(content=long_content, role="assistant", user_id="user_123")
        assert "..." in str(ep)
        assert len(str(ep)) < 100

    def test_optional_org_id(self):
        """org_id should be optional."""
        ep = Episode(content="Test", role="user", user_id="user_123")
        assert ep.org_id is None
        ep_with_org = Episode(content="Test", role="user", user_id="user_123", org_id="org_456")
        assert ep_with_org.org_id == "org_456"

    def test_optional_session_id(self):
        """session_id should be optional."""
        ep = Episode(content="Test", role="user", user_id="user_123")
        assert ep.session_id is None

    def test_optional_embedding(self):
        """embedding should be optional."""
        ep = Episode(content="Test", role="user", user_id="user_123")
        assert ep.embedding is None
        ep_with_embedding = Episode(
            content="Test", role="user", user_id="user_123", embedding=[0.1, 0.2, 0.3]
        )
        assert ep_with_embedding.embedding == [0.1, 0.2, 0.3]


class TestSemanticMemory:
    """Tests for SemanticMemory model."""

    def test_create_semantic_memory(self):
        """Basic semantic memory creation."""
        mem = SemanticMemory(
            content="User works in software development",
            user_id="user_123",
        )
        assert mem.content == "User works in software development"
        assert mem.id.startswith("sem_")

    def test_default_confidence_is_inferred(self):
        """Default confidence should be for inferred content."""
        mem = SemanticMemory(content="Test", user_id="user_123")
        assert mem.confidence.value == 0.6
        assert mem.confidence.extraction_method == ExtractionMethod.INFERRED

    def test_default_consolidation_strength(self):
        """Default consolidation_strength should be 0.0 (newly created)."""
        mem = SemanticMemory(content="Test", user_id="user_123")
        assert mem.consolidation_strength == 0.0

    def test_add_link(self):
        """add_link should add unique related IDs."""
        mem = SemanticMemory(content="Test", user_id="user_123")
        mem.add_link("mem_1")
        mem.add_link("mem_2")
        mem.add_link("mem_1")  # Duplicate
        assert mem.related_ids == ["mem_1", "mem_2"]

    def test_strengthen(self):
        """strengthen should increment score and passes."""
        mem = SemanticMemory(content="Test", user_id="user_123")
        assert mem.consolidation_strength == 0.0
        assert mem.consolidation_passes == 0  # Fresh memory, no consolidation yet
        mem.strengthen()
        assert mem.consolidation_strength == 0.1
        assert mem.consolidation_passes == 1  # After first strengthen

    def test_strengthen_caps_at_one(self):
        """Consolidation strength should cap at 1.0."""
        mem = SemanticMemory(content="Test", user_id="user_123", consolidation_strength=0.95)
        mem.strengthen(delta=0.1)
        assert mem.consolidation_strength == 1.0

    def test_weaken(self):
        """weaken should decrement score."""
        mem = SemanticMemory(content="Test", user_id="user_123", consolidation_strength=0.5)
        mem.weaken()
        assert mem.consolidation_strength == 0.4

    def test_weaken_caps_at_zero(self):
        """Consolidation strength should cap at 0.0."""
        mem = SemanticMemory(content="Test", user_id="user_123", consolidation_strength=0.05)
        mem.weaken(delta=0.1)
        assert mem.consolidation_strength == 0.0

    def test_str_representation(self):
        """String representation should show content preview."""
        mem = SemanticMemory(content="Short content", user_id="user_123")
        assert "Short content" in str(mem)

    # A-MEM inspired tests
    def test_default_amem_fields(self):
        """A-MEM fields should have sensible defaults."""
        mem = SemanticMemory(content="Test", user_id="user_123")
        assert mem.keywords == []
        assert mem.tags == []
        assert mem.context == ""
        assert mem.retrieval_count == 0
        assert mem.last_accessed is None
        assert mem.evolution_history == []

    def test_create_with_amem_fields(self):
        """Can create memory with A-MEM fields."""
        mem = SemanticMemory(
            content="User prefers Python",
            user_id="user_123",
            keywords=["python", "programming", "preference"],
            tags=["preference", "technical"],
            context="programming languages",
        )
        assert mem.keywords == ["python", "programming", "preference"]
        assert mem.tags == ["preference", "technical"]
        assert mem.context == "programming languages"

    def test_record_access(self):
        """record_access should update activation tracking."""
        mem = SemanticMemory(content="Test", user_id="user_123")
        assert mem.retrieval_count == 0
        assert mem.last_accessed is None

        mem.record_access()
        assert mem.retrieval_count == 1
        assert mem.last_accessed is not None

        first_access = mem.last_accessed
        mem.record_access()
        assert mem.retrieval_count == 2
        assert mem.last_accessed >= first_access

    def test_add_tag(self):
        """add_tag should add unique tags."""
        mem = SemanticMemory(content="Test", user_id="user_123")
        mem.add_tag("preference")
        mem.add_tag("technical")
        mem.add_tag("preference")  # Duplicate
        assert mem.tags == ["preference", "technical"]

    def test_add_keyword(self):
        """add_keyword should add unique keywords (case-insensitive)."""
        mem = SemanticMemory(content="Test", user_id="user_123")
        mem.add_keyword("Python")
        mem.add_keyword("coding")
        mem.add_keyword("python")  # Duplicate (case-insensitive)
        assert len(mem.keywords) == 2
        assert "Python" in mem.keywords
        assert "coding" in mem.keywords

    def test_evolve_tags(self):
        """evolve should add tags and record history."""
        mem = SemanticMemory(content="Test", user_id="user_123")
        mem.evolve(
            trigger_memory_id="sem_trigger",
            field="tags",
            new_value="preference,technical",
            reason="Related to user preferences",
        )
        assert "preference" in mem.tags
        assert "technical" in mem.tags
        assert len(mem.evolution_history) == 1
        assert mem.evolution_history[0].field_changed == "tags"
        assert mem.evolution_history[0].trigger_memory_id == "sem_trigger"

    def test_evolve_keywords(self):
        """evolve should add keywords and record history."""
        mem = SemanticMemory(content="Test", user_id="user_123")
        mem.evolve(
            trigger_memory_id="sem_trigger",
            field="keywords",
            new_value="python,coding",
        )
        assert "python" in mem.keywords
        assert "coding" in mem.keywords
        assert len(mem.evolution_history) == 1

    def test_evolve_context(self):
        """evolve should update context and record history."""
        mem = SemanticMemory(content="Test", user_id="user_123", context="programming")
        mem.evolve(
            trigger_memory_id="sem_trigger",
            field="context",
            new_value="data science",
        )
        assert "programming" in mem.context
        assert "data science" in mem.context
        assert len(mem.evolution_history) == 1

    def test_evolve_content_raises_error(self):
        """evolve should not allow changing content (immutable)."""
        mem = SemanticMemory(content="Original", user_id="user_123")
        with pytest.raises(ValueError, match="Cannot evolve content"):
            mem.evolve(
                trigger_memory_id="sem_trigger",
                field="content",
                new_value="Modified",
            )

    def test_evolution_history_records_old_value(self):
        """Evolution should record the old value for audit."""
        mem = SemanticMemory(content="Test", user_id="user_123", tags=["old_tag"])
        mem.evolve(
            trigger_memory_id="sem_trigger",
            field="tags",
            new_value="new_tag",
        )
        assert mem.evolution_history[0].old_value == "old_tag"
        assert mem.evolution_history[0].new_value == "new_tag"


class TestProceduralMemory:
    """Tests for ProceduralMemory model."""

    def test_create_procedural_memory(self):
        """Basic procedural memory creation."""
        mem = ProceduralMemory(
            content="User prefers Python code examples",
            user_id="user_123",
        )
        assert mem.content == "User prefers Python code examples"
        assert mem.id.startswith("proc_")

    def test_default_retrieval_count(self):
        """Default retrieval count should be 0."""
        mem = ProceduralMemory(content="Test", user_id="user_123")
        assert mem.retrieval_count == 0

    def test_reinforce(self):
        """reinforce() should increment retrieval count and update last_accessed."""
        mem = ProceduralMemory(content="Test", user_id="user_123")
        assert mem.last_accessed is None
        mem.reinforce()
        assert mem.retrieval_count == 1
        assert mem.last_accessed is not None
        first_access = mem.last_accessed
        mem.reinforce()
        assert mem.retrieval_count == 2
        assert mem.last_accessed >= first_access

    def test_record_access_alias(self):
        """record_access() should be an alias for reinforce()."""
        mem = ProceduralMemory(content="Test", user_id="user_123")
        mem.record_access()
        assert mem.retrieval_count == 1
        assert mem.last_accessed is not None

    def test_add_link(self):
        """add_link should add unique related IDs."""
        mem = ProceduralMemory(content="Test", user_id="user_123")
        mem.add_link("mem_1")
        mem.add_link("mem_2")
        mem.add_link("mem_1")  # Duplicate
        assert mem.related_ids == ["mem_1", "mem_2"]

    def test_trigger_context(self):
        """trigger_context should be configurable."""
        mem = ProceduralMemory(
            content="User likes detailed explanations",
            trigger_context="when explaining code",
            user_id="user_123",
        )
        assert mem.trigger_context == "when explaining code"


class TestAuditEntry:
    """Tests for AuditEntry model."""

    def test_create_audit_entry(self):
        """Basic audit entry creation."""
        entry = AuditEntry(event="encode", user_id="user_123")
        assert entry.event == "encode"
        assert entry.user_id == "user_123"
        assert entry.id.startswith("audit_")

    def test_for_encode(self):
        """for_encode factory should create encode audit entry."""
        entry = AuditEntry.for_encode(
            user_id="user_123",
            episode_id="ep_456",
            facts_count=3,
            duration_ms=150,
        )
        assert entry.event == "encode"
        assert entry.user_id == "user_123"
        assert entry.details["episode_id"] == "ep_456"
        assert entry.details["facts_count"] == 3
        assert entry.duration_ms == 150

    def test_for_recall(self):
        """for_recall factory should create recall audit entry."""
        entry = AuditEntry.for_recall(
            user_id="user_123",
            query_hash="abc123",
            results_count=5,
            memory_types=["episodic", "factual"],
        )
        assert entry.event == "recall"
        assert entry.details["query_hash"] == "abc123"
        assert entry.details["results_count"] == 5
        assert entry.details["memory_types"] == ["episodic", "factual"]

    def test_for_consolidate(self):
        """for_consolidate factory should create consolidate audit entry."""
        entry = AuditEntry.for_consolidate(
            user_id="user_123",
            episode_ids=["ep_1", "ep_2"],
            facts_created=2,
            links_created=1,
        )
        assert entry.event == "consolidate"
        assert entry.details["episode_ids"] == ["ep_1", "ep_2"]
        assert entry.details["facts_created"] == 2
        assert entry.details["links_created"] == 1

    def test_for_decay(self):
        """for_decay factory should create decay audit entry."""
        entry = AuditEntry.for_decay(
            user_id="user_123",
            memories_updated=10,
            memories_archived=2,
        )
        assert entry.event == "decay"
        assert entry.details["memories_updated"] == 10
        assert entry.details["memories_archived"] == 2

    def test_str_representation(self):
        """String representation should show event and user."""
        entry = AuditEntry(event="encode", user_id="user_123")
        assert "encode" in str(entry)
        assert "user_123" in str(entry)

    def test_optional_fields(self):
        """Optional fields should be None by default."""
        entry = AuditEntry(event="encode", user_id="user_123")
        assert entry.org_id is None
        assert entry.session_id is None
        assert entry.duration_ms is None
