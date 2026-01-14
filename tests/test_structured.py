"""Tests for StructuredMemory model and related functionality."""

from engram.models import (
    ConfidenceScore,
    Episode,
    ExtractionMethod,
    Negation,
    Person,
    Preference,
    QuickExtracts,
    ResolvedDate,
    StructuredMemory,
)


class TestStructuredMemoryModel:
    """Tests for the StructuredMemory Pydantic model."""

    def test_create_minimal(self):
        """StructuredMemory can be created with minimal required fields."""
        struct = StructuredMemory(
            source_episode_id="ep_123",
            user_id="user_1",
        )
        assert struct.id.startswith("struct_")
        assert struct.source_episode_id == "ep_123"
        assert struct.user_id == "user_1"
        assert struct.org_id is None
        assert struct.summary == ""
        assert struct.keywords == []
        assert struct.consolidated is False

    def test_create_with_all_fields(self):
        """StructuredMemory can be created with all extraction fields."""
        struct = StructuredMemory(
            source_episode_id="ep_123",
            user_id="user_1",
            org_id="org_1",
            emails=["john@example.com"],
            phones=["+1-555-1234"],
            urls=["https://example.com"],
            dates=[ResolvedDate(raw="next Tuesday", resolved="2025-01-14")],
            people=[Person(name="John Smith", role="manager")],
            organizations=["Acme Corp"],
            locations=["New York"],
            preferences=[Preference(topic="database", value="PostgreSQL")],
            negations=[Negation(content="doesn't use MongoDB", pattern="MongoDB")],
            summary="User discussed database preferences.",
            keywords=["database", "PostgreSQL"],
            embedding=[0.1] * 1536,
        )
        assert struct.emails == ["john@example.com"]
        assert struct.phones == ["+1-555-1234"]
        assert len(struct.dates) == 1
        assert struct.dates[0].resolved == "2025-01-14"
        assert struct.people[0].name == "John Smith"
        assert struct.negations[0].pattern == "MongoDB"

    def test_from_episode_factory(self):
        """from_episode factory method creates StructuredMemory correctly."""
        struct = StructuredMemory.from_episode(
            source_episode_id="ep_123",
            user_id="user_1",
            emails=["test@example.com"],
            phones=[],
            urls=[],
            dates=[],
            people=[Person(name="Alice")],
            organizations=[],
            locations=[],
            preferences=[],
            negations=[],
            summary="Test summary",
            keywords=["test"],
        )
        assert struct.source_episode_id == "ep_123"
        assert struct.emails == ["test@example.com"]
        assert struct.people[0].name == "Alice"
        # Confidence should be weighted based on extractions
        assert 0.8 <= struct.confidence.value <= 0.95

    def test_confidence_weighting_deterministic_only(self):
        """Confidence is higher (0.9) when only regex extractions present."""
        struct = StructuredMemory.from_episode(
            source_episode_id="ep_123",
            user_id="user_1",
            emails=["test@example.com"],
            phones=["+1-555-1234"],
            urls=["https://example.com"],
            # No LLM extractions
        )
        # With only deterministic extractions, confidence should be ~0.9
        assert struct.confidence.value >= 0.85

    def test_confidence_weighting_llm_only(self):
        """Confidence is 0.8 when only LLM extractions present."""
        struct = StructuredMemory.from_episode(
            source_episode_id="ep_123",
            user_id="user_1",
            # No deterministic extractions
            people=[Person(name="Alice")],
            summary="Test",
        )
        # With only LLM extractions, confidence should be ~0.8
        assert 0.75 <= struct.confidence.value <= 0.85

    def test_to_embedding_text(self):
        """to_embedding_text generates searchable text representation."""
        struct = StructuredMemory(
            source_episode_id="ep_123",
            user_id="user_1",
            summary="User prefers PostgreSQL.",
            keywords=["database", "PostgreSQL"],
            people=[Person(name="John", role="DBA")],
            organizations=["Acme Corp"],
            preferences=[Preference(topic="database", value="PostgreSQL")],
            negations=[Negation(content="Not MongoDB", pattern="MongoDB")],
            emails=["john@example.com"],
        )
        text = struct.to_embedding_text()
        assert "User prefers PostgreSQL" in text
        assert "Keywords:" in text
        assert "PostgreSQL" in text
        assert "John (DBA)" in text
        assert "Acme Corp" in text
        assert "database: PostgreSQL" in text
        assert "Not MongoDB" in text
        assert "john@example.com" in text

    def test_str_representation(self):
        """String representation shows summary preview and counts."""
        struct = StructuredMemory(
            source_episode_id="ep_123",
            user_id="user_1",
            summary="This is a very long summary that should be truncated in display.",
            emails=["a@b.com", "c@d.com"],
            phones=["+1-555-1234"],
        )
        s = str(struct)
        assert "This is a very long summary" in s
        assert "3 extracts" in s  # 2 emails + 1 phone


class TestSubModels:
    """Tests for StructuredMemory sub-models."""

    def test_resolved_date(self):
        """ResolvedDate stores raw and resolved date."""
        date = ResolvedDate(
            raw="next Tuesday",
            resolved="2025-01-14",
            context="meeting scheduled",
        )
        assert date.raw == "next Tuesday"
        assert date.resolved == "2025-01-14"
        assert date.context == "meeting scheduled"

    def test_person(self):
        """Person stores name and optional role."""
        person = Person(name="John Smith", role="manager", context="mentioned as boss")
        assert person.name == "John Smith"
        assert person.role == "manager"
        assert person.context == "mentioned as boss"

    def test_person_minimal(self):
        """Person can be created with just name."""
        person = Person(name="Alice")
        assert person.name == "Alice"
        assert person.role is None
        assert person.context == ""

    def test_preference(self):
        """Preference stores topic, value, and sentiment."""
        pref = Preference(topic="database", value="PostgreSQL", sentiment="positive")
        assert pref.topic == "database"
        assert pref.value == "PostgreSQL"
        assert pref.sentiment == "positive"

    def test_negation(self):
        """Negation stores content and pattern for filtering."""
        neg = Negation(
            content="User doesn't use MongoDB",
            pattern="MongoDB",
            context="correction",
        )
        assert neg.content == "User doesn't use MongoDB"
        assert neg.pattern == "MongoDB"
        assert neg.context == "correction"


class TestQuickExtracts:
    """Tests for QuickExtracts on Episode."""

    def test_quick_extracts_creation(self):
        """QuickExtracts stores immediate regex results."""
        qe = QuickExtracts(
            emails=["test@example.com", "other@example.com"],
            phones=["+1-555-1234"],
            urls=["https://example.com"],
        )
        assert len(qe.emails) == 2
        assert len(qe.phones) == 1
        assert len(qe.urls) == 1

    def test_quick_extracts_empty(self):
        """QuickExtracts defaults to empty lists."""
        qe = QuickExtracts()
        assert qe.emails == []
        assert qe.phones == []
        assert qe.urls == []

    def test_episode_with_quick_extracts(self):
        """Episode can have quick_extracts attached."""
        qe = QuickExtracts(emails=["test@example.com"])
        ep = Episode(
            content="My email is test@example.com",
            role="user",
            user_id="user_1",
            embedding=[0.1] * 1536,
            quick_extracts=qe,
        )
        assert ep.quick_extracts is not None
        assert ep.quick_extracts.emails == ["test@example.com"]

    def test_episode_structured_fields(self):
        """Episode has structured tracking fields."""
        ep = Episode(
            content="Test",
            role="user",
            user_id="user_1",
            embedding=[0.1] * 1536,
            structured=True,
            structured_into="struct_123",
        )
        assert ep.structured is True
        assert ep.structured_into == "struct_123"


class TestStructuredMemoryConsolidation:
    """Tests for StructuredMemory consolidation tracking."""

    def test_consolidated_defaults_false(self):
        """consolidated defaults to False."""
        struct = StructuredMemory(
            source_episode_id="ep_123",
            user_id="user_1",
        )
        assert struct.consolidated is False
        assert struct.consolidated_into is None

    def test_consolidated_can_be_set(self):
        """consolidated and consolidated_into can be set."""
        struct = StructuredMemory(
            source_episode_id="ep_123",
            user_id="user_1",
            consolidated=True,
            consolidated_into="sem_456",
        )
        assert struct.consolidated is True
        assert struct.consolidated_into == "sem_456"


class TestStructuredMemoryConfidence:
    """Tests for StructuredMemory confidence scoring."""

    def test_default_confidence(self):
        """Default confidence is 0.8 (LLM extraction base)."""
        struct = StructuredMemory(
            source_episode_id="ep_123",
            user_id="user_1",
        )
        assert struct.confidence.value == 0.8
        assert struct.confidence.extraction_method == ExtractionMethod.INFERRED
        assert struct.confidence.extraction_base == 0.8

    def test_custom_confidence(self):
        """Custom confidence can be set."""
        conf = ConfidenceScore(
            value=0.95,
            extraction_method=ExtractionMethod.EXTRACTED,
            extraction_base=0.95,
        )
        struct = StructuredMemory(
            source_episode_id="ep_123",
            user_id="user_1",
            confidence=conf,
        )
        assert struct.confidence.value == 0.95


class TestStructuredMemoryJSON:
    """Tests for StructuredMemory serialization."""

    def test_model_dump_json(self):
        """StructuredMemory can be serialized to JSON-compatible dict."""
        struct = StructuredMemory(
            source_episode_id="ep_123",
            user_id="user_1",
            summary="Test summary",
            people=[Person(name="John")],
            negations=[Negation(content="Not X", pattern="X")],
        )
        data = struct.model_dump(mode="json")
        assert data["source_episode_id"] == "ep_123"
        assert data["summary"] == "Test summary"
        assert data["people"][0]["name"] == "John"
        assert data["negations"][0]["pattern"] == "X"
        # derived_at should be ISO string
        assert isinstance(data["derived_at"], str)

    def test_model_validate(self):
        """StructuredMemory can be deserialized from dict."""
        data = {
            "id": "struct_test",
            "source_episode_id": "ep_123",
            "user_id": "user_1",
            "summary": "Restored summary",
            "keywords": ["test"],
            "emails": [],
            "phones": [],
            "urls": [],
            "dates": [],
            "people": [{"name": "Alice", "role": None, "context": ""}],
            "organizations": [],
            "locations": [],
            "preferences": [],
            "negations": [],
            "derived_at": "2025-01-01T00:00:00+00:00",
            "confidence": {
                "value": 0.8,
                "extraction_method": "inferred",
                "extraction_base": 0.8,
            },
            "consolidated": False,
            "consolidated_into": None,
            "embedding": None,
        }
        struct = StructuredMemory.model_validate(data)
        assert struct.id == "struct_test"
        assert struct.summary == "Restored summary"
        assert struct.people[0].name == "Alice"
