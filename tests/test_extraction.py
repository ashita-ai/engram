"""Unit tests for Engram extraction layer."""

import pytest

from engram.extraction import (
    DateExtractor,
    EmailExtractor,
    ExtractionPipeline,
    Extractor,
    PhoneExtractor,
    URLExtractor,
    default_pipeline,
)
from engram.models import Episode, Fact
from engram.models.base import ExtractionMethod


def make_episode(content: str) -> Episode:
    """Create a test episode with given content."""
    return Episode(
        content=content,
        role="user",
        user_id="user_123",
    )


class TestExtractorBase:
    """Tests for Extractor base class."""

    def test_extractor_is_abstract(self):
        """Extractor cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Extractor()  # type: ignore[abstract]

    def test_create_fact_sets_confidence(self):
        """_create_fact should set EXTRACTED confidence."""
        extractor = EmailExtractor()
        episode = make_episode("test@example.com")
        fact = extractor._create_fact("test@example.com", episode)

        assert fact.confidence.extraction_method == ExtractionMethod.EXTRACTED
        assert fact.confidence.extraction_base == 0.9

    def test_create_fact_links_episode(self):
        """_create_fact should link to source episode."""
        extractor = EmailExtractor()
        episode = make_episode("test@example.com")
        fact = extractor._create_fact("test@example.com", episode)

        assert fact.source_episode_id == episode.id
        assert fact.user_id == episode.user_id

    def test_create_fact_inherits_org_id(self):
        """_create_fact should inherit org_id from episode."""
        extractor = EmailExtractor()
        episode = Episode(
            content="test@example.com",
            role="user",
            user_id="user_123",
            org_id="org_456",
        )
        fact = extractor._create_fact("test@example.com", episode)

        assert fact.org_id == "org_456"


class TestEmailExtractor:
    """Tests for EmailExtractor."""

    def test_extract_simple_email(self):
        """Should extract a simple email address."""
        extractor = EmailExtractor()
        episode = make_episode("Contact me at user@example.com please")
        facts = extractor.extract(episode)

        assert len(facts) == 1
        assert facts[0].content == "user@example.com"
        assert facts[0].category == "email"

    def test_extract_multiple_emails(self):
        """Should extract multiple email addresses."""
        extractor = EmailExtractor()
        episode = make_episode("Email alice@example.com or bob@example.org")
        facts = extractor.extract(episode)

        assert len(facts) == 2
        emails = {f.content for f in facts}
        assert emails == {"alice@example.com", "bob@example.org"}

    def test_extract_email_with_plus(self):
        """Should handle plus addressing."""
        extractor = EmailExtractor()
        episode = make_episode("Use user+tag@example.com")
        facts = extractor.extract(episode)

        assert len(facts) == 1
        assert facts[0].content == "user+tag@example.com"

    def test_extract_email_with_subdomain(self):
        """Should handle subdomains."""
        extractor = EmailExtractor()
        episode = make_episode("Email user@mail.example.co.uk")
        facts = extractor.extract(episode)

        assert len(facts) == 1
        assert facts[0].content == "user@mail.example.co.uk"

    def test_normalize_to_lowercase(self):
        """Should normalize emails to lowercase."""
        extractor = EmailExtractor()
        episode = make_episode("Email User@Example.COM")
        facts = extractor.extract(episode)

        assert facts[0].content == "user@example.com"

    def test_deduplicate_emails(self):
        """Should deduplicate repeated emails."""
        extractor = EmailExtractor()
        episode = make_episode("user@example.com and user@example.com again")
        facts = extractor.extract(episode)

        assert len(facts) == 1

    def test_no_match_returns_empty(self):
        """Should return empty list when no emails found."""
        extractor = EmailExtractor()
        episode = make_episode("No email here")
        facts = extractor.extract(episode)

        assert len(facts) == 0


class TestPhoneExtractor:
    """Tests for PhoneExtractor."""

    def test_extract_us_format_parens(self):
        """Should extract (123) 456-7890 format."""
        extractor = PhoneExtractor()
        episode = make_episode("Call me at (555) 123-4567")
        facts = extractor.extract(episode)

        assert len(facts) == 1
        assert facts[0].content == "5551234567"
        assert facts[0].category == "phone"

    def test_extract_us_format_dashes(self):
        """Should extract 123-456-7890 format."""
        extractor = PhoneExtractor()
        episode = make_episode("Phone: 555-123-4567")
        facts = extractor.extract(episode)

        assert len(facts) == 1
        assert facts[0].content == "5551234567"

    def test_extract_us_format_dots(self):
        """Should extract 123.456.7890 format."""
        extractor = PhoneExtractor()
        episode = make_episode("Fax: 555.123.4567")
        facts = extractor.extract(episode)

        assert len(facts) == 1
        assert facts[0].content == "5551234567"

    def test_extract_international(self):
        """Should extract international format with +."""
        extractor = PhoneExtractor()
        episode = make_episode("International: +1-555-123-4567")
        facts = extractor.extract(episode)

        assert len(facts) == 1
        assert facts[0].content == "+15551234567"

    def test_extract_ten_digits(self):
        """Should extract plain 10-digit number."""
        extractor = PhoneExtractor()
        episode = make_episode("Call 5551234567 now")
        facts = extractor.extract(episode)

        assert len(facts) == 1
        assert facts[0].content == "5551234567"

    def test_filter_short_numbers(self):
        """Should filter numbers with less than 7 digits."""
        extractor = PhoneExtractor()
        episode = make_episode("Code 12345 is not a phone")
        facts = extractor.extract(episode)

        assert len(facts) == 0

    def test_deduplicate_phones(self):
        """Should deduplicate normalized phone numbers."""
        extractor = PhoneExtractor()
        episode = make_episode("(555) 123-4567 or 555-123-4567")
        facts = extractor.extract(episode)

        assert len(facts) == 1


class TestURLExtractor:
    """Tests for URLExtractor."""

    def test_extract_https_url(self):
        """Should extract https:// URLs."""
        extractor = URLExtractor()
        episode = make_episode("Visit https://example.com/page")
        facts = extractor.extract(episode)

        assert len(facts) == 1
        assert facts[0].content == "https://example.com/page"
        assert facts[0].category == "url"

    def test_extract_http_url(self):
        """Should extract http:// URLs."""
        extractor = URLExtractor()
        episode = make_episode("See http://example.org")
        facts = extractor.extract(episode)

        assert len(facts) == 1
        assert facts[0].content == "http://example.org"

    def test_extract_www_url(self):
        """Should extract www. URLs and add https://."""
        extractor = URLExtractor()
        episode = make_episode("Go to www.example.com")
        facts = extractor.extract(episode)

        assert len(facts) == 1
        assert facts[0].content == "https://www.example.com"

    def test_extract_url_with_path(self):
        """Should extract URLs with paths."""
        extractor = URLExtractor()
        episode = make_episode("https://example.com/path/to/page.html")
        facts = extractor.extract(episode)

        assert len(facts) == 1
        assert facts[0].content == "https://example.com/path/to/page.html"

    def test_extract_url_with_query(self):
        """Should extract URLs with query strings."""
        extractor = URLExtractor()
        episode = make_episode("https://example.com/search?q=test&page=1")
        facts = extractor.extract(episode)

        assert len(facts) == 1
        assert facts[0].content == "https://example.com/search?q=test&page=1"

    def test_normalize_domain_lowercase(self):
        """Should lowercase domain portion."""
        extractor = URLExtractor()
        episode = make_episode("https://EXAMPLE.COM/Path")
        facts = extractor.extract(episode)

        assert facts[0].content == "https://example.com/Path"

    def test_strip_trailing_punctuation(self):
        """Should strip trailing punctuation."""
        extractor = URLExtractor()
        episode = make_episode("Check https://example.com.")
        facts = extractor.extract(episode)

        assert facts[0].content == "https://example.com"

    def test_deduplicate_urls(self):
        """Should deduplicate URLs."""
        extractor = URLExtractor()
        episode = make_episode("https://example.com and https://example.com again")
        facts = extractor.extract(episode)

        assert len(facts) == 1


class TestDateExtractor:
    """Tests for DateExtractor."""

    def test_extract_iso_date(self):
        """Should extract ISO format dates."""
        extractor = DateExtractor()
        episode = make_episode("Meeting on 2024-01-15")
        facts = extractor.extract(episode)

        assert len(facts) == 1
        assert facts[0].content == "2024-01-15"
        assert facts[0].category == "date"

    def test_extract_us_date(self):
        """Should extract US format dates (MM/DD/YYYY)."""
        extractor = DateExtractor()
        episode = make_episode("Due date: 01/15/2024")
        facts = extractor.extract(episode)

        assert len(facts) == 1
        assert facts[0].content == "2024-01-15"

    def test_extract_european_date(self):
        """Should extract European format when day > 12."""
        extractor = DateExtractor()
        episode = make_episode("Meeting 25/01/2024")
        facts = extractor.extract(episode)

        assert len(facts) == 1
        assert facts[0].content == "2024-01-25"

    def test_extract_named_month(self):
        """Should extract dates with month names."""
        extractor = DateExtractor()
        episode = make_episode("Born on January 15, 2024")
        facts = extractor.extract(episode)

        assert len(facts) == 1
        assert facts[0].content == "2024-01-15"

    def test_extract_abbreviated_month(self):
        """Should extract dates with abbreviated months."""
        extractor = DateExtractor()
        episode = make_episode("Event on Jan 15 2024")
        facts = extractor.extract(episode)

        assert len(facts) == 1
        assert facts[0].content == "2024-01-15"

    def test_extract_day_month_year(self):
        """Should extract day-first format with month name."""
        extractor = DateExtractor()
        episode = make_episode("15th January 2024")
        facts = extractor.extract(episode)

        assert len(facts) == 1
        assert facts[0].content == "2024-01-15"

    def test_extract_ordinal_suffix(self):
        """Should handle ordinal suffixes (st, nd, rd, th)."""
        extractor = DateExtractor()
        episode = make_episode("March 1st, 2024 and April 2nd, 2024")
        facts = extractor.extract(episode)

        assert len(facts) == 2
        dates = {f.content for f in facts}
        assert dates == {"2024-03-01", "2024-04-02"}

    def test_validate_invalid_date(self):
        """Should reject invalid dates like Feb 30."""
        extractor = DateExtractor()
        episode = make_episode("February 30, 2024")
        facts = extractor.extract(episode)

        assert len(facts) == 0

    def test_deduplicate_dates(self):
        """Should deduplicate same dates in different formats."""
        extractor = DateExtractor()
        episode = make_episode("2024-01-15 and January 15, 2024")
        facts = extractor.extract(episode)

        assert len(facts) == 1
        assert facts[0].content == "2024-01-15"


class TestExtractionPipeline:
    """Tests for ExtractionPipeline."""

    def test_empty_pipeline(self):
        """Empty pipeline should return no facts."""
        pipeline = ExtractionPipeline()
        episode = make_episode("test@example.com")
        facts = pipeline.run(episode)

        assert len(facts) == 0

    def test_add_extractor(self):
        """Should be able to add extractors after creation."""
        pipeline = ExtractionPipeline()
        pipeline.add_extractor(EmailExtractor())
        episode = make_episode("test@example.com")
        facts = pipeline.run(episode)

        assert len(facts) == 1

    def test_run_multiple_extractors(self):
        """Should run all extractors and combine results."""
        pipeline = ExtractionPipeline([
            EmailExtractor(),
            PhoneExtractor(),
        ])
        episode = make_episode("Email test@example.com or call 555-123-4567")
        facts = pipeline.run(episode)

        assert len(facts) == 2
        categories = {f.category for f in facts}
        assert categories == {"email", "phone"}

    def test_run_with_results(self):
        """run_with_results should return grouped results."""
        pipeline = ExtractionPipeline([
            EmailExtractor(),
            PhoneExtractor(),
        ])
        episode = make_episode("Email test@example.com")
        results = pipeline.run_with_results(episode)

        assert len(results) == 2
        assert results[0].extractor_name == "email"
        assert len(results[0].facts) == 1
        assert results[1].extractor_name == "phone"
        assert len(results[1].facts) == 0

    def test_extractors_property(self):
        """Should expose extractors list."""
        extractors = [EmailExtractor(), PhoneExtractor()]
        pipeline = ExtractionPipeline(extractors)

        assert len(pipeline.extractors) == 2


class TestDefaultPipeline:
    """Tests for default_pipeline factory."""

    def test_default_pipeline_has_all_extractors(self):
        """default_pipeline should include all extractors."""
        pipeline = default_pipeline()

        assert len(pipeline.extractors) == 4
        names = {e.name for e in pipeline.extractors}
        assert names == {"email", "phone", "url", "date"}

    def test_default_pipeline_extracts_all_types(self):
        """default_pipeline should extract all fact types."""
        pipeline = default_pipeline()
        episode = make_episode(
            "Contact user@example.com or (555) 123-4567. "
            "Visit https://example.com by 2024-01-15."
        )
        facts = pipeline.run(episode)

        assert len(facts) == 4
        categories = {f.category for f in facts}
        assert categories == {"email", "phone", "url", "date"}
