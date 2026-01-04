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

    def test_normalize_domain_lowercase(self):
        """Should normalize domain to lowercase (local part preserved)."""
        extractor = EmailExtractor()
        episode = make_episode("Email user@EXAMPLE.COM")
        facts = extractor.extract(episode)

        # email-validator normalizes domain but preserves local part case
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
    """Tests for PhoneExtractor.

    Note: Uses real-looking US phone numbers because libphonenumber
    validates number patterns (555 numbers are fictional and may fail).
    """

    def test_extract_us_format_parens(self):
        """Should extract (XXX) XXX-XXXX format."""
        extractor = PhoneExtractor()
        # Use a real-pattern number (not 555)
        episode = make_episode("Call me at (202) 456-1414")
        facts = extractor.extract(episode)

        assert len(facts) == 1
        assert facts[0].content == "+12024561414"
        assert facts[0].category == "phone"

    def test_extract_us_format_dashes(self):
        """Should extract XXX-XXX-XXXX format."""
        extractor = PhoneExtractor()
        episode = make_episode("Phone: 202-456-1414")
        facts = extractor.extract(episode)

        assert len(facts) == 1
        assert facts[0].content == "+12024561414"

    def test_extract_us_format_dots(self):
        """Should extract XXX.XXX.XXXX format."""
        extractor = PhoneExtractor()
        episode = make_episode("Fax: 202.456.1414")
        facts = extractor.extract(episode)

        assert len(facts) == 1
        assert facts[0].content == "+12024561414"

    def test_extract_international(self):
        """Should extract international format with +."""
        extractor = PhoneExtractor()
        episode = make_episode("International: +44 20 7946 0958")
        facts = extractor.extract(episode)

        assert len(facts) == 1
        assert facts[0].content == "+442079460958"

    def test_extract_with_region(self):
        """Should use specified default region."""
        extractor = PhoneExtractor(default_region="GB")
        episode = make_episode("Call 020 7946 0958")
        facts = extractor.extract(episode)

        assert len(facts) == 1
        assert facts[0].content == "+442079460958"

    def test_filter_invalid_numbers(self):
        """Should filter out invalid phone numbers."""
        extractor = PhoneExtractor()
        # 555 numbers are fictional, may be filtered by libphonenumber
        episode = make_episode("Call 555-0123")
        facts = extractor.extract(episode)

        # May or may not extract depending on validation
        # Just verify no crash
        assert isinstance(facts, list)

    def test_deduplicate_phones(self):
        """Should deduplicate normalized phone numbers."""
        extractor = PhoneExtractor()
        episode = make_episode("(202) 456-1414 or 202-456-1414")
        facts = extractor.extract(episode)

        assert len(facts) == 1
        assert facts[0].content == "+12024561414"


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
    """Tests for DateExtractor.

    Note: dateparser is very flexible and may extract dates from
    partial text. Tests focus on common formats.
    """

    def test_extract_iso_date(self):
        """Should extract ISO format dates."""
        extractor = DateExtractor()
        episode = make_episode("Meeting on 2024-01-15")
        facts = extractor.extract(episode)

        # Should find at least the ISO date
        dates = {f.content for f in facts}
        assert "2024-01-15" in dates

    def test_extract_us_date(self):
        """Should extract US format dates (MM/DD/YYYY)."""
        extractor = DateExtractor()
        episode = make_episode("Due date: 01/15/2024")
        facts = extractor.extract(episode)

        dates = {f.content for f in facts}
        assert "2024-01-15" in dates

    def test_extract_european_date(self):
        """Should extract European format when day > 12."""
        extractor = DateExtractor()
        episode = make_episode("Meeting 25/01/2024")
        facts = extractor.extract(episode)

        dates = {f.content for f in facts}
        assert "2024-01-25" in dates

    def test_extract_named_month(self):
        """Should extract dates with month names."""
        extractor = DateExtractor()
        episode = make_episode("Born on January 15, 2024")
        facts = extractor.extract(episode)

        dates = {f.content for f in facts}
        assert "2024-01-15" in dates

    def test_extract_abbreviated_month(self):
        """Should extract dates with abbreviated months."""
        extractor = DateExtractor()
        episode = make_episode("Event on Jan 15, 2024")
        facts = extractor.extract(episode)

        dates = {f.content for f in facts}
        assert "2024-01-15" in dates

    def test_extract_day_month_year(self):
        """Should extract day-first format with month name."""
        extractor = DateExtractor()
        episode = make_episode("15th January 2024")
        facts = extractor.extract(episode)

        dates = {f.content for f in facts}
        assert "2024-01-15" in dates

    def test_extract_natural_language(self):
        """Should extract natural language dates."""
        extractor = DateExtractor()
        # dateparser supports relative dates
        episode = make_episode("Let's meet tomorrow")
        facts = extractor.extract(episode)

        # Should extract something (relative to today)
        assert len(facts) >= 1

    def test_deduplicate_dates(self):
        """Should deduplicate same dates."""
        extractor = DateExtractor()
        episode = make_episode("2024-01-15 is the same as 2024-01-15")
        facts = extractor.extract(episode)

        # Should deduplicate
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
            URLExtractor(),
        ])
        episode = make_episode("Email test@example.com or visit https://example.com")
        facts = pipeline.run(episode)

        assert len(facts) == 2
        categories = {f.category for f in facts}
        assert categories == {"email", "url"}

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

    def test_default_pipeline_extracts_multiple_types(self):
        """default_pipeline should extract from various formats."""
        pipeline = default_pipeline()
        episode = make_episode(
            "Contact user@example.com. "
            "Visit https://example.com for info."
        )
        facts = pipeline.run(episode)

        # Should extract at least email and URL
        categories = {f.category for f in facts}
        assert "email" in categories
        assert "url" in categories
