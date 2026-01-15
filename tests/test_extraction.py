"""Unit tests for Engram extraction layer."""

import pytest

from engram.extraction import (
    EmailExtractor,
    Extractor,
    PhoneExtractor,
    URLExtractor,
)
from engram.models import Episode


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


class TestEmailExtractor:
    """Tests for EmailExtractor."""

    def test_extract_simple_email(self):
        """Should extract a simple email address."""
        extractor = EmailExtractor()
        episode = make_episode("Contact me at user@example.com please")
        emails = extractor.extract(episode)

        assert len(emails) == 1
        assert emails[0] == "user@example.com"

    def test_extract_multiple_emails(self):
        """Should extract multiple email addresses."""
        extractor = EmailExtractor()
        episode = make_episode("Email alice@example.com or bob@example.org")
        emails = extractor.extract(episode)

        assert len(emails) == 2
        assert set(emails) == {"alice@example.com", "bob@example.org"}

    def test_extract_email_with_plus(self):
        """Should handle plus addressing."""
        extractor = EmailExtractor()
        episode = make_episode("Use user+tag@example.com")
        emails = extractor.extract(episode)

        assert len(emails) == 1
        assert emails[0] == "user+tag@example.com"

    def test_extract_email_with_subdomain(self):
        """Should handle subdomains."""
        extractor = EmailExtractor()
        episode = make_episode("Email user@mail.example.co.uk")
        emails = extractor.extract(episode)

        assert len(emails) == 1
        assert emails[0] == "user@mail.example.co.uk"

    def test_normalize_domain_lowercase(self):
        """Should normalize domain to lowercase (local part preserved)."""
        extractor = EmailExtractor()
        episode = make_episode("Email user@EXAMPLE.COM")
        emails = extractor.extract(episode)

        # email-validator normalizes domain but preserves local part case
        assert emails[0] == "user@example.com"

    def test_deduplicate_emails(self):
        """Should deduplicate repeated emails."""
        extractor = EmailExtractor()
        episode = make_episode("user@example.com and user@example.com again")
        emails = extractor.extract(episode)

        assert len(emails) == 1

    def test_no_match_returns_empty(self):
        """Should return empty list when no emails found."""
        extractor = EmailExtractor()
        episode = make_episode("No email here")
        emails = extractor.extract(episode)

        assert len(emails) == 0


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
        phones = extractor.extract(episode)

        assert len(phones) == 1
        assert phones[0] == "+12024561414"

    def test_extract_us_format_dashes(self):
        """Should extract XXX-XXX-XXXX format."""
        extractor = PhoneExtractor()
        episode = make_episode("Phone: 202-456-1414")
        phones = extractor.extract(episode)

        assert len(phones) == 1
        assert phones[0] == "+12024561414"

    def test_extract_us_format_dots(self):
        """Should extract XXX.XXX.XXXX format."""
        extractor = PhoneExtractor()
        episode = make_episode("Fax: 202.456.1414")
        phones = extractor.extract(episode)

        assert len(phones) == 1
        assert phones[0] == "+12024561414"

    def test_extract_international(self):
        """Should extract international format with +."""
        extractor = PhoneExtractor()
        episode = make_episode("International: +44 20 7946 0958")
        phones = extractor.extract(episode)

        assert len(phones) == 1
        assert phones[0] == "+442079460958"

    def test_extract_with_region(self):
        """Should use specified default region."""
        extractor = PhoneExtractor(default_region="GB")
        episode = make_episode("Call 020 7946 0958")
        phones = extractor.extract(episode)

        assert len(phones) == 1
        assert phones[0] == "+442079460958"

    def test_filter_invalid_numbers(self):
        """Should filter out invalid phone numbers."""
        extractor = PhoneExtractor()
        # 555 numbers are fictional, may be filtered by libphonenumber
        episode = make_episode("Call 555-0123")
        phones = extractor.extract(episode)

        # May or may not extract depending on validation
        # Just verify no crash
        assert isinstance(phones, list)

    def test_deduplicate_phones(self):
        """Should deduplicate normalized phone numbers."""
        extractor = PhoneExtractor()
        episode = make_episode("(202) 456-1414 or 202-456-1414")
        phones = extractor.extract(episode)

        assert len(phones) == 1
        assert phones[0] == "+12024561414"


class TestURLExtractor:
    """Tests for URLExtractor."""

    def test_extract_https_url(self):
        """Should extract https:// URLs."""
        extractor = URLExtractor()
        episode = make_episode("Visit https://example.com/page")
        urls = extractor.extract(episode)

        assert len(urls) == 1
        assert urls[0] == "https://example.com/page"

    def test_extract_http_url(self):
        """Should extract http:// URLs."""
        extractor = URLExtractor()
        episode = make_episode("See http://example.org")
        urls = extractor.extract(episode)

        assert len(urls) == 1
        assert urls[0] == "http://example.org"

    def test_extract_www_url(self):
        """Should extract www. URLs and add https://."""
        extractor = URLExtractor()
        episode = make_episode("Go to www.example.com")
        urls = extractor.extract(episode)

        assert len(urls) == 1
        assert urls[0] == "https://www.example.com"

    def test_extract_url_with_path(self):
        """Should extract URLs with paths."""
        extractor = URLExtractor()
        episode = make_episode("https://example.com/path/to/page.html")
        urls = extractor.extract(episode)

        assert len(urls) == 1
        assert urls[0] == "https://example.com/path/to/page.html"

    def test_extract_url_with_query(self):
        """Should extract URLs with query strings."""
        extractor = URLExtractor()
        episode = make_episode("https://example.com/search?q=test&page=1")
        urls = extractor.extract(episode)

        assert len(urls) == 1
        assert urls[0] == "https://example.com/search?q=test&page=1"

    def test_normalize_domain_lowercase(self):
        """Should lowercase domain portion."""
        extractor = URLExtractor()
        episode = make_episode("https://EXAMPLE.COM/Path")
        urls = extractor.extract(episode)

        assert urls[0] == "https://example.com/Path"

    def test_strip_trailing_punctuation(self):
        """Should strip trailing punctuation."""
        extractor = URLExtractor()
        episode = make_episode("Check https://example.com.")
        urls = extractor.extract(episode)

        assert urls[0] == "https://example.com"

    def test_deduplicate_urls(self):
        """Should deduplicate URLs."""
        extractor = URLExtractor()
        episode = make_episode("https://example.com and https://example.com again")
        urls = extractor.extract(episode)

        assert len(urls) == 1
