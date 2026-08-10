"""Unit tests for the text-building helpers in vectordb/embeddings.py.

These only test the plain-Python text assembly/filtering logic, not the
actual embedding model (sentence-transformers/torch aren't required to run
these -- see the module's own docstring on why the model import is lazy).
"""

import unittest

from vectordb.embeddings import (
    clean_wikipedia_comment,
    gdelt_article_text,
    is_substantive_comment,
    substantive_comment_text,
    wikipedia_anomaly_text,
)


class GdeltArticleTextTests(unittest.TestCase):
    def test_appends_snippet_when_present(self):
        article = {
            "title": "Wildfires force evacuations in LA County",
            "snippet": "evacuation order issued for canyon residents",
        }

        text = gdelt_article_text(article)

        self.assertEqual(
            text,
            "Wildfires force evacuations in LA County. evacuation order "
            "issued for canyon residents",
        )

    def test_falls_back_to_title_only_when_snippet_missing(self):
        article = {"title": "Wildfires force evacuations in LA County"}

        self.assertEqual(
            gdelt_article_text(article),
            "Wildfires force evacuations in LA County",
        )

    def test_falls_back_to_title_only_when_snippet_blank(self):
        article = {
            "title": "Wildfires force evacuations in LA County",
            "snippet": "   ",
        }

        self.assertEqual(
            gdelt_article_text(article),
            "Wildfires force evacuations in LA County",
        )

    def test_handles_missing_title(self):
        self.assertEqual(gdelt_article_text({}), "")
        # No title to prefix, but the snippet alone is still real content
        # worth embedding rather than discarding.
        self.assertEqual(
            gdelt_article_text({"snippet": "some fragment"}), "some fragment"
        )


class WikipediaAnomalyTextTests(unittest.TestCase):
    def test_combines_title_and_recent_comments(self):
        anomaly = {
            "page_title": "2026_California_wildfires",
            "recent_comments": "updated casualty figures per official reports",
        }

        text = wikipedia_anomaly_text(anomaly)

        self.assertEqual(
            text,
            "2026 California wildfires updated casualty figures per "
            "official reports",
        )

    def test_falls_back_to_title_only_when_recent_comments_missing(self):
        anomaly = {"page_title": "2026_California_wildfires"}

        self.assertEqual(
            wikipedia_anomaly_text(anomaly), "2026 California wildfires"
        )

    def test_falls_back_to_title_only_when_recent_comments_blank(self):
        anomaly = {
            "page_title": "2026_California_wildfires",
            "recent_comments": "",
        }

        self.assertEqual(
            wikipedia_anomaly_text(anomaly), "2026 California wildfires"
        )

    def test_handles_missing_title(self):
        self.assertEqual(wikipedia_anomaly_text({}), "")


class IsSubstantiveCommentTests(unittest.TestCase):
    def test_rejects_single_jargon_token(self):
        self.assertFalse(is_substantive_comment("ce"))
        self.assertFalse(is_substantive_comment("rv"))
        self.assertFalse(is_substantive_comment("RV"))

    def test_rejects_multi_token_jargon_only_comment(self):
        self.assertFalse(is_substantive_comment("rv top"))
        self.assertFalse(is_substantive_comment("ce, mos"))

    def test_rejects_empty_or_blank_comment(self):
        self.assertFalse(is_substantive_comment(""))
        self.assertFalse(is_substantive_comment("   "))
        self.assertFalse(is_substantive_comment(None))

    def test_rejects_jargon_only_section_marker(self):
        # MediaWiki auto-prepends "/* Section name */"; if the section name
        # itself is just jargon and nothing follows, there's still no real
        # topical signal.
        self.assertFalse(is_substantive_comment("/* top */"))

    def test_accepts_comment_with_any_non_jargon_token(self):
        # A single unrecognized token -- however short -- must keep the
        # whole comment; jargon words alongside it don't cancel that out.
        self.assertTrue(is_substantive_comment("rv vandalism"))
        self.assertTrue(is_substantive_comment("war"))
        self.assertTrue(is_substantive_comment("fire"))
        self.assertTrue(
            is_substantive_comment("ce: fixed wildfire casualty count")
        )

    def test_accepts_descriptive_section_marker_content(self):
        self.assertTrue(
            is_substantive_comment(
                "/* Casualties */ updated death toll to 45"
            )
        )

    def test_rejects_generic_maintenance_section_markers(self):
        # These are near-universal Wikipedia section headers -- unlike "/*
        # Casualties */", they name no actual topic, just page structure.
        self.assertFalse(is_substantive_comment("/* See also */"))
        self.assertFalse(is_substantive_comment("/* Attendances */"))
        self.assertFalse(is_substantive_comment("/* External links */"))
        self.assertFalse(
            is_substantive_comment(
                "/* See also */ | /* Attendances */ | /* See also */"
            )
        )

    def test_rejects_generic_page_creation_boilerplate(self):
        self.assertFalse(is_substantive_comment("created article"))
        self.assertFalse(is_substantive_comment("Created page"))

    def test_rejects_pure_revert_boilerplate_with_jargon_reason(self):
        # Stripping the auto-generated "Undid revision N by [[...]]:" prefix
        # leaves only jargon ("rv"), so this should now be rejected -- prior
        # to boilerplate-stripping it would have incorrectly passed on the
        # username tokens inside the wikilink.
        self.assertFalse(
            is_substantive_comment(
                "Undid revision 123 by [[Special:Contributions/X|X]] "
                "(talk): rv"
            )
        )

    def test_accepts_revert_boilerplate_with_real_reason(self):
        self.assertTrue(
            is_substantive_comment(
                "Restored revision 456 by "
                "[[Special:Contributions/Y|Y]] ([[User talk:Y|talk]]): "
                "Removing fringe theories introduced by IP"
            )
        )


class CleanWikipediaCommentTests(unittest.TestCase):
    def test_strips_revert_boilerplate_prefix(self):
        cleaned = clean_wikipedia_comment(
            "Restored revision 456 by "
            "[[Special:Contributions/Y|Y]] ([[User talk:Y|talk]]): "
            "Removing fringe theories introduced by IP"
        )
        self.assertEqual(cleaned, "Removing fringe theories introduced by IP")

    def test_unwraps_section_marker_as_plain_words(self):
        self.assertEqual(
            clean_wikipedia_comment("/* Casualties */ updated death toll"),
            "Casualties updated death toll",
        )

    def test_converts_wikilink_to_display_text(self):
        self.assertEqual(
            clean_wikipedia_comment("moved content to [[Middle Passage]]"),
            "moved content to Middle Passage",
        )
        self.assertEqual(
            clean_wikipedia_comment(
                "see [[Atlantic slave trade|the main article]]"
            ),
            "see the main article",
        )

    def test_strips_template_markup_but_keeps_words(self):
        cleaned = clean_wikipedia_comment(
            "Created page with '{{Short description|Downloadable content "
            "for the 2019 video game Control}}'"
        )
        self.assertNotIn("{{", cleaned)
        self.assertNotIn("}}", cleaned)
        self.assertNotIn("|", cleaned)
        self.assertIn("Downloadable content", cleaned)

    def test_handles_empty_or_none(self):
        self.assertEqual(clean_wikipedia_comment(""), "")
        self.assertEqual(clean_wikipedia_comment(None), "")
        self.assertEqual(clean_wikipedia_comment("   "), "")


class SubstantiveCommentTextTests(unittest.TestCase):
    def test_returns_cleaned_text_when_substantive(self):
        self.assertEqual(
            substantive_comment_text(
                "Restored revision 456 by "
                "[[Special:Contributions/Y|Y]] ([[User talk:Y|talk]]): "
                "Removing fringe theories introduced by IP"
            ),
            "Removing fringe theories introduced by IP",
        )

    def test_returns_none_when_jargon_or_maintenance_only(self):
        self.assertIsNone(substantive_comment_text("rv"))
        self.assertIsNone(substantive_comment_text("/* See also */"))
        self.assertIsNone(substantive_comment_text(""))
        self.assertIsNone(substantive_comment_text(None))


if __name__ == "__main__":
    unittest.main()
