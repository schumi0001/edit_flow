"""Shared text-embedding utilities for comparing GDELT articles against
Wikipedia edit anomalies.

`news-topic` events carry a `snippet` of real quadgrams from the article
body (see producer/gdelt_producer.py), and `wikipedia-anomalies` records
carry `recent_comments` sampled from real edit summaries during the
anomaly's window (see spark/ml_inference_stream.py) -- both appended to
their respective titles below. A page title being edited anomalously (e.g.
"2026_California_wildfires") and a news article titled "Wildfires force
evacuations in LA County" won't share keywords, which is exactly why this
uses semantic embeddings + cosine similarity rather than a lexical match.
"""

import os
import re
from functools import lru_cache

import numpy as np

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def load_model(model_name: str | None = None):
    """Lazily load and cache the sentence-transformers model.

    Imported inside the function (not at module scope) so tests can patch
    this function directly without requiring sentence-transformers/torch to
    be installed.
    """
    from sentence_transformers import SentenceTransformer

    resolved_name = model_name or os.environ.get(
        "EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
    )
    return SentenceTransformer(resolved_name)


def embed_texts(texts: list[str], model_name: str | None = None) -> np.ndarray:
    """Embed a batch of texts as L2-normalized vectors.

    Normalizing at encode time means cosine similarity between any two
    embeddings reduces to a plain dot product everywhere downstream.
    """
    model = load_model(model_name)
    return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)


def embed_text(text: str, model_name: str | None = None) -> np.ndarray:
    return embed_texts([text], model_name=model_name)[0]


def wikipedia_anomaly_text(anomaly: dict) -> str:
    """Build the text to embed for a flagged Wikipedia page.

    MediaWiki page titles use underscores in place of spaces (e.g.
    "Barack_Obama"), so those are restored before embedding. `recent_comments`
    (a handful of real, substantive edit summaries collected during the
    anomaly's window by spark/ml_inference_stream.py -- see
    substantive_comment_text below) is appended when present, giving the
    embedding real topical text beyond the bare page title. Absent/empty
    `recent_comments` (e.g. messages published before this field existed)
    falls back to title-only, unchanged from before.
    """
    title = (anomaly.get("page_title") or "").replace("_", " ").strip()
    recent_comments = (anomaly.get("recent_comments") or "").strip()
    parts = [part for part in (title, recent_comments) if part]
    return " ".join(parts)


def gdelt_article_text(article: dict) -> str:
    """Build the text to embed for one GDELT news article.

    `snippet` (a handful of real quadgrams -- 4-word phrases -- pulled from
    the article's body text by producer/gdelt_producer.py) is appended when
    present, giving the embedding real article content beyond the headline
    alone. Absent/empty `snippet` (e.g. the companion ngrams file was
    missing, or events published before this field existed) falls back to
    title-only, unchanged from before.
    """
    title = (article.get("title") or "").strip()
    snippet = (article.get("snippet") or "").strip()
    if not title:
        return snippet
    if not snippet:
        return title
    return f"{title}. {snippet}"


# Wikipedia's own documented shorthand for edit summaries (see
# https://en.wikipedia.org/wiki/Wikipedia:Edit_summary_legend/Quick_reference),
# plus generic maintenance/section-header vocabulary observed in practice
# (e.g. "/* See also */", "created article") that carries no real-world
# topical signal even though it isn't in Wikipedia's own jargon legend. Used
# by substantive_comment_text to recognize comments that are *only* editing
# mechanics/boilerplate.
_JARGON_TOKENS = frozenset(
    {
        "add", "addition", "alpha", "abc", "cap", "capital", "cpt", "lc",
        "lcase", "uc", "ucase", "cat", "recat", "cl", "clean", "tidy", "cm",
        "cmt", "re", "copyedit", "cpyed", "ced", "ce", "mce", "creation",
        "new", "dab", "disamb", "disam", "disambig", "byp", "dup", "dupe",
        "fmt", "frmt", "mo", "mos", "ft", "ref", "refs", "reference",
        "references", "rm", "rmv", "del", "rv", "rvt", "rvrt", "rvv", "sp",
        "typo", "typos", "talk", "wfy", "wky", "wkfy", "top", "undo", "undid",
        "minor", "misc", "na", "n/a",
        # Generic section headers: descriptive-sounding but near-universal,
        # so they carry no signal about *what* changed (unlike e.g. "/*
        # Casualties */", which names the actual topic under discussion).
        "see", "also", "external", "links", "link", "notes", "note",
        "further", "reading", "attendances", "attendance", "infobox",
        "categories", "category",
        # Generic page-creation/maintenance boilerplate.
        "created", "creating", "page", "pages", "article", "articles",
        "restored", "revision", "revisions", "wikify", "wikification",
    }
)

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

# MediaWiki auto-generates this exact prefix for rollback/undo actions, e.g.
# "Undid revision 123 by [[Special:Contributions/X|X]] (talk): rv vandalism"
# or "Restored revision 123 by [[...]] ([[...]]): reason". The wikilink
# targets are usernames/talk pages, never topical content, so the whole
# prefix is stripped rather than tokenized.
_REVISION_BOILERPLATE_PATTERN = re.compile(
    r"^(?:Undid|Restored)\s+revisions?\s+\d+"
    r"(?:\s*(?:through|to|-)\s*\d+)?\s+by\s+\[\[[^\]]*\]\]"
    r"(?:\s*\(\[\[[^\]]*\]\]\))?\s*:?\s*",
    re.IGNORECASE,
)

# "[[target|display text]]" or "[[target]]" -> keeps only the human-readable
# part, dropping internal wiki targets/namespaces that add no topical signal.
_WIKI_LINK_PATTERN = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]]+)\]\]")


def clean_wikipedia_comment(comment: str | None) -> str:
    """Strip MediaWiki markup/boilerplate from a raw edit comment.

    Removes the auto-generated revert/rollback prefix, unwraps "/* Section
    name */" (keeping the section text as plain words), converts wikilinks
    to their display text, and drops template/table syntax ("{{", "}}",
    "|") while preserving every word inside -- e.g. auto-generated
    page-creation summaries like "Created page with '{{Short
    description|Downloadable content for ...}}'" keep the actual content
    words instead of the raw markup. Returns "" for empty/blank input.
    """
    if not comment:
        return ""

    text = comment.strip()
    text = _REVISION_BOILERPLATE_PATTERN.sub("", text)
    text = re.sub(r"/\*\s*(.*?)\s*\*/", r"\1", text)
    text = _WIKI_LINK_PATTERN.sub(r"\1", text)
    text = text.replace("{{", " ").replace("}}", " ").replace("|", " ")
    return re.sub(r"\s+", " ", text).strip()


def substantive_comment_text(comment: str | None) -> str | None:
    """Return a cleaned comment (see clean_wikipedia_comment) if it carries
    real topical signal, or None if it's blank or pure editing jargon.

    Tokenizes the cleaned text on non-alphanumeric boundaries and returns
    None only when *every* resulting token exactly (whole-word, never
    substring) matches _JARGON_TOKENS, or there are no tokens at all (empty/
    blank comment). Any single unrecognized token -- regardless of length --
    keeps the comment: this is deliberately biased toward never discarding a
    real, possibly short, content word by mistake.
    """
    cleaned = clean_wikipedia_comment(comment)
    if not cleaned:
        return None

    tokens = _TOKEN_PATTERN.findall(cleaned.lower())
    if not tokens:
        return None

    if any(token not in _JARGON_TOKENS for token in tokens):
        return cleaned
    return None


def is_substantive_comment(comment: str | None) -> bool:
    """Return False only when a Wikipedia edit comment is pure editing jargon
    or boilerplate. See substantive_comment_text for the full semantics.
    """
    return substantive_comment_text(comment) is not None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two embeddings.

    Assumes both vectors are already L2-normalized (true for anything
    produced by embed_texts/embed_text), in which case this is just a dot
    product.
    """
    return float(np.dot(a, b))
