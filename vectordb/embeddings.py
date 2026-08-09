"""Shared text-embedding utilities for comparing GDELT articles against
Wikipedia edit anomalies.

Both `news-topic` and `wikipedia-anomalies` currently only carry titles (no
article body or edit diff text), so this module embeds titles as-is. A page
title being edited anomalously (e.g. "2026_California_wildfires") and a news
article titled "Wildfires force evacuations in LA County" won't share
keywords, which is exactly why this uses semantic embeddings + cosine
similarity rather than a lexical match.
"""

import os
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
    "Barack_Obama"), so those are restored before embedding.
    """
    title = anomaly.get("page_title") or ""
    return title.replace("_", " ").strip()


def gdelt_article_text(article: dict) -> str:
    title = article.get("title") or ""
    return title.strip()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two embeddings.

    Assumes both vectors are already L2-normalized (true for anything
    produced by embed_texts/embed_text), in which case this is just a dot
    product.
    """
    return float(np.dot(a, b))
