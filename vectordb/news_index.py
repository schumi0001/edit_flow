"""Rolling in-memory index of recent GDELT article embeddings.

At WikiPulse's data volume (thousands of articles/day), an in-memory index
with a single matrix-vector product per lookup is plenty fast -- no need for
a dedicated vector database. Entries are time-pruned so a Wikipedia anomaly
only ever gets matched against genuinely recent news, not something GDELT
published weeks ago.
"""

from datetime import datetime, timedelta, timezone

import numpy as np

DEFAULT_RETENTION_HOURS = 24


class NewsEmbeddingIndex:
    def __init__(self, retention: timedelta = timedelta(hours=DEFAULT_RETENTION_HOURS)):
        self.retention = retention
        self._articles: list[dict] = []
        self._embeddings: list[np.ndarray] = []
        self._added_at: list[datetime] = []

    def __len__(self) -> int:
        return len(self._articles)

    def add(self, article: dict, embedding: np.ndarray, now: datetime | None = None) -> None:
        self._articles.append(article)
        self._embeddings.append(embedding)
        self._added_at.append(now or datetime.now(timezone.utc))

    def prune(self, now: datetime | None = None) -> None:
        now = now or datetime.now(timezone.utc)
        cutoff = now - self.retention
        keep_indices = [i for i, added_at in enumerate(self._added_at) if added_at >= cutoff]

        self._articles = [self._articles[i] for i in keep_indices]
        self._embeddings = [self._embeddings[i] for i in keep_indices]
        self._added_at = [self._added_at[i] for i in keep_indices]

    def best_match(self, query_embedding: np.ndarray) -> tuple[float, dict] | None:
        """Return (similarity_score, article) for the closest match, or None if empty."""
        if not self._embeddings:
            return None

        matrix = np.vstack(self._embeddings)
        scores = matrix @ query_embedding
        best_index = int(np.argmax(scores))
        return float(scores[best_index]), self._articles[best_index]
