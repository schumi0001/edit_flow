"""Qdrant-backed rolling index of recent GDELT article embeddings.

Backed by a real vector database (Qdrant) rather than an in-process list, so
the index is persisted across restarts, survives more than one consumer
process, and its similarity search is a real ANN index rather than a
brute-force scan -- meaning it stays fast as the number of stored articles
grows well past what fits comfortably in one process's memory.

Entries are time-pruned so a Wikipedia anomaly only ever gets matched
against genuinely recent news, not something GDELT published weeks ago.
"""

import os
import uuid
from datetime import datetime, timedelta, timezone

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

DEFAULT_RETENTION_HOURS = 24
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_COLLECTION_NAME = "news_articles"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2's output size

# Namespace fixed articles' event_id to a stable UUID, keeping re-processed
# articles (e.g. after a Kafka redelivery) as upserts instead of duplicates.
_ID_NAMESPACE = uuid.UUID("6f0a3f2e-6b9d-4c9a-9b0e-7c3a5b1d2e4f")


class NewsEmbeddingIndex:
    def __init__(
        self,
        retention: timedelta = timedelta(hours=DEFAULT_RETENTION_HOURS),
        url: str | None = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        vector_size: int = EMBEDDING_DIM,
    ):
        self.retention = retention
        self.collection_name = collection_name
        self.client = QdrantClient(
            url=url or os.environ.get("QDRANT_URL", DEFAULT_QDRANT_URL)
        )
        self._ensure_collection(vector_size)

    def _ensure_collection(self, vector_size: int) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=vector_size, distance=qmodels.Distance.COSINE
                ),
            )

    def __len__(self) -> int:
        return self.client.count(
            collection_name=self.collection_name, exact=True
        ).count

    def add(self, article: dict, embedding: np.ndarray, now: datetime | None = None) -> None:
        now = now or datetime.now(timezone.utc)
        event_id = article.get("event_id")
        point_id = str(uuid.uuid5(_ID_NAMESPACE, event_id)) if event_id else str(uuid.uuid4())

        payload = dict(article)
        payload["_added_at"] = now.timestamp()

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                qmodels.PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload=payload,
                )
            ],
        )

    def prune(self, now: datetime | None = None) -> None:
        now = now or datetime.now(timezone.utc)
        cutoff = (now - self.retention).timestamp()

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="_added_at", range=qmodels.Range(lt=cutoff)
                        )
                    ]
                )
            ),
        )

    def best_match(self, query_embedding: np.ndarray) -> tuple[float, dict] | None:
        """Return (similarity_score, article) for the closest match, or None if empty."""
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            limit=1,
        ).points

        if not hits:
            return None

        hit = hits[0]
        article = {key: value for key, value in hit.payload.items() if key != "_added_at"}
        return float(hit.score), article
