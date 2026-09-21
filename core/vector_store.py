from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from core.types import Chunk


class VectorStore:
    def __init__(self, path: Path, profile: str):
        self.client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=f"research_{profile}",
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
        )

    def add(self, generation: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Each chunk must have exactly one embedding.")
        if chunks:
            self.collection.upsert(
                ids=[f"{generation}_{chunk.chunk_id}" for chunk in chunks],
                embeddings=vectors,
                metadatas=[
                    {"generation": generation, "document_id": chunk.document_id,
                     "chunk_id": chunk.chunk_id} for chunk in chunks
                ],
            )

    def search(
        self, vector: list[float], generations: list[str], limit: int
    ) -> list[tuple[str, float]]:
        if not generations or self.collection.count() == 0:
            return []
        result = self.collection.query(
            query_embeddings=[vector],
            where={"generation": {"$in": generations}},
            n_results=min(limit, self.collection.count()),
            include=["metadatas", "distances"],
        )
        metadata = result["metadatas"]
        distances = result["distances"]
        if metadata is None or distances is None:
            raise RuntimeError("Chroma returned no metadata or distances.")
        return [
            (str(item["chunk_id"]), 1.0 - distance)
            for item, distance in zip(metadata[0], distances[0], strict=True)
        ]

    def delete_generation(self, generation: str) -> None:
        self.collection.delete(where={"generation": generation})

    def close(self) -> None:
        self.client.close()

    @staticmethod
    def delete_document_everywhere(path: Path, document_id: str) -> None:
        client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        try:
            for collection in client.list_collections():
                if collection.name.startswith("research_"):
                    client.get_collection(collection.name, embedding_function=None).delete(
                        where={"document_id": document_id}
                    )
        finally:
            client.close()
