from typing import List, Optional

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from rag.config import CHROMA_PERSIST_DIR, DEFAULT_EMBEDDING_MODEL, NUM_RETRIEVED_DOCS

COLLECTION_NAME = "rag_documents"


class VectorStoreManager:
    """Thin wrapper around a persistent Chroma collection using local embeddings."""

    def __init__(self, embedding_model: str = DEFAULT_EMBEDDING_MODEL) -> None:
        # Runs locally — no API key required
        self._embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self._store: Optional[Chroma] = None
        self._init_store()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def _init_store(self) -> None:
        self._store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self._embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )

    # ── Mutations ─────────────────────────────────────────────────────────────

    def add_documents(self, documents: List[Document]) -> None:
        """Embed and persist a list of document chunks."""
        self._store.add_documents(documents)

    def reset(self) -> None:
        """Drop all documents and reinitialise an empty collection."""
        self._store.delete_collection()
        self._init_store()

    # ── Querying ──────────────────────────────────────────────────────────────

    def get_retriever(self, k: int = NUM_RETRIEVED_DOCS) -> VectorStoreRetriever:
        return self._store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )

    # ── Metadata ──────────────────────────────────────────────────────────────

    @property
    def chunk_count(self) -> int:
        try:
            return self._store._collection.count()
        except Exception:
            return 0
