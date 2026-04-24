"""
RAG (Retrieval-Augmented Generation) Module
============================================
Provides a ChromaDB-backed vector store with Google text-embedding-004
for embedding business review data and retrieving relevant context.

Usage:
    from summarizer_agent.rag import build_index, search, add_to_index

    # Index local files
    build_index("data/")

    # Search for relevant context
    results = search("parking complaints Chicago")

    # Dynamically add live search results
    add_to_index("Great pizza but terrible parking...", source="serpapi")
"""

import os
import glob
from typing import List, Dict

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Google Embedding Function (text-embedding-004)
# ---------------------------------------------------------------------------

class GoogleEmbeddingFunction:
    """
    Custom embedding function for ChromaDB that uses Google's
    text-embedding-004 model via the google-genai client.
    """

    def __init__(self):
        self._client = None

    @staticmethod
    def name() -> str:
        """Required by ChromaDB for embedding function identification."""
        return "google_text_embedding_004"

    @staticmethod
    def build_config_from_collection(*args, **kwargs):
        """Required by ChromaDB for persistence config validation."""
        return {}

    @property
    def client(self):
        """Lazy-init so the API key is read after dotenv is loaded."""
        if self._client is None:
            from google import genai
            api_key = os.getenv("GOOGLE_API_KEY", "")
            self._client = genai.Client(api_key=api_key)
        return self._client

    def __call__(self, input: List[str]) -> List[List[float]]:
        """Embed a list of texts and return a list of float vectors."""
        return self._embed_batch(input)

    def embed_documents(self, input: List[str]) -> List[List[float]]:
        """ChromaDB calls this when embedding documents for storage."""
        return self._embed_batch(input)

    def embed_query(self, input: List[str]) -> List[List[float]]:
        """ChromaDB calls this when embedding queries for search."""
        return self._embed_batch(input)

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Core embedding logic with batching."""
        all_embeddings: List[List[float]] = []
        batch_size = 20
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            result = self.client.models.embed_content(
                model="gemini-embedding-001",
                contents=batch,
            )
            all_embeddings.extend([e.values for e in result.embeddings])
        return all_embeddings


# ---------------------------------------------------------------------------
# Text Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 600, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: The text to split.
        chunk_size: Maximum characters per chunk.
        overlap: Number of overlapping characters between consecutive chunks.

    Returns:
        A list of non-empty text chunks.
    """
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ---------------------------------------------------------------------------
# ChromaDB Vector Store  (persisted under .adk/chromadb/)
# ---------------------------------------------------------------------------

CHROMA_DIR = os.path.join(
    os.path.dirname(__file__), "..", ".adk", "chromadb"
)
COLLECTION_NAME = "business_knowledge"

# Module-level singletons (lazy)
_chroma_client = None
_collection = None


def _get_collection():
    """Return (or create) the ChromaDB collection with Google embeddings."""
    global _chroma_client, _collection
    if _collection is None:
        import chromadb

        os.makedirs(CHROMA_DIR, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = _chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=GoogleEmbeddingFunction(),
        )
    return _collection


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_index(corpus_path: str) -> dict:
    """
    Ingest files from *corpus_path*, chunk, embed, and store in ChromaDB.

    Supported file types: .txt, .md, .csv

    Args:
        corpus_path: Directory containing files to index.

    Returns:
        A dict with status, file count, and chunk count.
    """
    collection = _get_collection()
    supported = ("*.txt", "*.md", "*.csv")
    files_processed = 0
    chunks_added = 0

    for ext in supported:
        pattern = os.path.join(corpus_path, "**", ext)
        for filepath in glob.glob(pattern, recursive=True):
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            except OSError:
                continue

            chunks = chunk_text(text)
            if not chunks:
                continue

            ids = [f"{os.path.basename(filepath)}_chunk{i}" for i in range(len(chunks))]
            metas = [
                {"source": filepath, "chunk_index": i, "filename": os.path.basename(filepath)}
                for i in range(len(chunks))
            ]
            collection.upsert(ids=ids, documents=chunks, metadatas=metas)
            chunks_added += len(chunks)
            files_processed += 1

    return {
        "status": "success",
        "files_processed": files_processed,
        "chunks_added": chunks_added,
        "total_documents_in_store": collection.count(),
    }


def search(query: str, top_k: int = 4) -> List[Dict]:
    """
    Retrieve the most relevant chunks for a given query.

    Args:
        query: Natural-language search query.
        top_k: Number of results to return.

    Returns:
        A list of dicts with keys: id, snippet, source, distance.
    """
    collection = _get_collection()

    if collection.count() == 0:
        return [
            {
                "info": "Knowledge base is empty. Add files to data/ and call build_index, "
                "or wait for the Researcher to populate it with live search results."
            }
        ]

    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, collection.count()),
    )

    output: List[Dict] = []
    for i in range(len(results["ids"][0])):
        output.append(
            {
                "id": results["ids"][0][i],
                "snippet": results["documents"][0][i],
                "source": results["metadatas"][0][i].get("source", "unknown"),
                "distance": (
                    results["distances"][0][i] if results.get("distances") else None
                ),
            }
        )
    return output


def add_to_index(text: str, source: str = "live_search") -> dict:
    """
    Dynamically add text (e.g. SerpAPI results) to the vector store
    so future queries can retrieve it.

    Args:
        text: The text content to index.
        source: A label indicating where the text came from.

    Returns:
        A dict with status and number of chunks added.
    """
    collection = _get_collection()
    chunks = chunk_text(text)
    if not chunks:
        return {"status": "no_content", "chunks_added": 0}

    ids = [f"{source}_{abs(hash(c)) % 999999}_c{i}" for i, c in enumerate(chunks)]
    metas = [{"source": source, "chunk_index": i} for i in range(len(chunks))]
    collection.upsert(ids=ids, documents=chunks, metadatas=metas)

    return {
        "status": "success",
        "chunks_added": len(chunks),
        "source": source,
        "total_documents_in_store": collection.count(),
    }
