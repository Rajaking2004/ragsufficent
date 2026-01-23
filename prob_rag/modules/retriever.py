"""
Module A: Standard Retriever

Implements vector-based retrieval using sentence transformers and FAISS.
Retrieves top-k relevant context chunks for a given query.
"""

import os
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None
    
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from ..config import RetrieverConfig


logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Container for retrieval results."""
    query: str
    contexts: List[str]
    scores: List[float]
    metadata: List[Dict[str, Any]]
    combined_context: str
    
    def __len__(self) -> int:
        return len(self.contexts)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "contexts": self.contexts,
            "scores": self.scores,
            "metadata": self.metadata,
            "combined_context": self.combined_context,
            "num_retrieved": len(self.contexts)
        }


class DocumentChunker:
    """Utility class for chunking documents."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str, doc_id: str = "") -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Input text to chunk
            doc_id: Document identifier for metadata
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text or not text.strip():
            return []
        
        words = text.split()
        chunks = []
        
        start = 0
        chunk_idx = 0
        
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)
            
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "doc_id": doc_id,
                    "chunk_idx": chunk_idx,
                    "start_word": start,
                    "end_word": end,
                    "num_words": len(chunk_words)
                }
            })
            
            # Move start position with overlap
            start = end - self.chunk_overlap if end < len(words) else end
            chunk_idx += 1
        
        return chunks
    
    def chunk_documents(self, documents: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of dicts with 'text' and optionally 'id' keys
            
        Returns:
            List of all chunks with metadata
        """
        all_chunks = []
        for idx, doc in enumerate(documents):
            doc_id = doc.get("id", f"doc_{idx}")
            text = doc.get("text", doc.get("content", ""))
            chunks = self.chunk_text(text, doc_id)
            all_chunks.extend(chunks)
        
        return all_chunks


class Retriever:
    """
    Module A: Standard Retriever
    
    Performs vector similarity search using sentence transformers embeddings
    and FAISS for efficient nearest neighbor search.
    """
    
    def __init__(self, config: RetrieverConfig):
        """
        Initialize the retriever.
        
        Args:
            config: RetrieverConfig with model and retrieval parameters
        """
        self.config = config
        self.encoder = None
        self.index = None
        self.documents: List[Dict[str, Any]] = []
        self.chunker = DocumentChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        
        self._load_encoder()
    
    def _load_encoder(self) -> None:
        """Load the sentence transformer model."""
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required. Install with: "
                "pip install sentence-transformers"
            )
        
        logger.info(f"Loading encoder: {self.config.model_name}")
        self.encoder = SentenceTransformer(self.config.model_name)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        logger.info(f"Encoder loaded. Embedding dim: {self.embedding_dim}")
    
    def _create_index(self, embeddings: np.ndarray) -> None:
        """Create FAISS index from embeddings."""
        if faiss is None:
            raise ImportError(
                "faiss is required. Install with: pip install faiss-cpu"
            )
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create index - using IndexFlatIP for inner product (cosine after normalization)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings)
        
        logger.info(f"FAISS index created with {self.index.ntotal} vectors")
    
    def index_documents(
        self,
        documents: List[Dict[str, str]],
        chunk: bool = True
    ) -> None:
        """
        Index documents for retrieval.
        
        Args:
            documents: List of document dicts with 'text' key
            chunk: Whether to chunk documents before indexing
        """
        logger.info(f"Indexing {len(documents)} documents...")
        
        # Chunk if needed
        if chunk:
            self.documents = self.chunker.chunk_documents(documents)
        else:
            self.documents = [
                {"text": d.get("text", d.get("content", "")), 
                 "metadata": d.get("metadata", {})}
                for d in documents
            ]
        
        if not self.documents:
            raise ValueError("No documents to index")
        
        # Extract texts and encode
        texts = [d["text"] for d in self.documents]
        logger.info(f"Encoding {len(texts)} chunks...")
        
        embeddings = self.encoder.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False  # We normalize in _create_index
        )
        
        # Create index
        self._create_index(embeddings.astype(np.float32))
        logger.info("Indexing complete")
    
    def index_from_contexts(self, contexts: List[str]) -> None:
        """
        Simple indexing from a list of context strings.
        Useful for per-query context from datasets.
        
        Args:
            contexts: List of context strings
        """
        documents = [{"text": ctx, "id": f"ctx_{i}"} for i, ctx in enumerate(contexts)]
        self.index_documents(documents, chunk=False)
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> RetrievalResult:
        """
        Retrieve relevant contexts for a query.
        
        Args:
            query: Query string
            top_k: Number of results (default: config.top_k)
            threshold: Minimum similarity threshold (default: config.similarity_threshold)
            
        Returns:
            RetrievalResult with contexts and scores
        """
        if self.index is None or len(self.documents) == 0:
            raise RuntimeError("No documents indexed. Call index_documents first.")
        
        top_k = top_k or self.config.top_k
        threshold = threshold if threshold is not None else self.config.similarity_threshold
        
        # Encode query
        query_embedding = self.encoder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)
        
        # Search
        scores, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))
        scores = scores[0]
        indices = indices[0]
        
        # Filter by threshold and collect results
        contexts = []
        filtered_scores = []
        metadata = []
        
        for score, idx in zip(scores, indices):
            if idx < 0:  # FAISS returns -1 for missing results
                continue
            if score < threshold:
                continue
            
            doc = self.documents[idx]
            contexts.append(doc["text"])
            filtered_scores.append(float(score))
            metadata.append(doc.get("metadata", {}))
        
        # Combine contexts
        combined = self._combine_contexts(contexts)
        
        return RetrievalResult(
            query=query,
            contexts=contexts,
            scores=filtered_scores,
            metadata=metadata,
            combined_context=combined
        )
    
    def _combine_contexts(self, contexts: List[str]) -> str:
        """
        Combine multiple contexts into a single string.
        
        Args:
            contexts: List of context strings
            
        Returns:
            Combined context string, truncated to max_context_length
        """
        if not contexts:
            return ""
        
        # Join with separators
        combined = "\n\n---\n\n".join(contexts)
        
        # Truncate if needed (rough word-based truncation)
        words = combined.split()
        if len(words) > self.config.max_context_length:
            combined = " ".join(words[:self.config.max_context_length])
            combined += "\n\n[Context truncated...]"
        
        return combined
    
    def save(self, path: str) -> None:
        """Save index and documents to disk."""
        os.makedirs(path, exist_ok=True)
        
        # Save FAISS index
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        
        # Save documents
        with open(os.path.join(path, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)
        
        # Save config
        with open(os.path.join(path, "config.pkl"), "wb") as f:
            pickle.dump(self.config, f)
        
        logger.info(f"Retriever saved to {path}")
    
    def load(self, path: str) -> None:
        """Load index and documents from disk."""
        # Load FAISS index
        index_path = os.path.join(path, "index.faiss")
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        
        # Load documents
        with open(os.path.join(path, "documents.pkl"), "rb") as f:
            self.documents = pickle.load(f)
        
        logger.info(f"Retriever loaded from {path}")


class PassthroughRetriever:
    """
    A simple retriever that passes through pre-retrieved contexts.
    Useful when contexts are already provided in the dataset.
    """
    
    def __init__(self, config: Optional[RetrieverConfig] = None):
        self.config = config or RetrieverConfig()
    
    def retrieve(
        self,
        query: str,
        contexts: Union[str, List[str]],
        **kwargs
    ) -> RetrievalResult:
        """
        Wrap pre-retrieved contexts in RetrievalResult format.
        
        Args:
            query: Query string
            contexts: Pre-retrieved context(s)
            
        Returns:
            RetrievalResult object
        """
        if isinstance(contexts, str):
            contexts = [contexts]
        
        # Dummy scores (1.0 for all since already retrieved)
        scores = [1.0] * len(contexts)
        metadata = [{"source": "provided"} for _ in contexts]
        combined = "\n\n---\n\n".join(contexts)
        
        return RetrievalResult(
            query=query,
            contexts=contexts,
            scores=scores,
            metadata=metadata,
            combined_context=combined
        )


def create_retriever(
    config: RetrieverConfig,
    use_passthrough: bool = False
) -> Union[Retriever, PassthroughRetriever]:
    """
    Factory function to create appropriate retriever.
    
    Args:
        config: RetrieverConfig
        use_passthrough: If True, create PassthroughRetriever
        
    Returns:
        Retriever instance
    """
    if use_passthrough:
        return PassthroughRetriever(config)
    return Retriever(config)
