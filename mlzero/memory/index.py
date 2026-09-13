"""
Lightweight Semantic Index using TF-IDF for Semantic Memory.
"""
import json
import math
from collections import defaultdict
from pathlib import Path

from mlzero.core.logger import setup_logger
from mlzero.schemas.memory import KnowledgeChunk

logger = setup_logger(__name__)


def tokenize(text: str) -> list[str]:
    """Simple tokenization for TF-IDF."""
    import re
    text = text.lower()
    return re.findall(r'\b\w+\b', text)


class SemanticIndex:
    """A lightweight local semantic index based on TF-IDF."""
    
    def __init__(self) -> None:
        self.chunks: list[KnowledgeChunk] = []
        self.doc_freqs: dict[str, int] = defaultdict(int)
        
    def _compute_tf(self, tokens: list[str]) -> dict[str, float]:
        tf: dict[str, float] = defaultdict(float)
        for token in tokens:
            tf[token] += 1.0
        # Normalize
        for token, val in list(tf.items()):
            tf[token] = val / len(tokens)
        return tf
        
    def add(self, chunks: list[KnowledgeChunk]) -> None:
        """Add chunks to the index."""
        for chunk in chunks:
            tokens = tokenize(chunk.content)
            if not tokens:
                continue
                
            tf = self._compute_tf(tokens)
            chunk.embedding_metadata = dict(tf)
            
            # Update document frequencies
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] += 1
                
            self.chunks.append(chunk)

    def search(self, query: str, top_k: int = 3, library_filter: str | None = None) -> list[tuple[KnowledgeChunk, float]]:
        """Search the index for relevant chunks."""
        if not self.chunks:
            return []
            
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
            
        query_tf = self._compute_tf(query_tokens)
        N = len(self.chunks)
        
        # Precompute IDF for query terms
        query_idf = {}
        for token in set(query_tokens):
            df = self.doc_freqs.get(token, 0)
            idf = math.log((N + 1) / (df + 1)) + 1
            query_idf[token] = idf
            
        scored_chunks = []
        for chunk in self.chunks:
            if library_filter and (not chunk.library_name or library_filter not in chunk.library_name):
                continue
                
            score = 0.0
            tf = chunk.embedding_metadata
            
            for token, q_tf in query_tf.items():
                if token in tf:
                    # TF-IDF dot product
                    chunk_tf_idf = tf[token] * query_idf[token]
                    query_tf_idf = q_tf * query_idf[token]
                    score += chunk_tf_idf * query_tf_idf
                    
            if score > 0:
                scored_chunks.append((chunk, score))
                
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:top_k]

    def save(self, path: Path | str) -> None:
        """Save the index to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "doc_freqs": self.doc_freqs,
            "chunks": [json.loads(chunk.model_dump_json()) for chunk in self.chunks]
        }
        path.write_text(json.dumps(data), encoding="utf-8")
        
    def load(self, path: Path | str) -> None:
        """Load the index from disk."""
        path = Path(path)
        if not path.exists():
            return
            
        data = json.loads(path.read_text(encoding="utf-8"))
        self.doc_freqs = defaultdict(int, data.get("doc_freqs", {}))
        
        self.chunks = []
        for chunk_data in data.get("chunks", []):
            self.chunks.append(KnowledgeChunk(**chunk_data))
