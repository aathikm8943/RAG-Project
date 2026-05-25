import re
import os
import warnings
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

import numpy as np
import yaml
import ollama
from pymilvus import MilvusClient
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# Suppress PyMilvus deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pymilvus")


class RetrievalSource(Enum):
    """Enum for retrieval source types."""
    DENSE = "dense"
    BM25 = "bm25"
    HYDE = "hyde"
    HYBRID = "hybrid"


@dataclass
class SearchResult:
    """Data class for search results."""
    text: str
    score: float
    source: str


class MilvusRetriever:
    """Production-grade retrieval system for RAG pipeline."""
    
    def __init__(self, collection_name: str = "nutrition_rag"):
        """Initialize the retriever with Milvus connection and configuration.
        
        Args:
            collection_name: Name of the Milvus collection to query.
        """
        # Load configuration
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, 'config.yml')
        
        print(f"Loading configuration from: {config_path}")
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize Milvus client
        self.client = MilvusClient(
            uri="http://localhost:19530",
            token=""
        )
        print("Successfully connected to Milvus")
        
        # Store collection name
        self.collection_name = collection_name
        
        # Initialize BM25 index
        self._initialize_bm25()
        
        # Initialize re-ranker
        self._initialize_reranker()
    
    def _initialize_bm25(self) -> None:
        """Initialize BM25 index with all chunks from collection."""
        try:
            print("Initializing BM25 index...")
            results = self.client.query(
                collection_name=self.collection_name,
                filter="id >= 0",
                output_fields=["text"]
            )
            
            self.all_chunks = [r["text"] for r in results]
            
            # Clean and tokenize chunks
            cleaned_chunks = [self._clean_text(chunk) for chunk in self.all_chunks]
            tokenized_chunks = [chunk.split() for chunk in cleaned_chunks]
            
            # Create BM25 index
            self.bm25 = BM25Okapi(tokenized_chunks)
            print(f"BM25 index initialized with {len(self.all_chunks)} chunks")
            
        except Exception as e:
            print(f"Warning: BM25 initialization failed: {e}")
            self.bm25 = None
    
    def _initialize_reranker(self) -> None:
        """Initialize cross-encoder re-ranker model."""
        try:
            print("Loading re-ranker model...")
            self.reranker = CrossEncoder("BAAI/bge-reranker-base")
            print("Re-ranker model loaded successfully")
        except Exception as e:
            print(f"Warning: Re-ranker initialization failed: {e}")
            self.reranker = None
    
    def _clean_text(self, text: str) -> str:
        """Clean text for BM25 processing.
        
        Args:
            text: Raw text to clean.
            
        Returns:
            Cleaned text in lowercase with special characters removed.
        """
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text
    
    def _convert_embeddings(self, prompt: str) -> np.ndarray:
        """Convert text to embeddings using configured model.
        
        Args:
            prompt: Text to convert to embeddings.
            
        Returns:
            Numpy array of embeddings.
        """
        response = ollama.embeddings(
            model=self.config['EMBEDDING_MODEL'],
            prompt=prompt
        )
        return np.array(response['embedding'])
    
    def _generate_hypothetical_answer(self, query: str) -> str:
        """Generate hypothetical answer using LLM for HyDE approach.
        
        Args:
            query: User query.
            
        Returns:
            Generated hypothetical answer.
        """
        try:
            response = ollama.chat(
                model="qwen3.5:cloud",
                messages=[
                    {
                        "role": "user",
                        "content": f"Answer briefly: {query}"
                    }
                ]
            )
            return response["message"]["content"]
        except Exception as e:
            print(f"Warning: HyDE generation failed: {e}")
            return query
    
    def dense_retrieval(
        self,
        query: str,
        top_k: int = 5,
        chunk_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Perform dense vector retrieval using semantic search.
        
        Args:
            query: Search query.
            top_k: Number of results to return.
            chunk_type: Optional filter for chunk type.
            
        Returns:
            List of search results with text, score, and source.
        """
        try:
            # Convert query to embedding
            query_embedding = self._convert_embeddings(query)
            
            # Build search parameters
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }
            
            # Build filter expression
            filter_expr = None
            if chunk_type:
                filter_expr = f'chunk_type == "{chunk_type}"'
            
            # Perform search using MilvusClient
            results = self.client.search(
                collection_name=self.collection_name,
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                filter=filter_expr,
                output_fields=["text"]
            )
            
            # Format results
            dense_results = []
            for hits in results:
                for hit in hits:
                    dense_results.append({
                        "text": hit.get("entity", {}).get("text"),
                        "score": float(hit["score"]),
                        "source": RetrievalSource.DENSE.value
                    })
            
            return dense_results
            
        except Exception as e:
            print(f"Dense retrieval error: {e}")
            return []
    
    def bm25_retrieval(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Perform BM25 keyword-based retrieval.
        
        Args:
            query: Search query.
            top_k: Number of results to return.
            
        Returns:
            List of search results with text, score, and source.
        """
        if self.bm25 is None:
            print("BM25 index not initialized")
            return []
        
        try:
            # Clean and tokenize query
            cleaned_query = self._clean_text(query)
            tokenized_query = cleaned_query.split()
            
            # Get BM25 scores
            scores = self.bm25.get_scores(tokenized_query)
            
            # Pair chunks with scores
            chunk_score_pairs = list(zip(self.all_chunks, scores))
            
            # Sort by score descending
            ranked_results = sorted(
                chunk_score_pairs,
                key=lambda x: x[1],
                reverse=True
            )
            
            # Get top-k results
            top_results = ranked_results[:top_k]
            
            # Format results
            formatted_results = []
            for chunk, score in top_results:
                formatted_results.append({
                    "text": chunk,
                    "score": float(score),
                    "source": RetrievalSource.BM25.value
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"BM25 retrieval error: {e}")
            return []
    
    def hybrid_retrieval(
        self,
        query: str,
        dense_k: int = 5,
        bm25_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Perform hybrid retrieval combining dense and BM25 results.
        
        Args:
            query: Search query.
            dense_k: Number of dense retrieval results.
            bm25_k: Number of BM25 retrieval results.
            
        Returns:
            List of unique search results from both methods.
        """
        try:
            # Get results from both methods
            dense_results = self.dense_retrieval(query, top_k=dense_k)
            bm25_results = self.bm25_retrieval(query, top_k=bm25_k)
            
            # Combine and deduplicate
            combined = dense_results + bm25_results
            
            unique_results = list(
                {item["text"]: item for item in combined}.values()
            )
            
            return unique_results
            
        except Exception as e:
            print(f"Hybrid retrieval error: {e}")
            return []
    
    def weighted_hybrid_retrieval(
        self,
        query: str,
        dense_weight: float = 0.7,
        bm25_weight: float = 0.3,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Perform weighted hybrid retrieval with score fusion.
        
        Args:
            query: Search query.
            dense_weight: Weight for dense retrieval scores.
            bm25_weight: Weight for BM25 scores.
            top_k: Number of results to return.
            
        Returns:
            List of search results with weighted scores.
        """
        try:
            dense_results = self.dense_retrieval(query, top_k=top_k)
            bm25_results = self.bm25_retrieval(query, top_k=top_k)
            
            # Combine scores
            combined = {}
            
            # Add dense results
            for r in dense_results:
                text = r["text"]
                combined[text] = {
                    "text": text,
                    "score": r["score"] * dense_weight
                }
            
            # Add BM25 results
            for r in bm25_results:
                text = r["text"]
                if text in combined:
                    combined[text]["score"] += r["score"] * bm25_weight
                else:
                    combined[text] = {
                        "text": text,
                        "score": r["score"] * bm25_weight
                    }
            
            # Rank by combined score
            ranked = sorted(
                combined.values(),
                key=lambda x: x["score"],
                reverse=True
            )
            
            return ranked[:top_k]
            
        except Exception as e:
            print(f"Weighted hybrid retrieval error: {e}")
            return []
    
    def hyde_retrieval(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Perform retrieval using HyDE (Hypothetical Document Embeddings).
        
        Args:
            query: Search query.
            top_k: Number of results to return.
            
        Returns:
            List of search results based on hypothetical answer embedding.
        """
        try:
            # Generate hypothetical answer
            hypothetical_answer = self._generate_hypothetical_answer(query)
            
            # Use hypothetical answer for retrieval
            query_embedding = self._convert_embeddings(hypothetical_answer)
            
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }
            
            results = self.client.search(
                collection_name=self.collection_name,
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["text"]
            )
            
            # Format results
            formatted_results = []
            for hits in results:
                for hit in hits:
                    formatted_results.append({
                        "text": hit.get("entity", {}).get("text"),
                        "score": float(hit["score"]),
                        "source": RetrievalSource.HYDE.value
                    })
            
            return formatted_results
            
        except Exception as e:
            print(f"HyDE retrieval error: {e}")
            return []
    
    def rerank(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Re-rank retrieved chunks using cross-encoder model.
        
        Args:
            query: Original search query.
            retrieved_chunks: List of retrieved chunks to re-rank.
            top_k: Optional number of top results to return.
            
        Returns:
            List of re-ranked search results.
        """
        if self.reranker is None:
            print("Re-ranker not initialized, returning original order")
            return retrieved_chunks
        
        try:
            # Create query-chunk pairs
            pairs = [
                [query, chunk["text"]]
                for chunk in retrieved_chunks
            ]
            
            # Get re-ranking scores
            scores = self.reranker.predict(pairs)
            
            # Combine chunks with new scores
            reranked = []
            for chunk, score in zip(retrieved_chunks, scores):
                reranked.append({
                    "text": chunk["text"],
                    "score": float(score),
                    "source": chunk.get("source", "reranked")
                })
            
            # Sort by re-ranking score
            reranked = sorted(
                reranked,
                key=lambda x: x["score"],
                reverse=True
            )
            
            # Return top-k if specified
            if top_k:
                return reranked[:top_k]
            
            return reranked
            
        except Exception as e:
            print(f"Re-ranking error: {e}")
            return retrieved_chunks
    
    def retrieve(
        self,
        query: str,
        method: str = "hybrid",
        top_k: int = 5,
        use_reranking: bool = False,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Main retrieval interface supporting multiple methods.
        
        Args:
            query: Search query.
            method: Retrieval method ('dense', 'bm25', 'hybrid', 
                    'weighted_hybrid', 'hyde').
            top_k: Number of results to return.
            use_reranking: Whether to apply cross-encoder re-ranking.
            **kwargs: Additional method-specific parameters.
            
        Returns:
            List of final search results.
        """
        # Select retrieval method
        if method == "dense":
            results = self.dense_retrieval(query, top_k=top_k, **kwargs)
        elif method == "bm25":
            results = self.bm25_retrieval(query, top_k=top_k)
        elif method == "hybrid":
            results = self.hybrid_retrieval(
                query,
                dense_k=kwargs.get('dense_k', top_k),
                bm25_k=kwargs.get('bm25_k', top_k)
            )
        elif method == "weighted_hybrid":
            results = self.weighted_hybrid_retrieval(
                query,
                dense_weight=kwargs.get('dense_weight', 0.7),
                bm25_weight=kwargs.get('bm25_weight', 0.3),
                top_k=top_k
            )
        elif method == "hyde":
            results = self.hyde_retrieval(query, top_k=top_k)
        else:
            print(f"Unknown retrieval method: {method}")
            return []
        
        # Apply re-ranking if requested
        if use_reranking and results:
            results = self.rerank(query, results, top_k=top_k)
        
        return results
    
    def close(self) -> None:
        """Close Milvus client connection."""
        try:
            self.client.close()
            print("Milvus connection closed")
        except Exception as e:
            print(f"Error closing connection: {e}")
