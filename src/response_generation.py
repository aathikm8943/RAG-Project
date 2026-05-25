import os
from typing import List, Dict, Any, Optional

import yaml
import ollama

from retrieval import MilvusRetriever


class ResponseGenerator:
    """Production-grade RAG response generation system."""
    
    def __init__(self, collection_name: str = "nutrition_rag"):
        """Initialize the response generator with retriever and configuration.
        
        Args:
            collection_name: Name of the Milvus collection to query.
        """
        # Load configuration
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, 'config.yml')
        
        print(f"Loading configuration from: {config_path}")
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize retriever
        self.retriever = MilvusRetriever(collection_name=collection_name)
        
        # Default LLM model
        self.llm_model = "kimi-k2.5:cloud"
    
    def build_context(self, results: List[Dict[str, Any]]) -> str:
        """Build context string from retrieved results.
        
        Args:
            results: List of retrieved search results.
            
        Returns:
            Formatted context string for LLM prompt.
        """
        context = ""
        for i, result in enumerate(results):
            chunk = result["text"]
            context += f"\n\nChunk {i+1}:\n"
            context += chunk
        
        return context
    
    def create_rag_prompt(
        self,
        query: str,
        context: str
    ) -> str:
        """Create RAG prompt with context and question.
        
        Args:
            query: User's question.
            context: Retrieved context chunks.
            
        Returns:
            Formatted RAG prompt.
        """
        prompt = f"""
            You are a nutrition assistant.

            Answer the question using the context provided below.

            If the answer is not available in the context, say:
            'I could not find relevant information.'

            ----------------------------
            Context:
            {context}

            ----------------------------
            Question:
            {query}

            Answer:
        """ 
        return prompt
    
    def generate_response(
        self,
        prompt: str,
        model: Optional[str] = None
    ) -> str:
        """Generate response from LLM.
        
        Args:
            prompt: Input prompt for the LLM.
            model: Optional model override.
            
        Returns:
            Generated response text.
        """
        try:
            response = ollama.chat(
                model=model or self.llm_model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            return response["message"]["content"]
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I encountered an error while generating the response."
    
    def query(
        self,
        query: str,
        retrieval_method: str = "hybrid",
        top_k: int = 5,
        use_reranking: bool = False,
        model: Optional[str] = None,
        **retrieval_kwargs
    ) -> Dict[str, Any]:
        """Main RAG pipeline interface.
        
        Args:
            query: User's question.
            retrieval_method: Retrieval method to use.
            top_k: Number of chunks to retrieve.
            use_reranking: Whether to apply re-ranking.
            model: Optional LLM model override.
            **retrieval_kwargs: Additional retrieval parameters.
            
        Returns:
            Dictionary with answer, retrieved chunks, and metadata.
        """
        try:
            # Retrieve relevant chunks
            print(f"Retrieving chunks using {retrieval_method} method...")
            retrieved_chunks = self.retriever.retrieve(
                query=query,
                method=retrieval_method,
                top_k=top_k,
                use_reranking=use_reranking,
                **retrieval_kwargs
            )
            
            if not retrieved_chunks:
                return {
                    "answer": "I could not find relevant information.",
                    "retrieved_chunks": [],
                    "metadata": {
                        "query": query,
                        "retrieval_method": retrieval_method,
                        "chunks_found": 0
                    }
                }
            
            # Build context from retrieved chunks
            context = self.build_context(retrieved_chunks)
            
            # Create RAG prompt
            prompt = self.create_rag_prompt(query, context)
            
            # Generate response
            print("Generating response...")
            answer = self.generate_response(prompt, model=model)
            
            # Prepare metadata
            metadata = {
                "query": query,
                "retrieval_method": retrieval_method,
                "top_k": top_k,
                "use_reranking": use_reranking,
                "chunks_found": len(retrieved_chunks),
                "model_used": model or self.llm_model
            }
            
            return {
                "answer": answer,
                "retrieved_chunks": retrieved_chunks,
                "context": context,
                "metadata": metadata
            }
            
        except Exception as e:
            print(f"RAG pipeline error: {e}")
            return {
                "answer": "I encountered an error while processing your query.",
                "retrieved_chunks": [],
                "error": str(e),
                "metadata": {
                    "query": query,
                    "error": str(e)
                }
            }
    
    def compare_retrieval_methods(
        self,
        query: str,
        methods: List[str] = None,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """Compare different retrieval methods for the same query.
        
        Args:
            query: User's question.
            methods: List of retrieval methods to compare.
            top_k: Number of chunks per method.
            
        Returns:
            Dictionary with responses from each method.
        """
        if methods is None:
            methods = ["dense", "bm25", "hybrid", "weighted_hybrid", "hyde"]
        
        results = {}
        
        for method in methods:
            print(f"\nTesting {method} retrieval...")
            try:
                response = self.query(
                    query=query,
                    retrieval_method=method,
                    top_k=top_k
                )
                results[method] = response
            except Exception as e:
                results[method] = {
                    "error": str(e),
                    "method": method
                }
        
        return results
    
    def close(self) -> None:
        """Close retriever connection."""
        self.retriever.close()
