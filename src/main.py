"""
Main entry point for the RAG pipeline.

This module provides command-line interface and demonstration functions
for testing the retrieval and response generation components.
"""

import argparse
import sys
from typing import Optional

from ingestion import PDFIngestion
from retrieval import MilvusRetriever
from response_generation import ResponseGenerator
from milvus_setup import MilvusSetup


def run_ingestion():
    """Run PDF ingestion pipeline."""
    print("=" * 60)
    print("RUNNING PDF INGESTION PIPELINE")
    print("=" * 60)
    
    try:
        ingestion = PDFIngestion()
        milvus_setup = MilvusSetup()
        
        # Create collection using new API
        milvus_setup.create_collection(
            collection_name="nutrition_rag",
            index_params={
                "index_type": "IVF_FLAT",
                "metric_type": "COSINE",
                "params": {"nlist": 128}
            }
        )
        
        # For ingestion, we still need to use the old Collection API
        # This will be updated in a future refactor
        from pymilvus import Collection, FieldSchema, CollectionSchema, DataType
        
        collection = Collection(name="nutrition_rag")
        ingestion.ingest_to_milvus(collection)
        print("\n✓ Ingestion completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Ingestion failed: {e}")
        sys.exit(1)


def test_retrieval(
    query: str,
    method: str = "hybrid",
    top_k: int = 5,
    use_reranking: bool = False
):
    """Test retrieval methods.
    
    Args:
        query: Search query.
        method: Retrieval method (dense, bm25, hybrid, weighted_hybrid, hyde).
        top_k: Number of results to retrieve.
        use_reranking: Whether to apply re-ranking.
    """
    print("=" * 60)
    print(f"TESTING RETRIEVAL: {method.upper()}")
    print("=" * 60)
    print(f"Query: {query}")
    print(f"Top-K: {top_k}")
    print(f"Use Re-ranking: {use_reranking}")
    print("-" * 60)
    
    try:
        retriever = MilvusRetriever(collection_name="nutrition_rag")
        
        results = retriever.retrieve(
            query=query,
            method=method,
            top_k=top_k,
            use_reranking=use_reranking
        )
        
        if not results:
            print("\nNo results found.")
            return
        
        print(f"\nFound {len(results)} results:\n")
        for i, result in enumerate(results, 1):
            print(f"RESULT #{i}")
            print("-" * 50)
            print(f"Source: {result['source']}")
            print(f"Score: {result['score']:.4f}")
            print(f"Text: {result['text'][:200]}...")
            print()
        
        retriever.close()
        
    except Exception as e:
        print(f"\n✗ Retrieval failed: {e}")
        sys.exit(1)


def test_rag_pipeline(
    query: str,
    retrieval_method: str = "hybrid",
    top_k: int = 5,
    use_reranking: bool = False,
    model: Optional[str] = None
):
    """Test complete RAG pipeline.
    
    Args:
        query: User question.
        retrieval_method: Retrieval method to use.
        top_k: Number of chunks to retrieve.
        use_reranking: Whether to apply re-ranking.
        model: LLM model to use.
    """
    print("=" * 60)
    print("TESTING RAG PIPELINE")
    print("=" * 60)
    print(f"Query: {query}")
    print(f"Retrieval Method: {retrieval_method}")
    print(f"Top-K: {top_k}")
    print(f"Use Re-ranking: {use_reranking}")
    print(f"Model: {model or 'default'}")
    print("-" * 60)
    
    try:
        rag = ResponseGenerator(collection_name="nutrition_rag")
        
        result = rag.query(
            query=query,
            retrieval_method=retrieval_method,
            top_k=top_k,
            use_reranking=use_reranking,
            model=model
        )
        
        print("\n" + "=" * 60)
        print("ANSWER:")
        print("=" * 60)
        print(result["answer"])
        
        print("\n" + "-" * 60)
        print("METADATA:")
        print("-" * 60)
        for key, value in result["metadata"].items():
            print(f"{key}: {value}")
        
        rag.close()
        
    except Exception as e:
        print(f"\n✗ RAG pipeline failed: {e}")
        sys.exit(1)


def compare_methods(query: str, top_k: int = 5):
    """Compare different retrieval methods.
    
    Args:
        query: User question.
        top_k: Number of chunks per method.
    """
    print("=" * 60)
    print("COMPARING RETRIEVAL METHODS")
    print("=" * 60)
    print(f"Query: {query}")
    print(f"Top-K: {top_k}")
    print("-" * 60)
    
    try:
        rag = ResponseGenerator(collection_name="nutrition_rag")
        
        results = rag.compare_retrieval_methods(
            query=query,
            top_k=top_k
        )
        
        for method, response in results.items():
            print(f"\n{'=' * 60}")
            print(f"METHOD: {method.upper()}")
            print("=" * 60)
            
            if "error" in response and len(response) == 1:
                print(f"Error: {response['error']}")
            else:
                print(f"Answer: {response['answer']}")
                print(f"Chunks Found: {response['metadata']['chunks_found']}")
        
        rag.close()
        
    except Exception as e:
        print(f"\n✗ Comparison failed: {e}")
        sys.exit(1)


def interactive_mode():
    """Run interactive Q&A mode."""
    print("=" * 60)
    print("INTERACTIVE RAG MODE")
    print("=" * 60)
    print("Type your questions below. Type 'quit' to exit.")
    print("-" * 60)
    
    try:
        rag = ResponseGenerator(collection_name="nutrition_rag")
        
        while True:
            query = input("\nYour question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not query:
                continue
            
            result = rag.query(
                query=query,
                retrieval_method="hybrid",
                top_k=5,
                use_reranking=True
            )
            
            print("\n" + "-" * 60)
            print("Answer:")
            print("-" * 60)
            print(result["answer"])
        
        rag.close()
        
    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Interactive mode failed: {e}")
        sys.exit(1)


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="RAG Pipeline for Nutrition Documents"
    )
    
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands"
    )
    
    # Ingestion command
    subparsers.add_parser(
        "ingest",
        help="Ingest PDFs into Milvus"
    )
    
    # Retrieval command
    retrieval_parser = subparsers.add_parser(
        "retrieve",
        help="Test retrieval methods"
    )
    retrieval_parser.add_argument(
        "query",
        type=str,
        help="Search query"
    )
    retrieval_parser.add_argument(
        "--method",
        type=str,
        default="hybrid",
        choices=["dense", "bm25", "hybrid", "weighted_hybrid", "hyde"],
        help="Retrieval method"
    )
    retrieval_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results"
    )
    retrieval_parser.add_argument(
        "--rerank",
        action="store_true",
        help="Apply re-ranking"
    )
    
    # RAG command
    rag_parser = subparsers.add_parser(
        "rag",
        help="Run complete RAG pipeline"
    )
    rag_parser.add_argument(
        "query",
        type=str,
        help="User question"
    )
    rag_parser.add_argument(
        "--method",
        type=str,
        default="hybrid",
        choices=["dense", "bm25", "hybrid", "weighted_hybrid", "hyde"],
        help="Retrieval method"
    )
    rag_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks"
    )
    rag_parser.add_argument(
        "--rerank",
        action="store_true",
        help="Apply re-ranking"
    )
    rag_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model"
    )
    
    # Compare command
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare retrieval methods"
    )
    compare_parser.add_argument(
        "query",
        type=str,
        help="User question"
    )
    compare_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks per method"
    )
    
    # Interactive command
    subparsers.add_parser(
        "interactive",
        help="Run interactive Q&A mode"
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    if args.command == "ingest":
        run_ingestion()
    
    elif args.command == "retrieve":
        test_retrieval(
            query=args.query,
            method=args.method,
            top_k=args.top_k,
            use_reranking=args.rerank
        )
    
    elif args.command == "rag":
        test_rag_pipeline(
            query=args.query,
            retrieval_method=args.method,
            top_k=args.top_k,
            use_reranking=args.rerank,
            model=args.model
        )
    
    elif args.command == "compare":
        compare_methods(
            query=args.query,
            top_k=args.top_k
        )
    
    elif args.command == "interactive":
        interactive_mode()


if __name__ == "__main__":
    main()
