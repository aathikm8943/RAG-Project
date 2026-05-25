# Simple RAG Pipeline Development

A production-grade Retrieval-Augmented Generation (RAG) pipeline for nutrition documents with multiple retrieval strategies and re-ranking capabilities.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Usage Guide](#usage-guide)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

This project implements a comprehensive RAG system for querying nutrition-related PDF documents. It supports multiple retrieval methods (dense vector search, BM25, hybrid approaches) and advanced techniques like HyDE (Hypothetical Document Embeddings) and cross-encoder re-ranking.

## ✨ Features

### Retrieval Methods
- **Dense Retrieval** - Semantic search using vector embeddings (nomic-embed-text)
- **BM25 Retrieval** - Keyword-based lexical search
- **Hybrid Retrieval** - Combination of dense and BM25 results
- **Weighted Hybrid** - Score fusion with configurable weights
- **HyDE Retrieval** - Hypothetical Document Embeddings for query expansion

### Advanced Features
- **Cross-Encoder Re-ranking** - BGE re-ranker for improved result quality
- **Multiple Chunking Strategies** - Semantic and recursive character chunking
- **Milvus Vector Database** - High-performance vector similarity search
- **Configurable Pipeline** - YAML-based configuration
- **CLI Interface** - Command-line tools for all operations

## 🏗️ Architecture

```
┌─────────────────┐
│   User Query    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│      ResponseGenerator              │
│  ┌─────────────────────────────┐    │
│  │   MilvusRetriever           │    │
│  │  ┌─────────┐ ┌──────────┐   │    │
│  │  │  Dense  │ │   BM25    │   │    │
│  │  └─────────┘ └──────────┘   │    │
│  │         \       /           │    │
│  │      Hybrid Fusion          │    │
│  │         │                   │    │
│  │      Re-ranker              │    │
│  └─────────┴───────────────────┘    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────┐
│   LLM (qwen3.5/kimi)    │
│   Response Generation   │
└─────────────────────────┘
```

## 📦 Prerequisites

- **Docker & Docker Compose** - For Milvus vector database
- **Python 3.10+** - Runtime environment
- **uv** - Package manager (recommended) or pip
- **Ollama** - For local embedding generation

### Install Docker Desktop
```bash
# Windows
# Download from https://www.docker.com/products/docker-desktop

# Verify installation
docker --version
docker-compose --version
```

### Install Ollama
```bash
# Windows (PowerShell)
winget install Ollama.Ollama

# Pull required models
ollama pull nomic-embed-text:v1.5
```

## 🚀 Installation

### 1. Clone the Repository
```bash
cd simple_rag_pipeline_development
```

### 2. Set Up Python Environment
```bash
# Using uv (recommended)
uv venv
source .venv/Scripts/activate  # Windows
uv sync

# Or using pip
python -m venv .venv
source .venv/Scripts/activate  # Windows
pip install -r requirements.txt
```

### 3. Start Milvus with Docker
```bash
# Start Milvus services
docker-compose up -d

# Verify services are running
docker-compose ps

# Expected output:
# NAME                STATUS
# milvus-etcd         Up (healthy)
# milvus-minio        Up (healthy)
# milvus-standalone   Up (healthy)
# attu                Up
```

### 4. Verify Installation
```bash
cd src
python main.py --help
```

## 📁 Project Structure

```
simple_rag_pipeline_development/
├── src/
│   ├── config.yml              # Configuration file
│   ├── ingestion.py            # PDF ingestion pipeline
│   ├── retrieval.py            # Retrieval methods
│   ├── response_generation.py  # RAG pipeline & LLM integration
│   ├── milvus_setup.py         # Milvus database setup
│   └── main.py                 # CLI entry point
├── experiments/
│   ├── ingestion_experiments.ipynb
│   └── retrieval_experiments.ipynb
├── data/
│   ├── raw_pdf/                # Input PDF files
│   └── processed/              # Processed data
├── volumes/                    # Milvus persistent data
├── docker-compose.yml          # Docker services configuration
├── pyproject.toml              # Python dependencies
└── README.md                   # This file
```

## ⚙️ Configuration

Edit `src/config.yml` to customize settings:

```yaml
# PDF ingestion settings
FOLDER_PATH: '../data/raw_pdf'

# RAG Pipeline Config
CHUNK_SIZE: 1000              # Characters per chunk
CHUNK_OVERLAP: 200            # Overlap between chunks
THRESHOLD: 0.75               # Semantic similarity threshold
EMBEDDING_MODEL: 'nomic-embed-text:v1.5'  # Ollama embedding model
```

## 📖 Usage Guide

### Quick Start

#### 1. Ingest PDF Documents
```bash
# Place your PDF files in data/raw_pdf/
# Then run:
cd src
python main.py ingest
```

#### 2. Test Retrieval
```bash
# Test hybrid retrieval
python main.py retrieve "pesticide residues from food products" --method hybrid --top-k 5

# Test with re-ranking
python main.py retrieve "dietary guidelines" --method hybrid --top-k 10 --rerank
```

#### 3. Run Complete RAG Pipeline
```bash
# Get AI-generated answer
python main.py rag "what are the nutrition recommendations?" --method hybrid --top-k 5
```

#### 4. Compare Retrieval Methods
```bash
# Compare all methods side-by-side
python main.py compare "what are the dietary guidelines?" --top-k 5
```

#### 5. Interactive Mode
```bash
# Chat-like interface
python main.py interactive
```

## 💡 Examples

### Example 1: Basic Retrieval
```bash
python main.py retrieve "how many dietary guidelines are available?" \
  --method dense \
  --top-k 5
```

**Output:**
```
============================================================
TESTING RETRIEVAL: DENSE
============================================================
Query: how many dietary guidelines are available?
Top-K: 5
------------------------------------------------------------

Found 5 results:

RESULT #1
--------------------------------------------------
Source: dense
Score: 0.8234
Text: The dietary guidelines provide recommendations...
```

### Example 2: RAG with Re-ranking
```bash
python main.py rag "what foods should I avoid for a healthy diet?" \
  --method weighted_hybrid \
  --top-k 10 \
  --rerank \
  --dense-weight 0.7 \
  --bm25-weight 0.3
```

### Example 3: Method Comparison
```bash
python main.py compare "impact of processed foods on health" --top-k 3
```

**Output:**
```
============================================================
METHOD: DENSE
============================================================
Answer: Processed foods high in saturated fats...
Chunks Found: 3

============================================================
METHOD: BM25
============================================================
Answer: Studies show correlation between processed food...
Chunks Found: 3

============================================================
METHOD: HYBRID
============================================================
Answer: Both semantic and keyword matching reveal...
Chunks Found: 5
```

### Example 4: Programmatic Usage
```python
from response_generation import ResponseGenerator

# Initialize
rag = ResponseGenerator()

# Query with custom parameters
response = rag.query(
    query="How does sodium affect blood pressure?",
    retrieval_method="hybrid",
    top_k=5,
    use_reranking=True,
    dense_weight=0.6,
    bm25_weight=0.4
)

# Access results
print(f"Answer: {response['answer']}")
print(f"Chunks used: {response['metadata']['chunks_found']}")
print(f"Retrieval method: {response['metadata']['retrieval_method']}")

# Cleanup
rag.close()
```

## 🧪 Running Experiments

The `experiments/` directory contains Jupyter notebooks for experimentation:

```bash
# Start Jupyter
uv run jupyter notebook

# Open in browser
# Navigate to experiments/retrieval_experiments.ipynb
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Milvus Connection Error
```bash
# Check if Milvus is running
docker-compose ps

# Restart Milvus
docker-compose down
docker-compose up -d

# Wait for health check
docker-compose logs -f milvus-standalone
```

#### 2. Missing Embeddings Model
```bash
# Pull the required model
ollama pull nomic-embed-text:v1.5

# Verify
ollama list
```

#### 3. No Results Found
```bash
# Check if data is ingested
docker-compose exec milvus-standalone milvus-cli query -c nutrition_rag -f "id >= 0"

# Re-ingest if needed
python main.py ingest
```

#### 4. PyMilvus Deprecation Warnings
The code uses `warnings.filterwarnings` to suppress deprecation warnings. If you still see them, ensure you're using:
- PyMilvus >= 2.4.0
- Updated imports in `retrieval.py`

#### 5. Out of Memory Errors
```bash
# Reduce batch size in config.yml
CHUNK_SIZE: 500  # Reduce from 1000

# Or limit retrieval results
python main.py rag "query" --top-k 3
```

### Performance Tips

1. **Use Re-ranking Sparingly** - Re-ranking improves quality but adds latency
2. **Adjust Top-K** - Start with 5-10 results for most queries
3. **Hybrid Methods** - Generally provide better coverage than single methods
4. **Monitor Milvus** - Use Attu UI at http://localhost:8000

## 📊 Monitoring

### Attu Web UI
Access Milvus management interface:
```
http://localhost:8000
```

### Check Collection Stats
```python
from pymilvus import MilvusClient

client = MilvusClient(uri="http://localhost:19530")
stats = client.get_collection_stats("nutrition_rag")
print(stats)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- [Milvus](https://milvus.io/) - Vector database
- [Ollama](https://ollama.ai/) - Local LLM and embeddings
- [LangChain](https://python.langchain.com/) - Document processing
- [BGE Re-ranker](https://huggingface.co/BAAI/bge-reranker-base) - Re-ranking model

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review experiment notebooks
3. Open an issue on GitHub
