# Project Documentation

This project contains three main components:
1. **Intent Classifier Training** - Trains transformer models for intent classification
2. **Knowledge Base Builder** - Builds a vector knowledge base from multiple datasets
3. **RAG Engine** - Retrieval-Augmented Generation system for grounded question answering

---

## Table of Contents

- [Setup](#setup)
- [Configuration](#configuration)
- [Intent Classifier Training](#intent-classifier-training)
- [Knowledge Base Builder](#knowledge-base-builder)
- [RAG Engine](#rag-engine)
- [End-to-End Workflow](#end-to-end-workflow)
- [File Structure](#file-structure)

---

## Setup

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Installation

1. **Clone the repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd Project
   ```

2. **Install all dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   This installs all required packages for both intent classifier training and knowledge base building.

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   ```
   
   Then edit `.env` and add your credentials:
   ```
   QDRANT_URL=your_qdrant_cloud_url_here
   QDRANT_API_KEY=your_qdrant_api_key_here
   COLLECTION_NAME=pattern_recognition
   OPENAI_API_KEY=your_openai_api_key_here
   ```

---

## Configuration

### Environment Variables (`.env` file)

The project requires the following credentials stored in a `.env` file:

**Qdrant Cloud** (for knowledge base):
- `QDRANT_URL`: Your Qdrant Cloud cluster URL
- `QDRANT_API_KEY`: Your Qdrant Cloud API key
- `COLLECTION_NAME`: Name of the collection (default: `pattern_recognition`)

**OpenAI** (for RAG engine):
- `OPENAI_API_KEY`: Your OpenAI API key for GPT-4o-mini

**Important**: Never commit the `.env` file to version control. The `.gitignore` file is configured to exclude it.

---

## Intent Classifier Training

### Overview

The intent classifier training scripts train transformer models to classify questions by intent type across multiple datasets (TREC, SQuAD, SciQ).

### Scripts

#### 1. `train_intent_classifier.py`

**Purpose**: Trains an intent classifier using DistilBERT-base-uncased model.

**What it does**:
- Loads TREC, SQuAD, and SciQ datasets
- Preprocesses questions and extracts intent labels
- Trains a DistilBERT-based sequence classification model
- Evaluates the model and saves it to `./intent_classifier_model/`

**Usage**:
```bash
python3 train_intent_classifier.py [options]
```

**Options**:
- `--model`: Model name (default: `distilbert-base-uncased`)
- `--output_dir`: Output directory (default: `./intent_classifier_model`)
- `--epochs`: Number of training epochs (default: `3`)
- `--batch_size`: Batch size (default: `16`)
- `--learning_rate`: Learning rate (default: `2e-5`)
- `--max_samples`: Maximum samples per dataset (default: `None`, uses all)

**Example**:
```bash
python3 train_intent_classifier.py --epochs 5 --batch_size 32 --max_samples 10000
```

#### 2. `train_intent_classifier_roberta.py`

**Purpose**: Trains an intent classifier using RoBERTa-base model (typically better performance than DistilBERT).

**What it does**:
- Same as `train_intent_classifier.py` but uses `roberta-base` model
- Generally produces better results but is slower to train
- Saves model to `./intent_classifier_model_roberta/`

**Usage**:
```bash
python3 train_intent_classifier_roberta.py [options]
```

**Options**: Same as `train_intent_classifier.py`

**Example**:
```bash
python3 train_intent_classifier_roberta.py --epochs 5 --batch_size 16
```

#### 3. `evaluate_per_dataset.py`

**Purpose**: Evaluates a trained intent classifier on each dataset separately to analyze performance differences.

**What it does**:
- Loads a trained model from a specified path
- Evaluates on TREC, SQuAD, and SciQ datasets individually
- Generates per-dataset metrics (accuracy, precision, recall, F1)
- Creates confusion matrices and classification reports
- Saves evaluation results to `per_dataset_evaluation.json`

**Usage**:
```bash
python3 evaluate_per_dataset.py --model_path <path_to_model> [options]
```

**Options**:
- `--model_path`: Path to trained model directory (required)
- `--batch_size`: Batch size for evaluation (default: `32`)

**Example**:
```bash
python3 evaluate_per_dataset.py --model_path ./intent_classifier_model_roberta
```

---

## Knowledge Base Builder

### Overview

The knowledge base builder creates a vector knowledge base by:
1. Loading documents from multiple datasets
2. Chunking documents into smaller pieces
3. Generating embeddings using sentence transformers
4. Storing embeddings in Qdrant Cloud for semantic search

### Components

#### 1. `dataset_loaders.py`

**Purpose**: Contains functions to load and process datasets from various sources.

**Functions**:
- `load_squad_contexts()`: Loads contexts from SQuAD dataset
- `load_sciq_contexts()`: Loads contexts from SciQ dataset
- `load_wikipedia_simple()`: Loads Wikipedia Simple articles (capped at 150,000)
- `load_hotpotqa()`: Loads HotpotQA paragraphs (capped at 10,000)

**Features**:
- Automatic deduplication using MD5 hashing
- Configurable sample limits to control storage usage
- Error handling with informative messages

#### 2. `knowledge_base.py`

**Purpose**: Main script for building and querying the knowledge base.

**What it does**:
- **DocumentChunker**: Token-based chunking (256 tokens per chunk, 25 token overlap)
- **KnowledgeBaseBuilder**: Main class that:
  - Connects to Qdrant Cloud
  - Loads embedding model (sentence-transformers/all-mpnet-base-v2)
  - Chunks documents
  - Generates embeddings
  - Stores in Qdrant
  - Provides retrieval functionality

**Usage**:

**Build knowledge base**:
```bash
python3 knowledge_base.py [options]
```

**Query knowledge base**:
```bash
python3 knowledge_base.py --query "your question here" --top_k 5
```

**Options**:
- `--qdrant_url`: Qdrant Cloud URL (default: from `.env`)
- `--qdrant_api_key`: Qdrant API key (default: from `.env`)
- `--collection_name`: Collection name (default: from `.env`)
- `--sources`: Data sources to include (default: `squad sciq wikipedia_simple hotpotqa`)
- `--max_samples`: Maximum samples per source (default: `None`)
- `--recreate`: Delete and recreate collection if it exists
- `--query`: Query text for retrieval (if provided, skips building)
- `--top_k`: Number of results to retrieve (default: `5`)

**Examples**:

Build with all sources:
```bash
python3 knowledge_base.py --recreate
```

Build with specific sources:
```bash
python3 knowledge_base.py --sources squad sciq --recreate
```

Query the knowledge base:
```bash
python3 knowledge_base.py --query "What is machine learning?" --top_k 10
```

#### 3. `run_knowledge_base.sh`

**Purpose**: Helper script to run knowledge base builder in a virtual environment.

**What it does**:
- Activates the `venv_kb` virtual environment
- Runs `knowledge_base.py` with all passed arguments
- Ensures correct Python environment is used

**Usage**:
```bash
./run_knowledge_base.sh [options]
```

**Examples**:
```bash
# Build knowledge base
./run_knowledge_base.sh --recreate

# Query knowledge base
./run_knowledge_base.sh --query "What is reinforcement learning?" --top_k 5
```

**Note**: Make sure the script is executable:
```bash
chmod +x run_knowledge_base.sh
```

---

## RAG Engine

### Overview

The RAG (Retrieval-Augmented Generation) Engine is a complete question-answering system that:
1. Classifies query intent using the trained intent classifier
2. Routes to optimal retrieval strategy based on intent
3. Retrieves relevant context from the Qdrant knowledge base
4. Processes retrieved chunks (cleaning, deduplication, ordering, merging)
5. Generates factually grounded answers using GPT-4o-mini
6. Falls back to general knowledge if retrieved context is insufficient

### Components

The RAG engine is split into two main files:

#### 1. `rag_components.py`

**Purpose**: Contains supporting classes for the RAG engine.

**Classes**:
- **IntentClassifier**: Wraps the trained intent classifier model to predict query intent
  - Loads and manages the intent classification model
  - Maps intent IDs to semantic names (factual, person, time, location, explanation, other)
  - Predicts intent for user queries
  
- **RAGRouter**: Routes retrieval strategy based on intent
  - Defines intent-based retrieval strategies (top-k, score thresholds)
  - Returns optimal retrieval parameters for each intent type
  
- **ContextProcessor**: Processes retrieved chunks
  - `clean_text()`: Removes extra whitespace and normalizes text
  - `deduplicate_chunks()`: Removes duplicate chunks based on text content
  - `order_chunks()`: Sorts chunks by relevance score (descending)
  - `merge_chunks()`: Combines chunks into coherent context block (max 4000 characters)

**Usage**: This file is imported by `rag_engine.py` and not run directly.

#### 2. `rag_engine.py`

**Purpose**: Main orchestrator that integrates all components and provides the CLI interface.

**What it does**:
- Initializes all RAG components (IntentClassifier, KnowledgeBaseBuilder, OpenAI client)
- Implements the complete RAG pipeline:
  - Intent classification
  - Context retrieval with intent-based routing
  - Context processing (deduplication, ordering, merging)
  - Grounded answer generation using custom strict prompt
  - Failure signal detection and fallback mechanism
- Provides command-line interface for querying

**Main Class**: `RAGEngine`
- `__init__()`: Initializes the engine with required components
- `retrieve_context()`: Retrieves relevant context based on query and intent
- `generate_grounded_answer()`: Generates strictly grounded answer using custom RAG prompt
- `generate_fallback_answer()`: Generates answer using general knowledge (fallback mode)
- `detect_failure_signals()`: Detects failure signals in RAG answer
- `answer_query()`: Complete RAG pipeline execution

### Intent-Based Retrieval Strategies

Different intents use different retrieval strategies:

| Intent | Top-K | Score Threshold | Rationale |
|--------|-------|-----------------|-----------|
| Factual | 5 | 0.3 | Balanced retrieval for factual lookups |
| Person | 3 | 0.4 | Precise matches for person queries |
| Time | 3 | 0.4 | Precise matches for time queries |
| Location | 3 | 0.4 | Precise matches for location queries |
| Explanation | 8 | 0.25 | More context needed for explanations |
| Other | 5 | 0.3 | Default balanced strategy |

### Usage

#### Running the RAG Engine

The RAG engine is run through `rag_engine.py`, which uses components from `rag_components.py`.

**Basic Usage**:
```bash
python3 rag_engine.py --query "What is machine learning?"
```

**Using the Shell Script** (Recommended):
```bash
./run_rag.sh --query "What is machine learning?"
```

**With Verbose Output**:
```bash
./run_rag.sh --query "What is machine learning?" --verbose
```

**With Custom Intent Model**:
```bash
./run_rag.sh --query "What is machine learning?" --intent_model ./intent_classifier_model
```

#### Command-Line Options

- `--query`: Query to answer (required)
- `--intent_model`: Path to intent classifier model (default: `./intent_classifier_model_roberta`)
- `--verbose`: Print detailed pipeline information including intent classification, retrieval details, and processing steps

#### Programmatic Usage

You can also use the RAG engine programmatically:

```python
from rag_engine import RAGEngine

# Initialize the engine
rag = RAGEngine(intent_model_path="./intent_classifier_model_roberta")

# Answer a query
response = rag.answer_query("What is machine learning?", verbose=False)

# Access the response
print(response['generation']['answer'])
print(f"Intent: {response['intent']['name']}")
print(f"Fallback used: {response['fallback_used']}")
```

### How It Works

1. **Intent Classification**: The query is classified into one of 6 intent types (factual, person, time, location, explanation, other)

2. **Retrieval Routing**: Based on the intent, the system selects optimal retrieval parameters:
   - Number of chunks to retrieve (top-k)
   - Minimum similarity score threshold

3. **Context Retrieval**: Semantically similar chunks are retrieved from Qdrant using the query embedding

4. **Context Processing**:
   - **Deduplication**: Removes duplicate chunks based on text content
   - **Ordering**: Sorts chunks by relevance score (descending)
   - **Merging**: Combines chunks into a coherent context block (max 4000 characters)

5. **Answer Generation**:
   - **Grounded Mode**: If sufficient context is retrieved, GPT-4o-mini generates an answer strictly based on the retrieved context
   - **Fallback Mode**: If context is insufficient, GPT-4o-mini uses its general knowledge

6. **Response**: Returns the answer along with metadata about intent, retrieval, and generation

### Example Output

```
============================================================
Processing Query: What is reinforcement learning?
============================================================
Predicted Intent: explanation (ID: 4)
Retrieved 8 chunks
Merged context length: 2847 characters
Using grounded RAG mode

============================================================
RAG Engine Response
============================================================

Query: What is reinforcement learning?
Intent: explanation
Retrieved Chunks: 8
Fallback Used: False

Answer:
Reinforcement learning is a type of machine learning where an agent learns to make decisions by interacting with an environment. The agent receives rewards or penalties for its actions and learns to maximize cumulative rewards over time...
```

### Fallback Mechanism

The system automatically uses fallback mode when any of these conditions are met:
- The grounded answer contains: "The answer is not found in the provided context"
- The grounded answer contains: "Additional information required"
- Retrieved context is too short (< 150 characters)

**How it works**:
1. First, the system generates a strictly grounded answer using the custom RAG prompt
2. The system checks for failure signals in the generated answer
3. If failure signals are detected, it automatically switches to fallback mode
4. Fallback mode uses GPT-4o-mini's general knowledge without strict grounding constraints
5. The original grounded answer is preserved in the response for reference

In fallback mode, GPT-4o-mini provides answers using its general knowledge, ensuring users always get a response even when the knowledge base doesn't contain relevant information.

---

## End-to-End Workflow

### Complete Setup and Run

1. **Initial Setup**:
   ```bash
   # Install dependencies
   pip install -r requirements.txt
   
   # Configure environment
   cp .env.example .env
   # Edit .env with your Qdrant credentials
   ```

2. **Train Intent Classifier** (Optional):
   ```bash
   # Train with DistilBERT
   python3 train_intent_classifier.py --epochs 3
   
   # Or train with RoBERTa (better performance)
   python3 train_intent_classifier_roberta.py --epochs 3
   ```

3. **Evaluate Intent Classifier** (Optional):
   ```bash
   python3 evaluate_per_dataset.py --model_path ./intent_classifier_model_roberta
   ```

4. **Build Knowledge Base**:
   ```bash
   # Using the shell script (recommended)
   ./run_knowledge_base.sh --recreate
   
   # Or directly with Python
   python3 knowledge_base.py --recreate
   ```

5. **Query Knowledge Base** (Optional - for direct retrieval):
   ```bash
   # Using the shell script
   ./run_knowledge_base.sh --query "What is artificial intelligence?" --top_k 5
   
   # Or directly with Python
   python3 knowledge_base.py --query "What is artificial intelligence?" --top_k 5
   ```

6. **Use RAG Engine** (Recommended - for complete Q&A):
   ```bash
   # Using the shell script
   ./run_rag.sh --query "What is artificial intelligence?"
   
   # With verbose output
   ./run_rag.sh --query "What is artificial intelligence?" --verbose
   
   # Or directly with Python
   python3 rag_engine.py --query "What is artificial intelligence?"
   ```

### Typical Workflow

1. **First Time Setup**:
   - Install dependencies: `pip install -r requirements.txt`
   - Create `.env` file with Qdrant and OpenAI credentials
   - Train intent classifier: `python3 train_intent_classifier_roberta.py` (optional but recommended)
   - Build knowledge base: `./run_knowledge_base.sh --recreate`

2. **Regular Usage (RAG Engine)**:
   - Ask questions: `./run_rag.sh --query "your question"`
   - The RAG engine handles intent classification, retrieval, and answer generation automatically

3. **Direct Knowledge Base Queries** (if needed):
   - Query: `./run_knowledge_base.sh --query "your question"`
   - Rebuild: `./run_knowledge_base.sh --recreate`

4. **Training/Evaluation** (if needed):
   - Train: `python3 train_intent_classifier_roberta.py`
   - Evaluate: `python3 evaluate_per_dataset.py --model_path ./intent_classifier_model_roberta`

---

## File Structure

```
Project/
├── README.md                          # This file
├── requirements.txt                   # All dependencies (install with: pip install -r requirements.txt)
├── .env                               # Environment variables (Qdrant & OpenAI credentials) - NOT in git
├── .env.example                       # Template for .env file
├── .gitignore                         # Git ignore file (excludes .env, venv, etc.)
│
├── Intent Classifier Training
│   ├── train_intent_classifier.py     # Train DistilBERT intent classifier
│   ├── train_intent_classifier_roberta.py  # Train RoBERTa intent classifier
│   ├── evaluate_per_dataset.py        # Evaluate model per dataset
│   └── intent_classifier_model*/      # Trained model directories (generated)
│
├── Knowledge Base Builder
│   ├── knowledge_base.py              # Main knowledge base script
│   ├── dataset_loaders.py             # Dataset loading functions
│   └── run_knowledge_base.sh          # Helper script to run in venv
│
├── RAG Engine
│   ├── rag_components.py              # Supporting classes (IntentClassifier, RAGRouter, ContextProcessor)
│   ├── rag_engine.py                  # Main RAG orchestrator and CLI
│   └── run_rag.sh                     # Helper script to run RAG engine
│
└── Other Files
    ├── build_knowledge_base.py        # Legacy file (use knowledge_base.py instead)
    └── requirements_*.txt            # Legacy requirements files (use requirements.txt)
```

---

## Storage Considerations

The knowledge base uses Qdrant Cloud with a 4GB storage limit. Current configuration:

| Source | Documents | Estimated Storage |
|--------|-----------|-------------------|
| SQuAD | ~21,000 | ~200 MB |
| SciQ | ~12,000 | ~95 MB |
| Wikipedia Simple | 150,000 (capped) | ~2.4 GB |
| HotpotQA | 10,000 (capped) | ~80 MB |
| **Total** | **~193,000** | **~2.8 GB** |

This leaves ~1.2 GB headroom for future expansion.

---

## Troubleshooting

### Common Issues

1. **Missing .env file**:
   ```
   ERROR: Missing required environment variables!
   ```
   Solution: Copy `.env.example` to `.env` and fill in your credentials.

2. **Version conflicts**:
   ```
   ERROR: Version conflict detected!
   ```
   Solution: Use a virtual environment and install from `requirements.txt`.

3. **Qdrant connection errors**:
   - Verify your `QDRANT_URL` and `QDRANT_API_KEY` in `.env`
   - Check your Qdrant Cloud account has available storage

4. **Dataset loading errors**:
   - Some datasets require internet connection for first-time download
   - Check your Hugging Face token if datasets are private

---

## Additional Notes

- **Tokenization**: The knowledge base uses token-based chunking (256 tokens). Install `tiktoken` for better tokenization, otherwise falls back to word-based.
- **Virtual Environment**: The `run_knowledge_base.sh` script uses `venv_kb`. You can create your own venv or modify the script.
- **Model Storage**: Trained intent classifier models are saved locally in their respective directories.
- **Deduplication**: All dataset loaders automatically deduplicate content using MD5 hashing.

---



