# Project Documentation

This project contains two main components:
1. **Intent Classifier Training** - Trains transformer models for intent classification
2. **Knowledge Base Builder** - Builds a vector knowledge base from multiple datasets

---

## Table of Contents

- [Setup](#setup)
- [Configuration](#configuration)
- [Intent Classifier Training](#intent-classifier-training)
- [Knowledge Base Builder](#knowledge-base-builder)
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
   
   Then edit `.env` and add your Qdrant Cloud credentials:
   ```
   QDRANT_URL=your_qdrant_cloud_url_here
   QDRANT_API_KEY=your_qdrant_api_key_here
   COLLECTION_NAME=pattern_recognition
   ```

---

## Configuration

### Environment Variables (`.env` file)

The knowledge base builder requires Qdrant Cloud credentials stored in a `.env` file:

- `QDRANT_URL`: Your Qdrant Cloud cluster URL
- `QDRANT_API_KEY`: Your Qdrant Cloud API key
- `COLLECTION_NAME`: Name of the collection (default: `pattern_recognition`)

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

5. **Query Knowledge Base**:
   ```bash
   # Using the shell script
   ./run_knowledge_base.sh --query "What is artificial intelligence?" --top_k 5
   
   # Or directly with Python
   python3 knowledge_base.py --query "What is artificial intelligence?" --top_k 5
   ```

### Typical Workflow

1. **First Time Setup**:
   - Install dependencies: `pip install -r requirements.txt`
   - Create `.env` file with Qdrant credentials
   - Build knowledge base: `./run_knowledge_base.sh --recreate`

2. **Regular Usage**:
   - Query knowledge base: `./run_knowledge_base.sh --query "your question"`
   - Rebuild if needed: `./run_knowledge_base.sh --recreate`

3. **Training Models** (if needed):
   - Train: `python3 train_intent_classifier_roberta.py`
   - Evaluate: `python3 evaluate_per_dataset.py --model_path ./intent_classifier_model_roberta`

---

## File Structure

```
Project/
├── README.md                          # This file
├── requirements.txt                   # All dependencies (install with: pip install -r requirements.txt)
├── .env                               # Environment variables (Qdrant credentials) - NOT in git
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



