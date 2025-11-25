#!/bin/bash
# Helper script to run RAG engine in virtual environment

cd "$(dirname "$0")"

# Activate virtual environment
source venv_kb/bin/activate

# Run the RAG engine with all arguments passed through
python3 rag_engine.py "$@"

