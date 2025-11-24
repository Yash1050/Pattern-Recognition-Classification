#!/bin/bash
# Helper script to run knowledge base builder in virtual environment

cd "$(dirname "$0")"

# Activate virtual environment
source venv_kb/bin/activate

# Run the script with all arguments passed through
python3 knowledge_base.py "$@"

