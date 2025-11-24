#!/bin/bash
# Quick fix script to downgrade datasets library for TREC compatibility

echo "Checking datasets library version..."
pip show datasets | grep Version

echo ""
echo "Downgrading datasets library to version < 3.0.0 (supports dataset scripts)..."
pip install 'datasets<3.0.0'

echo ""
echo "✓ Done! You can now run train_intent_classifier.py"
echo ""
echo "To verify, run: python train_intent_classifier.py"

