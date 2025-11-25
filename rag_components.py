#!/usr/bin/env python3
"""
RAG Components
--------------
Supporting classes for the RAG engine:
- IntentClassifier: Wraps intent classification model
- RAGRouter: Routes retrieval strategy based on intent
- ContextProcessor: Processes retrieved chunks (cleaning, deduplication, ordering, merging)
"""

import os
import json
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


class IntentClassifier:
    """Wrapper for intent classifier model."""
    
    def __init__(self, model_path: str = "./intent_classifier_model_roberta"):
        """
        Initialize intent classifier.
        
        Args:
            model_path: Path to trained intent classifier model
        """
        self.model_path = model_path
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.eval()
            
            # Load label mapping
            label_mapping_path = os.path.join(model_path, "label_mapping.json")
            if os.path.exists(label_mapping_path):
                with open(label_mapping_path, 'r') as f:
                    loaded_mapping = json.load(f)
                    # Check if loaded mapping has generic names (intent_0, intent_1, etc.)
                    # If so, use semantic names instead
                    first_value = list(loaded_mapping.values())[0] if loaded_mapping else ""
                    if first_value.startswith("intent_") or first_value.startswith("LABEL_"):
                        # Use semantic mapping based on training code
                        self.label_mapping = {
                            "0": "factual",
                            "1": "person",
                            "2": "time",
                            "3": "location",
                            "4": "explanation",
                            "5": "other"
                        }
                    else:
                        self.label_mapping = loaded_mapping
            else:
                # Default mapping based on training code
                self.label_mapping = {
                    "0": "factual",
                    "1": "person",
                    "2": "time",
                    "3": "location",
                    "4": "explanation",
                    "5": "other"
                }
        except Exception as e:
            print(f"Warning: Could not load intent classifier: {e}")
            print("RAG engine will use default retrieval strategy")
            self.model = None
            self.tokenizer = None
            self.label_mapping = {}
    
    def predict_intent(self, query: str) -> Tuple[int, str]:
        """
        Predict intent for a query.
        
        Args:
            query: User query text
            
        Returns:
            Tuple of (intent_id, intent_name)
        """
        if self.model is None or self.tokenizer is None:
            return (5, "other")  # Default to "other" if classifier unavailable
        
        try:
            inputs = self.tokenizer(query, return_tensors="pt", truncation=True, max_length=128)
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                intent_id = predictions.argmax().item()
                intent_name = self.label_mapping.get(str(intent_id), f"intent_{intent_id}")
                confidence = predictions[0][intent_id].item()
                
                return (intent_id, intent_name)
        except Exception as e:
            return (5, "other")


class RAGRouter:
    """Routes retrieval strategy based on intent."""
    
    # Intent-based retrieval strategies
    STRATEGIES = {
        "factual": {"top_k": 5, "score_threshold": 0.3},  # Factual lookup: fewer, high-quality chunks
        "person": {"top_k": 3, "score_threshold": 0.4},   # Person queries: precise matches
        "time": {"top_k": 3, "score_threshold": 0.4},     # Time queries: precise matches
        "location": {"top_k": 3, "score_threshold": 0.4}, # Location queries: precise matches
        "explanation": {"top_k": 8, "score_threshold": 0.25},  # Explanations: more context needed
        "other": {"top_k": 5, "score_threshold": 0.3}     # Default strategy
    }
    
    @staticmethod
    def get_strategy(intent_name: str) -> Dict:
        """Get retrieval strategy for intent."""
        return RAGRouter.STRATEGIES.get(intent_name, RAGRouter.STRATEGIES["other"])


class ContextProcessor:
    """Processes retrieved chunks: cleaning, deduplication, ordering, merging."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean text by removing extra whitespace and normalizing."""
        if not text:
            return ""
        # Remove extra whitespace
        text = " ".join(text.split())
        return text.strip()
    
    @staticmethod
    def deduplicate_chunks(chunks: List[Dict]) -> List[Dict]:
        """Remove duplicate chunks based on text content."""
        seen = set()
        unique_chunks = []
        
        for chunk in chunks:
            text = chunk.get('text', '').strip()
            text_hash = hash(text)
            if text_hash not in seen:
                seen.add(text_hash)
                unique_chunks.append(chunk)
        
        return unique_chunks
    
    @staticmethod
    def order_chunks(chunks: List[Dict]) -> List[Dict]:
        """Order chunks by relevance score (descending)."""
        return sorted(chunks, key=lambda x: x.get('score', 0.0), reverse=True)
    
    @staticmethod
    def merge_chunks(chunks: List[Dict], max_length: int = 4000) -> str:
        """
        Merge chunks into coherent context block.
        
        Args:
            chunks: List of chunk dictionaries
            max_length: Maximum character length for merged context
            
        Returns:
            Merged context string
        """
        merged = []
        current_length = 0
        
        for chunk in chunks:
            text = ContextProcessor.clean_text(chunk.get('text', ''))
            if not text:
                continue
            
            # Check if adding this chunk would exceed max length
            if current_length + len(text) + 2 > max_length:  # +2 for newline
                break
            
            merged.append(text)
            current_length += len(text) + 2
        
        return "\n\n".join(merged)

