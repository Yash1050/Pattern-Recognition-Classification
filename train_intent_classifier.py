#!/usr/bin/env python3
"""
Intent Classifier Training Script
---------------------------------
Trains an intent classifier on TREC, SQuAD, and SciQ datasets.

This script:
1. Loads TREC, SQuAD, and SciQ datasets using Hugging Face datasets
2. Preprocesses the data for intent classification
3. Trains a transformer-based classifier
4. Evaluates and saves the model
"""

import os
import json
import requests
import tempfile
from typing import Dict, List, Tuple
from datasets import load_dataset, Dataset, concatenate_datasets
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np


def load_trec_dataset():
    """Load TREC dataset using modern Hugging Face approach."""
    print("Loading TREC dataset...")
    
    # Check datasets library version
    try:
        import datasets
        datasets_version = datasets.__version__
        print(f"  Using datasets library version: {datasets_version}")
        
        # If version is 3.0.0 or higher, scripts are not supported
        major_version = int(datasets_version.split('.')[0])
        if major_version >= 3:
            print("  ⚠️  Warning: datasets >= 3.0.0 does not support dataset scripts.")
            print("  Trying alternative loading methods...")
    except:
        pass
    
    # Try loading with trust_remote_code=True first (required for script-based datasets)
    approaches = [
        ('trec', None, True),  # Default with trust_remote_code
        ('trec', 'coarse', True),  # Coarse labels with trust_remote_code
        ('trec', 'fine', True),  # Fine labels with trust_remote_code
    ]
    
    for dataset_name, config, trust_code in approaches:
        try:
            if config:
                dataset = load_dataset(dataset_name, config, trust_remote_code=trust_code)
            else:
                dataset = load_dataset(dataset_name, trust_remote_code=trust_code)
            
            # Verify dataset has required splits
            if 'train' in dataset and 'test' in dataset:
                config_str = f" ({config})" if config else ""
                print(f"✓ TREC dataset loaded{config_str}: {len(dataset['train'])} train, {len(dataset['test'])} test samples")
                return dataset
        except Exception as e:
            error_msg = str(e)
            if "Dataset scripts are no longer supported" in error_msg:
                print(f"  ⚠️  Dataset script format not supported by this datasets version.")
                break  # No point trying other configs if scripts aren't supported
            else:
                print(f"  Attempt with {dataset_name}{f' ({config})' if config else ''} failed: {error_msg[:100]}")
            continue
    
    # Try alternative community-maintained sources that don't use scripts
    print("  Trying alternative TREC dataset sources (community-maintained)...")
    alternative_sources = [
        'SetFit/trec',
        'jxm/trec',
    ]
    
    for alt_source in alternative_sources:
        try:
            print(f"  Attempting to load from {alt_source}...")
            dataset = load_dataset(alt_source, trust_remote_code=False)
            if 'train' in dataset and 'test' in dataset:
                print(f"✓ TREC dataset loaded from {alt_source}: {len(dataset['train'])} train, {len(dataset['test'])} test samples")
                return dataset
        except Exception as e:
            print(f"  Failed to load from {alt_source}: {str(e)[:100]}")
            continue
    
    # If all approaches fail, provide clear instructions
    print("\n" + "="*60)
    print("ERROR: Could not load TREC dataset")
    print("="*60)
    print("\nThe TREC dataset uses an older script format that is not supported")
    print("by newer versions of the datasets library (>= 3.0.0).")
    print("\nSOLUTIONS:")
    print("\n1. Downgrade datasets library to a version that supports scripts:")
    print("   pip install 'datasets<3.0.0'")
    print("\n2. Or use a community-maintained version (already attempted above)")
    print("\n3. Or manually download and process TREC from its original source")
    print("="*60)
    
    raise RuntimeError(
        "Failed to load TREC dataset. Please downgrade datasets library:\n"
        "  pip install 'datasets<3.0.0'\n"
        "Then run this script again."
    )


def load_squad_dataset():
    """Load SQuAD dataset."""
    print("Loading SQuAD dataset...")
    try:
        # SQuAD v1.1
        dataset = load_dataset('squad', trust_remote_code=False)
        print(f"✓ SQuAD dataset loaded: {len(dataset['train'])} train, {len(dataset['validation'])} validation samples")
        return dataset
    except Exception as e:
        print(f"Error loading SQuAD dataset: {e}")
        raise


def load_sciq_dataset():
    """Load SciQ dataset."""
    print("Loading SciQ dataset...")
    try:
        dataset = load_dataset('sciq', trust_remote_code=False)
        print(f"✓ SciQ dataset loaded: {len(dataset['train'])} train, {len(dataset['validation'])} validation, {len(dataset['test'])} test samples")
        return dataset
    except Exception as e:
        print(f"Error loading SciQ dataset: {e}")
        raise


def preprocess_trec(example):
    """Preprocess TREC dataset for intent classification."""
    # TREC has 'text' and 'coarse_label' or 'fine_label'
    text = example.get('text', '')
    label = example.get('coarse_label', example.get('fine_label', 0))
    return {
        'text': text,
        'label': label,
        'dataset': 'trec'
    }


def preprocess_squad(example):
    """Preprocess SQuAD dataset for intent classification."""
    # SQuAD has 'question' and 'context'
    # We'll use question as the text and create a simple intent label
    # For intent classification, we can use question type or create a generic label
    question = example.get('question', '')
    # Use a simple heuristic: question words indicate intent
    question_lower = question.lower()
    if any(word in question_lower for word in ['what', 'which']):
        intent = 0  # Factual
    elif any(word in question_lower for word in ['who', 'whom']):
        intent = 1  # Person
    elif any(word in question_lower for word in ['when']):
        intent = 2  # Time
    elif any(word in question_lower for word in ['where']):
        intent = 3  # Location
    elif any(word in question_lower for word in ['why', 'how']):
        intent = 4  # Explanation
    else:
        intent = 5  # Other
    
    return {
        'text': question,
        'label': intent,
        'dataset': 'squad'
    }


def preprocess_sciq(example):
    """Preprocess SciQ dataset for intent classification."""
    # SciQ has 'question', 'correct_answer', 'distractor1', etc.
    question = example.get('question', '')
    # Similar intent classification based on question type
    question_lower = question.lower()
    if any(word in question_lower for word in ['what', 'which']):
        intent = 0  # Factual
    elif any(word in question_lower for word in ['who', 'whom']):
        intent = 1  # Person
    elif any(word in question_lower for word in ['when']):
        intent = 2  # Time
    elif any(word in question_lower for word in ['where']):
        intent = 3  # Location
    elif any(word in question_lower for word in ['why', 'how']):
        intent = 4  # Explanation
    else:
        intent = 5  # Other
    
    return {
        'text': question,
        'label': intent,
        'dataset': 'sciq'
    }


def combine_datasets(trec_data, squad_data, sciq_data, max_samples_per_dataset=None):
    """Combine and preprocess all three datasets."""
    print("\nPreprocessing and combining datasets...")
    
    # Preprocess TREC
    trec_train = trec_data['train'].map(preprocess_trec, remove_columns=trec_data['train'].column_names)
    trec_test = trec_data['test'].map(preprocess_trec, remove_columns=trec_data['test'].column_names)
    
    # Preprocess SQuAD
    squad_train = squad_data['train'].map(preprocess_squad, remove_columns=squad_data['train'].column_names)
    squad_val = squad_data['validation'].map(preprocess_squad, remove_columns=squad_data['validation'].column_names)
    
    # Preprocess SciQ
    sciq_train = sciq_data['train'].map(preprocess_sciq, remove_columns=sciq_data['train'].column_names)
    sciq_val = sciq_data['validation'].map(preprocess_sciq, remove_columns=sciq_data['validation'].column_names)
    sciq_test = sciq_data['test'].map(preprocess_sciq, remove_columns=sciq_data['test'].column_names)
    
    # Limit samples if specified
    if max_samples_per_dataset:
        trec_train = trec_train.select(range(min(max_samples_per_dataset, len(trec_train))))
        trec_test = trec_test.select(range(min(max_samples_per_dataset, len(trec_test))))
        squad_train = squad_train.select(range(min(max_samples_per_dataset, len(squad_train))))
        squad_val = squad_val.select(range(min(max_samples_per_dataset, len(squad_val))))
        sciq_train = sciq_train.select(range(min(max_samples_per_dataset, len(sciq_train))))
        sciq_val = sciq_val.select(range(min(max_samples_per_dataset, len(sciq_val))))
        sciq_test = sciq_test.select(range(min(max_samples_per_dataset, len(sciq_test))))
    
    # Combine training sets
    train_combined = concatenate_datasets([trec_train, squad_train, sciq_train])
    
    # Combine validation/test sets
    val_combined = concatenate_datasets([squad_val, sciq_val])
    test_combined = concatenate_datasets([trec_test, sciq_test])
    
    print(f"✓ Combined dataset:")
    print(f"  - Train: {len(train_combined)} samples")
    print(f"  - Validation: {len(val_combined)} samples")
    print(f"  - Test: {len(test_combined)} samples")
    
    # Get unique labels
    unique_labels = sorted(set(train_combined['label']))
    num_labels = len(unique_labels)
    print(f"  - Number of intent classes: {num_labels}")
    
    return train_combined, val_combined, test_combined, num_labels


def tokenize_function(examples, tokenizer):
    """Tokenize the text examples."""
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)


def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def train_intent_classifier(
    model_name='distilbert-base-uncased',
    output_dir='./intent_classifier_model',
    num_epochs=3,
    batch_size=16,
    learning_rate=2e-5,
    max_samples_per_dataset=None
):
    """Main training function."""
    print("=" * 60)
    print("Intent Classifier Training")
    print("=" * 60)
    
    # Load datasets
    trec_data = load_trec_dataset()
    squad_data = load_squad_dataset()
    sciq_data = load_sciq_dataset()
    
    # Combine datasets
    train_dataset, val_dataset, test_dataset, num_labels = combine_datasets(
        trec_data, squad_data, sciq_data, max_samples_per_dataset
    )
    
    # Load tokenizer and model
    print(f"\nLoading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    
    # Tokenize datasets
    print("Tokenizing datasets...")
    train_tokenized = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=['text', 'dataset']
    )
    val_tokenized = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=['text', 'dataset']
    )
    test_tokenized = test_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=['text', 'dataset']
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Calculate steps per epoch for older API compatibility
    import torch
    num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if num_devices == 0:
        num_devices = 1
    steps_per_epoch = len(train_tokenized) // (batch_size * num_devices)
    if steps_per_epoch == 0:
        steps_per_epoch = 1
    
    # Training arguments
    # Try to use newer parameter names first, fall back to older ones if they fail
    # This handles version differences more reliably than version checking
    
    # Build base training args dict with common parameters
    base_args = {
        'output_dir': output_dir,
        'num_train_epochs': num_epochs,
        'per_device_train_batch_size': batch_size,
        'per_device_eval_batch_size': batch_size,
        'learning_rate': learning_rate,
        'weight_decay': 0.01,
        'logging_dir': f'{output_dir}/logs',
        'logging_steps': 100,
        'load_best_model_at_end': True,
        'metric_for_best_model': 'f1',
        'greater_is_better': True,
        'save_total_limit': 2,
    }
    
    # Try newer parameter names (transformers >= 4.19.0)
    try:
        training_args = TrainingArguments(
            **base_args,
            eval_strategy='epoch',
            save_strategy='epoch',
        )
    except TypeError:
        # Fall back to older parameter names (transformers < 4.19.0)
        print("  Using older transformers API (evaluation_strategy instead of eval_strategy)")
        # For older versions, try save_strategy='epoch' first
        try:
            training_args = TrainingArguments(
                **base_args,
                evaluation_strategy='epoch',
                save_strategy='epoch',
            )
        except (TypeError, ValueError):
            # If save_strategy doesn't work, calculate save_steps to match epoch boundaries
            # This ensures load_best_model_at_end works correctly
            print(f"  Calculating save_steps to match epoch boundaries ({steps_per_epoch} steps/epoch)")
            training_args = TrainingArguments(
                **base_args,
                evaluation_strategy='epoch',
                save_steps=steps_per_epoch,  # Save at end of each epoch
            )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(test_tokenized)
    print(f"\nTest Results:")
    for key, value in test_results.items():
        if 'loss' not in key:
            print(f"  {key}: {value:.4f}")
    
    # Save model
    print(f"\nSaving model to {output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Save label mapping
    label_mapping = {i: f'intent_{i}' for i in range(num_labels)}
    with open(f'{output_dir}/label_mapping.json', 'w') as f:
        json.dump(label_mapping, f, indent=2)
    
    print(f"\n✓ Training complete! Model saved to {output_dir}")
    return trainer, test_results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train intent classifier on TREC, SQuAD, and SciQ datasets')
    parser.add_argument('--model', type=str, default='distilbert-base-uncased',
                        help='Model name or path (default: distilbert-base-uncased)')
    parser.add_argument('--output_dir', type=str, default='./intent_classifier_model',
                        help='Output directory for the model (default: ./intent_classifier_model)')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs (default: 3)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate (default: 2e-5)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum samples per dataset (default: None, use all)')
    
    args = parser.parse_args()
    
    train_intent_classifier(
        model_name=args.model,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_samples_per_dataset=args.max_samples
    )

