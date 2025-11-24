#!/usr/bin/env python3
"""
Per-Dataset Evaluation Script
------------------------------
Evaluates a trained intent classifier on each dataset separately to check for:
- Overfitting
- Domain gaps
- Per-dataset performance differences
- Per-class metrics
"""

import os
import json
import argparse
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


def load_trec_dataset():
    """Load TREC dataset."""
    print("Loading TREC dataset...")
    try:
        dataset = load_dataset('trec', trust_remote_code=True)
        return dataset
    except:
        try:
            dataset = load_dataset('trec', 'coarse', trust_remote_code=True)
            return dataset
        except:
            # Try alternatives
            for alt in ['SetFit/trec', 'jxm/trec']:
                try:
                    dataset = load_dataset(alt, trust_remote_code=False)
                    return dataset
                except:
                    continue
            raise RuntimeError("Could not load TREC dataset")


def load_squad_dataset():
    """Load SQuAD dataset."""
    print("Loading SQuAD dataset...")
    return load_dataset('squad', trust_remote_code=False)


def load_sciq_dataset():
    """Load SciQ dataset."""
    print("Loading SciQ dataset...")
    return load_dataset('sciq', trust_remote_code=False)


def preprocess_trec(example):
    """Preprocess TREC dataset."""
    text = example.get('text', '')
    label = example.get('coarse_label', example.get('fine_label', 0))
    return {
        'text': text,
        'label': int(label),
        'dataset': 'trec'
    }


def preprocess_squad(example):
    """Preprocess SQuAD dataset."""
    question = example.get('question', '')
    question_lower = question.lower()
    if any(word in question_lower for word in ['what', 'which']):
        intent = 0
    elif any(word in question_lower for word in ['who', 'whom']):
        intent = 1
    elif any(word in question_lower for word in ['when']):
        intent = 2
    elif any(word in question_lower for word in ['where']):
        intent = 3
    elif any(word in question_lower for word in ['why', 'how']):
        intent = 4
    else:
        intent = 5
    
    return {
        'text': question,
        'label': intent,
        'dataset': 'squad'
    }


def preprocess_sciq(example):
    """Preprocess SciQ dataset."""
    question = example.get('question', '')
    question_lower = question.lower()
    if any(word in question_lower for word in ['what', 'which']):
        intent = 0
    elif any(word in question_lower for word in ['who', 'whom']):
        intent = 1
    elif any(word in question_lower for word in ['when']):
        intent = 2
    elif any(word in question_lower for word in ['where']):
        intent = 3
    elif any(word in question_lower for word in ['why', 'how']):
        intent = 4
    else:
        intent = 5
    
    return {
        'text': question,
        'label': intent,
        'dataset': 'sciq'
    }


def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def evaluate_dataset(model_path, dataset_name, dataset_split, preprocess_fn, tokenizer, model):
    """Evaluate model on a specific dataset."""
    print(f"\n{'='*60}")
    print(f"Evaluating on {dataset_name} ({dataset_split})")
    print(f"{'='*60}")
    
    # Load and preprocess dataset
    if dataset_name == 'trec':
        raw_data = load_trec_dataset()
        data = raw_data[dataset_split]
    elif dataset_name == 'squad':
        raw_data = load_squad_dataset()
        data = raw_data[dataset_split]
    elif dataset_name == 'sciq':
        raw_data = load_sciq_dataset()
        data = raw_data[dataset_split]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Preprocess
    processed_data = data.map(preprocess_fn, remove_columns=data.column_names)
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)
    
    tokenized_data = processed_data.map(tokenize_function, batched=True, remove_columns=['text', 'dataset'])
    
    # Evaluate
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    
    # Get predictions
    predictions = trainer.predict(tokenized_data)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, pred_labels, average=None, zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='weighted', zero_division=0
    )
    
    # Print results
    print(f"\nOverall Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Weighted F1: {f1_weighted:.4f}")
    print(f"  Weighted Precision: {precision_weighted:.4f}")
    print(f"  Weighted Recall: {recall_weighted:.4f}")
    print(f"  Total samples: {len(true_labels)}")
    
    # Per-class metrics
    print(f"\nPer-Class Metrics:")
    unique_labels = sorted(set(true_labels))
    for label in unique_labels:
        idx = label
        print(f"  Class {label}:")
        print(f"    Precision: {precision[idx]:.4f}")
        print(f"    Recall: {recall[idx]:.4f}")
        print(f"    F1: {f1[idx]:.4f}")
        print(f"    Support: {support[idx]}")
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=unique_labels)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(true_labels, pred_labels, labels=unique_labels, zero_division=0))
    
    return {
        'dataset': dataset_name,
        'split': dataset_split,
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'per_class_metrics': {
            int(label): {
                'precision': float(precision[idx]),
                'recall': float(recall[idx]),
                'f1': float(f1[idx]),
                'support': int(support[idx])
            }
            for idx, label in enumerate(unique_labels)
        },
        'confusion_matrix': cm.tolist(),
        'total_samples': int(len(true_labels))
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate intent classifier per dataset')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model directory')
    parser.add_argument('--output_file', type=str, default='per_dataset_evaluation.json',
                        help='Output file for detailed results (default: per_dataset_evaluation.json)')
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Try to load label mapping to get num_labels
    label_mapping_path = os.path.join(args.model_path, 'label_mapping.json')
    if os.path.exists(label_mapping_path):
        with open(label_mapping_path, 'r') as f:
            label_mapping = json.load(f)
        num_labels = len(label_mapping)
    else:
        # Default to 6 if not found
        num_labels = 6
        print(f"  Warning: label_mapping.json not found, using num_labels={num_labels}")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        num_labels=num_labels
    )
    
    # Evaluate on each dataset
    results = {}
    
    # TREC test set
    try:
        results['trec'] = evaluate_dataset(
            args.model_path, 'trec', 'test', preprocess_trec, tokenizer, model
        )
    except Exception as e:
        print(f"Error evaluating TREC: {e}")
        results['trec'] = {'error': str(e)}
    
    # SQuAD validation set
    try:
        results['squad'] = evaluate_dataset(
            args.model_path, 'squad', 'validation', preprocess_squad, tokenizer, model
        )
    except Exception as e:
        print(f"Error evaluating SQuAD: {e}")
        results['squad'] = {'error': str(e)}
    
    # SciQ test set
    try:
        results['sciq'] = evaluate_dataset(
            args.model_path, 'sciq', 'test', preprocess_sciq, tokenizer, model
        )
    except Exception as e:
        print(f"Error evaluating SciQ: {e}")
        results['sciq'] = {'error': str(e)}
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("SUMMARY COMPARISON")
    print(f"{'='*60}")
    
    if 'accuracy' in results.get('trec', {}):
        print(f"\nTREC Test Set:")
        print(f"  Accuracy: {results['trec']['accuracy']:.4f}")
        print(f"  F1: {results['trec']['f1_weighted']:.4f}")
    
    if 'accuracy' in results.get('squad', {}):
        print(f"\nSQuAD Validation Set:")
        print(f"  Accuracy: {results['squad']['accuracy']:.4f}")
        print(f"  F1: {results['squad']['f1_weighted']:.4f}")
    
    if 'accuracy' in results.get('sciq', {}):
        print(f"\nSciQ Test Set:")
        print(f"  Accuracy: {results['sciq']['accuracy']:.4f}")
        print(f"  F1: {results['sciq']['f1_weighted']:.4f}")
    
    # Save detailed results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Detailed results saved to {args.output_file}")
    
    # Analysis
    print(f"\n{'='*60}")
    print("ANALYSIS")
    print(f"{'='*60}")
    
    accuracies = []
    if 'accuracy' in results.get('trec', {}):
        accuracies.append(('TREC', results['trec']['accuracy']))
    if 'accuracy' in results.get('squad', {}):
        accuracies.append(('SQuAD', results['squad']['accuracy']))
    if 'accuracy' in results.get('sciq', {}):
        accuracies.append(('SciQ', results['sciq']['accuracy']))
    
    if len(accuracies) > 1:
        acc_values = [acc for _, acc in accuracies]
        max_acc = max(acc_values)
        min_acc = min(acc_values)
        diff = max_acc - min_acc
        
        print(f"\nAccuracy Range: {min_acc:.4f} - {max_acc:.4f}")
        print(f"Difference: {diff:.4f} ({diff*100:.2f} percentage points)")
        
        if diff > 0.10:  # More than 10% difference
            print(f"\n WARNING: Large performance gap detected!")
            print(f"   This suggests potential overfitting or domain gaps.")
        elif diff > 0.05:  # More than 5% difference
            print(f"\n  CAUTION: Moderate performance gap detected.")
        else:
            print(f"\n✓ Performance is relatively consistent across datasets.")


if __name__ == '__main__':
    main()

