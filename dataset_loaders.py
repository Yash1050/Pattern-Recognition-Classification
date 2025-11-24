#!/usr/bin/env python3
"""
Dataset Loaders
---------------
Functions to load and process datasets from various sources.
"""

import hashlib
from typing import List, Dict, Optional
from datasets import load_dataset


def load_squad_contexts(max_samples: Optional[int] = None) -> List[Dict]:
    """Load contexts from SQuAD dataset."""
    print("Loading SQuAD dataset...")
    dataset = load_dataset('squad', trust_remote_code=False)
    
    contexts = []
    seen_contexts = set()
    
    for split in ['train', 'validation']:
        data = dataset[split]
        if max_samples:
            data = data.select(range(min(max_samples, len(data))))
        
        for item in data:
            context = item['context'].strip()
            context_hash = hashlib.md5(context.encode()).hexdigest()
            if context_hash not in seen_contexts:
                seen_contexts.add(context_hash)
                contexts.append({
                    'text': context,
                    'metadata': {
                        'source': 'squad',
                        'split': split,
                        'title': item.get('title', ''),
                        'id': item.get('id', '')
                    }
                })
    
    print(f"Loaded {len(contexts)} unique SQuAD contexts")
    return contexts


def load_sciq_contexts(max_samples: Optional[int] = None) -> List[Dict]:
    """Load contexts from SciQ dataset."""
    print("Loading SciQ dataset...")
    dataset = load_dataset('sciq', trust_remote_code=False)
    
    contexts = []
    seen_contexts = set()
    
    for split in ['train', 'validation', 'test']:
        data = dataset[split]
        if max_samples:
            data = data.select(range(min(max_samples, len(data))))
        
        for item in data:
            support = item.get('support', '').strip()
            if support:
                context_hash = hashlib.md5(support.encode()).hexdigest()
                if context_hash not in seen_contexts:
                    seen_contexts.add(context_hash)
                    contexts.append({
                        'text': support,
                        'metadata': {
                            'source': 'sciq',
                            'split': split,
                            'question': item.get('question', ''),
                        }
                    })
    
    print(f"Loaded {len(contexts)} unique SciQ contexts")
    return contexts


def load_wikipedia_simple(max_samples: Optional[int] = None) -> List[Dict]:
    """Load Wikipedia Simple articles (sampled to control storage)."""
    print("Loading Wikipedia Simple dataset...")
    try:
        dataset = load_dataset(
            "wikimedia/wikipedia",
            "20231101.simple"
        )
        
        max_allowed = 150000
        if max_samples is None:
            effective_max = max_allowed
        else:
            effective_max = min(max_samples, max_allowed)
        
        contexts = []
        seen_contexts = set()
        
        data = dataset["train"]
        if effective_max:
            print(f"  Wikipedia Simple samples limited to {effective_max}")
            data = data.select(range(min(effective_max, len(data))))
        
        for item in data:
            text = item.get("text", "").strip()
            if text:
                text_hash = hashlib.md5(text.encode()).hexdigest()
                if text_hash not in seen_contexts:
                    seen_contexts.add(text_hash)
                    contexts.append({
                        "text": text,
                        "metadata": {
                            "source": "wikipedia_simple",
                            "title": item.get("title", ""),
                        }
                    })
        
        print(f"Loaded {len(contexts)} unique Wikipedia Simple articles")
        return contexts
    except Exception as e:
        print(f"Could not load Wikipedia Simple: {e}")
        return []


def load_hotpotqa(max_samples: Optional[int] = None) -> List[Dict]:
    """Load HotpotQA paragraphs (context passages)."""
    print("Loading HotpotQA dataset...")
    try:
        dataset = load_dataset("hotpot_qa", "distractor")
        data = dataset["train"]
        
        max_allowed = 10000
        if max_samples is None:
            effective_max = max_allowed
        else:
            effective_max = min(max_samples, max_allowed)
        
        if effective_max:
            print(f"  HotpotQA samples limited to {effective_max}")
            data = data.select(range(min(effective_max, len(data))))
        
        contexts = []
        for item in data:
            ctx = item["context"]
            titles = ctx["title"]
            paragraphs = ctx["sentences"]
            
            for t, s_list in zip(titles, paragraphs):
                paragraph = " ".join(s_list).strip()
                if paragraph:
                    contexts.append({
                        "text": paragraph,
                        "metadata": {
                            "source": "hotpotqa",
                            "title": t
                        }
                    })
        
        print(f"Loaded {len(contexts)} HotpotQA paragraphs")
        return contexts
    except Exception as e:
        print(f"Could not load HotpotQA: {e}")
        return []

