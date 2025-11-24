#!/usr/bin/env python3
"""
Knowledge Base Builder
----------------------
Builds a vector knowledge base from multiple sources:
1. Chunks documents from various corpora
2. Generates embeddings using sentence-transformers
3. Stores embeddings in Qdrant Cloud
4. Provides retrieval functionality
"""

import os
import sys
import json
import argparse
import time
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration from environment variables
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pattern_recognition")

# Validate required environment variables
if not QDRANT_URL or not QDRANT_API_KEY:
    print("ERROR: Missing required environment variables!")
    print("Please create a .env file with QDRANT_URL and QDRANT_API_KEY")
    print("See .env.example for reference")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    if 'cached_download' in str(e):
        print("ERROR: Version conflict detected!")
        print("The system-installed sentence-transformers is incompatible with huggingface_hub.")
        print("\nPlease run:")
        print("  pip install --upgrade 'sentence-transformers>=2.7.0' --user")
        print("\nOr if that fails, use a virtual environment:")
        print("  python3 -m venv venv")
        print("  source venv/bin/activate")
        print("  pip install -r requirements_knowledge_base.txt")
        sys.exit(1)
    else:
        raise

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)
from qdrant_client.http import models
import hashlib
from tqdm import tqdm

from dataset_loaders import (
    load_squad_contexts,
    load_sciq_contexts,
    load_wikipedia_simple,
    load_hotpotqa
)


class DocumentChunker:
    """Chunks documents into smaller pieces for embedding using token-based chunking."""
    
    def __init__(self, chunk_size_tokens: int = 256, chunk_overlap_tokens: int = 25):
        """
        Initialize chunker with token-based chunking.
        
        Args:
            chunk_size_tokens: Maximum number of tokens per chunk
            chunk_overlap_tokens: Number of tokens to overlap between chunks
        """
        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        
        try:
            import tiktoken
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            self.use_tiktoken = True
        except ImportError:
            print("tiktoken not available, using simple word-based tokenization")
            print("   For better tokenization, install tiktoken: pip install tiktoken")
            self.tokenizer = None
            self.use_tiktoken = False
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Split text into chunks based on tokens.
        
        Args:
            text: Text to chunk
            metadata: Metadata to attach to each chunk
            
        Returns:
            List of chunk dictionaries with 'text' and 'metadata' keys
        """
        if not text or len(text.strip()) == 0:
            return []
        
        chunks = []
        
        if self.use_tiktoken:
            tokens = self.tokenizer.encode(text)
            total_tokens = len(tokens)
            
            start_token = 0
            while start_token < total_tokens:
                end_token = min(start_token + self.chunk_size_tokens, total_tokens)
                chunk_tokens = tokens[start_token:end_token]
                chunk_text = self.tokenizer.decode(chunk_tokens).strip()
                
                if chunk_text:
                    chunk_metadata = metadata.copy() if metadata else {}
                    chunk_metadata['chunk_index'] = len(chunks)
                    chunk_metadata['chunk_start_tokens'] = start_token
                    chunk_metadata['chunk_end_tokens'] = end_token
                    chunk_metadata['chunk_size_tokens'] = len(chunk_tokens)
                    
                    chunks.append({
                        'text': chunk_text,
                        'metadata': chunk_metadata
                    })
                
                start_token = end_token - self.chunk_overlap_tokens if end_token < total_tokens else end_token
        else:
            words = text.split()
            total_words = len(words)
            words_per_chunk = self.chunk_size_tokens
            words_overlap = self.chunk_overlap_tokens
            
            start_word = 0
            while start_word < total_words:
                end_word = min(start_word + words_per_chunk, total_words)
                chunk_words = words[start_word:end_word]
                chunk_text = ' '.join(chunk_words).strip()
                
                if chunk_text:
                    chunk_metadata = metadata.copy() if metadata else {}
                    chunk_metadata['chunk_index'] = len(chunks)
                    chunk_metadata['chunk_start_words'] = start_word
                    chunk_metadata['chunk_end_words'] = end_word
                    
                    chunks.append({
                        'text': chunk_text,
                        'metadata': chunk_metadata
                    })
                
                start_word = end_word - words_overlap if end_word < total_words else end_word
        
        return chunks


class KnowledgeBaseBuilder:
    """Builds and manages the knowledge base in Qdrant."""
    
    def __init__(
        self,
        qdrant_url: str,
        qdrant_api_key: str,
        collection_name: str = "knowledge_base",
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        vector_size: int = 768
    ):
        """
        Initialize knowledge base builder.
        
        Args:
            qdrant_url: Qdrant Cloud cluster URL
            qdrant_api_key: Qdrant API key
            collection_name: Name of the Qdrant collection
            embedding_model: Sentence transformer model name
            vector_size: Dimension of embeddings (768 for all-mpnet-base-v2)
        """
        self.collection_name = collection_name
        self.vector_size = vector_size
        
        print(f"Connecting to Qdrant Cloud...")
        self.client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )
        print(f"Connected to Qdrant Cloud")
        
        print(f"Loading embedding model: {embedding_model}...")
        self.embedding_model = SentenceTransformer(embedding_model)
        print(f"Embedding model loaded")
        
        self.chunker = DocumentChunker()
    
    def create_collection(self, recreate: bool = False):
        """Create or recreate the Qdrant collection."""
        collections = self.client.get_collections()
        collection_exists = any(c.name == self.collection_name for c in collections.collections)
        
        if collection_exists:
            if recreate:
                print(f"Deleting existing collection: {self.collection_name}...")
                self.client.delete_collection(self.collection_name)
                print(f"Collection deleted")
            else:
                print(f"Collection '{self.collection_name}' already exists. Use --recreate to delete and recreate.")
                return
        
        print(f"Creating collection: {self.collection_name}...")
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE
            )
        )
        print(f"Collection created")
    
    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """Chunk all documents."""
        print(f"\nChunking {len(documents)} documents...")
        all_chunks = []
        
        for doc in tqdm(documents, desc="Chunking", unit="doc", mininterval=0.5):
            text = doc.get('text', '')
            if text and len(text.strip()) > 0:
                chunks = self.chunker.chunk_text(text, doc.get('metadata', {}))
                all_chunks.extend(chunks)
        
        print(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks
    
    def generate_embeddings(self, chunks: List[Dict], batch_size: int = 32) -> List[List[float]]:
        """Generate embeddings for all chunks."""
        print(f"\nGenerating embeddings for {len(chunks)} chunks...")
        
        texts = [chunk['text'] for chunk in chunks]
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch_texts,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            embeddings.extend(batch_embeddings.tolist())
        
        print(f"Generated {len(embeddings)} embeddings")
        return embeddings
    
    def store_in_qdrant(self, chunks: List[Dict], embeddings: List[List[float]]):
        """Store chunks and embeddings in Qdrant."""
        print(f"\nStoring {len(chunks)} chunks in Qdrant...")
        
        points = []
        seen_ids = set()
        duplicate_ids = []
        for idx, (chunk, embedding) in enumerate(tqdm(zip(chunks, embeddings), total=len(chunks), desc="Storing")):
            text_content = chunk['text']
            metadata_str = json.dumps(chunk.get('metadata', {}), sort_keys=True)
            unique_string = f"{text_content}|{metadata_str}|{idx}"
            hash_bytes = hashlib.md5(unique_string.encode()).digest()
            point_id = abs(int.from_bytes(hash_bytes[:8], byteorder='big')) % (2**63 - 1)
            
            if point_id in seen_ids:
                duplicate_ids.append((point_id, idx))
                collision_counter = len([d for d in duplicate_ids if d[0] == point_id])
                unique_string = f"{unique_string}|collision_{collision_counter}"
                hash_bytes = hashlib.md5(unique_string.encode()).digest()
                point_id = abs(int.from_bytes(hash_bytes[:8], byteorder='big')) % (2**63 - 1)
            seen_ids.add(point_id)
            
            payload = {
                'text': chunk['text'],
                **chunk.get('metadata', {})
            }
            
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                )
            )
        
        batch_size = 200
        total_batches = (len(points) + batch_size - 1) // batch_size
        print(f"Uploading {len(points)} points in {total_batches} batches (batch size: {batch_size})...")
        
        for i in tqdm(range(0, len(points), batch_size), desc="Uploading to Qdrant", total=total_batches, unit="batch"):
            batch_points = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch_points
            )
        
        unique_points_count = len(set(p.id for p in points))
        print(f"Stored {len(points)} points ({unique_points_count} unique IDs) in Qdrant collection '{self.collection_name}'")
        return unique_points_count
    
    def build_knowledge_base(
        self,
        sources: List[str],
        max_samples_per_source: Optional[int] = None,
        recreate_collection: bool = False
    ):
        """Build the complete knowledge base from specified sources."""
        print("=" * 60)
        print("Building Knowledge Base")
        print("=" * 60)
        
        self.create_collection(recreate=recreate_collection)
        
        all_documents = []
        source_counts = {}
        
        print(f"\nLoading documents from {len(sources)} source(s): {', '.join(sources)}")
        
        if 'squad' in sources:
            docs = load_squad_contexts(max_samples_per_source)
            all_documents.extend(docs)
            source_counts['squad'] = len(docs)
        
        if 'sciq' in sources:
            docs = load_sciq_contexts(max_samples_per_source)
            all_documents.extend(docs)
            source_counts['sciq'] = len(docs)
        
        if 'wikipedia_simple' in sources:
            docs = load_wikipedia_simple(max_samples_per_source)
            all_documents.extend(docs)
            source_counts['wikipedia_simple'] = len(docs)
        
        if 'hotpotqa' in sources:
            docs = load_hotpotqa(max_samples_per_source)
            all_documents.extend(docs)
            source_counts['hotpotqa'] = len(docs)
        
        print(f"\nLoading Summary:")
        for source, count in source_counts.items():
            status = "OK" if count > 0 else "FAILED"
            print(f"  {status} {source}: {count} documents")
        
        if not all_documents:
            print("No documents loaded. Check your source list and error messages above.")
            return
        
        print(f"\nTotal documents loaded: {len(all_documents)}")
        
        chunks = self.chunk_documents(all_documents)
        embeddings = self.generate_embeddings(chunks)
        stored_count = self.store_in_qdrant(chunks, embeddings)
        
        time.sleep(2)
        count_result = self.client.count(
            collection_name=self.collection_name,
            exact=True
        )
        exact_count = count_result.count
        
        print(f"\nKnowledge base built successfully!")
        print(f"  Collection: {self.collection_name}")
        print(f"  Points stored: {stored_count}")
        print(f"  Exact count: {exact_count} points")
        print(f"  Vector size: {self.vector_size}")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query text
            top_k: Number of results to return
            score_threshold: Minimum similarity score (0-1)
            filter_metadata: Metadata filters (e.g., {'source': 'squad'})
        
        Returns:
            List of retrieved documents with scores
        """
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True).tolist()
        
        query_filter = None
        if filter_metadata:
            conditions = []
            for key, value in filter_metadata.items():
                conditions.append(
                    models.Filter(
                        must=[
                            models.FieldCondition(
                                key=key,
                                match=models.MatchValue(value=value)
                            )
                        ]
                    )
                )
            if conditions:
                query_filter = models.Filter(must=conditions)
        
        search_results = None
        if hasattr(self.client, "search") and callable(getattr(self.client, "search")):
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=query_filter
            )
        else:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=query_filter,
                with_payload=True
            )
            search_results = response.points
        
        results = []
        for result in search_results:
            results.append({
                'text': result.payload.get('text', ''),
                'score': result.score,
                'metadata': {k: v for k, v in result.payload.items() if k != 'text'}
            })
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Build and manage knowledge base in Qdrant Cloud')
    
    parser.add_argument('--qdrant_url', type=str, default=QDRANT_URL,
                        help=f'Qdrant Cloud cluster URL (default: from config)')
    parser.add_argument('--qdrant_api_key', type=str, default=QDRANT_API_KEY,
                        help=f'Qdrant API key (default: from config)')
    parser.add_argument('--collection_name', type=str, default=COLLECTION_NAME,
                        help=f'Collection name (default: {COLLECTION_NAME})')
    
    parser.add_argument('--sources', type=str, nargs='+',
                        default=['squad', 'sciq', 'wikipedia_simple', 'hotpotqa'],
                        choices=['squad', 'sciq', 'wikipedia_simple', 'hotpotqa'],
                        help='Data sources to include (default: squad sciq wikipedia_simple hotpotqa)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum samples per source (default: None, use all)')
    
    parser.add_argument('--recreate', action='store_true',
                        help='Delete and recreate collection if it exists')
    
    parser.add_argument('--query', type=str, default=None,
                        help='Query text for retrieval (if provided, skips building)')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of results to retrieve (default: 5)')
    
    args = parser.parse_args()
    
    builder = KnowledgeBaseBuilder(
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
        collection_name=args.collection_name
    )
    
    if args.query:
        print(f"Retrieving documents for query: '{args.query}'")
        results = builder.retrieve(args.query, top_k=args.top_k)
        
        print(f"\n{'='*60}")
        print(f"Retrieval Results (top {len(results)})")
        print(f"{'='*60}")
        
        for i, result in enumerate(results, 1):
            print(f"\n[{i}] Score: {result['score']:.4f}")
            print(f"Source: {result['metadata'].get('source', 'unknown')}")
            print(f"Text: {result['text'][:200]}...")
    else:
        builder.build_knowledge_base(
            sources=args.sources,
            max_samples_per_source=args.max_samples,
            recreate_collection=args.recreate
        )


if __name__ == '__main__':
    main()

