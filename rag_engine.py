#!/usr/bin/env python3
"""
RAG (Retrieval-Augmented Generation) Engine
-------------------------------------------
Main orchestrator that integrates intent classification, retrieval routing, and LLM generation.

This engine:
1. Classifies query intent using trained intent classifier
2. Routes to optimal retrieval strategy based on intent
3. Retrieves relevant context from Qdrant knowledge base
4. Processes and merges retrieved chunks
5. Generates grounded answers using GPT-4o-mini
6. Falls back to general knowledge if retrieval is insufficient
"""

import os
import sys
import argparse
from typing import List, Dict, Optional
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Import knowledge base and RAG components
from knowledge_base import KnowledgeBaseBuilder
from rag_components import IntentClassifier, RAGRouter, ContextProcessor


class RAGEngine:
    """Main RAG engine integrating all components."""
    
    def __init__(
        self,
        intent_model_path: str = "./intent_classifier_model_roberta",
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        collection_name: Optional[str] = None,
        openai_api_key: Optional[str] = None
    ):
        """
        Initialize RAG engine.
        
        Args:
            intent_model_path: Path to intent classifier model
            qdrant_url: Qdrant Cloud URL (from .env if None)
            qdrant_api_key: Qdrant API key (from .env if None)
            collection_name: Collection name (from .env if None)
            openai_api_key: OpenAI API key (from .env if None)
        """
        # Initialize intent classifier
        self.intent_classifier = IntentClassifier(intent_model_path)
        
        # Initialize knowledge base builder
        qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "")
        qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY", "")
        collection_name = collection_name or os.getenv("COLLECTION_NAME", "pattern_recognition")
        
        if not qdrant_url or not qdrant_api_key:
            raise ValueError("Qdrant credentials not found. Set QDRANT_URL and QDRANT_API_KEY in .env")
        
        self.kb_builder = KnowledgeBaseBuilder(
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            collection_name=collection_name
        )
        
        # Initialize OpenAI client
        openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
        if not openai_api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY in .env")
        
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.model_name = "gpt-4o-mini"
    
    def retrieve_context(self, query: str, intent_name: str) -> List[Dict]:
        """
        Retrieve relevant context based on query and intent.
        
        Args:
            query: User query
            intent_name: Predicted intent name
            
        Returns:
            List of retrieved chunks with scores
        """
        strategy = RAGRouter.get_strategy(intent_name)
        results = self.kb_builder.retrieve(
            query=query,
            top_k=strategy["top_k"],
            score_threshold=strategy["score_threshold"]
        )
        return results
    
    def generate_grounded_answer(self, query: str, context: str, chunks: List[Dict]) -> Dict:
        """
        Generate strictly grounded answer using custom RAG prompt.
        
        Args:
            query: User query
            context: Retrieved and merged context
            chunks: List of retrieved chunks with metadata (should have 'index' and 'metadata' fields)
            
        Returns:
            Dictionary with answer and metadata
        """
        # Build chunk reference list for citation
        chunk_references = []
        for chunk in chunks:
            idx = chunk.get('index', '?')
            source = chunk.get('metadata', {}).get('source', 'unknown')
            chunk_references.append(f"Chunk {idx}: source={source}")
        
        chunk_ref_text = "\n".join(chunk_references) if chunk_references else "No chunks available"
        
        # Custom strict grounding prompt
        prompt = f"""You are an evidence-based AI assistant. You answer questions ONLY using the retrieved context provided below. You must strictly follow these rules:

1. You are NOT allowed to use any outside, prior, or general knowledge. 
   You must ignore everything you know outside the retrieved context.

2. If the answer is not fully supported by the retrieved context, say exactly:
   "The answer is not found in the provided context."

3. Cite which parts of the context you used. Use the format:
   [source: <dataset>, chunk: <index>]

4. If the context provides only partial information, give a partial answer and 
   clearly state what information is missing.

5. If the question is ambiguous, list the possible interpretations and answer 
   only what is supported by the context.

6. Keep the answer concise, accurate, and strictly grounded in the retrieved evidence.

------------------------

Retrieved Context:

{context}

------------------------

Chunk References:
{chunk_ref_text}

------------------------

User Question:

{query}

------------------------

Grounded Answer:

"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content.strip()
            
            return {
                "answer": answer,
                "model": self.model_name,
                "tokens_used": response.usage.total_tokens if hasattr(response, 'usage') else None
            }
        except Exception as e:
            return {
                "answer": f"Error generating answer: {str(e)}",
                "model": self.model_name,
                "error": str(e)
            }
    
    def generate_fallback_answer(self, query: str) -> Dict:
        """
        Generate answer using general knowledge (fallback mode).
        
        Args:
            query: User query
            
        Returns:
            Dictionary with answer and metadata
        """
        system_prompt = "You are a helpful AI assistant. Answer the user's question using your general knowledge. Be accurate, informative, and comprehensive."
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.7,  # Higher temperature for more natural responses
                max_tokens=500
            )
            
            answer = response.choices[0].message.content.strip()
            
            return {
                "answer": answer,
                "model": self.model_name,
                "tokens_used": response.usage.total_tokens if hasattr(response, 'usage') else None
            }
        except Exception as e:
            return {
                "answer": f"Error generating fallback answer: {str(e)}",
                "model": self.model_name,
                "error": str(e)
            }
    
    def detect_failure_signals(self, answer: str, context: str, min_context_length: int = 150) -> bool:
        """
        Detect failure signals in RAG answer that indicate fallback should be used.
        
        Args:
            answer: Generated RAG answer
            context: Retrieved context
            min_context_length: Minimum context length threshold
            
        Returns:
            True if fallback should be activated
        """
        # Check for explicit failure phrases
        failure_phrases = [
            "The answer is not found in the provided context",
            "Additional information required"
        ]
        
        answer_lower = answer.lower()
        for phrase in failure_phrases:
            if phrase.lower() in answer_lower:
                return True
        
        # Check context length
        if not context or len(context.strip()) < min_context_length:
            return True
        
        return False
    
    
    def answer_query(self, query: str, verbose: bool = False) -> Dict:
        """
        Complete RAG pipeline: intent -> retrieval -> generation.
        
        Args:
            query: User query
            verbose: Whether to print intermediate steps
            
        Returns:
            Dictionary with answer and full pipeline metadata
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing Query: {query}")
            print(f"{'='*60}")
        
        # Step 1: Classify intent
        intent_id, intent_name = self.intent_classifier.predict_intent(query)
        if verbose:
            print(f"Predicted Intent: {intent_name} (ID: {intent_id})")
        
        # Step 2: Retrieve context
        retrieved_chunks = self.retrieve_context(query, intent_name)
        if verbose:
            print(f"Retrieved {len(retrieved_chunks)} chunks")
            for i, chunk in enumerate(retrieved_chunks[:3], 1):
                print(f"  [{i}] Score: {chunk['score']:.4f} | Source: {chunk['metadata'].get('source', 'unknown')}")
        
        # Step 3: Process context
        # Deduplicate
        unique_chunks = ContextProcessor.deduplicate_chunks(retrieved_chunks)
        if verbose and len(unique_chunks) < len(retrieved_chunks):
            print(f"Deduplicated: {len(retrieved_chunks)} -> {len(unique_chunks)} chunks")
        
        # Order by score
        ordered_chunks = ContextProcessor.order_chunks(unique_chunks)
        
        # Merge into context
        context = ContextProcessor.merge_chunks(ordered_chunks, max_length=4000)
        if verbose:
            print(f"Merged context length: {len(context)} characters")
        
        # Step 4: Generate grounded RAG answer first
        if verbose:
            print("Generating grounded RAG answer...")
        
        # Prepare chunks with indices for citation
        chunks_with_indices = []
        for idx, chunk in enumerate(ordered_chunks, 1):
            chunk_copy = chunk.copy()
            chunk_copy['index'] = idx
            chunks_with_indices.append(chunk_copy)
        
        # Generate grounded answer using custom strict prompt
        grounded_result = self.generate_grounded_answer(query, context, chunks_with_indices)
        grounded_answer = grounded_result['answer']
        
        # Step 5: Check for failure signals and activate fallback if needed
        use_fallback = self.detect_failure_signals(grounded_answer, context, min_context_length=150)
        
        if use_fallback:
            if verbose:
                print("Failure signals detected - activating fallback mode")
            # Generate fallback answer using general knowledge
            fallback_result = self.generate_fallback_answer(query)
            final_answer = fallback_result['answer']
            generation_result = {
                "answer": final_answer,
                "model": self.model_name,
                "fallback_used": True,
                "grounded_answer": grounded_answer,  # Keep original for reference
                "tokens_used": fallback_result.get('tokens_used')
            }
        else:
            if verbose:
                print("Grounded answer accepted")
            generation_result = {
                "answer": grounded_answer,
                "model": self.model_name,
                "fallback_used": False,
                "tokens_used": grounded_result.get('tokens_used')
            }
        
        result = generation_result
        
        # Compile full response
        response = {
            "query": query,
            "intent": {
                "id": intent_id,
                "name": intent_name
            },
            "retrieval": {
                "chunks_retrieved": len(retrieved_chunks),
                "chunks_after_dedup": len(unique_chunks),
                "context_length": len(context),
                "top_chunks": [
                    {
                        "text": chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text'],
                        "score": chunk['score'],
                        "source": chunk['metadata'].get('source', 'unknown')
                    }
                    for chunk in ordered_chunks[:3]
                ]
            },
            "generation": result,
            "fallback_used": result.get("fallback_used", False)
        }
        
        # Add grounded answer to response if fallback was used
        if result.get("fallback_used") and "grounded_answer" in result:
            response["grounded_answer"] = result["grounded_answer"]
        
        return response


def main():
    parser = argparse.ArgumentParser(description='RAG Engine for grounded question answering')
    
    parser.add_argument('--query', type=str, required=True,
                        help='Query to answer')
    parser.add_argument('--intent_model', type=str,
                        default='./intent_classifier_model_roberta',
                        help='Path to intent classifier model (default: ./intent_classifier_model_roberta)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed pipeline information')
    
    args = parser.parse_args()
    
    try:
        # Initialize RAG engine
        rag = RAGEngine(intent_model_path=args.intent_model)
        
        # Process query
        response = rag.answer_query(args.query, verbose=args.verbose)
        
        # Print results
        print(f"\n{'='*60}")
        print("RAG Engine Response")
        print(f"{'='*60}")
        print(f"\nQuery: {response['query']}")
        print(f"Intent: {response['intent']['name']}")
        print(f"Retrieved Chunks: {response['retrieval']['chunks_retrieved']}")
        print(f"Fallback Used: {response['fallback_used']}")
        print(f"\nAnswer:")
        print(f"{response['generation']['answer']}")
        
        if args.verbose:
            print(f"\n{'='*60}")
            print("Detailed Information")
            print(f"{'='*60}")
            print(f"Context Length: {response['retrieval']['context_length']} characters")
            print(f"Top Retrieved Chunks:")
            for i, chunk in enumerate(response['retrieval']['top_chunks'], 1):
                print(f"  [{i}] Score: {chunk['score']:.4f} | Source: {chunk['source']}")
                print(f"      Text: {chunk['text']}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

