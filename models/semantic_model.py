import os
import gc
import torch
import numpy as np
from typing import Dict, Any
from sentence_transformers import SentenceTransformer

# Limit PyTorch CPU threads to conserve memory in containerized environments
try:
    torch.set_num_threads(1)
except Exception:
    pass

# In-memory cache for the SentenceTransformer model
_semantic_model = None

def load_semantic_model() -> SentenceTransformer:
    """
    Loads and caches the Sentence-BERT (all-MiniLM-L6-v2) model in-memory.
    Uses local_files_only=True first to avoid hanging on network requests.
    """
    global _semantic_model
    if _semantic_model is not None:
        return _semantic_model
    
    try:
        print("Loading Sentence-BERT model (all-MiniLM-L6-v2) from local cache...")
        _semantic_model = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)
        print("Sentence-BERT model loaded successfully from cache.")
        return _semantic_model
    except Exception as cache_err:
        print(f"Local cache lookup failed ({cache_err}). Loading from Hugging Face hub...")
        try:
            _semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Sentence-BERT model downloaded and loaded successfully.")
            return _semantic_model
        except Exception as e:
            print(f"Error loading Sentence-BERT model: {e}")
            raise

def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Computes cosine similarity between two 1D numpy arrays.
    Handles division by zero edge-cases.
    """
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
        
    return float(dot_product / (norm_a * norm_b))

def evaluate_understanding(student_transcript: str, reference_text: str) -> Dict[str, Any]:
    """
    Evaluates semantic similarity between the student's explanation and reference text.
    
    Returns a dictionary containing:
    - 'similarity_score': similarity as a percentage (0.0 to 100.0)
    - 'understanding_level': string ('Excellent', 'Good', 'Average', 'Poor')
    """
    if not student_transcript.strip() or not reference_text.strip():
        return {
            "similarity_score": 0.0,
            "understanding_level": "Poor",
            "error": "Empty input text provided"
        }

    try:
        # Load the sentence transformer model
        model = load_semantic_model()
        
        # Calculate embeddings
        embeddings = model.encode([student_transcript, reference_text])
        student_emb = embeddings[0]
        ref_emb = embeddings[1]
        
        # Calculate Cosine Similarity
        similarity = calculate_cosine_similarity(student_emb, ref_emb)
        
        # Free memory buffers immediately
        del embeddings, student_emb, ref_emb
        gc.collect()
        
        # Bound similarity between 0.0 and 1.0, then scale to 100
        bounded_sim = max(0.0, min(1.0, float(similarity)))
        score_percentage = round(bounded_sim * 100.0, 2)
        
        # Determine qualitative understanding level based on academic thresholds
        if score_percentage >= 80.0:
            level = "Excellent"
        elif score_percentage >= 65.0:
            level = "Good"
        elif score_percentage >= 50.0:
            level = "Average"
        else:
            level = "Poor"
            
        return {
            "similarity_score": score_percentage,
            "understanding_level": level,
            "error": None
        }
        
    except Exception as e:
        print(f"Semantic evaluation failed: {e}")
        return {
            "similarity_score": 0.0,
            "understanding_level": "Poor",
            "error": str(e)
        }
