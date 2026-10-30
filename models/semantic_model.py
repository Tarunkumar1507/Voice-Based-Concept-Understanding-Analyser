import numpy as np
from typing import Dict, Any
from sentence_transformers import SentenceTransformer

# In-memory cache for the SentenceTransformer model
_semantic_model = None

def load_semantic_model() -> SentenceTransformer:
    """
    Loads and caches the Sentence-BERT (all-MiniLM-L6-v2) model in-memory.
    """
    global _semantic_model
    if _semantic_model is not None:
        return _semantic_model
    
    try:
        print("Loading Sentence-BERT model (all-MiniLM-L6-v2)...")
        _semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Sentence-BERT model loaded successfully.")
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
        
        # Calculate cosine similarity
        similarity = calculate_cosine_similarity(student_emb, ref_emb)
        
        # Convert similarity to percentage (cosine similarity ranges from -1 to 1;
        # however, in sentence transformers, negative similarities are rare for meaningful texts.
        # We'll normalize it to a 0-100 range, clamping at 0 and 100)
        similarity_percentage = max(0.0, min(100.0, similarity * 100.0))
        
        # Determine understanding level
        # Excellent: >=80%, Good: >=60%, Average: >=40%, Poor: <40%
        if similarity_percentage >= 80.0:
            level = "Excellent"
        elif similarity_percentage >= 60.0:
            level = "Good"
        elif similarity_percentage >= 40.0:
            level = "Average"
        else:
            level = "Poor"
            
        return {
            "similarity_score": round(similarity_percentage, 2),
            "understanding_level": level
        }
        
    except Exception as e:
        print(f"Error during semantic similarity evaluation: {e}")
        return {
            "similarity_score": 0.0,
            "understanding_level": "Poor",
            "error": f"Evaluation error: {str(e)}"
        }
