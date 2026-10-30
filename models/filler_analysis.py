import re
from typing import Dict, Any, List

# Standard filler words and phrases to analyze
FILLER_WORDS = ["um", "uh", "like", "actually", "basically"]
FILLER_PHRASES = ["you know", "kind of", "sort of"]

def clean_and_tokenize(text: str) -> List[str]:
    """
    Cleans punctuation and tokenizes the text into lowercase words.
    """
    # Replace punctuation with spaces except apostrophes for contractions (e.g. don't)
    cleaned = re.sub(r"[^\w\s']", " ", text.lower())
    # Split by whitespace
    tokens = [w.strip() for w in cleaned.split() if w.strip()]
    return tokens

def analyze_fillers(transcript: str, duration_seconds: float = 0.0) -> Dict[str, Any]:
    """
    Analyzes the transcript to count filler words and compute frequencies/percentages.
    
    Returns a dictionary with:
    - 'total_fillers': int
    - 'filler_percentage': float (fillers as % of total words)
    - 'fillers_per_minute': float
    - 'fillers_per_100_words': float
    - 'filler_counts': Dict[str, int] (counts per individual filler)
    - 'total_words': int
    """
    if not transcript.strip():
        return {
            "total_fillers": 0,
            "filler_percentage": 0.0,
            "fillers_per_minute": 0.0,
            "fillers_per_100_words": 0.0,
            "filler_counts": {},
            "total_words": 0
        }

    tokens = clean_and_tokenize(transcript)
    total_words = len(tokens)
    
    if total_words == 0:
        return {
            "total_fillers": 0,
            "filler_percentage": 0.0,
            "fillers_per_minute": 0.0,
            "fillers_per_100_words": 0.0,
            "filler_counts": {},
            "total_words": 0
        }

    filler_counts = {item: 0 for item in FILLER_WORDS + FILLER_PHRASES}
    
    # 1. Count single-word fillers
    for token in tokens:
        if token in FILLER_WORDS:
            filler_counts[token] += 1
            
    # 2. Count multi-word filler phrases (using sliding windows or string search)
    normalized_text = " ".join(tokens)
    for phrase in FILLER_PHRASES:
        # Use regex to find non-overlapping occurrences of the phrase
        occurrences = len(re.findall(r'\b' + re.escape(phrase) + r'\b', normalized_text))
        filler_counts[phrase] = occurrences

    total_fillers = sum(filler_counts.values())
    
    # Percentage of total words
    filler_percentage = (total_fillers / total_words) * 100.0 if total_words > 0 else 0.0
    
    # Fillers per 100 words
    fillers_per_100_words = (total_fillers / total_words) * 100.0 if total_words > 0 else 0.0
    
    # Fillers per minute (if duration is provided)
    if duration_seconds > 0:
        duration_minutes = duration_seconds / 60.0
        fillers_per_minute = total_fillers / duration_minutes
    else:
        fillers_per_minute = 0.0
        
    return {
        "total_fillers": total_fillers,
        "filler_percentage": round(filler_percentage, 2),
        "fillers_per_minute": round(fillers_per_minute, 2),
        "fillers_per_100_words": round(fillers_per_100_words, 2),
        "filler_counts": {k: v for k, v in filler_counts.items() if v > 0},
        "total_words": total_words
    }
