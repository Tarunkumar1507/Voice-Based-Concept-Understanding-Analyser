import os
import sys
import gc
import torch
from typing import Optional

# Limit PyTorch CPU threads to conserve memory in containerized / low-spec environments
try:
    torch.set_num_threads(1)
except Exception:
    pass

# Global variable to cache the whisper model in-memory
_whisper_model = None

def get_whisper_model_name() -> str:
    """Returns configured model, defaulting to 'tiny' for low memory."""
    return os.getenv("WHISPER_MODEL", "tiny")

def load_whisper_model(model_name: Optional[str] = None):
    """
    Loads and caches the Whisper model in-memory.
    If the requested model fails (e.g. file missing / network issue), automatically falls back to an alternative.
    """
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    
    if model_name is None:
        model_name = get_whisper_model_name()
    
    try:
        import whisper
    except ImportError:
        print("Error: 'openai-whisper' package is not installed. Please install it via pip.")
        raise ImportError("openai-whisper is not installed. Check requirements.txt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Whisper '{model_name}' model on device: {device}...")
    
    try:
        _whisper_model = whisper.load_model(model_name, device=device)
        print(f"Whisper '{model_name}' model loaded successfully.")
        return _whisper_model
    except Exception as primary_err:
        print(f"Failed to load Whisper '{model_name}': {primary_err}")
        # Try fallback: if tiny failed try base, if base failed try tiny
        fallback_model = "base" if model_name == "tiny" else "tiny"
        print(f"Attempting fallback to '{fallback_model}' model...")
        try:
            gc.collect()
            _whisper_model = whisper.load_model(fallback_model, device=device)
            print(f"Fallback Whisper '{fallback_model}' model loaded successfully.")
            return _whisper_model
        except Exception as fallback_err:
            print(f"Fallback to '{fallback_model}' also failed: {fallback_err}")
            raise primary_err

def transcribe_audio(file_path: str, model_name: Optional[str] = None) -> str:
    """
    Transcribes the input audio file (WAV/MP3) to plain text using OpenAI Whisper.
    Includes hardware checks, error handling, and memory cleanup.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found at: {file_path}")

    if os.path.getsize(file_path) == 0:
        raise ValueError("The uploaded audio file is empty (0 bytes).")
        
    try:
        model = load_whisper_model(model_name)
        print(f"Transcribing audio file: {file_path}...")
        
        # Load audio using librosa at 16000Hz
        import librosa
        try:
            audio_data, _ = librosa.load(file_path, sr=16000)
        except Exception as load_err:
            raise RuntimeError(
                f"Failed to load audio file: {load_err}. "
                "Please verify that the audio file is valid, and that ffmpeg is installed if using complex codecs."
            )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        use_fp16 = (device == "cuda")
        
        result = model.transcribe(audio_data, fp16=use_fp16)
        
        # Clean up memory buffers immediately
        del audio_data
        gc.collect()
        
        transcript = result.get("text", "").strip()
        if not transcript:
            print("Warning: Whisper transcription returned empty text.")
            return ""
            
        return transcript
        
    except Exception as e:
        print(f"Whisper transcription failed: {e}")
        raise RuntimeError(f"Whisper transcription failed: {str(e)}")
