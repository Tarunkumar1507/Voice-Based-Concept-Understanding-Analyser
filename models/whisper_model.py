import os
import sys
import gc
import torch
from typing import Optional

# Limit PyTorch CPU threads to conserve memory in low-RAM environments (like Render free tier)
try:
    torch.set_num_threads(1)
except Exception:
    pass

# Global variable to cache the whisper model in-memory
_whisper_model = None

def get_whisper_model_name() -> str:
    """Returns the configured Whisper model name, defaulting to 'tiny' for memory efficiency."""
    return os.getenv("WHISPER_MODEL", "tiny")

def load_whisper_model(model_name: Optional[str] = None):
    """
    Loads and caches the Whisper model in-memory.
    Automatically checks for CPU/GPU availability.
    Falls back to 'tiny' if a larger model fails to load due to memory.
    """
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    
    if model_name is None:
        model_name = get_whisper_model_name()
    
    # Try to import whisper inside function to allow import errors to be caught gracefully
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
    except Exception as e:
        print(f"Failed to load Whisper model '{model_name}': {e}")
        if model_name != "tiny":
            print("Attempting fallback to lightweight 'tiny' model...")
            try:
                gc.collect()
                _whisper_model = whisper.load_model("tiny", device=device)
                print("Fallback Whisper 'tiny' model loaded successfully.")
                return _whisper_model
            except Exception as fallback_err:
                print(f"Fallback to 'tiny' model also failed: {fallback_err}")
        raise

def transcribe_audio(file_path: str, model_name: Optional[str] = None) -> str:
    """
    Transcribes the input audio file (WAV/MP3) to plain text using OpenAI Whisper.
    Includes hardware checks, error handling, and memory cleanup.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found at: {file_path}")

    # Check file size (extremely small files might be empty/corrupt)
    if os.path.getsize(file_path) == 0:
        raise ValueError("The uploaded audio file is empty (0 bytes).")
        
    try:
        model = load_whisper_model(model_name)
        print(f"Transcribing audio file: {file_path}...")
        
        # Load audio using librosa at 16000Hz to bypass Whisper's internal command-line ffmpeg dependency
        import librosa
        try:
            audio_data, _ = librosa.load(file_path, sr=16000)
        except Exception as load_err:
            raise RuntimeError(
                f"Failed to load audio file: {load_err}. "
                "Please verify that the audio file is valid, and that ffmpeg is installed if using complex codecs."
            )
        
        # Perform transcription
        # fp16=False enforces float32 execution (prevents warnings on CPU environments)
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
