import os
import sys
import torch
from typing import Optional

# Global variable to cache the whisper model in-memory
_whisper_model = None

def load_whisper_model(model_name: str = "base"):
    """
    Loads and caches the Whisper model in-memory.
    Automatically checks for CPU/GPU availability.
    """
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    
    # Try to import whisper inside function to allow import errors to be caught gracefully
    try:
        import whisper
    except ImportError:
        print("Error: 'openai-whisper' package is not installed. Please install it via pip.")
        raise ImportError("openai-whisper is not installed. Check requirements.txt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Whisper '{model_name}' model on device: {device}...")
    try:
        # Load the whisper model locally
        _whisper_model = whisper.load_model(model_name, device=device)
        print("Whisper model loaded successfully.")
        return _whisper_model
    except Exception as e:
        print(f"Failed to load Whisper model: {e}")
        raise

def transcribe_audio(file_path: str) -> str:
    """
    Transcribes the input audio file (WAV/MP3) to plain text using OpenAI Whisper.
    Includes hardware checks, error handling, and file validation.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found at: {file_path}")

    # Check file size (extremely small files might be empty/corrupt)
    if os.path.getsize(file_path) == 0:
        raise ValueError("The uploaded audio file is empty (0 bytes).")
        
    try:
        model = load_whisper_model("base")
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
        
        transcript = result.get("text", "").strip()
        if not transcript:
            print("Warning: Whisper transcription returned empty text.")
            return ""
            
        return transcript
        
    except Exception as e:
        print(f"Whisper transcription failed: {e}")
        raise RuntimeError(f"Whisper transcription failed: {str(e)}")
