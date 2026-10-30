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

def transcribe_with_gemini(file_path: str) -> Optional[str]:
    """
    Attempts to transcribe audio directly using the Google Gemini 1.5 Flash API.
    Runs in the cloud (2-3 seconds, 0 CPU / 0 RAM on Render), preventing timeouts.
    Returns None if key is missing or call fails, allowing automatic Whisper fallback.
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key.lower().startswith("your_") or len(api_key) < 15:
        return None
        
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        mime_type = "audio/mp3" if file_path.lower().endswith(".mp3") else "audio/wav"
        with open(file_path, "rb") as f:
            audio_bytes = f.read()
            
        print("Transcribing audio using Google Gemini API (cloud acceleration)...")
        model = genai.GenerativeModel("gemini-1.5-flash")
        audio_part = {
            "mime_type": mime_type,
            "data": audio_bytes
        }
        prompt = (
            "Listen to this audio recording and transcribe every spoken word verbatim into plain text. "
            "Output ONLY the exact transcribed words. Do not summarize, translate, or add commentary."
        )
        response = model.generate_content([audio_part, prompt], request_options={"timeout": 15.0})
        transcript = response.text.strip()
        if transcript:
            print(f"Gemini cloud transcription successful ({len(transcript)} chars).")
            return transcript
    except Exception as e:
        print(f"Gemini cloud transcription not available ({e}). Falling back to local Whisper...")
    return None

def load_whisper_model(model_name: Optional[str] = None):
    """
    Loads and caches the Whisper model in-memory.
    If the requested model fails, automatically falls back to an alternative.
    """
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    
    if model_name is None:
        model_name = get_whisper_model_name()
    
    try:
        import whisper
    except ImportError:
        print("Error: 'openai-whisper' package is not installed.")
        raise ImportError("openai-whisper is not installed. Check requirements.txt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Whisper '{model_name}' model on device: {device}...")
    
    try:
        _whisper_model = whisper.load_model(model_name, device=device)
        print(f"Whisper '{model_name}' model loaded successfully.")
        return _whisper_model
    except Exception as primary_err:
        print(f"Failed to load Whisper '{model_name}': {primary_err}")
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
    Transcribes the input audio file (WAV/MP3) to plain text.
    1. Uses Google Gemini API if GEMINI_API_KEY is configured (takes 2s in cloud).
    2. Falls back to local Whisper model seamlessly if offline or API key is absent.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found at: {file_path}")

    if os.path.getsize(file_path) == 0:
        raise ValueError("The uploaded audio file is empty (0 bytes).")
        
    # Attempt ultra-fast Gemini cloud transcription first
    gemini_result = transcribe_with_gemini(file_path)
    if gemini_result:
        return gemini_result
        
    # Fall back to local Whisper
    try:
        model = load_whisper_model(model_name)
        print(f"Transcribing audio with local Whisper: {file_path}...")
        
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
        
        del audio_data
        gc.collect()
        
        transcript = result.get("text", "").strip()
        if not transcript:
            print("Warning: Whisper transcription returned empty text.")
            return ""
            
        return transcript
        
    except Exception as e:
        print(f"Whisper transcription failed: {e}")
        raise RuntimeError(f"Transcription failed: {str(e)}")
