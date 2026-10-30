import os
import librosa
import numpy as np
from typing import Dict, Any

def extract_audio_features(audio_path: str, total_words: int = 0) -> Dict[str, Any]:
    """
    Extracts structural audio features from an audio file using Librosa.
    
    Parameters:
    - audio_path: Path to the WAV/MP3 file.
    - total_words: Number of transcribed words (used to calculate speaking rate).
    
    Returns a dictionary containing:
    - 'duration': float (seconds)
    - 'speaking_time': float (seconds)
    - 'silent_time': float (seconds)
    - 'pause_ratio': float (ratio of silence to total duration)
    - 'rms_energy': float (mean RMS energy)
    - 'zcr': float (mean Zero Crossing Rate)
    - 'speaking_rate': float (words per minute)
    - 'mean_pitch': float (estimated mean fundamental frequency in Hz)
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at: {audio_path}")

    # Set default values for silent/empty audio
    default_metrics = {
        "duration": 0.0,
        "speaking_time": 0.0,
        "silent_time": 0.0,
        "pause_ratio": 0.0,
        "rms_energy": 0.0,
        "zcr": 0.0,
        "speaking_rate": 0.0,
        "mean_pitch": 0.0
    }

    try:
        # Load audio (sr=None preserves native sampling rate)
        y, sr = librosa.load(audio_path, sr=None)
        
        # Check if the audio array is empty
        if len(y) == 0:
            return default_metrics
            
        # 1. Duration
        duration = float(librosa.get_duration(y=y, sr=sr))
        if duration == 0.0:
            return default_metrics

        # 2. Speaking Time & Silent Time (using energy threshold split)
        # top_db=25 means frame is considered silent if it is 25dB below the peak
        non_silent_intervals = librosa.effects.split(y, top_db=25)
        
        speaking_samples = 0
        for interval in non_silent_intervals:
            speaking_samples += (interval[1] - interval[0])
            
        speaking_time = float(speaking_samples) / sr
        
        # Clip speaking time to total duration
        speaking_time = min(speaking_time, duration)
        silent_time = max(0.0, duration - speaking_time)
        
        # 3. Pause Ratio
        pause_ratio = silent_time / duration if duration > 0 else 0.0

        # 4. RMS Energy
        rms = librosa.feature.rms(y=y)
        rms_energy = float(np.mean(rms))

        # 5. Zero Crossing Rate (ZCR)
        zcr = librosa.feature.zero_crossing_rate(y=y)
        mean_zcr = float(np.mean(zcr))

        # 6. Estimated Speaking Rate (Words Per Minute)
        # Using speaking_time is more accurate than total duration because it excludes long silences
        if speaking_time > 0:
            speaking_rate = float(total_words) / (speaking_time / 60.0)
        else:
            speaking_rate = 0.0

        # 7. Pitch Estimation (F0 using Yin algorithm)
        # Yin can be slow for high sample rates/durations. We downsample or run it with try/catch.
        mean_pitch = 0.0
        try:
            # Downsample to 16000Hz for pitch detection to make it much faster and stable
            y_16k = librosa.resample(y, orig_sr=sr, target_sr=16000)
            # Standard vocal range limits: 75Hz to 300Hz
            pitches = librosa.yin(y_16k, fmin=75, fmax=300, sr=16000)
            # Filter out non-voiced frames or unrealistic estimates
            valid_pitches = pitches[(pitches >= 75) & (pitches <= 300)]
            if len(valid_pitches) > 0:
                mean_pitch = float(np.mean(valid_pitches))
        except Exception as pitch_err:
            print(f"Pitch estimation failed: {pitch_err}, falling back to 0.0")
            mean_pitch = 0.0

        return {
            "duration": round(duration, 2),
            "speaking_time": round(speaking_time, 2),
            "silent_time": round(silent_time, 2),
            "pause_ratio": round(pause_ratio, 3),
            "rms_energy": round(rms_energy, 4),
            "zcr": round(mean_zcr, 4),
            "speaking_rate": round(speaking_rate, 2),
            "mean_pitch": round(mean_pitch, 2)
        }

    except Exception as e:
        print(f"Error extracting audio features: {e}")
        return default_metrics
