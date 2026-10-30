import os
import json
import librosa
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple

def load_concepts(concepts_path: str = None) -> Dict[str, str]:
    """
    Loads reference concepts and their definitions from the JSON file.
    """
    if concepts_path is None:
        concepts_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "reference_concepts",
            "concepts.json"
        )
    try:
        if os.path.exists(concepts_path):
            with open(concepts_path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            print(f"Concepts file not found at {concepts_path}")
            return {}
    except Exception as e:
        print(f"Error loading reference concepts: {e}")
        return {}

def save_uploaded_file(uploaded_file, upload_dir: str) -> str:
    """
    Saves an uploaded Streamlit file to the upload directory.
    Returns the absolute path of the saved file.
    """
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def plot_waveform(audio_path: str, output_image_path: str) -> str:
    """
    Generates a modern, stylish waveform visualization and saves it as an image.
    Uses matplotlib to create a high-quality visualization.
    """
    try:
        # Load audio using librosa
        y, sr = librosa.load(audio_path, sr=None)
        
        # Create figure and axis
        plt.figure(figsize=(10, 3), dpi=150)
        
        # Convert index to time
        time = np.linspace(0, len(y) / sr, num=len(y))
        
        # Plot with modern styling (glassmorphism feel, cyan/blue colors)
        plt.plot(time, y, color="#00d2ff", alpha=0.8, linewidth=0.8)
        plt.fill_between(time, y, color="#00d2ff", alpha=0.3)
        
        # Styling details
        plt.title("Speech Waveform Representation", fontsize=10, color="#1e293b", fontweight="bold")
        plt.xlabel("Time (seconds)", fontsize=8, color="#64748b")
        plt.ylabel("Amplitude", fontsize=8, color="#64748b")
        
        # Remove top and right borders for a cleaner look
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cbd5e1')
        ax.spines['bottom'].set_color('#cbd5e1')
        ax.tick_params(colors='#64748b', labelsize=8)
        
        plt.grid(True, linestyle="--", alpha=0.2, color="#cbd5e1")
        plt.tight_layout()
        
        # Ensure target folder exists
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        
        # Save output image
        plt.savefig(output_image_path, bbox_inches="tight", transparent=True)
        plt.close()
        return output_image_path
    except Exception as e:
        print(f"Error plotting waveform: {e}")
        # Generate dummy fallback image
        plt.figure(figsize=(10, 3), dpi=150)
        plt.text(0.5, 0.5, "Waveform Unavailable", ha='center', va='center', fontsize=12, color='#ef4444')
        plt.axis('off')
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        plt.savefig(output_image_path, bbox_inches="tight")
        plt.close()
        return output_image_path

def cleanup_files(*file_paths: str) -> None:
    """
    Deletes temporary files safely.
    """
    for file_path in file_paths:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")
