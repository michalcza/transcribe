from tqdm import tqdm
from threading import Thread
from dotenv import load_dotenv
from sklearn.cluster import KMeans, DBSCAN
from pydub import AudioSegment
from io import BytesIO
from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import fcluster, linkage
from diarization_utils import diarize_with_resemblyzer
import whisper
import os
import argparse
import signal
import sys
import psutil
import GPUtil
import time
import warnings
import torch
import queue
import torchaudio
import numpy as np
import requests
import onnxruntime

def download_silero_model():
    model_url = "https://models.silero.ai/models/vad/silero_vad.jit"
    local_path = "silero_vad.jit"
    if not os.path.exists(local_path):
        print("\n📥 Downloading Silero VAD model...")
        r = requests.get(model_url)
        with open(local_path, "wb") as f:
            f.write(r.content)
    return local_path

def load_silero_vad():
    path = download_silero_model()
    model = torch.jit.load(path)
    model.eval()
    return model

vad_model = None  # Only initialize if diarization is enabled

# Load environment variables from .env
load_dotenv()

# Set CUDA device from .env
cuda_devices = os.getenv("CUDA_VISIBLE_DEVICES", "0")  # Default to GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n💻 Using device: {device.upper()}")

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning)  
warnings.filterwarnings("ignore", module="whisper")  
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  
os.environ["TOKENIZERS_PARALLELISM"] = "false"  

monitoring = True  # Global flag for monitoring thread

# Gracefully handle Ctrl+C interruptions
def signal_handler(sig, frame):
    global monitoring
    monitoring = False
    print("\nProcess interrupted. Exiting gracefully.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def format_timestamp(seconds):
    # Convert seconds to HH:MM:SS format.
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

    # Run VAD + Clustering if gpu_usage is not None:
    usage_message += f" | GPU: {gpu_usage:.1f}%"
    print('\r' + ' ' * 80, end='\r')  # Clear line with spaces
    print(usage_message, end="", flush=True)  # Keep updates on the same line
    time.sleep(2)  # Update every 2 seconds

def monitor_system():
    """
    Monitors CPU, RAM, and GPU usage while transcription runs.
    Prints a live updating line to the console.
    """
    global monitoring
    process = psutil.Process(os.getpid())  # Current process info

    while monitoring:
        cpu_usage = process.cpu_percent(interval=1) / psutil.cpu_count()
        ram_usage = process.memory_info().rss / (1024 * 1024)  # MB

        gpu_usage = None
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu_usage = gpus[0].load * 100  # First GPU

        usage_message = f"\r🔹 CPU: {cpu_usage:.2f}% | RAM: {ram_usage:.1f}MB"
        if gpu_usage is not None:
            usage_message += f" | GPU: {gpu_usage:.1f}%"
        
        print('\r' + ' ' * 80, end='\r')  # Clear line with spaces
        print(usage_message, end="", flush=True)
        time.sleep(2)
        
# Convert input audio to WAV (16kHz mono)
def convert_audio_to_wav(input_path):
    # Convert input audio to 16kHz mono WAV for diarization compatibility.
    output_path = os.path.splitext(input_path)[0] + ".wav"

    if input_path.endswith(".wav"):
        return input_path  # Already WAV

    print(f"🔹 Converting {input_path} to {output_path} (16kHz mono)")
    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(output_path, format="wav")
        return output_path
    except Exception as e:
        print(f"\n❌ Conversion error: {e}")
        return None

def transcribe_audio(file_path, language=None, model_size="medium", enable_diarization=False, enable_monitoring=True):
    global monitoring, vad_model

    monitoring = True  # Enable monitoring if used

    print(f"🔹 Loading Whisper model: {model_size}")
    model = whisper.load_model(model_size)
    model.to(device)

    print(f"🎙️ Transcribing file: {file_path}")
    if language:
        print(f"🌍 Forcing language: {language}")

    # Convert to WAV if needed
    wav_file = convert_audio_to_wav(file_path)

    # Start monitoring thread
    if enable_monitoring:
        monitor_thread = Thread(target=monitor_system, daemon=True)
        monitor_thread.start()

    # Transcription with Whisper
    with torch.no_grad():
        result = model.transcribe(wav_file, language=language)

    # Base transcript text with time stamps
    transcript_text = [
        f"[{format_timestamp(segment['start'])} - {format_timestamp(segment['end'])}] {segment['text']}"
        for segment in result["segments"]
    ]
    final_transcript = "\n".join(transcript_text)

    # Stop monitoring
    monitoring = False

    # Diarization (Resemblyzer-based)
    if enable_diarization:
        print("\n🔹 Performing speaker diarization using Resemblyzer.")
        diarized_segments = diarize_with_resemblyzer(wav_file, num_speakers=args.speakers)

        # Format diarization output
        if diarized_segments:
            diarization_output = "\n\nDiarization Results:\n"
                
            for start, end, speaker_label in diarized_segments:
                diarization_output += f"[{format_timestamp(start)} - {format_timestamp(end)}] Speaker {speaker_label}: Speech detected\n"

            final_transcript += diarization_output
        else:
            final_transcript += "\n\nDiarization failed or returned no segments.\n"

    # Determine output file name based on forced language
    if language:
        lang_code = language.lower()
        output_file = os.path.splitext(file_path)[0] + f"_{lang_code}.txt"
    else:
        output_file = os.path.splitext(file_path)[0] + ".txt"

    # Save transcript to file
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(final_transcript)
        print(f"\n✅ Transcription complete.")
        print(f"\n📄 Saved transcript to: {output_file}")
    except Exception as e:
        print(f"\n❌ Error writing transcript to file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio with optional speaker separation and system monitoring.")
    parser.add_argument("--file", required=True, help="Path to the audio file.")
    parser.add_argument("--language", default=None, help="Force a specific language.")
    parser.add_argument("--model", choices=["tiny", "small", "medium", "large"], default="medium", help="Whisper model size.")
    parser.add_argument("--diarization", action="store_true", help="Enable speaker separation.")
    parser.add_argument("--no-monitor", action="store_true", help="Disable system resource monitoring.")
    parser.add_argument("--speakers", type=int, help="Estimated number of speakers for diarization.")

    args = parser.parse_args()
    transcribe_audio(args.file, args.language, args.model, args.diarization, not args.no_monitor)
