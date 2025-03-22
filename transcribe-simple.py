from tqdm import tqdm
from threading import Thread
from dotenv import load_dotenv
from sklearn.cluster import DBSCAN
from pydub import AudioSegment
from io import BytesIO
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
        print("📥 Downloading Silero VAD model...")
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
print(f"💻 Using device: {device.upper()}")

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

def vad_segmentation(audio_path):
    # Detects speech segments in an audio file.
    print("🔹 Running VAD-based diarization.")

    waveform, sample_rate = torchaudio.load(audio_path)
    segments = []

    for i in range(0, waveform.shape[1], sample_rate):
        chunk = waveform[:, i : i + sample_rate]
    
def detect_speech(model, chunk, sample_rate):
    # Convert to mono float32
    chunk = chunk.mean(dim=0).unsqueeze(0)  # (1, samples)
    # Resample to 16 kHz if needed
    if sample_rate != 16000:
        chunk = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(chunk)

    with torch.no_grad():
        prob = model(chunk)[0][0].item()
    return prob > 0.5  # Adjustable
    
def vad_segmentation(audio_path):
    print("🔹 Running Silero VAD-based diarization...")

    waveform, sample_rate = torchaudio.load(audio_path)
    segments = []

    for i in range(0, waveform.shape[1], sample_rate):
        chunk = waveform[:, i:i + sample_rate]
        if detect_speech(vad_model, chunk, sample_rate):
            segments.append((i / sample_rate, (i + sample_rate) / sample_rate))
    return segments

def cluster_speakers(segments):
    # Clusters speech segments into speakers.
    if not segments:
        return []

    segment_array = np.array([(start, end) for start, end in segments])
    clustering = DBSCAN(eps=3, min_samples=2).fit(segment_array)
    return clustering.labels_

# Run VAD + Clustering
def vad_diarization(file_path):
    # Applies VAD + Clustering for speaker diarization.
    segments = vad_segmentation(file_path)
    speaker_labels = cluster_speakers(segments)

    diarized_transcript = []
    for i, (start, end) in enumerate(segments):
        speaker = speaker_labels[i] if len(speaker_labels) > i else "Unknown"
        diarized_transcript.append(f"[{start:.2f} - {end:.2f}] Speaker {speaker}: Speech detected")

    return "\n".join(diarized_transcript)

# Monitor CPU, RAM, and GPU usage in a separate thread
def monitor_system():
    # Monitors CPU, RAM, and GPU usage while transcription runs.
    global monitoring
    process = psutil.Process(os.getpid())  # Get current Python process

    while monitoring:
        cpu_usage = process.cpu_percent() / psutil.cpu_count()  # Per-core CPU usage
        ram_usage = process.memory_info().rss / (1024 * 1024)  # RAM usage in MB

        gpu_usage = None
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu_usage = gpus[0].load * 100  # GPU utilization

        # Keep monitoring output on the same line
        usage_message = f"\r🔹 CPU: {cpu_usage:.2f}% | RAM: {ram_usage:.1f}MB"
        if gpu_usage is not None:
            usage_message += f" | GPU: {gpu_usage:.1f}%"

        print(usage_message, end="", flush=True)  # Keep updates on the same line
        time.sleep(2)  # Update every 2 seconds

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
        print(f"❌ Conversion error: {e}")
        return None

def transcribe_audio(file_path, language=None, model_size="medium", enable_diarization=False, enable_monitoring=True):
    global monitoring, vad_model
    
    monitoring = True  # Ensure monitoring is enabled

    print(f"🔹 Loading Whisper model: {model_size}")
    model = whisper.load_model(model_size)
    model.to(device)  # Move model to GPU

    print(f"🎙️ Transcribing file: {file_path}")
    if language:
        print(f"🌍 Forcing language: {language}")

    # Convert to WAV for diarization
    if enable_diarization:
        file_path = convert_audio_to_wav(file_path)

        # Only load VAD model if diarization is enabled
        if vad_model is None:
            print("🔹 Loading Silero VAD model.")
            vad_model = load_silero_vad()

    # Start system monitor if enabled
    if enable_monitoring:
        monitor_thread = Thread(target=monitor_system, daemon=True)
        monitor_thread.start()

    # Transcription
    with torch.no_grad():
        result = model.transcribe(file_path, language=language)

    transcript_text = [
        f"[{format_timestamp(segment['start'])} - {format_timestamp(segment['end'])}] {segment['text']}"
        for segment in result["segments"]
    ]

    monitoring = False  # Stop system monitoring

    # Apply diarization if enabled
    final_transcript = "\n".join(transcript_text)
    if enable_diarization:
        print("🔹 Using VAD-based speaker diarization.")
        diarization_result = vad_diarization(file_path)
        final_transcript += "\n\nDiarization Results:\n" + diarization_result

    # Save transcript
    # Determine output file name based on forced language
    if language:
        lang_code = language.lower()
        output_file = os.path.splitext(file_path)[0] + f"_{lang_code}.txt"
    else:
        output_file = os.path.splitext(file_path)[0] + ".txt"
        
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(final_transcript)
        print(f"\n✅ Transcription complete.")
        print(f"📄 Saved transcript to: {output_file}")
    except Exception as e:
        print(f"❌ Error writing transcript to file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio with optional speaker separation and system monitoring.")
    parser.add_argument("--file", required=True, help="Path to the audio file.")
    parser.add_argument("--language", default=None, help="Force a specific language.")
    parser.add_argument("--model", choices=["tiny", "small", "medium", "large"], default="medium", help="Whisper model size.")
    parser.add_argument("--diarization", action="store_true", help="Enable speaker separation.")
    parser.add_argument("--no-monitor", action="store_true", help="Disable system resource monitoring.")
    args = parser.parse_args()
    transcribe_audio(args.file, args.language, args.model, args.diarization, not args.no_monitor)
