from tqdm import tqdm
from threading import Thread
from dotenv import load_dotenv
from sklearn.cluster import KMeans, DBSCAN
from pydub import AudioSegment
from io import BytesIO
from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import fcluster, linkage
from diarization_utils import diarize_with_resemblyzer, plot_embeddings
from pathlib import Path

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
import json

# Hide PyTorch + tokenizer + future deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

def assign_speakers_to_segments(segments, diarized_segments):
    """
    Assigns speaker labels to Whisper segments by overlap.
    """
    labeled = []
    for seg in segments:
        seg_start, seg_end = seg['start'], seg['end']
        # Default speaker if not matched
        speaker = "Speaker ?"
        for d_start, d_end, spk in diarized_segments:
            if d_start <= seg_start <= d_end or d_start <= seg_end <= d_end:
                speaker = f"Speaker {spk}"
                break
        labeled.append({
            "start": seg_start,
            "end": seg_end,
            "speaker": speaker,
            "text": seg['text'].strip()
        })
    return labeled

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
        segments = result["segments"]

    # Base transcript text with time stamps
    transcript_text = [
        f"[{format_timestamp(segment['start'])} - {format_timestamp(segment['end'])}] {segment['text']}"
        for segment in result["segments"]
    ]
    final_transcript = "\n".join(transcript_text)

    # Stop monitoring
    monitoring = False
    
    # Assign speaker labels to segments
    if enable_diarization:
        print("🔹 Performing speaker diarization using Resemblyzer...")

        if args.plot:
            diarized_segments, embeds, labels = diarize_with_resemblyzer(
                wav_file,
                num_speakers=args.speakers,
                return_plot_data=True
            )
            plot_path = Path(file_path).with_suffix(".png")
            plot_embeddings(embeds, labels, out_path=plot_path, show=False)








        # if args.plot:
            # diarized_segments, embeds, labels = diarize_with_resemblyzer(
                # wav_file,
                # num_speakers=args.speakers,
                # return_plot_data=True
            # )
            # plot_embeddings(embeds, labels)
            # plot_path = Path(file_path).with_suffix(".png")
            
            
            
            #plot_embeddings(embeds, labels, out_path=plot_path, show=False)
        else:
            diarized_segments = diarize_with_resemblyzer(
                wav_file,
                num_speakers=args.speakers
            )

        final_segments = assign_speakers_to_segments(segments, diarized_segments)

    else:
        # No diarization: assign default speaker
        final_segments = [
            {
                "start": seg['start'],
                "end": seg['end'],
                "speaker": "Speaker 0",
                "text": seg['text'].strip()
            }
            for seg in segments
        ]
        
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

    base = Path(file_path)
    base_output_path = base.with_suffix("")

    if language:
        lang_code = language.lower()
        base_output_path = base_output_path.with_name(base_output_path.name + f"_{lang_code}")

    # Save transcript as plain text
    output_txt = base_output_path.with_suffix(".txt")
    try:
        with open(output_txt, "w", encoding="utf-8") as f:
            f.write(final_transcript)
        print(f"\n✅ Transcription complete.")
        print(f"📄 Saved transcript to: {output_txt}")
    except Exception as e:
        print(f"\n❌ Error writing transcript to text file: {e}")

    # Save transcript as JSON
    output_json = base_output_path.with_suffix(".json")
    try:
        with open(output_json, "w", encoding="utf-8") as jf:
            json.dump(final_segments, jf, indent=2)
        print(f"📄 Saved transcript JSON to: {output_json}")
    except Exception as e:
        print(f"\n❌ Error writing transcript to JSON: {e}")

    # # Optionally: save a second JSON version with custom `_pl` suffix
    # json_path_pl = base_output_path.with_name(base_output_path.name + "_pl").with_suffix(".json")
    # try:
        # with open(json_path_pl, "w", encoding="utf-8") as jf:
            # json.dump(final_segments, jf, indent=2)
        # print(f"📄 Saved alternate JSON to: {json_path_pl}")
    # except Exception as e:
        # print(f"\n❌ Error writing alternate JSON to: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio with optional speaker separation and system monitoring.")
    parser.add_argument("--file", required=True, help="Path to the audio file.")
    parser.add_argument("--language", default=None, help="Force a specific language.")
    parser.add_argument("--model", choices=["tiny", "small", "medium", "large"], default="medium", help="Whisper model size.")
    parser.add_argument("--diarization", action="store_true", help="Enable speaker separation.")
    parser.add_argument("--no-monitor", action="store_true", help="Disable system resource monitoring.")
    parser.add_argument("--speakers", type=int, help="Estimated number of speakers for diarization.")
    parser.add_argument("--plot", action="store_true", help="Visualize speaker embeddings after diarization.")

    args = parser.parse_args()
    transcribe_audio(args.file, args.language, args.model, args.diarization, not args.no_monitor)
