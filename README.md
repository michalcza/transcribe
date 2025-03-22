# Transcribe & Diarize

A Python-based command-line tool that:
- Transcribes audio files using OpenAI's Whisper
- Performs speaker diarization using Resemblyzer and KMeans
- Monitors CPU, RAM, and GPU usage during execution
- Outputs readable, timestamped transcripts with optional speaker labels

## Features

- Whisper transcription  
- Speaker diarization using voice embeddings  
- Real-time system resource monitoring  
- Audio preprocessing & format conversion  
- Supports `.wav`, `.m4a`, `.mp3`, and more  
- Configurable language and model size  
- Optional speaker count override

### Files

/-----------------------|------------------------------------------------------\
| File                  | Description                                          |
|-----------------------|------------------------------------------------------|
|.\requirements.txt     | List of prerequisites.                               |
|.\transcribe.py        | Main applicaiton.                                    |
|.\transcribe-simple.py | Simplified application without diarization.          |
|.\diarization_utils.py | Loaded by .\transcribe.py                            |
|.\.env                 | Private keys and variables                           |
|.\gpu-test.py          | Standalone used to test for GPU and configuration.   |
|.\README.md            | This file                                            |
\-----------------------|------------------------------------------------------/

## Installation

### 1. Clone the repo
```bash
git clone https://github.com/michalcza/transcribe
cd transcribe
```

### 2. Setup virtual enviroment
```
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

## Usage
```
python transcribe_refactored.py --file path/to/audio.m4a --model medium --language en --diarization --speakers 2
```

### CLI Options
/----------------|--------------------------------------------------|----------\
| Argument       | Description                                      | Default  |
|----------------|--------------------------------------------------|----------|
| `--file`       | Path to audio file                               | required |
| `--language`   | Force language (e.g., `en`, `es`)                | auto     |
| `--model`      | Whisper model size [`tiny|small|medium|large`]   | medium   |
| `--diarization`| Enable speaker separation                        | off      |
| `--speakers`   | Estimated number of speakers (for KMeans)        | optional |
| `--no-monitor` | Disable CPU/RAM/GPU system monitoring            | enabled  |
\----------------|--------------------------------------------------|----------/

## Output
Transcripts are saved as .txt files in the same directory as your audio, with optional diarization info:
```
[00:00:00 - 00:00:05] Hello, welcome to the meeting.
[00:00:05 - 00:00:12] Speaker 1: Speech detected
```

## Output
- openai-whisper
- resemblyzer
- pydub
- torch
- scikit-learn
- psutil
- GPUtil
- tqdm
- argparse

Install with:
```
pip install -r requirements.txt
```

## Known Issues
- Very short or low-volume audio may result in only one speaker being detected.
- Ensure audio is at least 3–5 seconds long for effective diarization.
- Accuracy improves with higher Whisper models (e.g., medium, large).

## To Do
- Merge speaker labels into Whisper transcript
- Export .srt / .vtt subtitle files
- Use DBSCAN or BIC for automatic speaker count
- Visualize speaker clusters (PCA, t-SNE)
- Streamlit web interface

## Author
- Developed by Michal Czarnecki (mczarnecki@gmail.com)
- Adapted from OpenAI Whisper + Resemblyzer
