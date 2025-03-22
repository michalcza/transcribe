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
- Optional Speaker Embedding Clustering via PCA and plotting.

### Files

/-----------------------|------------------------------------------------------\
| File                  | Description                                          |
|-----------------------|------------------------------------------------------|
|.\audio\               | Folder for audio input files (and output)            |
|.\plots\               | Folder for PCA plotting output graphs                |
|.\.env                 | Private keys and variables (excluded)                |
|.\.env.example         | Private keys and variables example file              |
|.\diarization_utils.py | Imported by .\transcribe.py                          |
|.\gpu-test.py          | Standalone used to test for GPU and configuration.   |
|.\plot_test.py         | PCA plotting test.                                   |
|.\README.md            | This file                                            |
|.\requirements.txt     | List of prerequisites.                               |
|.\transcribe.py        | Main applicaiton.                                    |
|.\transcribe-simple.py | Simplified application without diarization.          |
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
| `--plot`       | Enable PCA speaker visualization                 | off      |
\----------------|--------------------------------------------------|----------/

## Output

### Text File
Files are appended with forced language.
```audio_pl.txt```
```audio_pl.json```
If no forced language selection, output files equal input file.
```audio.txt``` 

Transcripts are saved as .txt files in the same directory as your audio, with optional diarization info:
```
[00:00:00 - 00:00:05] Hello, welcome to the meeting.
[00:00:05 - 00:00:12] Speaker 0: Speech detected.
...
Diarization Results:
[00:00:00 - 00:15:01] Speaker 0: Speech detected
```
### JSON
Transcripts are saved as .json files in the same directory as your audio, with optional diarization info:
```
[
  {
    "start": 0.0,
    "end": 4.1,
    "speaker": "Speaker 0",
    "text": "Nie jest Wizard no\u3067\u3059\u306d returns."
  },
]
```

##Speaker Embedding Clustering via PCA
This plot shows voice embeddings projected into 2D. Each point is a speech segment, colored by assigned speaker. PCA1 and PCA2 are principal components capturing the dominant variance in speaker identity. Clear separation indicates confident diarization.
Plot file written to `./plots/` folder and same root name as output file `audio_pl.png`.

## Dependencies
-openai-whisper
-resemblyzer
-pydub
-torch
-torchaudio
-scikit-learn
-psutil
-GPUtil
-tqdm
-matplotlib
-requests
-python-dotenv

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
- Use DBSCAN or BIC for automatic speaker count
- Visualize speaker clusters (PCA, t-SNE)
- Streamlit web interface

## Author
- Developed by Michal Czarnecki (mczarnecki@gmail.com)
- Adapted from OpenAI Whisper + Resemblyzer
