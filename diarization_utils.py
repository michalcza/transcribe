from resemblyzer import preprocess_wav, VoiceEncoder
import numpy as np
from sklearn.cluster import KMeans
import os

def diarize_with_resemblyzer(audio_path, num_speakers=None):
    print(f"🔹 Running speaker diarization using Resemblyzer on {os.path.basename(audio_path)}.")


    wav = preprocess_wav(audio_path)
    encoder = VoiceEncoder()

    # Safely unpack only what we need
    result = encoder.embed_utterance(wav, return_partials=True)
    print(f"Type: {type(result)}, Length: {len(result)}")

    if isinstance(result, tuple):
        print(f"Shape of result[0]: {result[0].shape}")  # Expect (N, 256)

    result = encoder.embed_utterance(wav, return_partials=True)

    # Handle tuple of unknown length defensively
    if isinstance(result, tuple) and len(result) >= 2:
        embeds = result[0]
        timestamps = result[1]
    else:
        raise ValueError("Unexpected output format from embed_utterance")

    # Check for single embedding (1D)
    if embeds.ndim == 1:
        print("⚠️ Only one embedding found — treating as a single speaker.")
        embeds = embeds.reshape(1, -1)
        timestamps = [(0.0, len(wav) / 16000)]
        labels = [0]  # One speaker only
    else:
        if len(embeds) < num_speakers:
            print(f"⚠️ Only {len(embeds)} embeddings found, reducing num_speakers to {len(embeds)}")
            num_speakers = len(embeds)

        kmeans = KMeans(n_clusters=num_speakers, random_state=0).fit(embeds)
        labels = kmeans.labels_

    segments = [(start, end, label) for (start, end), label in zip(timestamps, labels)]
    return segments

