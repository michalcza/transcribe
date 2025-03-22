import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from resemblyzer import preprocess_wav, VoiceEncoder
from pathlib import Path

def diarize_with_resemblyzer(audio_path, num_speakers=None, return_plot_data=False):

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
    
    if return_plot_data:
        return segments, embeds, labels
    else:
        return segments

def plot_embeddings(embeds, labels, out_path=None, title="Speaker Clustering (PCA)", show=False):
    embeds = np.array(embeds)
    labels = np.array(labels)

    # Fake a second point if there's only one
    if len(embeds) < 2:
        print("⚠️ Only one embedding found — injecting dummy point for visualization.")
        embeds = np.vstack([embeds, embeds[0] + 1e-6])  # add tiny offset
        labels = np.append(labels, labels[0])  # reuse label

    # Reduce to 2D using PCA
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeds)

    # Plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        reduced[:, 0], reduced[:, 1],
        c=labels,
        cmap="tab10", s=40
    )
    plt.title(title)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.grid(True)
    plt.colorbar(scatter, ticks=range(max(labels) + 1), label="Speaker")
    plt.tight_layout()

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path)
        print(f"📊 Saved speaker embedding plot to: {out_path}")

    if show:
        plt.show()

plt.close()

