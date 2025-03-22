import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path

def plot_embeddings(embeds, labels, out_path=None, title="Speaker Clustering (PCA)", show=False):
    embeds = np.array(embeds)
    labels = np.array(labels)

    # Inject a second point if we only have one
    if len(embeds) < 2:
        print("⚠️ Only one embedding found — injecting dummy point for visualization.")
        embeds = np.vstack([embeds, embeds[0] + 1e-6])
        labels = np.append(labels, labels[0])

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeds)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        reduced[:, 0], reduced[:, 1],
        c=labels, cmap="tab10", s=40
    )
    plt.title(title)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.grid(True)
    plt.colorbar(scatter, ticks=range(np.max(labels)+1), label="Speaker")
    plt.tight_layout()

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path)
        print(f"📊 Saved speaker plot to: {out_path}")

    if show:
        plt.show()

    plt.close()

# ---- TEST CODE ----

if __name__ == "__main__":
    print("🧪 Testing speaker embedding plot...")

    # Generate fake speaker data
    np.random.seed(42)
    speaker_0 = np.random.normal(0, 1, (30, 256))  # 30 embeddings
    speaker_1 = np.random.normal(5, 1, (30, 256))  # 30 embeddings
    embeds = np.vstack((speaker_0, speaker_1))
    labels = [0]*30 + [1]*30

    # Output plot
    plot_embeddings(embeds, labels, out_path="output/speaker_test.png", show=True)
