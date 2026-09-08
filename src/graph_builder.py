import numpy as np
import librosa
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data


def build_segment_graph(segment_features, top_k=1):

    n = len(segment_features)
    edges = []

    for i in range(n - 1):
        edges.append((i, i + 1))
        edges.append((i + 1, i))

    sim_matrix = cosine_similarity(segment_features)
    for i in range(n):
        candidates = [(j, sim_matrix[i, j]) for j in range(n) if abs(j - i) > 1]
        candidates.sort(key=lambda x: -x[1])
        for j, _ in candidates[:top_k]:
            if (i, j) not in edges:
                edges.append((i, j))
                edges.append((j, i))

    return edges, sim_matrix


def to_pyg_data(segment_features, edges):
    
    x = torch.tensor(segment_features, dtype=torch.float)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)


def process_track(audio_path, sr=22050, segment_sec=5, top_k=1, n_chroma=12):
    
    y, _ = librosa.load(audio_path, sr=sr)
    segment_samples = segment_sec * sr
    n_segments = len(y) // segment_samples

    if n_segments < 2:
        return None

    segment_features = []
    for i in range(n_segments):
        start = i * segment_samples
        end = start + segment_samples
        seg = y[start:end]
        chroma = librosa.feature.chroma_stft(y=seg, sr=sr, n_chroma=n_chroma)
        segment_features.append(chroma.mean(axis=1))
    segment_features = np.array(segment_features)

    edges, _ = build_segment_graph(segment_features, top_k=top_k)
    return to_pyg_data(segment_features, edges)
