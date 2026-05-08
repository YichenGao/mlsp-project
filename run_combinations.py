import time
import numpy as np
import pandas as pd
import librosa
from collections import Counter, defaultdict
from sklearn.decomposition import PCA, NMF
from sklearn.neighbors import BallTree
from tqdm import tqdm

song_index_path = "/orcd/pool/003/dbertsim_shared/kondylis/mlsp-project/song_index.csv"
results_path = "/orcd/pool/003/dbertsim_shared/kondylis/mlsp-project/grid_results_all.csv"
sample_rate = 22050
hop_frac = 0.25
n_queries_per_song = 3

def extract_feature(y, sample_rate, feature_type):
    if feature_type == "mel":
        feats = librosa.feature.melspectrogram(y=y, sr=sample_rate, n_mels=128, fmin=20, fmax=8000)
    elif feature_type == "chroma":
        feats = librosa.feature.chroma_stft(y=y, sr=sample_rate, n_fft=2048, hop_length=512)
    elif feature_type == "spec":
        feats = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    return feats.flatten().astype(np.float32)


def build_windows(song_df, duration_sec, feature_type, sample_rate=22050, hop_frac=0.25):
    window_size = int(duration_sec * sample_rate)
    hop_size = int(window_size * hop_frac)
    rows, features = [], []
    for idx, row in tqdm(song_df.iterrows(), total = len(song_df), desc = f"Using {feature_type} and {duration_sec}s"):
        song_id, path = row["song_id"], row["path"]
        y, sr = librosa.load(path, sr = sample_rate, mono=True)
        for start in range(0, len(y) - window_size + 1, hop_size):
            end = start + window_size
            features.append(extract_feature(y[start:end], sample_rate, feature_type))
            rows.append({"song_id": song_id, "start_time_sec": start / sample_rate})
    return pd.DataFrame(rows), np.stack(features)


def fit_reducer(features, method, n_components):
    if method == "pca":
        reducer = PCA(n_components=n_components, whiten=True).fit(features)
        fit_metric = reducer.explained_variance_ratio_.sum()
        print(f"PCA n={n_components}: explained var = {fit_metric:.3f}")
    elif method == "nmf":
        reducer = NMF(n_components=n_components, max_iter=300, random_state=0).fit(features)
        embeddings_full = reducer.transform(features)
        recon = reducer.inverse_transform(embeddings_full)
        fit_metric = np.linalg.norm(features - recon) / np.linalg.norm(features)
        print(f"NMF n={n_components}: relative recon error = {fit_metric:.3f}")

    embeddings = reducer.transform(features).astype(np.float32)
    tree = BallTree(embeddings, leaf_size=40)
    return reducer, tree


def query_simple(snippet, reducer, tree, windows_df, feature_type, sample_rate, duration_sec, hop_frac=0.25, k=20):
    window_size = int(duration_sec * sample_rate)
    hop_size = int(window_size * hop_frac)
    if len(snippet) < window_size:
        return []
    query_feats = []
    for start in range(0, len(snippet) - window_size + 1, hop_size):
        query_feats.append(extract_feature(snippet[start:start + window_size], sample_rate, feature_type))
    query_emb = reducer.transform(np.stack(query_feats)).astype(np.float32)
    _, neighbor_idxs = tree.query(query_emb, k=k)
    candidate_ids = windows_df.iloc[neighbor_idxs.flatten()]["song_id"].values
    votes = Counter(candidate_ids)
    return [sid for sid, _ in votes.most_common()]


def build_test_set(song_df, snippet_dur=5.0, n_queries_per_song=3, sample_rate=22050, seed=0):
    rng = np.random.default_rng(seed)
    snippet_size = int(snippet_dur * sample_rate)
    test_set = []
    for _, row in tqdm(song_df.iterrows(), total=len(song_df), desc="build test set"):
        true_id, path = row["song_id"], row["path"]
        y, sr = librosa.load(path, sr=sample_rate, mono=True)
        if len(y) < snippet_size:
            continue
        starts = rng.integers(0, len(y) - snippet_size, size=n_queries_per_song)
        for s in starts:
            test_set.append({"true_id": true_id, "snippet": y[s:s + snippet_size].copy()})
    return test_set


def evaluate(test_set, reducer, tree, windows_df, feature_type, sample_rate, duration_sec):
    top1, top5, mrr_sum, total, latency_sum = 0, 0, 0.0, 0, 0.0
    for item in test_set:
        true_id, snippet = item["true_id"], item["snippet"]
        t0 = time.perf_counter()
        ranked = query_simple(snippet, reducer, tree, windows_df, feature_type, sample_rate, duration_sec, k=20)
        latency_sum += time.perf_counter() - t0
        total += 1
        if ranked and ranked[0] == true_id:
            top1 += 1
        if true_id in ranked[:5]:
            top5 += 1
        if true_id in ranked:
            mrr_sum += 1.0 / (ranked.index(true_id) + 1)
    return {"top1": top1 / total, "top5": top5 / total, "mrr": mrr_sum / total, "latency_sec": latency_sum / total}


song_df = pd.read_csv(song_index_path)

durations = [5, 10]
feature_types = ["mel", "chroma"]
methods = ["pca", "nmf"]
dims = [128, 256, 512]

results = []
for duration_sec in durations:
    test_set = build_test_set(song_df, snippet_dur = duration_sec, n_queries_per_song = n_queries_per_song, sample_rate = sample_rate, seed=42)
    for feature_type in feature_types:
        windows_df, features = build_windows(song_df, duration_sec, feature_type, sample_rate=sample_rate, hop_frac=hop_frac)
        for method in methods:
            for n_components in dims:
                reducer, tree = fit_reducer(features, method, n_components)
                metrics = evaluate(test_set, reducer, tree, windows_df, feature_type, sample_rate, duration_sec)
                row = {"duration_sec": duration_sec, "feature": feature_type,
                       "method": method, "dims": n_components,
                       "top1": metrics["top1"],
                       "top5": metrics["top5"],
                       "mrr": metrics["mrr"],
                       "latency_sec": metrics["latency_sec"]}
                print(row)
                results.append(row)
                pd.DataFrame(results).to_csv(results_path, index=False)
