import os
import time
import random
import sqlite3
import struct
import numpy as np
import librosa
import pandas as pd
from collections import Counter
from scipy.ndimage import maximum_filter

from build_db import *
# SR = 22050
# N_FFT = 2048
# HOP_LENGTH = 512
# PEAK_NEIGHBORHOOD_SIZE = 50
# TARGET_ZONE_T_MIN = 1
# TARGET_ZONE_T_MAX = 50
# TARGET_ZONE_F_MIN = -30
# TARGET_ZONE_F_MAX = 30
DB_PATH = "fingerprints.db"

DURATIONS = [5, 10]  # Snippet lengths in seconds
SNRS_DB = [0, 5, 10, 15, 20, 25, 30, float('inf')]
PITCH_SHIFTS = list(range(-10, 11))  # Semitones
NUM_SAMPLES_PER_TRACK = 3  # Number of random crops per song
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

def add_awgn(y, snr_db):
    if snr_db == float('inf'):
        return y
    rms = np.sqrt(np.mean(y**2))
    y_norm = y / rms if rms > 0 else y
    noise = np.random.randn(len(y_norm))
    noise_variance = 10 ** (-snr_db / 10)
    y_noisy = y_norm + noise * np.sqrt(noise_variance)
    return np.clip(y_noisy, -1.0, 1.0)

def apply_pitch_shift(y, sr, n_steps):
    if n_steps == 0:
        return y
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def get_query_hashes(y):
    """Signal processing pipeline: Spectrogram -> Constellation -> Triplet Hashes."""
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    local_max = maximum_filter(S, size=PEAK_NEIGHBORHOOD_SIZE) == S
    threshold = np.percentile(S, 95)
    peaks = (S > threshold) & local_max
    freq_indices, time_indices = np.where(peaks)
    cmap = sorted(list(zip(freq_indices, time_indices)), key=lambda x: x[1])
    
    hashes = []
    for i in range(len(cmap)):
        f_anchor, t_anchor = cmap[i]
        for j in range(i + 1, len(cmap)):
            f_target, t_target = cmap[j]
            delta_t = t_target - t_anchor
            if TARGET_ZONE_T_MIN <= delta_t <= TARGET_ZONE_T_MAX:
                if TARGET_ZONE_F_MIN <= (f_target - f_anchor) <= TARGET_ZONE_F_MAX:
                    hashes.append((f"{f_anchor}|{f_target}|{delta_t}", t_anchor))
            elif delta_t > TARGET_ZONE_T_MAX:
                break
    return hashes

def retrieve_ranked_list(y):
    """Queries the database and returns a list of song names ranked by alignment score."""
    query_hashes = get_query_hashes(y)
    if not query_hashes:
        return []
        
    matches = {} # {song_id: [offsets]}
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    for hash_val, t_query in query_hashes:
        cursor.execute("SELECT song_id, offset FROM hashes WHERE hash=?", (hash_val,))
        for song_id, t_db in cursor.fetchall():
            if isinstance(t_db, bytes):
                t_db = struct.unpack('<q', t_db)[0]
            if song_id not in matches:
                matches[song_id] = []
            matches[song_id].append(int(t_db) - int(t_query))

    scores = []
    for song_id, offsets in matches.items():
        # Score = count of the most frequent time-offset (temporal alignment consensus)
        _, count = Counter(offsets).most_common(1)[0]
        cursor.execute("SELECT name FROM songs WHERE id=?", (song_id,))
        name = cursor.fetchone()[0]
        scores.append((count, name))
    
    conn.close()
    scores.sort(key=lambda x: x[0], reverse=True)
    return [s[1] for s in scores]

def run_tests(audio_dir):
    audio_files = []
    for root, _, files in os.walk(audio_dir):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3')):
                audio_files.append(os.path.join(root, file))

    results = []
    for d in DURATIONS:
        d_samples = d * SR
        print(f"Testing {d}s snippets...")

        for file_path in audio_files:
            true_name = os.path.basename(file_path)
            y_full, _ = librosa.load(file_path, sr=SR)
            if len(y_full) < d_samples: continue

            for _ in range(NUM_SAMPLES_PER_TRACK):
                start = random.randint(0, len(y_full) - d_samples)
                snippet = y_full[start : start + d_samples]

                # --- AWGN TESTS ---
                for snr in SNRS_DB:
                    y_test = add_awgn(snippet.copy(), snr)
                    start_time = time.time()
                    ranking = retrieve_ranked_list(y_test)
                    latency = time.time() - start_time
                    rank = ranking.index(true_name) + 1 if true_name in ranking else float('inf')
                    results.append({'duration': d, 'aug': 'awgn', 'val': snr, 'rank': rank, 'latency': latency})

                # --- PITCH SHIFT TESTS ---
                for shift in PITCH_SHIFTS:
                    y_test = apply_pitch_shift(snippet.copy(), SR, shift)
                    start_time = time.time()
                    ranking = retrieve_ranked_list(y_test)
                    latency = time.time() - start_time
                    rank = ranking.index(true_name) + 1 if true_name in ranking else float('inf')
                    results.append({'duration': d, 'aug': 'pitch', 'val': shift, 'rank': rank, 'latency': latency})

    return pd.DataFrame(results)

def get_metrics(df):
    """Calculates evaluation criteria from the results dataframe."""
    top1 = (df['rank'] == 1).mean()
    top5 = (df['rank'] <= 5).mean()
    mrr = (1.0 / df['rank']).mean()
    latency = df['latency'].mean()
    return {'Top-1': top1, 'Top-5': top5, 'MRR': mrr, 'Avg Latency': latency}


df_results = run_tests("./audio")
df_results.to_csv("results.csv", index=False)

print("METRICS")
metrics = get_metrics(df_results)
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")
