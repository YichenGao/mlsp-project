from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import librosa

music_dir = "/orcd/pool/003/dbertsim_shared/kondylis/mlsp-project/music_mlsp"
out_dir = "/orcd/pool/003/dbertsim_shared/kondylis/mlsp-project/processed_music_index"

sample_rate = 22050
hop_frac = 0.25

def song_index_df(music_dir):
    rows = []
    for song_id, file in enumerate(os.listdir(music_dir)):
        if file[-4:] == ".mp3":
            mp3_path = os.path.join(music_dir, file)
            duration = librosa.get_duration(path=mp3_path)
            rows.append({"song_id": song_id, "path": mp3_path, "duration_sec": duration,})
    song_idx_df = pd.DataFrame(rows)
    return song_idx_df


song_df = song_index_df(music_dir)
song_index_path = "/orcd/pool/003/dbertsim_shared/kondylis/mlsp-project/song_index.csv"
song_df.to_csv(song_index_path, index=False)

def build_window_feature_database(song_df, sample_rate=22050, hop_frac=0.25, duration_sec=5):
    window_size = int(duration_sec * sample_rate)
    hop_size = int(window_size * hop_frac)

    rows, features = [], []
    for index, row in tqdm(song_df.iterrows(), total=len(song_df)):
        song_id, path = row["song_id"], row["path"]
        y, sr = librosa.load(path, sr=sample_rate, mono=True)

        for start in range(0, len(y) - window_size + 1, hop_size):
            end = start + window_size
            mel = librosa.feature.melspectrogram(y = y[start : end], sr = sample_rate, n_mels=128, fmin=20, fmax=8000)
            rows.append({"song_id": song_id, "start_sample": start, "end_sample": end, "start_time_sec": start / sample_rate, "duration_sec": duration_sec})
            features.append(mel.flatten().astype(np.float32))

    windows_df = pd.DataFrame(rows)
    features = np.stack(features)

    windows_df.to_csv(f"/orcd/pool/003/dbertsim_shared/kondylis/mlsp-project/windows_5s.csv", index=False)
    np.save(f"/orcd/pool/003/dbertsim_shared/kondylis/mlsp-project/features_logmel_5s.npy", features)
    return windows_df, features

song_index_path = "/orcd/pool/003/dbertsim_shared/kondylis/mlsp-project/song_index.csv"
song_df = pd.read_csv(song_index_path)

build_window_feature_database(song_df=song_df, sample_rate=22050, hop_frac=0.25, duration_sec=5)