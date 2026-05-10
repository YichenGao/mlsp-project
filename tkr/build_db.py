import os
import sqlite3
import librosa
import numpy as np
from scipy.ndimage import maximum_filter


# Parameters for peak picking and hashing
SR = 22050 # Sampling rate
N_FFT = 2048 # Number of points in DFT/FFT
HOP_LENGTH = 512 # Hop length for STFT
PEAK_NEIGHBORHOOD_SIZE = 50 # Size of local neighborhood
TARGET_ZONE_T_MIN = 1 # Minimum time frames ahead for target zone
TARGET_ZONE_T_MAX = 50 # Maximum time frames ahead for target zone
TARGET_ZONE_F_MIN = -30 # Frequency bin lower bound for target zone
TARGET_ZONE_F_MAX = 30 # Frequency bin upper bound for target zone


def get_constellation_map(audio_path):
    """Make constellation map of time-frequency peaks from audio file."""
    y, _ = librosa.load(audio_path, sr=SR)
    
    # Compute spectrogram.
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    
    # Peak picking: Find local maxima in a neighborhood.
    local_max = maximum_filter(S, size=PEAK_NEIGHBORHOOD_SIZE) == S
    
    # Ignore small peaks.
    threshold = np.percentile(S, 90) 
    peaks = (S > threshold) & local_max
    
    # Get coordinates of peaks (frequency_bin, time_frame).
    frequencies, times = np.where(peaks)
    return list(zip(frequencies, times))


def generate_hashes(constellation_map):
    """Make triplet hashes from a constellation map."""
    hashes = []
    # Sort by time to easily look ahead
    constellation_map.sort(key=lambda x: x[1]) 
    
    for i in range(len(constellation_map)):
        f_anchor, t_anchor = constellation_map[i]
        
        # Look ahead to form target zones
        for j in range(i + 1, len(constellation_map)):
            f_target, t_target = constellation_map[j]
            delta_t = t_target - t_anchor
            
            # Check if target is within the valid time-frequency zone
            if TARGET_ZONE_T_MIN <= delta_t <= TARGET_ZONE_T_MAX:
                if TARGET_ZONE_F_MIN <= (f_target - f_anchor) <= TARGET_ZONE_F_MAX:
                    # Create the triplet hash: (f_anchor, f_target, delta_t)
                    hash_str = f"{f_anchor}|{f_target}|{delta_t}"
                    hashes.append((hash_str, t_anchor))
            elif delta_t > TARGET_ZONE_T_MAX:
                break # Moved past the target zone
    return hashes


def build_database(audio_dir, db_path="fingerprints.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables. The 'hashes' table acts as our inverted list.
    cursor.execute("CREATE TABLE IF NOT EXISTS songs (id INTEGER PRIMARY KEY, filepath TEXT, name TEXT)")
    cursor.execute("CREATE TABLE IF NOT EXISTS hashes (hash TEXT, offset INTEGER, song_id INTEGER)")
    
    # Walk through the nested directories
    song_id = 1
    for root, _, files in os.walk(audio_dir):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3')):
                filepath = os.path.join(root, file)
                print(f"Processing: {filepath}")
                
                try:
                    cmap = get_constellation_map(filepath)
                    hashes = generate_hashes(cmap)
                    
                    # Insert song info
                    cursor.execute("INSERT INTO songs (id, filepath, name) VALUES (?, ?, ?)", 
                                   (song_id, filepath, file))
                    
                    # Insert hashes (Batch insert for speed)
                    hash_records = [(h, int(offset), song_id) for h, offset in hashes]
                    cursor.executemany("INSERT INTO hashes (hash, offset, song_id) VALUES (?, ?, ?)", hash_records)
                    
                    song_id += 1
                except Exception as e:
                    print(f"Error processing {file}: {e}")

    # Create an index on the hash column for extremely fast lookups
    print("Building database index...")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_hash ON hashes(hash)")
    conn.commit()
    conn.close()
    print("Finished.")


# Build database
if __name__ == '__main__':
    build_database("./audio")
