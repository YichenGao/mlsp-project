import sqlite3
import librosa
from collections import Counter
from build_db import get_constellation_map, generate_hashes, SR


def query_database(audio_path, offset_sec, duration_sec, k=3, db_path="fingerprints.db"):
    print(f"Extracting query snippet: {offset_sec}s to {offset_sec + duration_sec}s...")
    
    # Load audio snippet
    y, _ = librosa.load(audio_path, sr=SR, offset=offset_sec, duration=duration_sec)
    
    # Get constellation map
    cmap = get_constellation_map(audio_path) # We process the snippet directly
    # Adjust timeframes
    query_hashes = generate_hashes(cmap)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Matches structure: {song_id: [list of (db_offset - query_offset)]}
    matches = {}
    
    print(f"Searching database with {len(query_hashes)} query hashes...")
    for hash_val, query_offset in query_hashes:
        # Look up the inverted list for this specific hash
        cursor.execute("SELECT song_id, offset FROM hashes WHERE hash=?", (hash_val,))
        results = cursor.fetchall()
        
        for song_id, db_offset in results:
            if song_id not in matches:
                matches[song_id] = []

            if isinstance(db_offset, bytes):
                db_offset = struct.unpack('<q', db_offset)[0]
            
            # Calculate the time shift / delta
            delta = int(db_offset) - int(query_offset)
            matches[song_id].append(delta)
            
    # Score the matches
    scores = []
    for song_id, offsets in matches.items():
        # The true match will have many identical (db_offset - query_offset) values
        # representing the continuous alignment of the transparent plastic over the strip chart
        if offsets:
            most_common_offset, count = Counter(offsets).most_common(1)[0]
            scores.append((count, song_id))
            
    # Sort by the highest number of geometrically aligned hash matches
    scores.sort(reverse=True)
    
    print(f"\n--- Top {k} Matches ---")
    for i in range(min(k, len(scores))):
        match_count, song_id = scores[i]
        cursor.execute("SELECT name FROM songs WHERE id=?", (song_id,))
        song_name = cursor.fetchone()[0]
        print(f"{i+1}. {song_name} (Score: {match_count} aligned points)")

    conn.close()

    
if __name__ == "__main__":
    TEST_FILE = "./audio/miscellaneous/all_star.mp3"
    query_database(TEST_FILE, offset_sec=45.0, duration_sec=10.0, k=5)
