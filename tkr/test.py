from imports import *
from spectrograms import *
from hashing import *

# Load audio
song_path = 'audio/miscellaneous/all_star.mp3'
offset, duration = 0.0, 120.0 # Offset, duration (seconds)
s, rate_ = librosa.load(song_path, offset=offset, duration=duration) # Signal, rate

# Resample audio signal
rate = 8000.0 # New sampling rate
s = resample(s, rate_in=rate_, rate_out=rate) # Resampling

# Compute spectrogram
time, freq, spec = make_spectrogram(s, rate=rate, norm=True)
c = 1e-4
spec = 10.0 * np.log10(spec + c)
show_spectrogram(time, freq, spec)

# Compute constellation map
d_time, d_freq = 0.5, 100. # seconds, hertz
threshold = -20.0 # dB
constellation = make_constellation(spec, d_time=d_time, d_freq=d_freq, threshold=threshold)
show_spectrogram(time, freq, constellation, colormap='gray', title='constellation', zlabel='value')

# Now, make a hash list
coordinates = get_coordinates(constellation, order='(time, frequency)')
dt, df = 100, 100
triplets = make_triplets(coordinates, dt=dt, df=df)
hash_list = make_hash_list(triplets)



# Query
offset, duration = 50.0, 10.0 # Offset, duration (seconds)
q, rate_ = librosa.load(song_path, offset=offset, duration=duration) # Signal, rate
q = resample(q, rate_in=rate_, rate_out=rate) # Resample

# Compute spectrogram
time, freq, spec = make_spectrogram(q, rate=rate, norm=True)
spec = 10.0 * np.log10(spec + c)
show_spectrogram(time, freq, spec, colormap='viridis')

# Compute constellation map and query
constellation = make_constellation(spec, d_time=d_time, d_freq=d_freq, threshold=threshold)
show_spectrogram(time, freq, constellation, colormap='gray', title='constellation', zlabel='value')

# Make a hash list
coordinates = get_coordinates(constellation, order='(time, frequency)')
triplets = make_triplets(coordinates, dt=dt, df=df)
query = make_hash_list(triplets)



# Try to match
matching_function = make_matching_function(hash_list, query)
match_indices, match_score = get_match(matching_function)
print(f'match_indices = {match_indices}, match_score = {match_score}')

plt.figure()
plt.plot(matching_function)
plt.grid()
plt.show()
