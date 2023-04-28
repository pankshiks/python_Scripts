import pandas as pd
import numpy as np
import os
os.environ['LIBROSA_CACHE_DIR'] = '/tmp/librosa_cache'
import librosa

# Define a function to extract features from a song
def extract_features(song_path):
    y, sr = librosa.load(song_path) 
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)  
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)  
    rmse = librosa.feature.rms(y=y)  
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)  
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)  
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr) 
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)  
    mfcc = librosa.feature.mfcc(y=y, sr=sr) 
    return [tempo, chroma_stft.mean(), rmse.mean(), spectral_centroid.mean(),
            spectral_bandwidth.mean(), spectral_contrast.mean(), spectral_rolloff.mean()] + mfcc.mean(axis=1).tolist()

# Define a function to process a directory of songs and create a CSV file
def create_csv(songs_dir, csv_path):
    features = []
    for song_name in os.listdir(songs_dir):
        print(song_name)
        if song_name.endswith(".mp3") or song_name.endswith(".wav"):
            song_path = os.path.join(songs_dir, song_name)
            y, sr = librosa.load(song_path, duration=200) 
            # extract the features
            # chroma_new = librosa.feature.chroma_stft(y=y, sr=sr)
            # mfccs_new = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=1)
            # zcr_new = librosa.feature.zero_crossing_rate(y)

            mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=1))
            chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=1))
            zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
            print(song_name, mfccs, chroma, zcr)

            # store the features in a dictionary
            song_features = {
            'song': song_name,
            'mfccs_new': mfccs,
            'chroma_new': chroma,
            # 'bandwidth_new': bandwidth_new.mean(),
            # 'contrast_new': contrast_new.mean(),
            # 'rolloff_new': rolloff_new.mean(),
            'zcr_new': zcr
            }

            # append the dictionary to the features list
            features.append(song_features)
            # song_features = extract_features(song_path)
            # song_features.insert(0, song_name)    
            # features.append(song_features)
    df = pd.DataFrame(features)

    # df = pd.DataFrame(features, columns=["song", "chroma",] + [f"mfcc{i}" for i in range(1, 21)])
    df.to_csv(csv_path, index=False)

# Usage example
create_csv("songs/", "songs.csv")
