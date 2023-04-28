import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import librosa
from sklearn.tree import DecisionTreeClassifier

# Load the data
data = pd.read_csv('songs.csv')

def predict_mood(features, df):
    Xt = df.iloc[:, 1:4].values
    yt = df.iloc[:, -3:].values

    clf = DecisionTreeClassifier()
    clf.fit(Xt, yt)

    mood = clf.predict([features])[0]
    return np.array(mood)

song_file = 'songs/song_30.wav'
clip_start = 20 
clip_end = 40  
y, sr = librosa.load(song_file, duration=200)


mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=1), axis=1)
chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=1), axis=1)
zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))

features = np.concatenate([mfccs, chroma, [zcr]])
print(features)
mood = predict_mood(features, data)
print('########################\n')
print( "Mood is {}, Genre is {}, Instruments is {}".format(mood[0], mood[1], mood[2]))
print('\n########################')
