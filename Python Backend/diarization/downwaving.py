import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import Audio
import soundfile as sf

y, s = librosa.load('audio/example1.wav', sr=16000)
# y, s = librosa.load('speakers/S0-0.wav')
print(y, s)
sf.write('audio/example1.wav', y, s)

data, samplerate = sf.read('audio/example1.wav')
print(data, samplerate)


