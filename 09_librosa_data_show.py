from ctypes.wintypes import RECT

import pyaudio
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

CHUNK = 1024
FORMAT = pyaudio.paInt16
RATE = 44100
sr = RATE
CHANNELS = 1
T = 1 / RATE
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output16.wav"

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

frames = []
print("* recording")
for i in range(int(RATE/CHUNK)*RECORD_SECONDS):
    y = np.fromstring(stream.read(CHUNK), dtype=np.int16)
    frames.append(y)


frames = np.asarray(frames, dtype=np.float32)
frames = np.reshape(frames, frames.shape[0] * frames.shape[1])

print(frames.shape)
print(frames.dtype)


# D = np.abs(librosa.stft(frames))
# librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), y_axis='linear', x_axis='time')
# plt.show()

# S = librosa.feature.melspectrogram(y=frames, sr=44100, n_mels=128)
# librosa.display.specshow(librosa.power_to_db(S, ref=np.max), sr=sr, y_axis='mel', x_axis='time')
# plt.show()

frequencies, times, spectrogram = signal.spectrogram(frames, 44100, nfft=10240, nperseg=1024)
plt.pcolormesh(times, frequencies, spectrogram)
print(frequencies.shape)
print(frequencies[1:100])
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.ylim([0,800])
plt.show()

print("* done")

