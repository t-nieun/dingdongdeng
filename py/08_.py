import pyaudio
import numpy as np

CHUNK = 1024
RATE = 44100
N = 1024
T = 1.0/44100
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)
data2 = []
i = 0
data = stream.read(CHUNK)
#data = np.fromstring(stream.read(CHUNK), dtype=np.int16)

print(data)