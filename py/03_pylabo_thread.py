import pyaudio
import wave
import numpy as np
import winsound
import threading

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 1

WAVE_OUTPUT_FILENAME = "output.wav"
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

frames = []

def audioread():
    while 1:
        frames = []
        print("* recording")
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        print("* done recording")
        #stream.stop_stream()
        #stream.close()
        #p.terminate()
        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

def audiowrite():
    while 1:
        winsound.PlaySound("output.wav", 0)

t = threading.Thread(target=audioread)
tt = threading.Thread(target=audiowrite)

t.start()
tt.start()
print("Main Thread")
