import pyaudio
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks
import cv2
CHUNK = 2048
FORMAT = pyaudio.paInt16
RATE = 44100
sr = RATE
CHANNELS = 11
T = 1 / RATE
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output16.wav"

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

frames = []

y, sr = librosa.load(librosa.util.example_audio_file())
print(y.shape)
print(y.dtype)
print(sr)


print("* recording")
for i in range(200):
    data = np.fromstring(stream.read(CHUNK), dtype=np.int16)
    data = np.asarray(data, dtype=np.float32)
    frames.append(data)

    n = len(data)
    now_rmse = np.linalg.norm(data - 0) / np.sqrt(n)
    std_peaks, _ = find_peaks(data, height=1000)  # 1500을 넘는 peak값을 찾는다. (max를 찾기 위한 표준 peak들)

    if now_rmse > 1000:  # 피아노 소리가 들리지 않을 때는 계산하지 않음 (들어온 데이터의 크기로 분석)
        if len(std_peaks) > 0 and now_rmse > 1000:

                ## CQT(Constant-Q Transform)
                # C = np.abs(librosa.cqt(data, sr=sr))
                # print(C.shape)
                # print(C.dtype)
                # librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max), sr=sr, x_axis='time', y_axis='cqt_note')
                # plt.colorbar(format='%+2.0f dB')
                # plt.title('Constant-Q power spectrum')
                # plt.tight_layout()
                # plt.show()

                ## MFCCs(Mel-frequency cepstral coefficients)
                # S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
                # mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
                # plt.figure(figsize=(10, 4))
                # librosa.display.specshow(mfccs, x_axis='time')
                # plt.colorbar()
                # plt.title('MFCC')
                # plt.tight_layout()
                # plt.show()

                ## 스펙트로그램으로 분석
                frequencies, times, spectrogram = signal.spectrogram(data, 44100, nfft=1024, nperseg=1024)
                plt.pcolormesh(times, frequencies, spectrogram)
                int_spectrogram = np.asarray(spectrogram, dtype=np.int32)

                idx = np.argmax(int_spectrogram, axis=0)
                print(frequencies[idx])
                plt.ylabel('Frequency [Hz]')
                plt.xlabel('Time [sec]')
                plt.ylim([0, 1600])
                plt.show()

frames = np.asarray(frames, dtype=np.float32)
frames = np.reshape(frames, frames.shape[0] * frames.shape[1])

frequencies, times, spectrogram = signal.spectrogram(frames, 44100, nfft=1024, nperseg=1024)
plt.pcolormesh(times, frequencies, spectrogram)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.ylim([0, 1600])
plt.show()


print("* done")
