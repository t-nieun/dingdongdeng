import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

def find(array_, std):
    result = []
    for x in array_:
        if x > std:
            i = x
            result.append(i)
    return result

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]

CHUNK = 2 ** 10
RATE = 44100
N = 1024
T = 1.0 / 44100.0
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True,
                frames_per_buffer=CHUNK)
data2 = []  # 1000이 넘는 peak값들
i = 0
data = np.fromstring(stream.read(CHUNK), dtype=np.int16)
n = len(data)

ind_y = np.where(y_peak == y)[0]
x = np.linspace(0, 44100 / 2, n // 2)
y = np.fft.fft(data) / n
y = np.absolute(y)
y = y[range(int(n / 2))]
y_peak = find(y, 1200)
#print(y_peak)
print(ind_y)
frequency_ = x[ind_y]
#print("계이름은 : ", frequency_)
# ind_x = find_nearest(x, y_max)
# print(ind_y)
# print(x(ind_x))
# plt.plot(x, y)
# plt.xlim(0, 3000)
# plt.show()


stream.stop_stream()
print('빠져나옴')
stream.close()
p.terminate()