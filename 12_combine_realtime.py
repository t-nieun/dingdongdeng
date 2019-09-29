import copy
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.signal import find_peaks
from scipy.fftpack import fft

# array내에 value값과 가장 가까운 값을 찾아주는 함수
def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()  # array 내의 값에서 value를 뺀 후 0에 가장 가까운 값을 value와 가장 가깝다고 판단
    return array[idx]


def find_nearest_idx(array, value):
    idx = (np.abs(array - value)).argmin()  # array 내의 값에서 value를 뺀 후 0에 가장 가까운 값을 value와 가장 가깝다고 판단
    return idx


# fre_key(인자)가 해당하는 계이름을 찾아주는 함수
def find_interval(fre_key):
    interval_Dict = {32.7032: 'C1', 34.6478: 'C#1', 36.7081: 'D1', 38.8909: 'D#1', 41.2034: 'E1', 43.6535: 'F1',
                     46.2493: 'F#1', 48.9994: 'G1', 51.9130: 'G#1', 55.0000: 'A1', 58.2705: 'A#1', 61.7354: 'B1',
                     65.4064: 'C2', 69.2957: 'C#2', 73.4162: 'D2', 77.7817: 'D#2', 82.4069: 'E2', 87.3071: 'F2',
                     92.4986: 'F#2', 97.9989: 'G2', 103.8262: 'G#2', 110.0000: 'A2', 116.5409: 'A#2', 123.4708: 'B2',
                     130.8128: 'C3', 138.5913: 'C#3', 146.8324: 'D3', 155.5635: 'D#3', 164.8138: 'E3', 174.6141: 'F3',
                     184.9972: 'F#3', 195.9977: 'G3', 207.6523: 'G#3', 220.0000: 'A3', 226.0819: 'A#3', 246.9417: 'B3',
                     261.6256: 'C4', 277.1826: 'C#4', 293.6648: 'D4', 311.1270: 'D#4', 329.6276: 'E4', 349.2282: 'F4',
                     369.9944: 'F#4', 391.9954: 'G4', 415.3047: 'G#4', 440.0000: 'A4', 466.1638: 'A#4', 493.8833: 'B4',
                     523.2511: 'C5', 554.3653: 'C#5', 587.3295: 'D5', 622.2540: 'D#5', 659.2551: 'E5', 698.4565: 'F5',
                     739.9888: 'F#5', 783.9909: 'G5', 830.6094: 'G#5', 880.0000: 'A5', 932.3275: 'A#5', 987.7666: 'B5',
                     1108.502: 'C6', 1108.731: 'C#6', 1174.659: 'D6', 1244.508: 'D#6', 1318.510: 'E6', 1396.913: 'F6',
                     1479.978: 'F#6', 1567.982: 'G6', 1661.219: 'G#6', 1760.000: 'A6', 1864.655: 'A#6', 1975.533: 'B6',
                     2093.005: 'C7', 2217.461: 'C#7', 2349.318: 'D7', 2489.016: 'D#7', 2637.020: 'E7', 2793.826: 'F7',
                     2959.955: 'F#7', 3135.963: 'G7', 3322.438: 'G#7', 3520.000: 'A7', 3729.310: 'A#7', 3951.066: 'B7',
                     4186.009: 'C8', 4434.922: 'C#8', 4698.636: 'D8', 4978.032: 'D#8', 5274.041: 'E8', 5587.652: 'F8',
                     5919.911: 'F#8', 6271.927: 'G8', 6644.875: 'G#8', 7040.000: 'A8', 7458.620: 'A#8', 7902.133: 'B8'}
    # A3    C6 1046
    return interval_Dict[fre_key]


# 양자화 해주는 함수
def scale(note):
    fre_array = np.array(
        [32.7032, 34.6478, 36.7081, 38.8909, 41.2034, 43.6535, 46.2493, 48.9994, 51.9130, 55.0000, 58.2705, 61.7354,
         65.4064, 69.2957, 73.4162, 77.7817, 82.4069, 87.3071, 92.4986, 97.9989, 103.8262, 110.0000, 116.5409, 123.4708,
         130.8128, 138.5913, 146.8324, 155.5635, 164.8138, 174.6141, 184.9972, 195.9977, 207.6523, 220.0000, 266.0819,
         246.9417, 261.6256, 277.1826, 293.6648, 311.1270, 329.6276, 349.2282, 369.9944, 391.9954, 415.3047, 440.0000,
         466.1638, 493.8833, 523.2511, 554.3653, 587.3295, 622.2540, 659.2551, 698.4565, 739.9888, 783.9909, 830.6094,
         880.0000, 932.3275, 987.7666, 1108.502, 1108.731, 1174.659, 1244.508, 1318.510, 1396.913, 1479.978, 1567.982,
         1661.219, 1760.000, 1864.655, 1975.533, 2093.005, 2217.461, 2349.318, 2489.016, 2637.020, 2793.826, 2959.955,
         3135.963, 3322.438, 3520.000, 3729.310, 3951.066, 4186.009, 4434.922, 4698.636, 4978.032, 5274.041, 5587.652,
         5919.911, 6271.927, 6644.875, 7040.000, 7458.620, 7902.133])
    sound_arr = []
    freq_arr = []
    freq_idx_arr = []
    for im in note:
        a = find_nearest(fre_array, im)
        freq_arr.append(a)
        b = find_nearest_idx(fre_array, im)
        freq_idx_arr.append(b)
        sound_arr.append(find_interval(a))
    return sound_arr, freq_arr, freq_idx_arr


# 배수로 감쇄하는 함수
# def multiple_freq_decrease(y_, origin_y_, peak_):
#     if len(peak_) == 0:
#         return -999
#     x, x_interval = np.linspace(0, 44100 / 2, 2048 / 2, retstep=True)  # x는 주파수 영역
#
#     for i in range(len(peak_) - 1):  # 모든 피크에 대해서
#         if x[peak_[0]] < 100:  # C1~B2
#             print('감지불가구간 : C1~B2')
#
#         elif x[peak_[0]] < 214:  # C3~A3
#             y_[peak_[0]] = y_[peak_[0]] * 3  # 저주파값 너무 낮아서 증폭시켜본것
#             for j in range(2, 6):  # 기준 피크로 부터 2배수는 자신의 1배만큼 감쇄하다가 뒤로갈수록 적게 감쇄 5배수경우 6/5배 감쇄
#                 if (peak_[i] * j + 1) < 512:
#                     y_[peak_[i] * j - 1] = (origin_y_[peak_[i] * j - 1]) - abs(origin_y_[peak_[i]] * (2 / j))
#                     y_[peak_[i] * j] = (origin_y_[peak_[i] * j]) - abs(origin_y_[peak_[i]] * (2 / j))
#                     y_[peak_[i] * j + 1] = (origin_y_[peak_[i] * j + 1]) - abs(origin_y_[peak_[i]] * (2 / j))
#                     y_[peak_[i] * j + 2] = (origin_y_[peak_[i] * j + 2]) - abs(origin_y_[peak_[i]] * (2 / j))
#
#         elif x[peak_[0]] < 391:  # B3~G4
#             for j in range(2, 6):  # 기준 피크로 부터 2배수는 자신의 4배만큼 감쇄하다가 뒤로갈수록 적게 감쇄 5배수경우 8/5배 감쇄
#                 if (peak_[i] * j + 1) < 512:
#                     y_[peak_[i] * j - 2] = (origin_y_[peak_[i] * j - 2]) - abs(origin_y_[peak_[i]] * (8 / j))
#                     y_[peak_[i] * j - 1] = (origin_y_[peak_[i] * j - 1]) - abs(origin_y_[peak_[i]] * (8 / j))
#                     y_[peak_[i] * j] = (origin_y_[peak_[i] * j]) - abs(origin_y_[peak_[i]] * (8 / j))
#                     y_[peak_[i] * j + 1] = (origin_y_[peak_[i] * j + 1]) - abs(origin_y_[peak_[i]] * (8 / j))
#                     y_[peak_[i] * j + 2] = (origin_y_[peak_[i] * j + 2]) - abs(origin_y_[peak_[i]] * (8 / j))
#                     y_[peak_[i] * j + 3] = (y_[peak_[i] * j + 3]) - abs(y_[peak_[i]] * (8 / j))
#                     y_[peak_[i] * j + 4] = (y_[peak_[i] * j + 4]) - abs(y_[peak_[i]] * (8 / j))
#
#         else:  # A4~C7
#             for j in range(2, 6):  # 기준 피크로 부터 4배수 까지 감쇄하는데 이때 감쇄하는 값의 양쪽 값과 자신을 감쇄
#                 if (peak_[i] * j + 1) < 512:
#                 # print('hell')
#                     y_[peak_[i] * j - 2] = (origin_y_[peak_[i] * j - 2]) - abs(origin_y_[peak_[i]] ** ((1) ** (j - 1)))
#                     y_[peak_[i] * j - 1] = (origin_y_[peak_[i] * j - 1]) - abs(origin_y_[peak_[i]] ** ((1) ** (j - 1)))
#                     y_[peak_[i] * j] = (origin_y_[peak_[i] * j]) - abs(origin_y_[peak_[i]] * ((1) ** (j - 1)))
#                     y_[peak_[i] * j + 1] = (origin_y_[peak_[i] * j + 1]) - abs(origin_y_[peak_[i]] * ((1) ** (j - 1)))
#                     y_[peak_[i] * j + 2] = (origin_y_[peak_[i] * j + 2]) - abs(origin_y_[peak_[i]] * ((1) ** (j - 1)))
#

# 실시간 그래프 출력을 위한 초기화 함수
def multiple_freq_decrease(y_, origin_y_, peak_):
    if len(peak_) == 0:
        return -999
    x, x_interval = np.linspace(0, 44100 / 2, 2048 / 2, retstep=True)  # x는 주파수 영역

    if len(peak_) == 0:
        return -999
    # print('peak :', peak_, 'x : ', peak_[0] * 21.533)
    for i in range(len(peak_) - 1):  # 모든 피크에 대해서
        if peak_[i] <= 7:  # D3
            y_[peak_[0]] = y_[peak_[0]] * 3  # 저주파값 너무 낮아서 증폭시켜본것
            for j in range(2, 6):  # 기준 피크로 부터 4배수 까지 감쇄하는데 이때 감쇄하는 값의 양쪽 값과 자신을 감쇄
                y_[peak_[i] * j - 1] = y_[peak_[i] * j - 1] - 5 * abs(y_[peak_[i]] * ((1 / 6) ** (j - 1)))
                y_[peak_[i] * j] = y_[peak_[i] * j] - 5 * abs(y_[peak_[i]] * ((1 / 6) * (j - 1)))
                y_[peak_[i] * j + 1] = y_[peak_[i] * j + 1] - 5 * abs(y_[peak_[i]] * ((1 / 6) ** (j - 1)))

        elif peak_[i] <= 9:  # G3
            y_[peak_[0]] = y_[peak_[0]] * 1.5  # 저주파값 너무 낮아서 증폭시켜본것
            for j in range(2, 6):  # 기준 피크로 부터 4배수 까지 감쇄하는데 이때 감쇄하는 값의 양쪽 값과 자신을 감쇄
                y_[peak_[i] * j - 1] = y_[peak_[i] * j - 1] - 5 * abs(origin_y_[peak_[i]] * ((1 / 4) ** (j - 1)))
                y_[peak_[i] * j] = y_[peak_[i] * j] - 5 * abs(origin_y_[peak_[i]] * ((1 / 4) * (j - 1)))
                y_[peak_[i] * j + 1] = y_[peak_[i] * j + 1] - 5 * abs(origin_y_[peak_[i]] * ((1 / 4) ** (j - 1)))

        elif peak_[i] <= 11:  # B3
            for j in range(2, 6):  # 기준 피크로 부터 4배수 까지 감쇄하는데 이때 감쇄하는 값의 양쪽 값과 자신을 감쇄
                y_[peak_[i] * j - 1] = y_[peak_[i] * j - 1] - 4 * abs(origin_y_[peak_[i]] * ((1 / 4) ** (j - 1)))
                y_[peak_[i] * j] = y_[peak_[i] * j] - 4 * abs(origin_y_[peak_[i]] * ((1 / 4) * (j - 1)))
                y_[peak_[i] * j + 1] = y_[peak_[i] * j + 1] - 4 * abs(origin_y_[peak_[i]] * ((1 / 4) ** (j - 1)))

        elif peak_[i] <= 15:  # D4
            for j in range(2, 6):  # 기준 피크로 부터 4배수 까지 감쇄하는데 이때 감쇄하는 값의 양쪽 값과 자신을 감쇄
                y_[peak_[i] * j - 1] = y_[peak_[i] * j - 1] - 3.5 * abs(origin_y_[peak_[i]] * ((1 / 6) ** (j - 1)))
                y_[peak_[i] * j] = y_[peak_[i] * j] - 3.5 * abs(origin_y_[peak_[i]] * ((1 / 6) * (j - 1)))
                y_[peak_[i] * j + 1] = y_[peak_[i] * j + 1] - 3.5 * abs(origin_y_[peak_[i]] * ((1 / 6) ** (j - 1)))

        elif peak_[i] <= 19:  # ~G4
            for j in range(2, 6):  # 기준 피크로 부터 4배수 까지 감쇄하는데 이때 감쇄하는 값의 양쪽 값과 자신을 감쇄
                y_[peak_[i] * j - 1] = y_[peak_[i] * j - 1] - abs(origin_y_[peak_[i]] * ((1 / 2) ** (j - 1)))
                y_[peak_[i] * j] = y_[peak_[i] * j] - abs(origin_y_[peak_[i]] * ((1 / 3) * (j - 1)))
                y_[peak_[i] * j + 1] = y_[peak_[i] * j + 1] - abs(origin_y_[peak_[i]] * ((1 / 2) ** (j - 1)))

        elif peak_[i] <= 36:  # E5
            y_[peak_[i]] = y_[peak_[i]] * 0.5  # 저주파값 너무 낮아서 증폭시켜본것
            for j in range(2, 6):  # 기준 피크로 부터 4배수 까지 감쇄하는데 이때 감쇄하는 값의 양쪽 값과 자신을 감쇄
                y_[peak_[i] * j - 1] = y_[peak_[i] * j - 1] - abs(origin_y_[peak_[i]] * ((1 / 16) ** (j - 1)))
                y_[peak_[i] * j] = y_[peak_[i] * j] - abs(origin_y_[peak_[i]] * ((1 / 16) * (j - 1)))
                y_[peak_[i] * j + 1] = y_[peak_[i] * j + 1] - abs(origin_y_[peak_[i]] * ((1 / 16) ** (j - 1)))

        else:
            for j in range(2, 6):  # 기준 피크로 부터 4배수 까지 감쇄하는데 이때 감쇄하는 값의 양쪽 값과 자신을 감쇄
                if (peak_[i] * j + 1) < 512:
                    y_[peak_[i] * j - 2] = y_[peak_[i] * j - 2] - abs(origin_y_[peak_[i]] * ((1 / 2) ** (j - 1)))
                    y_[peak_[i] * j - 1] = y_[peak_[i] * j - 1] - abs(origin_y_[peak_[i]] * ((1 / 2) ** (j - 1)))
                    y_[peak_[i] * j] = y_[peak_[i] * j] - abs(origin_y_[peak_[i]] * ((1 / 3) * (j - 1)))
                    y_[peak_[i] * j + 1] = y_[peak_[i] * j + 1] - abs(origin_y_[peak_[i]] * ((1 / 2) ** (j - 1)))
                    y_[peak_[i] * j + 2] = y_[peak_[i] * j + 2] - abs(origin_y_[peak_[i]] * ((1 / 2) ** (j - 1)))
    return y_
def init():
    line_origin.set_data([], [])
    line_decrease.set_data([], [])
    line_threshold.set_data([], [])
    line_peak.set_data([], [])
    return line_origin, line_decrease, line_threshold, line_peak

CHUNK = 2048
RATE = 44100
T = 1.0 / RATE
keep_rmse = 0
keep_keep_rmse = 0
keep_keep_keeep_rmse = 0
keep_gyename = 0
gye_name1 = 'n'
press_point_x_list = []
press_point_y_list = []
press_point_count = 0
rmse_list = []

before_origin_y = 0
before_decrease_y = 0
before_threshold = 0
before_peaks = []
after_peaks =[]
keep_peaks = 0
keep_peaks1 = 0
now_rmse_all_list = []
std_y = 0
peaks1 = 0
str_keep_peaks = []
str_keep_peaks1 = []
fig = plt.figure()
annotation2 = plt.annotate('after_peaks : %s' % str_keep_peaks1, xy=(11, 10), xytext=(4000, 2200000), size=10,
                             ha='right',
                             va='center')

def audio_read(i):
    global keep_rmse
    global keep_keep_rmse
    global keep_keep_keeep_rmse
    global keep_peaks
    global keep_peaks1
    global before_threshold
    global before_origin_y
    global before_decrease_y
    global after_peaks
    global std_y
    global peaks1
    global str_keep_peaks
    global str_keep_peaks1
    global gye_name1
    global annotation1
    global annotation2
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
    data = np.fromstring(stream.read(CHUNK), dtype=np.int16)  # 마이크에서 데이터를 읽어옴 (데이터 길이 1024)
    n = len(data)
    now_rmse = np.linalg.norm(data - 0) / np.sqrt(n)
    rmse_list.append(now_rmse)

    if now_rmse > 2000:  # 피아노 소리가 들리지 않을 때는 계산하지 않음 (들어온 데이터의 크기로 분석)

        n = len(data)
        x, x_interval = np.linspace(0, 44100 / 2, n / 2, retstep=True)  # x는 주파수 영역
        y = fft(data, n)  # 푸리에 변환

        y = np.absolute(y)
        y = y[range(int(n / 2))]
        origin_y = copy.copy(y)  # y값은 함수(..decrease)에 의해 변환되기 때문에 원래 y값을 미리 저장한다.

        # peak 값을 찾기 위한 임계점을 유동적으로 하기 위한 기준 잡기
        max_peak = 0
        std_peaks, _ = find_peaks(y, height=1500)  # 1500을 넘는 peak값을 찾는다. (max를 찾기 위한 표준 peak들)

        if len(std_peaks) > 0 and now_rmse > 4000:
            if keep_keep_rmse < keep_rmse and keep_rmse > now_rmse and keep_keep_keeep_rmse < keep_rmse and \
                    np.abs(keep_keep_keeep_rmse - keep_rmse) > 1000:
                std_y = np.ones(int(n / 2)) * before_threshold
                str_keep_peaks = str(scale(keep_peaks * x_interval)[0])
                print(keep_peaks1 * x_interval)
                str_keep_peaks1 = str(scale(keep_peaks1 * x_interval)[0])

                line_origin.set_data(x, before_origin_y)
                line_decrease.set_data(x, before_decrease_y)
                line_threshold.set_data(x, std_y)
                line_peak.set_data(after_peaks * x_interval, before_decrease_y[after_peaks])
                annotation2 = plt.annotate('Preditec Peak : %s' % str_keep_peaks1, xy=(11, 10), xytext=(4000, 6000000), size=10, ha='right', va='center')
                print(str_keep_peaks1)
                now_rmse_all_list.append(keep_rmse)

            max_peak = np.max(y[std_peaks])  # std_peaks에 있는 값들 중에서 가장 큰 값을 찾는다.
            if max_peak * 0.4 > 0.8 * 1e7:
                std_threshold = 1e7  # max_peak을 이용하여 임계값을 설정한다.
            else:
                std_threshold = max_peak * 0.4  # max_peak을 이용하여 임계값을 설정한다.
            peaks, _ = find_peaks(y, height=std_threshold)  # 임계값을 넘는 peak만 음으로 인식한다.
            keep_peaks = peaks
            gye_name = scale(peaks * x_interval)
            #  하모닉 음을 줄이기 위한 부분(치지 않은 음인데 친 음의 배수 라서 튄 계이름들)
            if not gye_name[0]:
                print('계이름이 없습니다.')
            else:
                multiple_freq_decrease(y, origin_y, peaks)

                peaks1, _ = find_peaks(y, height=std_threshold)
                keep_peaks1 = peaks1
                # print('peaks :  ', peaks1)

                gye_name1 = scale(peaks1 * x_interval)
                keep_gyename = gye_name1

            before_decrease_y = y
            before_origin_y = origin_y
            before_threshold = std_threshold
            before_peaks = copy.copy(peaks)
            after_peaks = copy.copy(peaks1)

        keep_keep_keeep_rmse = keep_keep_rmse
        keep_keep_rmse = keep_rmse
        keep_rmse = now_rmse

        stream.stop_stream()
        stream.close()
        p.terminate()

    return line_origin, line_decrease, line_threshold, line_peak, annotation2

plt.style.use('seaborn-pastel')

ax = plt.axes(xlim=(0, 4000), ylim=(0, 10000000))
line_origin, = ax.plot([], [], 'g--', label='Origin_Data')  # 감소 전 데이터 빨간색점선으로 표시
line_decrease, = ax.plot([], [], 'r', label='Decreased_Data')  # 감소 후 데이터 녹색으로 표시
line_threshold, = ax.plot([], [], 'b', label='Threshold')  # 쓰레스홀드
line_peak, = ax.plot([], [], "rx", label='one')  # 찾은 계이름 빨간색x로 표시

ax.legend()
ax.set_title('Pitch Prediction')
ax.set_xlabel('Freq(Hz)')
ax.set_ylabel('Amount')
ax.legend(('Origin_Data', 'Decreased_Data', 'Threshold', 'Peak'))

#실시간 그래프 애니메이션 출력
anim = animation.FuncAnimation(fig, audio_read, init_func=init, frames=1000, interval=1, blit=True)
plt.show()