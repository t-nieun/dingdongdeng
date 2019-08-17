import matplotlib.pyplot as plt
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks
import numpy as np
import librosa

x = electrocardiogram()[2000:4000]

print(x)
print(x.shape)
peaks, _ = find_peaks(x, height=100)
plt.plot(x)
plt.plot(peaks, x[peaks], "x")
plt.plot(np.zeros_like(x), "--", color="gray")
plt.show()