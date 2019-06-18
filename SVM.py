import scipy.io.wavfile as wav
from scipy.fftpack import fft
import numpy as np
import matplotlib.pyplot as plt

rate, data = wav.read('Attack Clap 01.wav')
fft_out = fft(data)
plt.plot(data, np.abs(fft_out))
plt.show()

