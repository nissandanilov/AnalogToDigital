import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.fftpack import fft

class AnalogToDigital():
    def __init__(self, filename):
        self.filename = filename
        self.data = None
        self.df = None

    def load_data(self):
        self.data = np.fromfile(self.filename, dtype=np.uint16)
        self.data = np.reshape(self.data, (8, -1), order='F')
        self.data = np.multiply(np.exp(-74.12), (self.data - np.float_power(2, 15 - 1)))

    def df_convert(self):
        self.load_data()
        self.df = pd.DataFrame(self.data.T)

    def apply_band_pass_filter_and_plot(self, low_f, high_f):
        Fs = 4000  # samples/sec
        N = len(self.df)  # number of samples
        T = 1.0 / Fs  # Time interval

        FILTER_ORDER = 5

        for ch_idx in self.df.columns:
          y = np.array(self.df[ch_idx])

          b, a = butter(FILTER_ORDER, [low_f, high_f], fs=Fs, btype='band')
          y_filtered = filtfilt(b, a, self.df[ch_idx], padlen=15, axis=0)
          yfft = fft(y)
          yfft_filtered = fft(y_filtered)
          xfft = np.linspace(0.0, 1.0/(2.0*T), N//2)

          plt.figure(figsize=(15,12))
          plt.plot(xfft, 2.0/N * np.abs(yfft[:N//2]), label=f"ch{ch_idx}", linewidth=0.1)
          plt.plot(xfft, 2.0/N * np.abs(yfft_filtered[:N//2]), label=f"ch{ch_idx} filtered", linewidth=0.1)
          plt.xlabel("Frequency [Hz]")
          plt.ylabel("Amplitude")
          plt.legend()
          plt.yscale('log')
          plt.show()


#   Example
# A2D = AnalogToDigital('file_path')
# A2D.load_data()
# A2D.df_convert()
# A2D.apply_band_pass_filter_and_plot(250,500)
