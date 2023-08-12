import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter
from scipy import signal


class AnalogToDigital():
    def __init__(self, filename):
        self.filename = filename
        self.data = None


    def load_data(self):
        self.data = np.fromfile(self.filename, dtype=np.uint16)
        self.data = np.reshape(self.data, (8, -1), order='F')
        self.data = np.multiply(np.exp(-74.12), (self.data - np.float_power(2, 15 - 1)))

    def df_convert(self):
        self.load_data()
        self.df = pd.DataFrame(self.data.T)

    def apply_band_pass_filter_and_plot(self, low_f, high_f):
        HEAD = 7000
        # HEAD = len(self.df)
        b, a = butter(2, [low_f, high_f], fs=4000, btype='band')
        np_filtered = signal.filtfilt(b, a, self.df, padlen=15, axis=0)


        for ch_idx in range(np_filtered.shape[1]):
          s = self.df[ch_idx]
          s_filtered = np_filtered[:,ch_idx]

          plt.figure(figsize=(15,7))
          plt.title(f"Plot of the first {HEAD} samples")
          plt.plot(list(range(len(s[:HEAD]))), s[:HEAD], linestyle='dashed', linewidth=1, label=f'ch{ch_idx}')
          plt.plot(list(range(len(s_filtered[:HEAD]))), s_filtered[:HEAD], label=f"y{ch_idx}")
          plt.legend()
          plt.show()

a2d = AnalogToDigital("C:\\Users\\nissa\Desktop\\X-trodes\\NEUR0000.DT8")
a2d.load_data()
a2d.df_convert()

a2d.apply_band_pass_filter_and_plot(low_f=100, high_f=500)

a2d.apply_band_pass_filter_and_plot(low_f=10, high_f=1500)

