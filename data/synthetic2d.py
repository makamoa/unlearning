import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

class TimeVaryingMultiSineWaveDataset(Dataset):
    def __init__(self, num_samples, seq_len, num_waves, freq_range=(1, 5), amp_range=(0.5, 1.5), phase_range=(0, 2 * np.pi)):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_waves = num_waves
        self.freq_range = freq_range
        self.amp_range = amp_range
        self.phase_range = phase_range
        self.data = self.generate_data()

    def generate_data(self):
        x = np.linspace(0, 4 * np.pi, self.seq_len)
        data = []
        for _ in range(self.num_samples):
            signal = np.zeros_like(x)
            for _ in range(self.num_waves):
                freq_start = np.random.uniform(*self.freq_range)
                freq_end = np.random.uniform(*self.freq_range)
                amp = np.random.uniform(*self.amp_range)
                phase = np.random.uniform(*self.phase_range)
                freq = np.linspace(freq_start, freq_end, self.seq_len)
                signal += amp * np.sin(freq * x + phase)
            data.append(signal)
        return np.array(data, dtype=np.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

class Sensor2DSignalDataset(Dataset):
    def __init__(self, num_samples, seq_len, num_sensors, signal_type='time_varying_multi_sine', num_waves=3, freq_range=(1, 5), amp_range=(0.5, 1.5), phase_range=(0, 2 * np.pi)):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_sensors = num_sensors
        self.signal_type = signal_type
        self.num_waves = num_waves
        self.freq_range = freq_range
        self.amp_range = amp_range
        self.phase_range = phase_range
        self.data = self.generate_data()

    def generate_data(self):
        dataset = TimeVaryingMultiSineWaveDataset(self.num_samples, self.seq_len, self.num_waves, self.freq_range, self.amp_range, self.phase_range)
        base_signals = [dataset[i] for i in range(self.num_samples)]
        data = []
        for base_signal in base_signals:
            sensor_signals = []
            for sensor in range(self.num_sensors):
                phase_shift = np.random.uniform(*self.phase_range)
                sensor_signal = base_signal * np.cos(phase_shift) + base_signal * np.sin(phase_shift)
                sensor_signals.append(sensor_signal)
            data.append(np.array(sensor_signals))
        return np.array(data, dtype=np.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

def plot_2d_time_domain(signal, sample_rate=1.0, title="2D Time Domain Signal"):
    plt.figure(figsize=(10, 4))
    for sensor_signal in signal:
        plt.plot(np.arange(len(sensor_signal)) / sample_rate, sensor_signal)
    plt.title(title)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # Parameters
    sample_rate = 100  # Example sample rate for plotting
    num_samples = 1000
    seq_len = 1000
    num_sensors = 5  # Number of sensors
    num_waves = 3  # Number of sine waves to combine

    # Generate example signals
    sensor_2d_dataset = Sensor2DSignalDataset(num_samples, seq_len, num_sensors, num_waves=num_waves)

    sensor_2d_signal = sensor_2d_dataset[0]

    # Plot 2D time-domain signals
    plot_2d_time_domain(sensor_2d_signal, sample_rate, title="2D Time Domain Sensor Signals")