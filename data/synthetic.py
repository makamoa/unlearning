from skimage.util import noise
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle

class SyntheticDataset(Dataset):
    def __init__(self,  *signals, time_mask=False, frequency_mask=False, noise=False, num_masks=2, mask_size=128, preprocess=True):
        self.signals = signals
        self.seq_len = signals[0].seq_len
        self.num_masks = num_masks
        self.mask_size = mask_size
        self.frequency_mask = frequency_mask
        self.noise = noise
        self.time_mask = time_mask
        self.input, self.output = self.stack_signals()

    def save(self,filename):
        with open(filename,'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        return data

    def stack_signals(self):
        data = []
        processed = []
        for i, signal in enumerate(self.signals):
            assert signal.seq_len == self.seq_len, f"Signal {signal.__class__.__name__} has a different seq_len"
            processed_ = signal.data.copy()
            if self.noise:
                processed_ = self.add_noise(processed_)
            if self.frequency_mask:
                processed_ = self.apply_frequency_mask(processed_)
            if self.time_mask:
                processed_ = self.apply_time_mask(processed_)
            data.append(signal.data)
            processed.append(processed_)
        return np.concatenate(processed, axis=0), np.concatenate(data, axis=0)

    def add_noise(self, signal):
        return signal

    def apply_time_mask(self, signal):
        masked_signal = signal.copy()
        for i in range(masked_signal.shape[0]):  # Iterate over each sample in the batch
            for _ in range(self.num_masks):
                start_idx = np.random.randint(0, self.seq_len - self.mask_size + 1)
                masked_signal[i, start_idx:start_idx + self.mask_size] = 0
        return masked_signal

    def apply_frequency_mask(self, signal):
        masked_signal = signal.copy()
        for i in range(masked_signal.shape[0]):  # Iterate over each sample in the batch
            freq_signal = np.fft.fft(masked_signal[i])
            for _ in range(self.num_masks):
                start_idx = np.random.randint(0, len(freq_signal) - self.mask_size + 1)
                freq_signal[start_idx:start_idx + self.mask_size] = 0
            masked_signal[i] = np.fft.ifft(
                freq_signal).real  # Taking the real part as the inverse FFT may result in small imaginary parts
        return masked_signal

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return self.input[idx][...,None], self.output[idx][...,None]

class SineWaveDataset(Dataset):
    def __init__(self, num_samples, seq_len, freq_range=(1, 5), amp_range=(0.5, 1.5), phase_range=(0, 2 * np.pi)):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.freq_range = freq_range
        self.amp_range = amp_range
        self.phase_range = phase_range
        self.data = self.generate_data()

    def generate_data(self):
        x = np.linspace(0, 4 * np.pi, self.seq_len)
        data = []
        for _ in range(self.num_samples):
            freq = np.random.uniform(*self.freq_range)
            amp = np.random.uniform(*self.amp_range)
            phase = np.random.uniform(*self.phase_range)
            signal = amp * np.sin(freq * x + phase)
            data.append(signal)
        return np.array(data, dtype=np.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

class MultiSineWaveDataset(Dataset):
    def __init__(self, num_samples, seq_len, num_waves=1, freq_range=(1, 500), amp_range=(0.5, 1.5), phase_range=(0, 2 * np.pi)):
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
                freq = np.random.uniform(*self.freq_range)
                amp = np.random.uniform(*self.amp_range)
                phase = np.random.uniform(*self.phase_range)
                signal += amp * np.sin(freq * x + phase)
            data.append(signal)
        return np.array(data, dtype=np.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

class EquispacedFreqMultiSineWaveDataset(Dataset):
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
            freqs = np.linspace(self.freq_range[0], self.freq_range[1], self.num_waves)
            for freq in freqs:
                amp = np.random.uniform(*self.amp_range)
                phase = np.random.uniform(*self.phase_range)
                signal += amp * np.sin(freq * x + phase)
            data.append(signal)
        return np.array(data, dtype=np.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

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

class SquareWaveDataset(Dataset):
    def __init__(self, num_samples, seq_len, freq_range=(1, 5), amp_range=(0.5, 1.5), duty_cycle_range=(0.4, 0.6)):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.freq_range = freq_range
        self.amp_range = amp_range
        self.duty_cycle_range = duty_cycle_range
        self.data = self.generate_data()

    def generate_data(self):
        x = np.linspace(0, 4 * np.pi, self.seq_len)
        data = []
        for _ in range(self.num_samples):
            freq = np.random.uniform(*self.freq_range)
            amp = np.random.uniform(*self.amp_range)
            duty_cycle = np.random.uniform(*self.duty_cycle_range)
            signal = amp * (np.sign(np.sin(freq * x)) * 0.5 + 0.5 * duty_cycle)
            data.append(signal)
        return np.array(data, dtype=np.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

class SawtoothWaveDataset(Dataset):
    def __init__(self, num_samples, seq_len, freq_range=(1, 5), amp_range=(0.5, 1.5)):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.freq_range = freq_range
        self.amp_range = amp_range
        self.data = self.generate_data()

    def generate_data(self):
        x = np.linspace(0, 4 * np.pi, self.seq_len)
        data = []
        for _ in range(self.num_samples):
            freq = np.random.uniform(*self.freq_range)
            amp = np.random.uniform(*self.amp_range)
            signal = amp * (2 * (x * freq / (2 * np.pi) % 1) - 1)
            data.append(signal)
        return np.array(data, dtype=np.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

class TimeVaryingSineWaveDataset(Dataset):
    def __init__(self, num_samples, seq_len, freq_range=(1, 5), amp_range=(0.5, 1.5), phase_range=(0, 2 * np.pi)):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.freq_range = freq_range
        self.amp_range = amp_range
        self.phase_range = phase_range
        self.data = self.generate_data()

    def generate_data(self):
        x = np.linspace(0, 4 * np.pi, self.seq_len)
        data = []
        for _ in range(self.num_samples):
            freq_start = np.random.uniform(*self.freq_range)
            freq_end = np.random.uniform(*self.freq_range)
            amp = np.random.uniform(*self.amp_range)
            phase = np.random.uniform(*self.phase_range)
            freq = np.linspace(freq_start, freq_end, self.seq_len)
            signal = amp * np.sin(freq * x + phase)
            data.append(signal)
        return np.array(data, dtype=np.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

class TimeVaryingSquareWaveDataset(Dataset):
    def __init__(self, num_samples, seq_len, freq_range=(1, 5), amp_range=(0.5, 1.5), duty_cycle_range=(0.4, 0.6)):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.freq_range = freq_range
        self.amp_range = amp_range
        self.duty_cycle_range = duty_cycle_range
        self.data = self.generate_data()

    def generate_data(self):
        x = np.linspace(0, 4 * np.pi, self.seq_len)
        data = []
        for _ in range(self.num_samples):
            freq_start = np.random.uniform(*self.freq_range)
            freq_end = np.random.uniform(*self.freq_range)
            amp = np.random.uniform(*self.amp_range)
            duty_cycle = np.random.uniform(*self.duty_cycle_range)
            freq = np.linspace(freq_start, freq_end, self.seq_len)
            signal = amp * (np.sign(np.sin(freq * x)) * 0.5 + 0.5 * duty_cycle)
            data.append(signal)
        return np.array(data, dtype=np.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

class TimeVaryingSawtoothWaveDataset(Dataset):
    def __init__(self, num_samples, seq_len, freq_range=(1, 5), amp_range=(0.5, 1.5)):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.freq_range = freq_range
        self.amp_range = amp_range
        self.data = self.generate_data()

    def generate_data(self):
        x = np.linspace(0, 4 * np.pi, self.seq_len)
        data = []
        for _ in range(self.num_samples):
            freq_start = np.random.uniform(*self.freq_range)
            freq_end = np.random.uniform(*self.freq_range)
            amp = np.random.uniform(*self.amp_range)
            freq = np.linspace(freq_start, freq_end, self.seq_len)
            signal = amp * (2 * (x * freq / (2 * np.pi) % 1) - 1)
            data.append(signal)
        return np.array(data, dtype=np.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

class GaussianWaveDataset(Dataset):
    def __init__(self, num_samples, seq_len, freq_range=(1, 5), amp_range=(0.5, 1.5), sigma_range=(0.1, 1.0)):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.freq_range = freq_range
        self.amp_range = amp_range
        self.sigma_range = sigma_range
        self.data = self.generate_data()

    def generate_data(self):
        x = np.linspace(0, 4 * np.pi, self.seq_len)
        data = []
        for _ in range(self.num_samples):
            freq = np.random.uniform(*self.freq_range)
            amp = np.random.uniform(*self.amp_range)
            sigma = np.random.uniform(*self.sigma_range)
            signal = amp * np.exp(-0.5 * ((x - np.pi) / sigma) ** 2) * np.sin(freq * x)
            data.append(signal)
        return np.array(data, dtype=np.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

class TimeVaryingGaussianWaveDataset(Dataset):
    def __init__(self, num_samples, seq_len, freq_range=(1, 5), amp_range=(0.5, 1.5), sigma_range=(0.1, 1.0)):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.freq_range = freq_range
        self.amp_range = amp_range
        self.sigma_range = sigma_range
        self.data = self.generate_data()

    def generate_data(self):
        x = np.linspace(0, 4 * np.pi, self.seq_len)
        data = []
        for _ in range(self.num_samples):
            freq_start = np.random.uniform(*self.freq_range)
            freq_end = np.random.uniform(*self.freq_range)
            amp = np.random.uniform(*self.amp_range)
            sigma_start = np.random.uniform(*self.sigma_range)
            sigma_end = np.random.uniform(*self.sigma_range)
            freq = np.linspace(freq_start, freq_end, self.seq_len)
            sigma = np.linspace(sigma_start, sigma_end, self.seq_len)
            signal = amp * np.exp(-0.5 * ((x - np.pi) / sigma) ** 2) * np.sin(freq * x)
            data.append(signal)
        return np.array(data, dtype=np.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]


class TimeVaryingEquispacedFreqMultiSineWaveDataset(Dataset):
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
            freqs_start = np.linspace(self.freq_range[0], self.freq_range[1], self.num_waves)
            freqs_end = np.linspace(self.freq_range[0], self.freq_range[1], self.num_waves)
            for freq_start, freq_end in zip(freqs_start, freqs_end):
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

if __name__ == '__main__':
    # Parameters
    num_samples = 1000
    seq_len = 100
    batch_size = 32

    # Create datasets
    sine_dataset = SineWaveDataset(num_samples, seq_len)
    square_dataset = SquareWaveDataset(num_samples, seq_len)
    sawtooth_dataset = SawtoothWaveDataset(num_samples, seq_len)

    # Create data loaders
    sine_loader = DataLoader(sine_dataset, batch_size=batch_size, shuffle=True)
    square_loader = DataLoader(square_dataset, batch_size=batch_size, shuffle=True)
    sawtooth_loader = DataLoader(sawtooth_dataset, batch_size=batch_size, shuffle=True)

    # Example usage: Iterate through data loaders
    for batch in sine_loader:
        print(batch)
        break

    for batch in square_loader:
        print(batch)
        break

    for batch in sawtooth_loader:
        print(batch)
        break
