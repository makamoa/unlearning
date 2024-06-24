import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram


def plot_time_domain(signal, sample_rate=1.0, title="Time Domain Signal"):
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(signal)) / sample_rate, signal)
    plt.title(title)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()


def plot_frequency_domain(signal, sample_rate=1.0, title="Frequency Domain Signal"):
    n = len(signal)
    freq = np.fft.fftfreq(n, d=1 / sample_rate)
    fft_signal = np.fft.fft(signal)

    plt.figure(figsize=(10, 4))
    plt.plot(freq[:n // 2], np.abs(fft_signal)[:n // 2])
    plt.title(title)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.show()


def plot_spectrogram(signal, sample_rate=1.0, title="Spectrogram"):
    f, t, Sxx = spectrogram(signal, fs=sample_rate)
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.title(title)
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.colorbar(label='Power/Frequency [dB/Hz]')
    plt.show()

def plot_time_domain_batch(signals, sample_rate=1.0, title="Time Domain Signals", max_figures_per_row=5):
    num_signals = len(signals)
    num_rows = (num_signals + max_figures_per_row - 1) // max_figures_per_row
    plt.figure(figsize=(20, 4 * num_rows))
    for i, signal in enumerate(signals):
        plt.subplot(num_rows, max_figures_per_row, i + 1)
        plt.plot(np.arange(len(signal)) / sample_rate, signal)
        plt.title(f"{title} - Signal {i+1}")
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_frequency_domain_batch(*all_signals, sample_rate=1.0, title="Frequency Domain Signals", max_figures_per_row=5):
    signals = all_signals[0]
    num_signals = len(signals)
    num_rows = (num_signals + max_figures_per_row - 1) // max_figures_per_row
    plt.figure(figsize=(20, 4 * num_rows))
    for j, signals in enumerate(all_signals):
        for i, signal in enumerate(signals):
            n = len(signal)
            freq = np.fft.fftfreq(n, d=1 / sample_rate)
            fft_signal = np.fft.fft(signal)
            plt.subplot(num_rows, max_figures_per_row, i + 1)
            plt.plot(freq[:n // 2], np.abs(fft_signal)[:n // 2], label=f"signal {j}")
            plt.title(f"{title} - Signal {i+1}")
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Magnitude')
            plt.grid(True)
            plt.legend()
    plt.tight_layout()
    plt.show()

def plot_spectrogram_batch(signals, sample_rate=1.0, title="Spectrograms", max_figures_per_row=5):
    num_signals = len(signals)
    num_rows = (num_signals + max_figures_per_row - 1) // max_figures_per_row
    plt.figure(figsize=(20, 4 * num_rows))
    for i, signal in enumerate(signals):
        f, t, Sxx = spectrogram(signal, fs=sample_rate)
        plt.subplot(num_rows, max_figures_per_row, i + 1)
        plt.pcolormesh(t, f, Sxx, shading='gouraud')
        plt.title(f"{title} - Signal {i+1}")
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        plt.colorbar(label='Power/Frequency [dB/Hz]')
    plt.tight_layout()
    plt.show()
