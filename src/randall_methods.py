import glob

import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.fft
from tqdm import tqdm
from scipy.fft import fft, ifft
from kurtogram import fast_kurtogram
import utils as ut
import data_utils as du


def randall_method_1(signal, fs):
    '''
    Simply the envelope analysis (squared envelope spectrum) of the full
    bandwidth raw signal. The justification is that there are no obvious
    masking sources in this case, and in fact many bearing faults were
    found to be easily diagnosable with this method. For such cases it
    means that the signals are not very demanding in terms of diagnostic
    power, and do not provide a good test for newly proposed algorithms
    (unless for specific purposes, such as the determination of fault size).

    inputs:
        signal: Sampled signal
        fs:     Sampling frequency
    outputs:
        f:      Squared envelope spectrum frequency
        X:      Squared envelope spectrum amplitude
    '''
    analytic_signal = scipy.signal.hilbert(signal)
    envelope = np.abs(analytic_signal)
    squared_envelope = envelope ** 2
    f, X = ut.oneside_fft(squared_envelope, fs)

    return f, X


def randall_method_2(signal, fs):
    '''
    Implemented as described in Signal Pre-whitening Using Cepstrum Editing (Liftering) to
    Enhance Fault Detection in Rolling Element Bearings
    '''

    t = np.linspace(0, len(signal)/fs, len(signal))
    X = scipy.fft.fft(signal)

    phase = np.angle(X)

    log_amplitude = np.log(np.abs(X) + 1e-10);

    real_cepstrum = scipy.fft.ifft(log_amplitude)

    edited_cepstrum = np.zeros_like(real_cepstrum)
    edited_cepstrum[0] = real_cepstrum[0]

    edited_log_amplitude_spectrum = scipy.fft.fft(edited_cepstrum)

    edited_log_spectrum = edited_log_amplitude_spectrum + 1j * phase

    X = np.exp(edited_log_spectrum)
    time_signal = np.real(scipy.fft.ifft(X))

  

    return t, time_signal

def DRS(signal, N, Delta):
    '''
    Step 1 of Randall's method 3
    Discrete/random separation (DRS) to remove deterministic (discrete
    frequency) components.

    Implemented as described in
    Unsupervised noise cancellation for vibration signals:
    part II—a novel frequency-domain algorithm
    '''
    # nom and denom of H
    nominator = np.zeros(N, dtype=complex)
    denominator = np.zeros(N, dtype=complex)
    for k in tqdm(range(2*N+Delta, len(signal), 100), desc="Building DRS filter"):
        window_idx = np.arange(k-N, k)
        delayed_window_idx = np.arange(k-2*N-Delta, k-N-Delta)

        # N length window of signal
        x = signal[window_idx]*scipy.signal.windows.parzen(N)
        # N length time delayed window of signal
        x_d = signal[delayed_window_idx]*scipy.signal.windows.parzen(N)

        X_k_f = scipy.fft.fft(x)
        X_d_k_f = scipy.fft.fft(x_d)

        #  Cross-power spectrum between sequence and delayed sequence
        nominator += X_d_k_f * X_k_f.conj()
        # Power spectrum of delayed sequence
        denominator += X_d_k_f * X_d_k_f.conj()

    # Compute the filter H(f) in the frequency domain
    H_f = nominator / denominator
    # Transform the frequency-domain filter back to the time domain
    h_n = scipy.fft.ifft(H_f).real  # Impulse response of the filter

    # Applies time domain filter to original signal
    # The phase correction is achieved by correctly aligning the delayed part
    # of the signal with the filter response
    deterministic_part = np.zeros(len(signal))
    random_part = np.zeros(len(signal))

    # signal = np.concatenate((np.zeros(N), signal))

    # for i in tqdm(range(0, len(deterministic_part), 1), desc="Computing the DRS"):
    #     deterministic_part[i] = signal[i:i+N] @ h_n
    #     random_part[i] = signal[i + N] - deterministic_part[i]

    # deterministic_part = np.zeros(len(signal))
    # random_part = np.zeros(len(signal))
    for i in tqdm(range(N+Delta, len(signal), 1), desc="Computing the DRS"):
        deterministic_part[i] = signal[i-N-Delta:i-Delta] @ h_n
        random_part[i] = signal[i] - deterministic_part[i]

    deterministic_part[:N+Delta] = 0
    random_part[:N+Delta] = 0



    return random_part, deterministic_part

def randall_method_3(signal, fs, N=16384, Delta=500, nlevel=2):
    '''
    1. Discrete/random separation (DRS) to remove deterministic
    (discrete frequency) components.
    2. Spectral kurtosis to determine the most impulsive band,
    followed by bandpass filtering.
    3. Envelope analysis (squared envelope spectrum) of the
    bandpass filtered signal.
    ''' 
    # 1. DRS Filtering
    random_part, deterministic_part = DRS(signal, N, Delta)
    filtered_signal = random_part[N+Delta:]
    t = np.linspace((N+Delta)/fs, len(signal)/fs, len(signal)-N-Delta)
    
    # 2. Kurtogram band selection and bandpass filtering
    _, _, _, fc, bandwidth = fast_kurtogram(filtered_signal, fs, nlevel=nlevel, verbose=True)
     
    filtered_signal = ut.bandpass_filter(filtered_signal, fs, fc, bandwidth)

    # 3. Squared envelope spectrum
    analytic_signal = scipy.signal.hilbert(filtered_signal)
    envelope = np.abs(analytic_signal)
    squared_envelope = envelope ** 2
    f, X = ut.oneside_fft(squared_envelope, fs)

    return t, filtered_signal, f, X


def plot_kurtogram(Kwav, freq_w, level_w):
    plt.imshow(Kwav, aspect='auto',
               extent=(freq_w[0], freq_w[-1], level_w[0], level_w[-1]),
               interpolation='none')
    plt.colorbar()
    plt.savefig("kurtogram.png", dpi=300)

    plt.show()


if __name__ == "__main__":


    # Parameters
    fs = 1000  # Sampling frequency (Hz)
    t = np.linspace(0, 1, fs, endpoint=False)  # Time vector (1 second duration)
    frequency = 50  # Frequency of the sine wave (Hz)
    amplitude = 1  # Amplitude of the sine wave
    noise_amplitude = 0.5  # Amplitude of the noise
    N = 100  # Window size for DRS
    Delta = 10  # Delay parameter for DRS
    alpha = 0.1  # Smoothing factor for first-order tracking (0 < alpha < 1)

    # Generate sine wave
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * t)

    # Generate random noise
    noise = noise_amplitude * np.random.normal(size=len(t))

    # Combine sine wave and noise
    signal = sine_wave + noise

    # Apply DRS (Deterministic-Random Separation)
    random_part, deterministic_part = DRS(signal, N, Delta)

    # Plot the results
    plt.figure(figsize=(8, 6))

    # Plot 1: Original signal
    plt.subplot(3, 1, 1)
    plt.xlim(0.2, 0.4)
    plt.plot(t, signal, label="Signal (Sine + Noise)", color="blue")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Original Signal")
    plt.legend()
    plt.grid(True)

    # Plot 2: Deterministic component
    plt.subplot(3, 1, 2)
    #x-limit
    plt.xlim(0.2, 0.4)
    plt.plot(t, deterministic_part, label="Deterministic Component (Extracted by DRS)", color="green")
    plt.plot(t, sine_wave, label="Pure Sine Wave", color="purple", linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Deterministic Component (Extracted by DRS)")
    plt.legend()
    plt.grid(True)

    # Plot 3: Random component (Fixed title)
    plt.subplot(3, 1, 3)
    plt.xlim(0.2, 0.4)
    plt.ylim(-1.6, 2)
    plt.plot(t, random_part, label="Random Component (Extracted by DRS)", color="red")
    plt.plot(t, noise, label="Noise", color="purple", linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)

    # Adjust layout and display
    plt.tight_layout()
    plt.show()
