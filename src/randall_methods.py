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
    '''mat_files = glob.glob("matlab_example_data/outer_fault.mat", recursive=True)
    print("\n".join(mat_files))

    fs = 97656
    # signal = scipy.io.loadmat(mat_files[0])['xOuter'].ravel()
    signal = scipy.io.loadmat('src/matlab_example_data/outer_fault.mat')
    signal = signal['xOuter'].ravel()
    print(signal)
    time = np.linspace(0, len(signal)/fs, len(signal))
    plt.plot(time, signal, c="b", linewidth=0.5)
    plt.xlim(0, 0.1)
    plt.show()

    analytic_signal = scipy.signal.hilbert(signal)
    envelope = np.abs(analytic_signal)
    squared_envelope = envelope ** 2
    f, X = ut.oneside_fft(squared_envelope, fs)

    plt.xlim(0, 1000)
    print(f[0], f[-1])
    f, X = scipy.signal.periodogram(squared_envelope, fs, window="hann")
    X = np.sqrt(X)
    plt.plot(f, X, c="b", linewidth=0.5)
    plt.show()'''
    # Set RPM manually
    # rpm = 1750  # RPM value

    # # Load signal
    # fs = 12e3
    # mat_files = glob.glob("/Users/elmo/Arotor/openConMo/src/CWRU-dataset/12k_Drive_End_Bearing_Fault_Data/B/028/3007_2.mat", recursive=True)

    # # Assuming your helper function exists:
    # signal, _, _, _ = du.extract_signals(mat_files[0], normal=False)

    # # Convert time (seconds) to shaft revolutions
    # time = np.linspace(0, len(signal) / fs, len(signal))
    # shaft_revs = time * (rpm / 60)  # Convert time in seconds to revolutions

    # # Apply DRS filtering
    # N = 8192
    # Delta = 500
    # random_part, deterministic_part = DRS(signal, N=N, Delta=Delta)


    # #plt.figure(figsize=(15, 9))
    # #plt.semilogy(np.abs(hn))
    # #plt.title("Filter weights")
    # # Plot
    # plt.figure(figsize=(15, 9))

    # # Plot the original vibration signal
    # plt.subplot(3, 1, 1)
    # plt.plot(shaft_revs, signal, label='Original signal')
    # plt.title("(a) Measured vibration signal")
    # plt.xlabel("Shaft Revolutions")
    # plt.ylabel("Amplitude")
    # plt.grid(True)

    # # Plot the extracted periodic (deterministic) part
    # plt.subplot(3, 1, 2)
    # plt.plot(shaft_revs, deterministic_part, label='Deterministic (periodic) part', color='green')
    # plt.title("(b) Extracted periodic part")
    # plt.xlabel("Shaft Revolutions")
    # plt.ylabel("Amplitude")
    # plt.grid(True)

    # # Plot the extracted non-deterministic (random) part
    # plt.subplot(3, 1, 3)
    # plt.plot(shaft_revs, random_part, label='Random (non-deterministic) part', color='orange')
    # plt.title("(c) Extracted non-deterministic part")
    # plt.xlabel("Shaft Revolutions")
    # plt.ylabel("Amplitude")
    # plt.grid(True)

    # plt.tight_layout()
    # plt.show()
    # Parameters

    '''version 1'''
    # fs = 1000  # Sampling frequency (Hz)
    # t = np.linspace(0, 1, fs, endpoint=False)  # Time vector (1 second duration)
    # frequency = 50  # Frequency of the sine wave (Hz)
    # amplitude = 1  # Amplitude of the sine wave
    # noise_amplitude = 0.5  # Amplitude of the noise
    # N = 100  # Window size for DRS
    # Delta = 10  # Delay parameter for DRS

    # # Generate sine wave
    # sine_wave = amplitude * np.sin(2 * np.pi * frequency * t)

    # # Generate random noise
    # noise = noise_amplitude * np.random.normal(size=len(t))

    # # Combine sine wave and noise
    # signal = sine_wave + noise

    # # Apply DRS (Deterministic-Random Separation)
    # random_part, deterministic_part = DRS(signal, N, Delta)

    # # Plot the results
    # plt.figure(figsize=(12, 8))

    # # Plot 1: Original signal
    # plt.subplot(3, 1, 1)
    # plt.plot(t, signal, label="Signal (Sine + Noise)", color="blue")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Amplitude")
    # plt.title("Original Signal (Sine + Noise)")
    # plt.legend()
    # plt.grid(True)

    # # Plot 2: Deterministic component
    # plt.subplot(3, 1, 2)
    # plt.plot(t, deterministic_part, label="Deterministic Component (Extracted by DRS)", color="green")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Amplitude")
    # plt.title("Deterministic Component (Extracted by DRS)")
    # plt.legend()
    # plt.grid(True)

    # # Plot 3: Random component
    # plt.subplot(3, 1, 3)
    # plt.plot(t, random_part, label="Random Component (Extracted by DRS)", color="red")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Amplitude")
    # plt.title("Random Component (Extracted by DRS)")
    # plt.legend()
    # plt.grid(True)

    # plt.tight_layout()
    # plt.show()

    '''Version 2'''
    def first_order_tracking(signal, alpha):
    # Recursive filter: y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
        smoothed_signal = scipy.signal.lfilter([alpha], [1, -(1 - alpha)], signal) * 3
        return smoothed_signal


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

    # Apply first-order tracking to the deterministic component
    smoothed_deterministic = first_order_tracking(deterministic_part, alpha)

    # Plot the results
    plt.figure(figsize=(12, 10))

    # Plot 1: Original signal
    plt.subplot(4, 1, 1)
    plt.plot(t, signal, label="Signal (Sine + Noise)", color="blue")
    plt.plot(t, sine_wave, label="Pure Sine Wave", color="purple", linestyle="--")
    plt.plot(t, noise, label="Noise", color="orange", linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Original Signal (Sine + Noise)")
    plt.legend()
    plt.grid(True)

    # Plot 2: Deterministic component
    plt.subplot(4, 1, 2)
    plt.plot(t, deterministic_part, label="Deterministic Component (Extracted by DRS)", color="green")
    plt.plot(t, sine_wave, label="Pure Sine Wave", color="purple", linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Deterministic Component (Extracted by DRS)")
    plt.legend()
    plt.grid(True)

    # Plot 3: Smoothed deterministic component (First-order tracking)
    plt.subplot(4, 1, 3)
    plt.plot(t, sine_wave, label="Pure Sine Wave", color="purple", linestyle="--")
    plt.plot(t, smoothed_deterministic, label="Smoothed Deterministic Component (First-Order Tracking)", color="orange")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Smoothed Deterministic Component (First-Order Tracking)")
    plt.legend()
    plt.grid(True)

    # Plot 4: Random component
    plt.subplot(4, 1, 4)
    plt.plot(t, random_part, label="Random Component (Extracted by DRS)", color="red")
    plt.plot(t, noise, label="Pure Sine Wave", color="purple", linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

