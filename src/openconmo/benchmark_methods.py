import glob

import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.fft
from tqdm import tqdm
from scipy.fft import fft, ifft
from kurtogram import fast_kurtogram
import utils as ut


def envelope(signal, fs):
    """
    Perform squared envelope spectrum analysis of a signal.

    This method applies envelope analysis using the squared envelope spectrum
    of the full-bandwidth signal. It is suitable when there are no strong
    masking sources in the signal, and bearing faults can be diagnosed directly.
    Often used as a simple baseline in fault detection.

    Parameters
    ----------
    signal : ndarray
        The input raw signal.
    fs : float
        Sampling frequency of the signal in Hz.

    Returns
    -------
    f : ndarray
        Frequency vector corresponding to the squared envelope spectrum.
    X : ndarray
        Amplitude of the squared envelope spectrum.
    """

    # Compute the analytic signal using the Hilbert transform
    analytic_signal = scipy.signal.hilbert(signal)

    # Extract the amplitude envelope
    envelope = np.abs(analytic_signal)

    # Square the envelope to enhance impulsive features
    squared_envelope = envelope ** 2

    # Compute the one-sided FFT of the squared envelope signal
    f, X = ut.oneside_fft(squared_envelope, fs)

    return f, X


def cepstrum_prewhitening(signal, fs):
    """
    Apply cepstrum editing (liftering) for signal pre-whitening.

    This method performs signal pre-whitening using cepstrum editing, as described in
    "Signal Pre-whitening Using Cepstrum Editing (Liftering) to Enhance Fault Detection
    in Rolling Element Bearings."

    Parameters
    ----------
    signal : ndarray
        The input time-domain signal.
    fs : float
        Sampling frequency of the signal in Hz.

    Returns
    -------
    t : ndarray
        Time vector corresponding to the reconstructed signal.
    time_signal : ndarray
        Time-domain signal after cepstrum-based pre-whitening.
    """
    # Create time vector
    t = np.linspace(0, len(signal)/fs, len(signal))

    # Compute FFT of the signal
    X = scipy.fft.fft(signal)

    # Extract the phase for later reconstruction
    phase = np.angle(X)

    # Compute the log-amplitude spectrum, avoiding log(0) with small offset
    log_amplitude = np.log(np.abs(X) + 1e-10);

    # Compute real cepstrum via IFFT of the log-amplitude spectrum
    real_cepstrum = scipy.fft.ifft(log_amplitude)

    # Initialize edited cepstrum _ keep only DC component (liftering step)
    edited_cepstrum = np.zeros_like(real_cepstrum)
    edited_cepstrum[0] = real_cepstrum[0]

    # Transform the edited cepstrum back to the frequency domain
    edited_log_amplitude_spectrum = scipy.fft.fft(edited_cepstrum)

    # Reconstuct the edited log spectrum using original phase
    edited_log_spectrum = edited_log_amplitude_spectrum + 1j * phase

    # Convert back to linear spectrum, then to time domain
    X = np.exp(edited_log_spectrum)
    time_signal = np.real(scipy.fft.ifft(X))

    return t, time_signal

def DRS(signal, N, Delta):
    """
    Discrete/Random Separation (DRS) to remove deterministic components.

    This method separates a signal into deterministic and random parts
    by constructing a frequency-domain filter based on delayed cross-power spectra.
    The approach is based on:
    "Unsupervised Noise Cancellation for Vibration Signals: Part II—
    A Novel Frequency-Domain Algorithm." https://doi.org/10.1016/S0888-3270(03)00013-X

    Parameters
    ----------
    signal : ndarray
        The input time-domain signal.
    N : int
        Window length used for spectral estimation.
    Delta : int
        Delay applied to the signal in the cross-correlation.

    Returns
    -------
    random_part : ndarray
        The component of the signal with deterministic content removed.
    deterministic_part : ndarray
        The reconstructed deterministic (periodic/discrete frequency) part of the signal.

    """
    # Initialize numerator and denominator for frequency-domain filter H(f)
    nominator = np.zeros(N, dtype=complex)
    denominator = np.zeros(N, dtype=complex)

    # Estiamte cross- and power spectra across the signal using overlapping windows
    for k in tqdm(range(2*N+Delta, len(signal), 100), desc="Building DRS filter"):

        # Define current and delayed windows of length N
        window_idx = np.arange(k-N, k)
        delayed_window_idx = np.arange(k-2*N-Delta, k-N-Delta)

        # Apply Parzen window to current and delayed signal segments
        x = signal[window_idx]*scipy.signal.windows.parzen(N) 
        x_d = signal[delayed_window_idx]*scipy.signal.windows.parzen(N) 

        # Cuompute FFTs of the current and delayed windows
        X_k_f = scipy.fft.fft(x)
        X_d_k_f = scipy.fft.fft(x_d)

        # Accumulate cross-power and auto-power spectra
        nominator += X_d_k_f * X_k_f.conj()  # Cross-power spectrum
        denominator += X_d_k_f * X_d_k_f.conj() # Power spectrum of the delayed signal

    # Compute the frequency response H(f) and transform to time domain
    H_f = nominator / denominator
    h_n = scipy.fft.ifft(H_f).real

    # Initialize output arrays for deterministic and random components
    deterministic_part = np.zeros(len(signal))
    random_part = np.zeros(len(signal))

    # Filter the signal to extract deterministic part; substract to get random part
    for i in tqdm(range(N+Delta, len(signal), 1), desc="Computing the DRS"):
        deterministic_part[i] = signal[i-N-Delta:i-Delta] @ h_n
        random_part[i] = signal[i] - deterministic_part[i]

    return random_part, deterministic_part

def benchmark_method(signal, fs, N=16384, Delta=500, nlevel=2):
    """
    Fault detection using DRS, kurtogram-based filtering, and envelope analysis.

    The approach is based on:
    "Rolling elment bearing diagnostics using the Case Western Reserve University data: A benchmark study" https://doi.org/10.1016/j.ymssp.2015.04.021

    This method combines three steps for enhancing impulsive features in vibration signals:
    1. Discrete/Random Separation (DRS) to remove deterministic frequency components.
    2. Spectral kurtosis analysis via a fast kurtogram to select the most impulsive frequency band.
    3. Envelope analysis (squared envelope spectrum) of the filtered signal to extract fault signatures.

    Parameters
    ----------
    signal : ndarray
        Input time-domain signal.
    fs : float
        Sampling frequency of the signal in Hz.
    N : int, optional
        Length of the window used for DRS filtering (default is 16384).
    Delta : int, optional
        Overlap or shift value used in DRS (default is 500).
    nlevel : int, optional
        Number of decomposition levels used in the fast kurtogram (default is 2).

    Returns
    -------
    t : ndarray
        Time vector corresponding to the filtered signal after DRS.
    filtered_signal : ndarray
        Signal after DRS and kurtogram-based bandpass filtering.
    f : ndarray
        Frequency vector of the squared envelope spectrum.
    X : ndarray
        Amplitude of the squared envelope spectrum.
    """
    
    # Step 1: Apply DRS to remove deterministic components
    random_part, deterministic_part = DRS(signal, N, Delta)
    filtered_signal = random_part[N+Delta:] # Remove transient startup region
    t = np.linspace((N+Delta)/fs, len(signal)/fs, len(signal)-N-Delta)
    
    # Step 2: Use fast kurtogram to find most impulsive frequency band
    _, _, _, fc, bandwidth = fast_kurtogram(filtered_signal, fs, nlevel=nlevel, verbose=True)
     
    # Apply bandpass filter around the selected frequency band
    filtered_signal = ut.bandpass_filter(filtered_signal, fs, fc, bandwidth)

    # Step 3: Perform envelope analysis on the filtered signal
    analytic_signal = scipy.signal.hilbert(filtered_signal)
    envelope = np.abs(analytic_signal)
    squared_envelope = envelope ** 2
    f, X = ut.oneside_fft(squared_envelope, fs)


    return t, filtered_signal, f, X

if __name__ == "__main__":
    print("This module contains benchmark methods for fault detection in vibration signals.")