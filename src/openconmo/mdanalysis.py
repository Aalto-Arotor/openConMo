import numpy as np
from sklearn.linear_model import Lasso
import cvxpy as cp


def quantize(signal, nbit=8):
    """Quantize a signal to n-bit.

    Parameters
    ----------
    signal : ndarray of shape (signal_length,)
        Original signal to be quantized.

    nbit : int, default=8
        Number of bits.

    Returns
    -------
    signal: ndarray of shape (signal_length,)

    """

    max_amplitude = np.max(signal)
    signal = (signal / max_amplitude * (2**nbit)).astype(int)
    signal = signal / 2**nbit
    return signal


def generate_signal(
    freqs=None,
    num_harmonics: int = 2,
    duration: float = 10,
    sample_rate: float = 40,
    SNR: float = 10,
    nbit: int = 8,
    seed: int = None,
    f_mod: float = 0,
):
    """Generate a signal with either constant or periodically modulated frequencies.
    The amplitudes and initial phases of the frequencies are randomly generated.

    Parameters
    ----------
    freqs : array-like of shape (num_sources,)
        Fundamental frequencies of the signal.

    num_harmonics : int, default=2
        Number of harmonics of the frequencies.

    duration : float, default=10
        Signal length in seconds.

    sample_rate : float, default=40
        Sampling rate of the signal.

    SNR : float, default=10
        Signal-to-noise ratio, measured in decibels.

    nbit : int, default=8
        Number of bits.

    seed : int, default=None
        Initial seed for randomization.

    f_mod : float, default=0
        Modulating frequency. For a stationary signal with constant frequency, set to default value 0.

    Returns
    -------
    signal : ndarray of shape (sample_rate*duration,)
        Generated signal.

    sample_rate : float
        Sampling rate of generated signal.

    true_spec : ndarray of shape (num_sources*num_harmonics,2)
        Containing the true frequencies in the first column and their amplitudes in the second column.

    base_freq : ndarray of shape (sample_rate*duration,)
        The reference frequencies for a signal. This is constant (and all entries equal 1) when the signal is stationary with
        constant frequencies. For signal with periodically modulated frequencies, this array equals each frequency component
        divided by its initial value.
    """

    if freqs is None:
        freqs = (0.3, 0.32, 0.34, 0.35, 0.4, 1, 1.025, 1.05)
    num_sources = len(freqs)
    N = sample_rate * duration

    if f_mod == 0:
        base_freq = np.ones(N)
    else:
        f_max = np.max(freqs) * num_harmonics
        a_mod = (sample_rate / 2 - f_max) / f_max / 2
        base_freq = (
            1 + a_mod - a_mod * np.cos(2 * np.pi * f_mod * np.arange(N) / sample_rate)
        )
    base_phase = np.cumsum(base_freq) / sample_rate

    if seed is not None:
        np.random.seed(seed)
    init_phase = np.random.randn(num_harmonics, num_sources) * np.pi
    amplitude = np.round(abs(np.random.rand(num_harmonics, num_sources) + 1), 2)
    signal = np.zeros(N)
    true_f, true_spec = [], []

    for p in range(num_sources):
        for l in range(num_harmonics):
            if freqs[p] * (l + 1) < sample_rate / 2:
                signal += amplitude[l, p] * np.cos(
                    2 * np.pi * freqs[p] * (l + 1) * base_phase + init_phase[l, p]
                )
                true_f.append(freqs[p] * (l + 1))
                true_spec.append(amplitude[l, p])

    true_spec = np.array([true_f, true_spec])
    signal_power = np.var(signal)
    noise_power = signal_power / (10 ** (SNR / 10))
    signal += np.random.randn(N) * np.sqrt(noise_power)
    signal = quantize(signal, nbit)
    return signal, sample_rate, true_spec, base_freq


def cov_matrix(signal, M: int = 50):
    """Calculate the covariance matrix of a signal using short frames of length M.

    Parameters
    ----------
    signal : ndarray of shape (signal_length,)
        Original signal to be quantized.

    M : int, default=50
        The size of the square covariance matrix.

    Returns
    -------
    R : ndarray of shape (M, M)
        Sample covariance matrix of the signal.
    """
    N = len(signal)
    if M < N:
        Y = np.array([signal[i : i + M] for i in range(N - M + 1)])
        R = Y.T @ Y.conj() / (N - M + 1)
    else:
        R = np.outer(signal, signal.conj())
    return R


def stationarize(y, base_phase, resolution=None):
    """Stationarize a nonstationary signal. This technique is designed specifically for signals whose frequency components change over time but remain proportional to each other.

    Parameters
    ----------
    signal : ndarray of shape (signal_length,)
        Original signal to be quantized.

    base_phase : ndarray of shape (signal_length,)
        The reference cumulative phase for a signal; this is linear for a stationary signal with constant frequencies. This is calculated as the cummulative sum of the reference frequency.

    resolution : int, default=signal_length
        The size of the frequency grid to transform the signal.

    Returns
    -------
    out : ndarray of shape (signal_length,)
        Stationarized output signal.
    """
    N = len(y)
    if resolution is None:
        resolution = N
    freqs = np.linspace(0, 2 * np.pi, resolution)
    A = np.exp(1j * base_phase.reshape(-1, 1) @ freqs.reshape(1, -1))
    B = np.exp(1j * np.arange(N).reshape(-1, 1) @ freqs.reshape(1, -1))
    out = B @ A.conj().T @ y / resolution
    if y.dtype == "float":
        out = np.real(out)
    return out


def detect_peaks(data, N=None):
    """Detect the indices of the peaks in the spectrum using derivatives.

    Parameters
    ----------
    data : ndarray of shape (data_length,)
        The array containing the spectrum whose peaks are to be identified.

    N : int, default=None
        Number of peaks to be identified. If this is not given, all points where the derivative flips from positive to negative will be counted as peaks.

    Returns
    -------
    peak_idx : ndarray of shape (num_peaks,)
        Indices of the peaks in the data spectrum.
    """
    derivative = np.diff(data)
    peak_idx = np.where((derivative[:-1] > 0) & (derivative[1:] <= 0))[0] + 1
    if N is not None:
        peak_idx = sorted(peak_idx, key=lambda x: data[x], reverse=True)
        peak_idx = peak_idx[:N]
        peak_idx.sort()
    return peak_idx


def peak_rmse(freqs, spec, true_peaks):
    """Calculate the RMSE between the spectrum/pseudospectrum resulted from an analysis and the true peaks of the simulated model.

    Parameters
    ----------
    freqs : ndarray of shape (spectrum_length,)
        The frequency axis of the spectrum.

    spec : ndarray of shape (spectrum_length,)
        The magnitude spectrum corresponding to the frequencies in **freqs**.

    true_peaks : ndarray of shape (num_peaks,)
        The correct peaks in the simulated signal model.

    Returns
    -------
    rmse : float
        RMSE between the known correct peaks in the simulated model and the peaks detected by analysis.
    """
    estimated_peaks = freqs[detect_peaks(spec)]
    error = np.zeros(len(true_peaks))
    for i, f in enumerate(true_peaks):
        error[i] = np.min(abs(f - estimated_peaks))
    rmse = np.mean(error**2) ** 0.5
    return rmse


class FFT:
    """Perform Fast-Fourier Transform on the signal.

    Parameters
    ----------
    y : ndarray of shape (signal_length,)
        The signal to be analyzed

    fs : float, default=2*np.pi
        The sampling rate of the signal.

    true_peaks : ndarray of shape (num_peaks,)
        The correct peaks in the simulated signal model.

    resolution : int, default=4*signal_length
        The resolution of the FFT analysis result, i.e., the size of the frequency domain array.
        This is defaulted to 4 times the original FFT resolution, which equals the signal length.

    Attributes
    ----------
    freqs : ndarray of shape (resolution,)
        The frequency axis of the spectrum.

    spec : ndarray of shape (resolution,)
        The normalized magnitude spectrum corresponding to the frequencies in **freqs**.

    Functions
    ---------
    rmse(true_peaks)
        Calculate the RMSE between the known correct peaks in the simulated model and the peaks detected by analysis.
    """

    def __init__(self, y, fs=2 * np.pi, true_peaks=None, resolution=None):
        self.name = "FFT*"
        N = len(y)
        if resolution is None:
            resolution = 4 * N
        fft_spec = np.fft.fft(y, n=resolution)
        self.spec = abs(fft_spec[: resolution // 2]) / np.max(
            abs(fft_spec[: resolution // 2])
        )
        self.freqs = np.linspace(0, fs / 2, resolution // 2)
        if true_peaks is not None:
            self.error = peak_rmse(self.freqs, self.spec, true_peaks)

    def rmse(self, true_peaks):
        """Calculate the RMSE between the spectrum resulted from analysis and the true peaks of the simulated model.

        Parameters
        ----------
        true_peaks : ndarray of shape (num_peaks,)
            The correct peaks in the simulated signal model.

        Returns
        -------
        rmse : float
            RMSE between the known correct peaks in the simulated model and the peaks detected by analysis.
        """
        self.error = peak_rmse(self.freqs, self.spec, true_peaks)
        return self.error


class MaximumLikelihood:
    """Perform Maximum Likelihood analysis on the signal.

    Parameters
    ----------
    y : ndarray of shape (signal_length,)
        The signal to be analyzed

    fs : float, default=2*np.pi
        The sampling rate of the signal.

    resolution : int
        The resolution of the analysis result, i.e., the size of the frequency domain array.
        This is defaulted to 4 times the original FFT resolution.

    freq_range : array_like of shape (2,), default=(0, fs/2)
        Range of frequency to analyze.

    phase : ndarray of shape (signal_length,)
        Reference phase for signal with inconstant but proportional frequencies. The reference phase is chosen so that the initial frequency is 1Hz.

    true_peaks : ndarray of shape (num_peaks,)
        The correct peaks in the simulated signal model.

    Attributes
    ----------
    freqs : ndarray of shape (resolution,)
        The frequency axis of the spectrum.

    spec : ndarray of shape (resolution,)
        The normalized magnitude spectrum corresponding to the frequencies in **freqs**.

    Functions
    ---------
    rmse(true_peaks)
        Calculate the RMSE between the known correct peaks in the simulated model and the peaks detected by analysis.
    """

    def __init__(
        self,
        y,
        fs=2 * np.pi,
        resolution=None,
        freq_range=None,
        phase=None,
        true_peaks=None,
    ):
        self.name = "ML*"
        N = len(y)
        if freq_range is None:
            freq_range = (0, fs / 2)
        if resolution is None:
            resolution = int(4 * np.diff(freq_range)[0] * N / fs)
        if phase is None:
            phase = np.arange(N)
        self.freqs = np.linspace(freq_range[0], freq_range[1], resolution)
        A = np.exp(
            1j * 2 * np.pi * phase.reshape(-1, 1) @ self.freqs.reshape(1, -1) / fs
        )
        self.spec = A.conj().T @ y
        self.spec = abs(self.spec) / np.max(abs(self.spec))
        if true_peaks is not None:
            self.error = peak_rmse(self.freqs, self.spec, true_peaks)

    def rmse(self, true_peaks):
        """Calculate the RMSE between the spectrum resulted from analysis and the true peaks of the simulated model.

        Parameters
        ----------
        true_peaks : ndarray of shape (num_peaks,)
            The correct peaks in the simulated signal model.

        Returns
        -------
        rmse : float
            RMSE between the known correct peaks in the simulated model and the peaks detected by analysis.
        """
        self.error = peak_rmse(self.freqs, self.spec, true_peaks)
        return self.error


class NLS:
    """Perform Nonlinear Least Squares analysis on the signal.

    Parameters
    ----------
    y : ndarray of shape (signal_length,)
        The signal to be analyzed

    fs : float, default=2*np.pi
        The sampling rate of the signal.

    resolution : int
        The resolution of the analysis result, i.e., the size of the frequency domain array.
        This is defaulted to 4 times the original FFT resolution.

    freq_range : array_like of shape (2,), default=(0, fs/2)
        Range of frequency to analyze.

    phase : ndarray of shape (signal_length,)
        Reference phase for signal with inconstant but proportional frequencies. The reference phase is chosen so that the initial frequency is 1Hz.

    true_peaks : ndarray of shape (num_peaks,)
        The correct peaks in the simulated signal model.

    Attributes
    ----------
    freqs : ndarray of shape (resolution,)
        The frequency axis of the spectrum.

    spec : ndarray of shape (resolution,)
        The normalized magnitude spectrum corresponding to the frequencies in **freqs**.

    Functions
    ---------
    rmse(true_peaks)
        Calculate the RMSE between the known correct peaks in the simulated model and the peaks detected by analysis.
    """

    def __init__(
        self,
        y,
        fs=2 * np.pi,
        resolution=None,
        freq_range=None,
        phase=None,
        true_peaks=None,
    ):
        self.name = "NLS*"
        N = len(y)
        if freq_range is None:
            freq_range = (0, fs / 2)
        if resolution is None:
            resolution = int(4 * np.diff(freq_range)[0] * N / fs)
        if phase is None:
            phase = np.arange(N)
        self.freqs = np.linspace(freq_range[0], freq_range[1], resolution)

        self.spec = np.zeros(resolution)
        for k in range(resolution):
            a = np.cos(2 * np.pi * self.freqs[k] * phase / fs)
            self.spec[k] = (np.dot(a.conj(), y) / np.linalg.norm(a)) ** 2
        self.spec = abs(self.spec) / np.max(abs(self.spec))
        if true_peaks is not None:
            self.error = peak_rmse(self.freqs, self.spec, true_peaks)

    def rmse(self, true_peaks):
        """Calculate the RMSE between the pseudospectrum resulted from analysis and the true peaks of the simulated model.

        Parameters
        ----------
        true_peaks : ndarray of shape (num_peaks,)
            The correct peaks in the simulated signal model.

        Returns
        -------
        rmse : float
            RMSE between the known correct peaks in the simulated model and the peaks detected by analysis.
        """
        self.error = peak_rmse(self.freqs, self.spec, true_peaks)
        return self.error


class Beamforming:
    """Perform Beamforming analysis on the signal.

    Parameters
    ----------
    y : ndarray of shape (signal_length,)
        The signal to be analyzed

    fs : float, default=2*np.pi
        The sampling rate of the signal.

    resolution : int
        The resolution of the analysis result, i.e., the size of the frequency domain array.
        This is defaulted to 4 times the original FFT resolution.

    freq_range : array_like of shape (2,), default=(0, fs/2)
        Range of frequency to analyze.

    R : ndarray of shape (M, M), default=None
        Precalculated covariance matrix for the signal. If not given, the covariance matrix is calculated base on the covariance matrix size parameter M.

    M : int, default=200
        Size of the covariance matrix. Only used if R=None.

    true_peaks : ndarray of shape (num_peaks,)
        The correct peaks in the simulated signal model.

    Attributes
    ----------
    freqs : ndarray of shape (resolution,)
        The frequency axis of the spectrum.

    spec : ndarray of shape (resolution,)
        The normalized magnitude spectrum corresponding to the frequencies in **freqs**.

    Functions
    ---------
    rmse(true_peaks)
        Calculate the RMSE between the known correct peaks in the simulated model and the peaks detected by analysis.
    """

    def __init__(
        self,
        y,
        fs=2 * np.pi,
        resolution=None,
        freq_range=None,
        R=None,
        M=200,
        true_peaks=None,
    ):
        self.name = "Beamforming"
        N = len(y)
        if R is None:
            if M > N:
                M = N
            R = cov_matrix(y, M)
        else:
            assert np.diff(R.shape) == 0, "Covariance matrix is not square!"
            M = R.shape[0]
        if freq_range is None:
            freq_range = (0, fs / 2)
        if resolution is None:
            resolution = int(4 * np.diff(freq_range)[0] * N / fs)
        self.freqs = np.linspace(freq_range[0], freq_range[1], resolution)

        self.spec = np.zeros(resolution, dtype=complex)
        for k in range(resolution):
            a = np.exp(1j * 2 * np.pi * np.arange(M) * self.freqs[k] / fs).reshape(
                -1, 1
            )
            self.spec[k] = ((a.conj().T @ R @ a) / (a.conj().T @ a))[0, 0]
        self.spec = abs(self.spec) / np.max(abs(self.spec))
        if true_peaks is not None:
            self.error = peak_rmse(self.freqs, self.spec, true_peaks)

    def rmse(self, true_peaks):
        """Calculate the RMSE between the pseudospectrum resulted from analysis and the true peaks of the simulated model.

        Parameters
        ----------
        true_peaks : ndarray of shape (num_peaks,)
            The correct peaks in the simulated signal model.

        Returns
        -------
        rmse : float
            RMSE between the known correct peaks in the simulated model and the peaks detected by analysis.
        """
        self.error = peak_rmse(self.freqs, self.spec, true_peaks)
        return self.error


class MVDR:
    """Perform MVDR (Minimum Variance Distortionless Response) analysis on the signal.

    Parameters
    ----------
    y : ndarray of shape (signal_length,)
        The signal to be analyzed

    fs : float, default=2*np.pi
        The sampling rate of the signal.

    resolution : int
        The resolution of the analysis result, i.e., the size of the frequency domain array.
        This is defaulted to 4 times the original FFT resolution.

    freq_range : array_like of shape (2,), default=(0, fs/2)
        Range of frequency to analyze.

    R : ndarray of shape (M, M), default=None
        Precalculated covariance matrix for the signal. If not given, the covariance matrix is calculated base on the covariance matrix size parameter M.

    M : int, default=200
        Size of the covariance matrix. Only used if R=None.

    true_peaks : ndarray of shape (num_peaks,)
        The correct peaks in the simulated signal model.

    Attributes
    ----------
    freqs : ndarray of shape (resolution,)
        The frequency axis of the spectrum.

    spec : ndarray of shape (resolution,)
        The normalized magnitude spectrum corresponding to the frequencies in **freqs**.

    Functions
    ---------
    rmse(true_peaks)
        Calculate the RMSE between the known correct peaks in the simulated model and the peaks detected by analysis.
    """

    def __init__(
        self,
        y,
        fs=2 * np.pi,
        resolution=None,
        freq_range=None,
        R=None,
        M=200,
        true_peaks=None,
    ):
        self.name = "MVDR"
        N = len(y)
        if R is None:
            if M > N:
                M = N
            invR = np.linalg.pinv(cov_matrix(y, M))
        else:
            assert np.diff(R.shape) == 0, "Covariance matrix is not square!"
            M = R.shape[0]
            invR = np.linalg.pinv(R)
        if freq_range is None:
            freq_range = (0, fs / 2)
        if resolution is None:
            resolution = int(4 * np.diff(freq_range)[0] * N / fs)
        self.freqs = np.linspace(freq_range[0], freq_range[1], resolution)

        self.spec = np.zeros(resolution, dtype=complex)
        for k in range(resolution):
            a = np.exp(1j * 2 * np.pi * np.arange(M) * self.freqs[k] / fs).reshape(
                -1, 1
            )
            self.spec[k] = 1 / np.real(a.conj().T @ invR @ a)
        self.spec = abs(self.spec) / np.max(abs(self.spec))
        if true_peaks is not None:
            self.error = peak_rmse(self.freqs, self.spec, true_peaks)

    def rmse(self, true_peaks):
        """Calculate the RMSE between the pseudospectrum resulted from analysis and the true peaks of the simulated model.

        Parameters
        ----------
        true_peaks : ndarray of shape (num_peaks,)
            The correct peaks in the simulated signal model.

        Returns
        -------
        rmse : float
            RMSE between the known correct peaks in the simulated model and the peaks detected by analysis.
        """
        self.error = peak_rmse(self.freqs, self.spec, true_peaks)
        return self.error


class MUSIC:
    """Perform MUSIC (MUltiple SIgnal Classification) analysis on the signal.

    Parameters
    ----------
    y : ndarray of shape (signal_length,)
        The signal to be analyzed

    fs : float, default=2*np.pi
        The sampling rate of the signal.

    resolution : int
        The resolution of the analysis result, i.e., the size of the frequency domain array.
        This is defaulted to 4 times the original FFT resolution.

    freq_range : array_like of shape (2,), default=(0, fs/2)
        Range of frequency to analyze.

    R : ndarray of shape (M, M), default=None
        Precalculated covariance matrix for the signal. If not given, the covariance matrix is calculated base on the covariance matrix size parameter M.

    M : int, default=200
        Size of the covariance matrix. Only used if R=None.

    num_sources : int, default=None
        Number of sources for analysis. If the number of sources is not given, it is selected automatically from the eigenvalue distribution.

    true_peaks : ndarray of shape (num_peaks,)
        The correct peaks in the simulated signal model.

    Attributes
    ----------
    freqs : ndarray of shape (resolution,)
        The frequency axis of the spectrum.

    spec : ndarray of shape (resolution,)
        The normalized magnitude spectrum corresponding to the frequencies in **freqs**.

    Functions
    ---------
    rmse(true_peaks)
        Calculate the RMSE between the known correct peaks in the simulated model and the peaks detected by analysis.
    """

    def __init__(
        self,
        y,
        fs=2 * np.pi,
        resolution=None,
        freq_range=None,
        R=None,
        M=200,
        num_sources=None,
        true_peaks=None,
    ):
        self.name = "MUSIC*"
        N = len(y)
        if R is None:
            if M > N:
                M = N
            R = cov_matrix(y, M)
        else:
            assert np.diff(R.shape) == 0, "Covariance matrix is not square!"
            M = R.shape[0]
        if freq_range is None:
            freq_range = (0, fs / 2)
        if resolution is None:
            resolution = int(4 * np.diff(freq_range)[0] * N / fs)
        self.freqs = np.linspace(freq_range[0], freq_range[1], resolution)

        eigvals, eigvecs = np.linalg.eig(R)
        eigvals = abs(eigvals)
        idx = np.argsort(eigvals)[::-1]
        if num_sources is None:
            num_sources = len(np.where(eigvals > np.mean(eigvals))[0])
        noise_subspace = eigvecs[:, idx[num_sources:]]

        self.spec = np.zeros(resolution)
        for i, f in enumerate(self.freqs):
            a = np.exp(1j * 2 * np.pi * np.arange(M) * f / fs)
            self.spec[i] = np.linalg.norm(noise_subspace.conj().T @ a) ** (-2)
        self.spec = abs(self.spec) / np.max(abs(self.spec))
        if true_peaks is not None:
            self.error = peak_rmse(self.freqs, self.spec, true_peaks)

    def rmse(self, true_peaks):
        """Calculate the RMSE between the pseudospectrum resulted from analysis and the true peaks of the simulated model.

        Parameters
        ----------
        true_peaks : ndarray of shape (num_peaks,)
            The correct peaks in the simulated signal model.

        Returns
        -------
        rmse : float
            RMSE between the known correct peaks in the simulated model and the peaks detected by analysis.
        """
        self.error = peak_rmse(self.freqs, self.spec, true_peaks)
        return self.error


class ESPRIT:
    """Perform ESPRIT (Estimation of Signal Parameters via Rotational Invariance Techniques) analysis on the signal.

    Parameters
    ----------
    y : ndarray of shape (signal_length,)
        The signal to be analyzed

    fs : float, default=2*np.pi
        The sampling rate of the signal.

    resolution : int
        The resolution to quantize the results, since ESPRIT does not produce a spectrum for an evenly
        spaced frequency grid. If not given, the raw analysis results are returned.

    R : ndarray of shape (M, M), default=None
        Precalculated covariance matrix for the signal. If not given, the covariance matrix is calculated base on the covariance matrix size parameter M.

    M : int, default=200
        Size of the covariance matrix. Only used if R=None.

    num_sources : int, default=None
        Number of sources for analysis. If the number of sources is not given, it is selected automatically from the eigenvalue distribution.

    true_peaks : ndarray of shape (num_peaks,)
        The correct peaks in the simulated signal model.

    Attributes
    ----------
    freqs : ndarray of shape (resolution,)
        The frequency axis of the spectrum.

    spec : ndarray of shape (resolution,)
        The normalized magnitude spectrum corresponding to the frequencies in **freqs**.

    Functions
    ---------
    rmse(true_peaks)
        Calculate the RMSE between the known correct peaks in the simulated model and the peaks detected by analysis.
    """

    def __init__(
        self,
        y,
        fs=2 * np.pi,
        resolution=None,
        R=None,
        M=200,
        num_sources=None,
        true_peaks=None,
    ):
        self.name = "ESPRIT"
        N = len(y)
        if R is None:
            if M > N:
                M = N
            R = cov_matrix(y, M)
        else:
            assert np.diff(R.shape) == 0, "Covariance matrix is not square!"
            M = R.shape[0]
        eigvals, eigvecs = np.linalg.eig(R)
        eigvals = abs(eigvals)
        idx = np.argsort(eigvals)[::-1]

        def inner(K):
            signal_subspace = eigvecs[:, idx[:K]]
            Phi = np.linalg.pinv(signal_subspace[:-1, :]) @ signal_subspace[1:, :]
            eigvals, _ = np.linalg.eig(Phi)
            return np.angle(eigvals) * fs / (2 * np.pi)

        if hasattr(num_sources, "__len__"):
            freqs = []
            for k in num_sources:
                freqs += list(inner(k))
            freqs = np.array(freqs)
        else:
            if num_sources is None:
                num_sources = len(np.where(eigvals > np.mean(eigvals))[0])
            freqs = inner(num_sources)
        freqs = np.sort(freqs[freqs >= 0])

        if resolution is None:
            self.quantize = False
            A = np.exp(
                1j * 2 * np.pi * np.arange(N).reshape(-1, 1) @ freqs.reshape(1, -1) / fs
            )
            self.spec = A.conj().T @ y
            self.freqs = freqs
        else:
            self.quantize = True
            self.freqs = np.linspace(0, fs / 2, resolution)
            self.spec = np.zeros_like(self.freqs, dtype=complex)
            for f in freqs:
                idx = np.argmin(np.abs(self.freqs - f))
                self.spec[idx] = np.dot(
                    np.exp(1j * 2 * np.pi * np.arange(N) * self.freqs[idx] / fs), y
                )
        self.spec = abs(self.spec) / np.max(abs(self.spec))
        if true_peaks is not None:
            self.rmse(true_peaks)

    def rmse(self, true_peaks):
        """Calculate the RMSE between the pseudospectrum resulted from analysis and the true peaks of the simulated model.

        Parameters
        ----------
        true_peaks : ndarray of shape (num_peaks,)
            The correct peaks in the simulated signal model.

        Returns
        -------
        rmse : float
            RMSE between the known correct peaks in the simulated model and the peaks detected by analysis.
        """
        if self.quantize:
            self.error = peak_rmse(self.freqs, self.spec, true_peaks)
        else:
            error = np.zeros(len(true_peaks))
            for i, f in enumerate(true_peaks):
                error[i] = np.min(abs(f - self.freqs))
            self.error = np.mean(error**2) ** 0.5
        return self.error


class CompressiveSensing:
    """Perform Compressive Sensing analysis on the signal.

    Parameters
    ----------
    y : ndarray of shape (signal_length,)
        The signal to be analyzed

    fs : float, default=2*np.pi
        The sampling rate of the signal.

    resolution : int
        The resolution of the analysis result, i.e., the size of the frequency domain array.
        This is defaulted to 4 times the original FFT resolution.

    freq_range : array_like of shape (2,), default=(0, fs/2)
        Range of frequency to analyze.

    alpha : float, default=0.1
        The amount of L1-norm regularization. Higher alpha means fewer peaks int he analysis results, and vice versa.

    method : {'sklearn', 'cvxpy'}, default='sklearn'
        Library for LASSO implementation. Use 'cvxpy' for complex signals, and 'sklearn' for faster analysis of real signals.

    phase : ndarray of shape (signal_length,)
        Reference phase for signal with inconstant but proportional frequencies. The reference phase is chosen so that the initial frequency is 1Hz.

    true_peaks : ndarray of shape (num_peaks,)
        The correct peaks in the simulated signal model.

    Attributes
    ----------
    freqs : ndarray of shape (resolution,)
        The frequency axis of the spectrum.

    spec : ndarray of shape (resolution,)
        The normalized magnitude spectrum corresponding to the frequencies in **freqs**.

    Functions
    ---------
    rmse(true_peaks)
        Calculate the RMSE between the known correct peaks in the simulated model and the peaks detected by analysis.
    """

    def __init__(
        self,
        y,
        fs=2 * np.pi,
        resolution=None,
        freq_range=None,
        alpha=0.1,
        method="sklearn",
        phase=None,
        true_peaks=None,
    ):
        self.name = "CS"
        N = len(y)
        if freq_range is None:
            freq_range = (0, fs / 2)
        if resolution is None:
            resolution = int(4 * np.diff(freq_range)[0] * N / fs)
        if phase is None:
            phase = np.arange(N)
        self.freqs = np.linspace(freq_range[0], freq_range[1], resolution)

        if method == "sklearn":
            A = 2 * np.pi * phase.reshape(-1, 1) @ self.freqs.reshape(1, -1) / fs
            A = np.concatenate((np.sin(A), np.cos(A)), axis=1)
            model = Lasso(alpha=alpha)
            model.fit(A, y)
            self.spec = np.sqrt(
                model.coef_[:resolution] ** 2 + model.coef_[resolution:] ** 2
            )
            self.spec = abs(self.spec) / np.max(abs(self.spec))
        elif method == "cvxpy":
            A = np.exp(
                1j * 2 * np.pi * phase.reshape(-1, 1) @ self.freqs.reshape(1, -1) / fs
            )
            coef = cp.Variable(resolution, complex=True)
            objective = cp.Minimize(
                cp.norm2(A @ coef - y) ** 2 + alpha * 1000 * cp.norm1(coef)
            )
            problem = cp.Problem(objective)
            problem.solve(solver=cp.SCS, warm_start=True)
            self.spec = np.abs(coef.value)
            self.spec = abs(self.spec) / np.max(abs(self.spec))
        else:
            print("Chosen method is not available! Please use 'sklearn' or 'cvxpy'.")
        if true_peaks is not None:
            self.error = peak_rmse(self.freqs, self.spec, true_peaks)

    def rmse(self, true_peaks):
        """Calculate the RMSE between the pseudospectrum resulted from analysis and the true peaks of the simulated model.

        Parameters
        ----------
        true_peaks : ndarray of shape (num_peaks,)
            The correct peaks in the simulated signal model.

        Returns
        -------
        rmse : float
            RMSE between the known correct peaks in the simulated model and the peaks detected by analysis.
        """
        self.error = peak_rmse(self.freqs, self.spec, true_peaks)
        return self.error


class WSF:
    """Perform WSF (Weighted Subspace Fitting) analysis on the signal.

    Parameters
    ----------
    y : ndarray of shape (signal_length,)
        The signal to be analyzed

    fs : float, default=2*np.pi
        The sampling rate of the signal.

    resolution : int
        The resolution of the analysis result, i.e., the size of the frequency domain array.
        This is defaulted to 4 times the original FFT resolution.

    freq_range : array_like of shape (2,), default=(0, fs/2)
        Range of frequency to analyze.

    num_sources : int, default=None
        Number of sources for analysis. If the number of sources is not given, it is selected automatically from the eigenvalue distribution.

    R : ndarray of shape (M, M), default=None
        Precalculated covariance matrix for the signal. If not given, the covariance matrix is calculated base on the covariance matrix size parameter M.

    M : int, default=200
        Size of the covariance matrix. Only used if R=None.

    true_peaks : ndarray of shape (num_peaks,)
        The correct peaks in the simulated signal model.

    Attributes
    ----------
    freqs : ndarray of shape (resolution,)
        The frequency axis of the spectrum.

    spec : ndarray of shape (resolution,)
        The normalized magnitude spectrum corresponding to the frequencies in **freqs**.

    Functions
    ---------
    rmse(true_peaks)
        Calculate the RMSE between the known correct peaks in the simulated model and the peaks detected by analysis.
    """

    def __init__(
        self,
        y,
        fs=2 * np.pi,
        resolution=None,
        freq_range=None,
        num_sources=None,
        R=None,
        M=200,
        true_peaks=None,
    ):
        self.name = "WSF"
        N = len(y)
        if R is None:
            if M > N:
                M = N
            R = cov_matrix(y, M)
        else:
            assert np.diff(R.shape) == 0, "Covariance matrix is not square!"
            M = R.shape[0]
        if freq_range is None:
            freq_range = (0, fs / 2)
        if resolution is None:
            resolution = int(4 * np.diff(freq_range)[0] * N / fs)
        self.freqs = np.linspace(freq_range[0], freq_range[1], resolution)

        eigvals, eigvecs = np.linalg.eig(R)
        eigvals = abs(eigvals)
        idx = np.argsort(eigvals)[::-1]
        eigvals_sort = np.sort(eigvals)[::-1]
        if num_sources is None:
            num_sources = len(np.where(eigvals > np.mean(eigvals))[0])
        signal_subspace = eigvecs[:, idx[:num_sources]]

        sigma_squared = np.mean(eigvals_sort[num_sources:])
        W_opt = np.diag(
            (eigvals_sort[:num_sources] - sigma_squared) ** 2
            / eigvals_sort[:num_sources]
        )
        A = np.exp(-1j * 2 * np.pi * np.outer(np.arange(M), self.freqs / fs))

        self.spec = np.zeros_like(self.freqs)
        for i in range(resolution):
            Pa = np.outer(A[:, i], A[:, i].conj()) / np.linalg.norm(A[:, i])
            self.spec[i] = np.real(
                np.trace(Pa @ signal_subspace @ W_opt @ signal_subspace.conj().T)
            )
        self.spec -= np.median(self.spec)
        self.spec = abs(self.spec) / np.max(abs(self.spec))
        if true_peaks is not None:
            self.error = peak_rmse(self.freqs, self.spec, true_peaks)

    def rmse(self, true_peaks):
        """Calculate the RMSE between the pseudospectrum resulted from analysis and the true peaks of the simulated model.

        Parameters
        ----------
        true_peaks : ndarray of shape (num_peaks,)
            The correct peaks in the simulated signal model.

        Returns
        -------
        rmse : float
            RMSE between the known correct peaks in the simulated model and the peaks detected by analysis.
        """
        self.error = peak_rmse(self.freqs, self.spec, true_peaks)
        return self.error


class IAA:
    """Perform IAA (Iterative Adaptive Approach) analysis on the signal.

    Parameters
    ----------
    y : ndarray of shape (signal_length,)
        The signal to be analyzed

    fs : float, default=2*np.pi
        The sampling rate of the signal.

    n_iter : int, default=15
        Number of iterations. 10-15 iterations should be enough for convergence.

    resolution : int
        The resolution of the analysis result, i.e., the size of the frequency domain array.
        This is defaulted to 4 times the original FFT resolution.

    phase : ndarray of shape (signal_length,)
        Reference phase for signal with inconstant but proportional frequencies. The reference phase is chosen so that the initial frequency is 1Hz.

    true_peaks : ndarray of shape (num_peaks,)
        The correct peaks in the simulated signal model.

    Attributes
    ----------
    freqs : ndarray of shape (resolution,)
        The frequency axis of the spectrum.

    spec : ndarray of shape (resolution,)
        The normalized magnitude spectrum corresponding to the frequencies in **freqs**.

    R : ndarray of shape (signal_length, signal_length)
        Full covariance matrix of the signall computed by IAA.

    Functions
    ---------
    rmse(true_peaks)
        Calculate the RMSE between the known correct peaks in the simulated model and the peaks detected by analysis.
    """

    def __init__(
        self, y, fs=2 * np.pi, n_iter=15, resolution=None, phase=None, true_peaks=None
    ):
        self.name = "IAA"
        N = len(y)
        if resolution is None:
            resolution = N
        if phase is None:
            phase = np.arange(N)
        self.freqs = np.linspace(0, fs, resolution)

        A = np.exp(
            1j * 2 * np.pi * phase.reshape(-1, 1) @ self.freqs.reshape(1, -1) / fs
        )
        coef = A.conj().T @ y / N

        for _ in range(n_iter):
            R = A @ np.diag(np.abs(coef) ** 2) @ A.conj().T
            coef = A.conj().T @ np.linalg.solve(R, y) * np.abs(coef) ** 2
        self.spec = np.abs(coef) / np.max(np.abs(coef))
        self.R = R
        if true_peaks is not None:
            self.error = peak_rmse(self.freqs, self.spec, true_peaks)

    def rmse(self, true_peaks):
        """Calculate the RMSE between the pseudospectrum resulted from analysis and the true peaks of the simulated model.

        Parameters
        ----------
        true_peaks : ndarray of shape (num_peaks,)
            The correct peaks in the simulated signal model.

        Returns
        -------
        rmse : float
            RMSE between the known correct peaks in the simulated model and the peaks detected by analysis.
        """
        self.error = peak_rmse(self.freqs, self.spec, true_peaks)
        return self.error
