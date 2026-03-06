import dash_mantine_components as dmc
import numpy as np
from scipy.signal import hilbert

from openconmo.benchmark_methods import (
    benchmark_method,
    cepstrum_prewhitening,
    envelope,
)

dmc.add_figure_templates()


def create_dummy_figure(title):
    return {
        "data": [],
        "layout": {"title": title, "template": "mantine_light", "height": 300},
    }


def create_time_series_plot(signal, fs, title="Time Series", unit=""):
    time = np.arange(len(signal)) / fs
    fig = {
        "data": [
            {
                "x": time,
                "y": signal,
                "type": "scatter",
                "mode": "lines",
                "name": "Signal",
                "showlegend": False,
            }
        ],
        "layout": {
            "title": title,
            "template": "mantine_light",
            "xaxis": {"title": {"text": "Time (s)"}},
            "yaxis": {"title": {"text": "Amplitude"}},
            "margin": {"l": 60, "r": 30, "t": 50, "b": 50},
            "hovermode": "x unified",
            "height": 350,
        },
    }
    return fig


def create_frequency_domain_plot(
    signal, fs, title="Frequency Spectrum", unit=""
):
    n = len(signal)
    signal = signal - np.mean(signal)
    fft = np.fft.rfft(signal)
    freq = np.fft.rfftfreq(n, d=1 / fs)
    magnitude = 2 / n * np.abs(fft)

    fig = {
        "data": [
            {
                "x": freq,
                "y": magnitude,
                "type": "scatter",
                "mode": "lines",
                "name": "Spectrum",
                "hovertemplate": "Freq: %{x:.2f} Hz<br>Mag: %{y:.2e} "
                + unit
                + "<extra></extra>",
            }
        ],
        "layout": {
            "title": title,
            "template": "mantine_light",
            "xaxis": {"title": {"text": "Frequency (Hz)"}},
            "yaxis": {"title": {"text": "Magnitude"}},
            "margin": {"l": 60, "r": 30, "t": 50, "b": 50},
            "hovermode": "x unified",
        },
    }
    return fig


def create_envelope_spectrum_plot(
    signal, fs, title="Envelope Spectrum", unit=""
):
    signal = signal - np.mean(signal)
    analytic_signal = hilbert(signal)
    env = np.abs(analytic_signal)
    env = env - np.mean(env)
    n = len(env)
    fft = np.fft.rfft(env)
    freq = np.fft.rfftfreq(n, d=1 / fs)
    magnitude = 2 / n * np.abs(fft)

    fig = {
        "data": [
            {
                "x": freq,
                "y": magnitude,
                "type": "scatter",
                "mode": "lines",
                "name": "Envelope",
                "hovertemplate": "Freq: %{x:.2f} Hz<br>Mag: %{y:.2e} "
                + unit
                + "<extra></extra>",
            }
        ],
        "layout": {
            "title": title,
            "template": "mantine_light",
            "xaxis": {"title": {"text": "Frequency (Hz)"}},
            "yaxis": {"title": {"text": "Envelope Mag"}},
            "margin": {"l": 60, "r": 30, "t": 50, "b": 50},
            "hovermode": "x unified",
        },
    }
    return fig


def squared_envelope_plot(
    signal, fs, title="Squared Envelope Spectrum", unit=""
):
    f, X = envelope(signal, fs)
    fig = {
        "data": [
            {
                "x": f,
                "y": X,
                "type": "scatter",
                "mode": "lines",
                "name": "Envelope",
                "hovertemplate": "Freq: %{x:.2f} Hz<br>Mag: %{y:.2e} "
                + unit
                + "<extra></extra>",
                "showlegend": False,
            }
        ],
        "layout": {
            "title": title,
            "template": "mantine_light",
            "xaxis": {"title": {"text": "Frequency (Hz)"}},
            "yaxis": {"title": {"text": "Envelope Mag"}},
            "margin": {"l": 60, "r": 30, "t": 50, "b": 50},
            "hovermode": "x unified",
        },
    }
    return fig


def cepstrum_prewhitening_plot(
    signal, fs, title="Cepstrum Prewhitening", unit=""
):
    _, time_signal = cepstrum_prewhitening(signal, fs)
    s_mid = time_signal[2000:-2000]
    f, X = envelope(s_mid, fs)
    fig = {
        "data": [
            {
                "x": f,
                "y": X,
                "type": "scatter",
                "mode": "lines",
                "name": "Cepstrum",
                "hovertemplate": "Freq: %{x:.2f} Hz<br>Mag: %{y:.2e} "
                + unit
                + "<extra></extra>",
                "showlegend": False,
            }
        ],
        "layout": {
            "title": title,
            "template": "mantine_light",
            "xaxis": {"title": {"text": "Frequency (Hz)"}},
            "yaxis": {"title": {"text": "Envelope Mag"}},
            "margin": {"l": 60, "r": 30, "t": 50, "b": 50},
            "hovermode": "x unified",
        },
    }
    return fig


def benchmark_plot(signal, fs, title="Benchmark Method", unit=""):
    t, s, f, X = benchmark_method(signal, fs, N=8192, Delta=500)
    fig = {
        "data": [
            {
                "x": f,
                "y": X,
                "type": "scatter",
                "mode": "lines",
                "name": "Benchmark",
                "hovertemplate": "Freq: %{x:.2f} Hz<br>Mag: %{y:.2e} "
                + unit
                + "<extra></extra>",
                "showlegend": False,
            }
        ],
        "layout": {
            "title": title,
            "template": "mantine_light",
            "xaxis": {"title": {"text": "Frequency (Hz)"}},
            "yaxis": {"title": {"text": "Envelope Mag"}},
            "margin": {"l": 60, "r": 30, "t": 50, "b": 50},
            "hovermode": "x unified",
        },
    }
    return fig


def add_sideband_lines(plot, harmonic_freq, f_sb_hz):
    """Add sideband lines around a given harmonic frequency on the plot."""
    ymax = (
        max(plot["data"][0]["y"])
        if plot.get("data") and plot["data"][0].get("y") is not None
        else 1
    )

    sideband_legend_already = any(
        (trace.get("name") == "1st-order sidebands at ± SB")
        and trace.get("showlegend")
        for trace in plot.get("data", [])
    )

    for idx, offset in enumerate([-f_sb_hz, f_sb_hz]):
        freq = harmonic_freq + offset

        showlegend = (not sideband_legend_already) and (idx == 0)
        name = "1st-order sidebands at ± SB" if showlegend else None

        plot["data"].append(
            {
                "x": [freq, freq],
                "y": [0, ymax],
                "type": "scatter",
                "mode": "lines",
                "line": {"color": "red", "dash": "dot", "width": 1},
                "name": name,
                "legendgroup": "cursor-overlay",
                "yaxis": "y",
                "hoverinfo": "name",
                "showlegend": showlegend,
            }
        )


def _interpolate_spectrum_y(plot, freq_hz):
    """Return interpolated spectrum amplitude at a given frequency."""
    if not plot.get("data"):
        return None

    x = np.asarray(plot["data"][0].get("x", []), dtype=float)
    y = np.asarray(plot["data"][0].get("y", []), dtype=float)
    if x.size < 2 or y.size < 2:
        return None

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 2:
        return None

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    if freq_hz < x[0] or freq_hz > x[-1]:
        return None

    return float(np.interp(freq_hz, x, y))


def add_time_period_cursor(plot, ff_hz, cursor_center_s):
    """Overlay repeating red dashed time-cursor lines with spacing = 1/FF."""
    if not plot or not plot.get("data"):
        return

    if not ff_hz or ff_hz <= 0:
        return

    plot.setdefault("layout", {})
    plot["layout"]["legend"] = {
        "orientation": "h",
        "yanchor": "top",
        "y": -0.25,
        "xanchor": "center",
        "x": 0.5,
    }
    margin = plot["layout"].setdefault("margin", {})
    margin["b"] = max(margin.get("b", 50), 90)

    x = np.asarray(plot["data"][0].get("x", []), dtype=float)
    y = np.asarray(plot["data"][0].get("y", []), dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if x.size < 2 or y.size < 2:
        return

    period_s = 1.0 / float(ff_hz)
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    y_min = float(np.min(y))
    y_max = float(np.max(y))

    if x_max <= x_min:
        return

    # Keep period visible inside current time window.
    max_period = max(x_max - x_min, 1e-12)
    period_s = min(period_s, max_period)

    default_center = x_min + 0.5 * period_s
    center = (
        float(cursor_center_s)
        if cursor_center_s is not None
        else default_center
    )

    # Normalize anchor phase into visible range so wheel movement can shift
    # phase freely without being pinned by edge clamping.
    if period_s > 0:
        center = x_min + ((center - x_min) % period_s)

    # Repeat this cursor window every period, left and right, to saturate plot.
    # Clamp number of windows for performance when period is very small.
    n_left = int(np.ceil((center - x_min) / period_s)) + 2
    n_right = int(np.ceil((x_max - center) / period_s)) + 2
    max_rects = 400
    if n_left + n_right + 1 > max_rects:
        scale = max_rects / float(n_left + n_right + 1)
        n_left = int(np.floor(n_left * scale))
        n_right = int(np.floor(n_right * scale))

    x_lines = []
    y_lines = []
    for k in range(-n_left, n_right + 1):
        c = center + k * period_s
        left = max(c - 0.5 * period_s, x_min)
        right = min(c + 0.5 * period_s, x_max)
        if right <= left:
            continue

        # Left edge line
        x_lines.extend([left, left, None])
        y_lines.extend([y_min, y_max, None])

        # Right edge line
        x_lines.extend([right, right, None])
        y_lines.extend([y_min, y_max, None])

    if x_lines:
        plot["data"].append(
            {
                "x": x_lines,
                "y": y_lines,
                "type": "scatter",
                "mode": "lines",
                "line": {"color": "red", "dash": "dash", "width": 1},
                "name": "Time cursor windows (T = 1/FF)",
                "legendgroup": "time-cursor-overlay",
                "hoverinfo": "name",
                "showlegend": True,
            }
        )


def add_harmonic_lines(plot, ff_hz, n_harmonics, rotating_freq_hz, f_sb_hz):
    """Overlay harmonic and optional sideband lines on frequency plot."""
    if not (ff_hz and n_harmonics and ff_hz > 0):
        return

    # FF input is in Hz in UI, so harmonics are integer multiples of FF.
    # Keep rotating_freq_hz in signature for backward compatibility.

    marker_freqs = []

    for i in range(1, n_harmonics + 1):
        harmonic_freq = ff_hz * i
        marker_freqs.append(harmonic_freq)
        ymax = (
            max(plot["data"][0]["y"])
            if plot.get("data") and plot["data"][0].get("y") is not None
            else 1
        )

        if i == 1:
            name = "Harmonics of expected FF"
            showlegend = True
        else:
            name = ""
            showlegend = False

        plot["data"].append(
            {
                "x": [harmonic_freq, harmonic_freq],
                "y": [0, ymax],
                "type": "scatter",
                "mode": "lines",
                "line": {"color": "red", "dash": "dash", "width": 1},
                "name": name,
                "legendgroup": "cursor-overlay",
                "yaxis": "y",
                "hoverinfo": "name",
                "showlegend": showlegend,
            }
        )

        if f_sb_hz and f_sb_hz > 0:
            add_sideband_lines(plot, harmonic_freq, f_sb_hz)
            marker_freqs.extend(
                [harmonic_freq - f_sb_hz, harmonic_freq + f_sb_hz]
            )

    marker_x = []
    marker_y = []
    for freq in marker_freqs:
        y_val = _interpolate_spectrum_y(plot, freq)
        if y_val is not None:
            marker_x.append(freq)
            marker_y.append(y_val)

    if marker_x:
        plot["data"].append(
            {
                "x": marker_x,
                "y": marker_y,
                "type": "scatter",
                "mode": "markers",
                "name": "Selected cursor points",
                "legendgroup": "cursor-overlay",
                "marker": {
                    "color": "red",
                    "size": 8,
                    "symbol": "circle",
                    "line": {"color": "white", "width": 1},
                },
                "hovertemplate": (
                    "Cursor: %{x:.2f} Hz<br>Mag: %{y:.2e}<extra></extra>"
                ),
                "showlegend": True,
            }
        )


def update_axis_ranges(
    plot, x_lim_1, x_lim_2, y_lim_1, y_lim_2, freq_scale, amp_scale
):
    """Adjust axis ranges and scaling types based on user input."""
    if x_lim_1 is not None and x_lim_2 is not None and x_lim_1 < x_lim_2:
        if freq_scale == "log":
            plot["layout"]["xaxis"]["range"] = [
                np.log10(max(1e-10, x_lim_1)),
                np.log10(max(1e-10, x_lim_2)),
            ]
        else:
            plot["layout"]["xaxis"]["range"] = [x_lim_1, x_lim_2]

    if y_lim_1 is not None and y_lim_2 is not None and y_lim_1 < y_lim_2:
        if amp_scale == "log":
            plot["layout"]["yaxis"]["range"] = [
                np.log10(max(1e-10, y_lim_1)),
                np.log10(max(1e-10, y_lim_2)),
            ]
        else:
            plot["layout"]["yaxis"]["range"] = [y_lim_1, y_lim_2]


def add_cursors_to_fig(fig, ff_hz, n_harmonics, rot_freq, sb_hz):
    """Backward-compatible wrapper for harmonic and sideband overlays."""
    add_harmonic_lines(fig, ff_hz, n_harmonics, rot_freq, sb_hz)
    return fig
