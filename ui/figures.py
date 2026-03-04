import numpy as np
import plotly.graph_objects as go
from scipy.signal import hilbert
from openconmo.benchmark_methods import envelope, cepstrum_prewhitening, benchmark_method
import dash_mantine_components as dmc

dmc.add_figure_templates()

def create_dummy_figure(title):
    return {
        'data': [],
        'layout': {
            'title': title,
            'template': 'mantine_light',
            'height': 300
        }
    }

def create_time_series_plot(signal, fs, title="Time Series", unit=""):
    time = np.arange(len(signal)) / fs
    fig = {
        'data': [{
            'x': time,
            'y': signal,
            'type': 'scatter',
            'mode': 'lines',
            'name': 'Signal'
        }],
        'layout': {
            'title': title,
            'template': 'mantine_light',
            'xaxis': {'title': {'text': 'Time (s)'}},
            'yaxis': {'title': {'text': f'Amplitude ({unit})'}},
            'margin': {'l': 60, 'r': 30, 't': 50, 'b': 50},
            'hovermode': 'x unified',
            'height': 350
        }
    }
    return fig

def create_frequency_domain_plot(signal, fs, title="Frequency Spectrum", unit=""):
    n = len(signal)
    signal = signal - np.mean(signal)
    fft = np.fft.rfft(signal)
    freq = np.fft.rfftfreq(n, d=1/fs)
    magnitude = 2/n * np.abs(fft)
    
    fig = {
        'data': [{
            'x': freq,
            'y': magnitude,
            'type': 'scatter',
            'mode': 'lines',
            'name': 'Spectrum',
            'hovertemplate': 'Freq: %{x:.2f} Hz<br>Mag: %{y:.2e} ' + unit + '<extra></extra>'
        }],
        'layout': {
            'title': title,
            'template': 'mantine_light',
            'xaxis': {'title': {'text': 'Frequency (Hz)'}},
            'yaxis': {'title': {'text': f'Magnitude ({unit})'}},
            'margin': {'l': 60, 'r': 30, 't': 50, 'b': 50},
            'hovermode': 'x unified'
        }
    }
    return fig

def create_envelope_spectrum_plot(signal, fs, title="Envelope Spectrum", unit=""):
    signal = signal - np.mean(signal)
    analytic_signal = hilbert(signal)
    env = np.abs(analytic_signal)
    env = env - np.mean(env)
    n = len(env)
    fft = np.fft.rfft(env)
    freq = np.fft.rfftfreq(n, d=1/fs)
    magnitude = 2/n * np.abs(fft)
    
    fig = {
        'data': [{
            'x': freq,
            'y': magnitude,
            'type': 'scatter',
            'mode': 'lines',
            'name': 'Envelope',
            'hovertemplate': 'Freq: %{x:.2f} Hz<br>Mag: %{y:.2e} ' + unit + '<extra></extra>'
        }],
        'layout': {
            'title': title,
            'template': 'mantine_light',
            'xaxis': {'title': {'text': 'Frequency (Hz)'}},
            'yaxis': {'title': {'text': f'Envelope Mag ({unit})'}},
            'margin': {'l': 60, 'r': 30, 't': 50, 'b': 50},
            'hovermode': 'x unified'
        }
    }
    return fig

def squared_envelope_plot(signal, fs, title="Squared Envelope Spectrum", unit=""):
    f, X = envelope(signal, fs)
    fig = {
        'data': [{
            'x': f,
            'y': X,
            'type': 'scatter',
            'mode': 'lines',
            'name': 'Envelope',
            'hovertemplate': 'Freq: %{x:.2f} Hz<br>Mag: %{y:.2e} ' + unit + '<extra></extra>',
            'showlegend': False
        }],
        'layout': {
            'title': title,
            'template': 'mantine_light',
            'xaxis': {'title': {'text': 'Frequency (Hz)'}},
            'yaxis': {'title': {'text': f'Envelope Mag ({unit})'}},
            'margin': {'l': 60, 'r': 30, 't': 50, 'b': 50},
            'hovermode': 'x unified'
    }}
    return fig
    
def cepstrum_prewhitening_plot(signal, fs, title="Cepstrum Prewhitening", unit=""):
    _, time_signal = cepstrum_prewhitening(signal, fs)
    s_mid = time_signal[2000:-2000]
    f, X = envelope(s_mid, fs)
    fig = {
        'data': [{
            'x': f,
            'y': X,
            'type': 'scatter',
            'mode': 'lines',
            'name': 'Cepstrum',
            'hovertemplate': 'Freq: %{x:.2f} Hz<br>Mag: %{y:.2e} ' + unit + '<extra></extra>',
            'showlegend': False
        }],
        'layout': {
            'title': title,
            'template': 'mantine_light',
            'xaxis': {'title': {'text': 'Frequency (Hz)'}},
            'yaxis': {'title': {'text': f'Envelope Mag ({unit})'}},
            'margin': {'l': 60, 'r': 30, 't': 50, 'b': 50},
            'hovermode': 'x unified'
    }}
    return fig

def benchmark_plot(signal, fs, title="Benchmark Method", unit=""):
    t, s, f, X = benchmark_method(signal, fs, N=8192, Delta=500)
    fig = {
        'data': [{
            'x': f,
            'y': X,
            'type': 'scatter',
            'mode': 'lines',
            'name': 'Benchmark',
            'hovertemplate': 'Freq: %{x:.2f} Hz<br>Mag: %{y:.2e} ' + unit + '<extra></extra>',
            'showlegend': False
        }],
        'layout': {
            'title': title,
            'template': 'mantine_light',
            'xaxis': {'title': {'text': 'Frequency (Hz)'}},
            'yaxis': {'title': {'text': f'Envelope Mag ({unit})'}},
            'margin': {'l': 60, 'r': 30, 't': 50, 'b': 50},
            'hovermode': 'x unified'
    }}
    return fig

def add_sideband_lines(plot, harmonic_freq, f_sb_hz):
    """Add sideband lines around a given harmonic frequency on the plot."""
    ymax = max(plot['data'][0]['y']) if plot.get('data') and plot['data'][0].get('y') is not None else 1

    sideband_legend_already = any(
        (trace.get('name') == '1st-order sidebands at ± SB') and trace.get('showlegend')
        for trace in plot.get('data', [])
    )

    for idx, offset in enumerate([-f_sb_hz, f_sb_hz]):
        freq = harmonic_freq + offset

        showlegend = (not sideband_legend_already) and (idx == 0)
        name = '1st-order sidebands at ± SB' if showlegend else None

        plot['data'].append({
            'x': [freq, freq],
            'y': [0, ymax],
            'type': 'scatter',
            'mode': 'lines',
            'line': {'color': 'red', 'dash': 'dot', 'width': 1},
            'name': name,
            'yaxis': 'y',
            'hoverinfo': 'name',
            'showlegend': showlegend
        })


def add_harmonic_lines(plot, ff_hz, n_harmonics, rotating_freq_hz, f_sb_hz):
    """Overlay harmonic and optional sideband lines on a frequency-domain plot."""
    if not (ff_hz and n_harmonics and ff_hz > 0):
        return

    rotating_freq_hz = rotating_freq_hz if rotating_freq_hz else 1

    for i in range(1, n_harmonics + 1):
        harmonic_freq = ff_hz * i * rotating_freq_hz
        ymax = max(plot['data'][0]['y']) if plot.get('data') and plot['data'][0].get('y') is not None else 1

        if i == 1:
            name = 'Harmonics of expected FF'
            showlegend = True
        else:
            name = ''
            showlegend = False

        plot['data'].append({
            'x': [harmonic_freq, harmonic_freq],
            'y': [0, ymax],
            'type': 'scatter',
            'mode': 'lines',
            'line': {'color': 'red', 'dash': 'dash', 'width': 1},
            'name': name,
            'yaxis': 'y',
            'hoverinfo': 'name',
            'showlegend': showlegend
        })

        if f_sb_hz and f_sb_hz > 0:
            add_sideband_lines(plot, harmonic_freq, f_sb_hz)


def update_axis_ranges(plot, x_lim_1, x_lim_2, y_lim_1, y_lim_2, freq_scale, amp_scale):
    """Adjust axis ranges and scaling types based on user input."""
    if x_lim_1 is not None and x_lim_2 is not None and x_lim_1 < x_lim_2:
        if freq_scale == 'log':
            plot['layout']['xaxis']['range'] = [
                np.log10(max(1e-10, x_lim_1)),
                np.log10(max(1e-10, x_lim_2)),
            ]
        else:
            plot['layout']['xaxis']['range'] = [x_lim_1, x_lim_2]

    if y_lim_1 is not None and y_lim_2 is not None and y_lim_1 < y_lim_2:
        if amp_scale == 'log':
            plot['layout']['yaxis']['range'] = [
                np.log10(max(1e-10, y_lim_1)),
                np.log10(max(1e-10, y_lim_2)),
            ]
        else:
            plot['layout']['yaxis']['range'] = [y_lim_1, y_lim_2]


def add_cursors_to_fig(fig, ff_hz, n_harmonics, rot_freq, sb_hz):
    """Backward-compatible wrapper for harmonic and sideband overlays."""
    add_harmonic_lines(fig, ff_hz, n_harmonics, rot_freq, sb_hz)
    return fig