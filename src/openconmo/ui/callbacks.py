from openconmo.benchmark_methods import envelope, cepstrum_prewhitening, benchmark_method
from openconmo.ui.figures import create_time_series_plot, create_frequency_domain_plot, create_dummy_figure, create_envelope_spectrum_plot
from openconmo.ui.utils import read_from_parquet

from dash.dependencies import Input, Output
from dash import html
import dash_mantine_components as dmc
import numpy as np

def register_callbacks(app):
    '''Register all Dash app callbacks for plot updates and metadata display.'''
    @app.callback(
        [Output('time-plot', 'figure'),
         Output('frequency-plot', 'figure'),
         Output('envelope-plot', 'figure')],
         
        [Input('upload-data', 'contents'),
         Input('dummy-dropdown-1', 'value'),
         Input('time-start', 'value'),
         Input('time-stop', 'value'),
         Input('x_lim_1', 'value'),
         Input('x_lim_2', 'value'),
         Input('y_lim_1', 'value'),
         Input('y_lim_2', 'value'),
         Input('ff_hz', 'value'),
         Input('n_harmonics', 'value'),
         Input('f_sb_hz', 'value'),
         Input('freq-scale', 'value'),
         Input('amp-scale', 'value')]
    )
    def update_plots(contents, dropdown_value, time_start, time_stop, 
                    x_lim_1, x_lim_2, y_lim_1, y_lim_2,
                    ff_hz, n_harmonics, f_sb_hz, freq_scale, amp_scale):
        '''Update time, frequency, and envelope plots based on uploaded data and user-selected parameters.'''

        if contents is None:
            dummy_fig = create_dummy_figure("Upload a file")
            return dummy_fig, dummy_fig, dummy_fig
        
        try:
            signal, fs, name, measurement_location, unit, meas_id, fault, fault_frequencies, rotating_freq_hz = read_from_parquet(contents)
            
            # Convert time to indices
            start_idx = max(0, int(time_start * fs))
            stop_idx = min(len(signal), int(time_stop * fs))
            
            # Slice the signal
            signal_slice = signal[start_idx:stop_idx]
            
            # Create plots
            title_upper = f"Time Domain - {name} - {measurement_location} (ID: {meas_id})"
            title_lower = f"Frequency Domain - {name} - {measurement_location} (ID: {meas_id})"
            title_envelope = f"Envelope Spectrum - {name} - {measurement_location} (ID: {meas_id})"
            
            upper_plot = create_time_series_plot(signal_slice, fs, title=title_upper, unit=unit)
            lower_plot = create_frequency_domain_plot(signal_slice, fs, title=title_lower, unit=unit)
            
            if dropdown_value == "1":
                print("Method 1\n")
            elif dropdown_value == "2":
                print("Method 2\n")
            elif dropdown_value == "3":
                print("Method 3\n")
            
            

            
            envelope_plot = create_envelope_spectrum_plot(signal_slice, fs, title=title_envelope, unit=unit)
            
            
            def add_harmonic_lines(plot, ff_hz, n_harmonics, rotating_freq_hz, f_sb_hz):
                '''Overlay harmonic and optional sideband lines on a frequency-domain plot.'''
                if not (ff_hz and n_harmonics and ff_hz > 0):
                    return

                for i in range(1, n_harmonics + 1):
                    harmonic_freq = ff_hz * i * rotating_freq_hz
                    
                    # Add harmonic line
                    plot['data'].append({
                        'x': [harmonic_freq, harmonic_freq],
                        'y': [0, 1],
                        'type': 'scatter',
                        'mode': 'lines',
                        'line': {'color': 'red', 'dash': 'dash', 'width': 1},
                        'name': f'{i}x {ff_hz} Hz',
                        'yaxis': 'y',
                        'hoverinfo': 'name',
                        'showlegend': True
                    })
                    
                    if f_sb_hz and f_sb_hz > 0:
                        add_sideband_lines(plot, harmonic_freq, f_sb_hz)

            def add_sideband_lines(plot, harmonic_freq, f_sb_hz):
                '''Add sideband lines around a given harmonic frequency on the plot.'''
                for offset, label in [(-f_sb_hz, 'SB-'), (f_sb_hz, 'SB+')]:
                    freq = harmonic_freq + offset
                    plot['data'].append({
                        'x': [freq, freq],
                        'y': [0, 1],
                        'type': 'scatter',
                        'mode': 'lines',
                        'line': {'color': 'green', 'dash': 'dot', 'width': 1},
                        'name': f'{label} {freq:.1f} Hz',
                        'yaxis': 'y',
                        'hoverinfo': 'name',
                        'showlegend': True
                    })

            def update_axis_ranges(plot, x_lim_1, x_lim_2, y_lim_1, y_lim_2, freq_scale, amp_scale):
                '''Adjust x and y axis ranges and scaling types based on user input.'''
                if x_lim_1 is not None and x_lim_2 is not None and x_lim_1 < x_lim_2:
                    if freq_scale == 'log':
                        plot['layout']['xaxis']['range'] = [np.log10(max(1e-10, x_lim_1)), np.log10(x_lim_2)]
                    else:
                        plot['layout']['xaxis']['range'] = [x_lim_1, x_lim_2]
                
                if y_lim_1 is not None and y_lim_2 is not None and y_lim_1 < y_lim_2:
                    if amp_scale == 'log':
                        plot['layout']['yaxis']['range'] = [np.log10(max(1e-10, y_lim_1)), np.log10(y_lim_2)]
                    else:
                        plot['layout']['yaxis']['range'] = [y_lim_1, y_lim_2]

            # Update scales and add harmonic lines for frequency plots
            for plot in [lower_plot, envelope_plot]:
                plot['layout']['xaxis']['type'] = freq_scale
                plot['layout']['yaxis']['type'] = amp_scale
                
                add_harmonic_lines(plot, ff_hz, n_harmonics, rotating_freq_hz, f_sb_hz)
                update_axis_ranges(plot, x_lim_1, x_lim_2, y_lim_1, y_lim_2, freq_scale, amp_scale)
                
                if ff_hz and n_harmonics and ff_hz > 0:
                    plot['layout']['showlegend'] = True

            return upper_plot, lower_plot, envelope_plot
            
        except Exception as e:
            print(f"Error reading file: {str(e)}")
            error_fig = create_dummy_figure(f"Error reading file: {str(e)}")
            return error_fig, error_fig, error_fig

    @app.callback(
        Output('metadata-display', 'children'),
        [Input('upload-data', 'contents')]
    )
    def update_metadata(contents):
        '''Display signal metadata from uploaded file in a formatted table.'''
        if contents is None:
            return [dmc.Text("No file uploaded yet", c="dimmed")]
        
        try:
            signal, fs, name, measurement_location, unit, meas_id, fault, fault_frequencies, rotating_freq_hz = read_from_parquet(contents)
            return [
                dmc.Table(
                    children=[
                        html.Tbody([
                            html.Tr([html.Td("Name:"), html.Td(name)]),
                            html.Tr([html.Td("Meas location:"), html.Td(measurement_location)]),
                            html.Tr([html.Td("Meas ID:"), html.Td(str(meas_id))]),
                            html.Tr([html.Td("Sampling Rate:"), html.Td(f"{fs} Hz")]),
                            html.Tr([html.Td("N samples:"), html.Td(f"{len(signal)}")]),
                            html.Tr([html.Td("Unit:"), html.Td(unit)]),
                            html.Tr([html.Td("Fault:"), html.Td(fault)]),
                            html.Tr([html.Td("Rotating Frequency:"), html.Td(f"{rotating_freq_hz:.2f} Hz")]),
                            html.Tr([
                                html.Td("Fault Frequencies:"), 
                                html.Td(
                                    dmc.Stack([
                                        dmc.Text(f"{type}: {val:.2f}")
                                        for _, freqs in fault_frequencies.items() 
                                        for type, val in freqs.items()
                                    ], justify="xs")
                                )
                            ])
                        ])
                    ],
                    striped=True,
                    highlightOnHover=True
                )
            ]
            
        except Exception as e:
            return [dmc.Alert(f"Error reading metadata: {str(e)}", c="red")] 