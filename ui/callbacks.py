from figures import (
    create_time_series_plot,
    create_dummy_figure,
    squared_envelope_plot,
    cepstrum_prewhitening_plot,
    benchmark_plot,
    add_harmonic_lines,
    update_axis_ranges,
)
from utils import read_from_parquet

from dash.dependencies import Input, Output, State
from dash import html
import dash_mantine_components as dmc
import numpy as np

def register_callbacks(app):
    '''Register all Dash app callbacks for plot updates and metadata display.'''
    
    @app.callback(
        [Output('time-plot', 'figure'),
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
        
        if contents is None:
            dummy_fig = create_dummy_figure("Upload a file")
            return dummy_fig, dummy_fig
        
        try:
            signal, fs, name, loc, unit, meas_id, fault, fault_freqs, rot_freq = read_from_parquet(contents)
            
            start_idx = max(0, int(time_start * fs))
            stop_idx = min(len(signal), int(time_stop * fs))
            signal_slice = signal[start_idx:stop_idx]
            
            title_upper = f"Time Domain - {name} - {loc} (ID: {meas_id})"
            title_env = f"Envelope Spectrum - {name} - {loc} (ID: {meas_id})"
            
            upper_plot = create_time_series_plot(signal_slice, fs, title=title_upper, unit=unit)
            
            if dropdown_value == "1":
                env_plot = squared_envelope_plot(signal_slice, fs, title=title_env, unit=unit)
            elif dropdown_value == "2":
                env_plot = cepstrum_prewhitening_plot(signal_slice, fs)
            else:
                env_plot = benchmark_plot(signal_slice, fs)

            for plot in [env_plot]:
                plot['layout']['xaxis']['type'] = freq_scale
                plot['layout']['yaxis']['type'] = amp_scale
                add_harmonic_lines(plot, ff_hz, n_harmonics, rot_freq, f_sb_hz)
                update_axis_ranges(plot, x_lim_1, x_lim_2, y_lim_1, y_lim_2, freq_scale, amp_scale)

                if ff_hz and n_harmonics and ff_hz > 0:
                    plot['layout']['showlegend'] = True
                    plot['layout']['legend'] = {
                        'orientation': 'h',
                        'yanchor': 'top',
                        'y': -0.5,
                        'xanchor': 'center',
                        'x': 0.5
                    }

            return upper_plot, env_plot
            
        except Exception as e:
            print(f"Error reading file: {str(e)}")
            error_fig = create_dummy_figure(f"Error: {str(e)}")
            return error_fig, error_fig

    @app.callback(
        Output("bearing-fault-results", "children"),
        Input("bearing-calc-btn", "n_clicks"),
        State("bearing-speed-rpm", "value"),
        State("bearing-n-rollers", "value"),
        State("bearing-ball-d-mm", "value"),
        State("bearing-pitch-d-mm", "value"),
        State("bearing-contact-angle-deg", "value"),
        prevent_initial_call=True,
    )
    def calculate_bearing_faults(n_clicks, rpm, n, d_mm, D_mm, theta_deg):
        if rpm is None or n is None or d_mm is None or D_mm is None or theta_deg is None:
            return dmc.Alert("Please fill all bearing inputs before calculating.", color="yellow", title="Missing Input")

        if rpm <= 0 or n < 1 or d_mm <= 0 or D_mm <= 0:
            return dmc.Alert(
                "Shaft speed, number of rolling elements, and diameters must be positive.",
                color="yellow",
                title="Invalid Input",
            )

        if d_mm >= D_mm:
            return dmc.Alert(
                "Rolling element diameter d must be smaller than pitch diameter D.",
                color="red",
                title="Invalid Bearing Geometry",
            )

        fr = rpm / 60.0
        ratio = (d_mm / D_mm) * np.cos(np.radians(theta_deg))
        
        return dmc.SimpleGrid(
            cols=2,
            spacing="xs",
            children=[
                dmc.Paper([dmc.Text("FTF", fw=700, size="xs"), dmc.Text(f"{0.5*fr*(1-ratio):.3f} Hz")], p="sm", withBorder=True),
                dmc.Paper([dmc.Text("BPFO", fw=700, size="xs"), dmc.Text(f"{0.5*n*fr*(1-ratio):.3f} Hz")], p="sm", withBorder=True),
                dmc.Paper([dmc.Text("BPFI", fw=700, size="xs"), dmc.Text(f"{0.5*n*fr*(1+ratio):.3f} Hz")], p="sm", withBorder=True),
                dmc.Paper([dmc.Text("BSF", fw=700, size="xs"), dmc.Text(f"{(D_mm/(2*d_mm))*fr*(1-ratio**2):.3f} Hz")], p="sm", withBorder=True),
            ],
        )

    @app.callback(
        Output('metadata-display', 'children'),
        Input('upload-data', 'contents')
    )
    def update_metadata(contents):
        if contents is None:
            return dmc.Text("No file uploaded yet", c="dimmed")
        
        try:
            signal, fs, name, loc, unit, m_id, fault, f_freqs, rot_freq = read_from_parquet(contents)
            
            return dmc.Table(
                striped=True,
                highlightOnHover=True,
                withTableBorder=True,
                children=[
                    dmc.TableTbody([
                        dmc.TableTr([dmc.TableTd("Name:"), dmc.TableTd(name)]),
                        dmc.TableTr([dmc.TableTd("Meas location:"), dmc.TableTd(loc)]),
                        dmc.TableTr([dmc.TableTd("Sampling Rate:"), dmc.TableTd(f"{fs} Hz")]),
                        dmc.TableTr([
                            dmc.TableTd("Fault Frequencies:"), 
                            dmc.TableTd(
                                dmc.Stack([
                                    dmc.Text(f"{f_type}: {val:.2f}")
                                    for _, freqs in f_freqs.items() 
                                    for f_type, val in freqs.items()
                                ], gap="xs")
                            )
                        ])
                    ])
                ]
            )
        except Exception as e:
            return dmc.Alert(f"Error: {str(e)}", color="red", title="Upload Error")