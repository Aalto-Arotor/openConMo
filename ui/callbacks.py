import copy

import dash_mantine_components as dmc
import numpy as np
from dash import html, no_update
from dash.dependencies import ALL, Input, Output, State
from figures import (
    add_harmonic_lines,
    add_time_period_cursor,
    benchmark_plot,
    cepstrum_prewhitening_plot,
    create_dummy_figure,
    create_time_series_plot,
    squared_envelope_plot,
    update_axis_ranges,
)
from utils import read_from_parquet


def register_callbacks(app):
    """Register all Dash app callbacks for
    plot updates and metadata display."""

    def _remove_cursor_overlays(fig):
        """Remove harmonic/sideband/marker overlay traces from figure."""
        if not fig or "data" not in fig:
            return
        fig["data"] = [
            trace
            for trace in fig["data"]
            if trace.get("legendgroup") != "cursor-overlay"
        ]

    def _remove_time_cursor_overlays(fig):
        """Remove time-window cursor overlay traces from a figure in-place."""
        if not fig or "data" not in fig:
            return
        fig["data"] = [
            trace
            for trace in fig["data"]
            if trace.get("legendgroup") != "time-cursor-overlay"
        ]

    def _normalize_method_value(value):
        """Normalize method selector value to one of: '1', '2', '3'."""
        if value is None:
            return "1"

        # Some component/state combinations may pass object-like values.
        if isinstance(value, dict):
            value = value.get("value", value.get("label", "1"))

        text = str(value).strip().lower()
        if text in {"1", "envelope"}:
            return "1"
        if text in {"2", "cepstrum prewhitening", "cepstrum-prewhitening"}:
            return "2"
        if text in {"3", "benchmark"}:
            return "3"
        return "1"

    @app.callback(
        Output("dummy-dropdown-1", "value"),
        Input("upload-data", "contents"),
        prevent_initial_call=True,
    )
    def reset_method_on_upload(contents):
        """Reset analysis method to Envelope for each new upload."""
        if contents is None:
            return no_update
        return "1"

    def _to_float(value, default):
        """Best-effort numeric coercion for callback values."""
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @app.callback(
        [
            Output("time-plot", "figure"),
            Output("envelope-plot", "figure"),
            Output("spectrum-x-hidden", "children"),
            Output("time-x-hidden", "children"),
        ],
        [
            Input("upload-data", "contents"),
            Input("dummy-dropdown-1", "value"),
            Input("time-start", "value"),
            Input("time-stop", "value"),
        ],
        [
            State("x_lim_1", "value"),
            State("x_lim_2", "value"),
            State("y_lim_1", "value"),
            State("y_lim_2", "value"),
            State("ff_hz", "value"),
            State("n_harmonics", "value"),
            State("f_sb_hz", "value"),
            State("t_cursor_s", "value"),
            State("freq-scale", "value"),
            State("amp-scale", "value"),
            State("time-plot", "figure"),
            State("envelope-plot", "figure"),
        ],
    )
    def update_plots(
        contents,
        dropdown_value,
        time_start,
        time_stop,
        x_lim_1,
        x_lim_2,
        y_lim_1,
        y_lim_2,
        ff_hz,
        n_harmonics,
        f_sb_hz,
        t_cursor_s,
        freq_scale,
        amp_scale,
        existing_time_plot,
        existing_env_plot,
    ):

        if contents is None:
            # Avoid overwriting already rendered plots with a dummy figure
            # during transient multi-callback ordering on upload/reset.
            if existing_time_plot is not None and existing_env_plot is not None:
                return no_update, no_update, no_update, no_update
            dummy_fig = create_dummy_figure("Upload a file")
            return dummy_fig, dummy_fig, None, None

        # Set reasonable defaults for None values
        # Normalize method value from UI (can arrive as None/int/str)
        # and default to Envelope on upload.
        dropdown_value = _normalize_method_value(dropdown_value)
        ff_hz = _to_float(ff_hz, 0.0)
        n_harmonics = int(_to_float(n_harmonics, 0.0))
        f_sb_hz = _to_float(f_sb_hz, 0.0)
        freq_scale = freq_scale or "linear"
        amp_scale = amp_scale or "linear"
        time_start = _to_float(time_start, 1.0)
        time_stop = _to_float(time_stop, 2.0)
        t_cursor_s = _to_float(t_cursor_s, 0.5)
        if time_stop <= time_start:
            time_stop = time_start + 1.0

        from dash import ctx

        triggered_id = ctx.triggered_id
        triggered_props = getattr(ctx, "triggered_prop_ids", {}) or {}
        upload_triggered = any(
            str(prop).startswith("upload-data.")
            for prop in triggered_props.keys()
        )

        # Always default to Envelope when a new file is uploaded.
        # Use triggered props (not only triggered_id), because Dash can
        # co-trigger multiple inputs on drop and report a non-upload id.
        if upload_triggered or triggered_id == "upload-data":
            dropdown_value = "1"

        try:
            (
                signal,
                fs,
                name,
                loc,
                unit,
                meas_id,
                fault,
                fault_freqs,
                rot_freq,
            ) = read_from_parquet(contents)

            start_idx = max(0, int(time_start * fs))
            stop_idx = min(len(signal), int(time_stop * fs))

            # Fallback to full signal when selected time window is empty
            # or too short for spectrum calculation.
            if stop_idx - start_idx < 2:
                start_idx = 0
                stop_idx = len(signal)

            signal_slice = signal[start_idx:stop_idx]

            title_upper = f"Time Domain - {name} - {loc} (ID: {meas_id})"
            title_env = f"Envelope Spectrum - {name} - {loc} (ID: {meas_id})"

            upper_plot = create_time_series_plot(
                signal_slice, fs, title=title_upper, unit=unit
            )
            add_time_period_cursor(upper_plot, ff_hz, t_cursor_s)

            if dropdown_value == "1":
                env_plot = squared_envelope_plot(
                    signal_slice, fs, title=title_env, unit=unit
                )
            elif dropdown_value == "2":
                env_plot = cepstrum_prewhitening_plot(signal_slice, fs)
            else:
                env_plot = benchmark_plot(signal_slice, fs)

            for plot in [env_plot]:
                _remove_cursor_overlays(plot)
                plot["layout"]["xaxis"]["type"] = freq_scale
                plot["layout"]["yaxis"]["type"] = amp_scale
                add_harmonic_lines(plot, ff_hz, n_harmonics, rot_freq, f_sb_hz)
                update_axis_ranges(
                    plot,
                    x_lim_1,
                    x_lim_2,
                    y_lim_1,
                    y_lim_2,
                    freq_scale,
                    amp_scale,
                )

                if ff_hz and n_harmonics and ff_hz > 0:
                    plot["layout"]["showlegend"] = True
                    plot["layout"]["legend"] = {
                        "orientation": "h",
                        "yanchor": "top",
                        "y": -0.5,
                        "xanchor": "center",
                        "x": 0.5,
                    }

            # Extract spectrum x values for client-side wheel calculations
            x_vals = (
                env_plot.get("data", [])[0].get("x", []) if env_plot else []
            )
            x_vals_list = [
                float(x)
                for x in x_vals
                if isinstance(x, (int, float)) and np.isfinite(x)
            ]

            # Interpolate for finer frequency resolution (10x more points)
            if len(x_vals_list) > 1:
                # Create interpolated array with 10x more points
                interpolated_x = np.interp(
                    np.linspace(
                        0, len(x_vals_list) - 1, len(x_vals_list) * 10
                    ),
                    np.arange(len(x_vals_list)),
                    x_vals_list,
                )
                x_vals_list = interpolated_x.tolist()

            # Store as JSON string in hidden div for JavaScript to read
            import json

            x_vals_json = json.dumps(x_vals_list)
            time_x_vals = (
                upper_plot.get("data", [])[0].get("x", [])
                if upper_plot
                else []
            )
            time_x_list = [
                float(t)
                for t in time_x_vals
                if isinstance(t, (int, float)) and np.isfinite(t)
            ]
            time_x_json = json.dumps(time_x_list)

            return upper_plot, env_plot, x_vals_json, time_x_json

        except Exception as e:
            print(f"Error reading file: {str(e)}")
            error_fig = create_dummy_figure(f"Error: {str(e)}")
            return error_fig, error_fig, None, None

    @app.callback(
        [
            Output("time-plot", "figure", allow_duplicate=True),
            Output("envelope-plot", "figure", allow_duplicate=True),
        ],
        [
            Input("ff_hz", "value"),
            Input("n_harmonics", "value"),
            Input("f_sb_hz", "value"),
            Input("t_cursor_s", "value"),
            Input("x_lim_1", "value"),
            Input("x_lim_2", "value"),
            Input("y_lim_1", "value"),
            Input("y_lim_2", "value"),
            Input("freq-scale", "value"),
            Input("amp-scale", "value"),
        ],
        [State("time-plot", "figure"), State("envelope-plot", "figure")],
        prevent_initial_call=True,
    )
    def update_plot_overlays(
        ff_hz,
        n_harmonics,
        f_sb_hz,
        t_cursor_s,
        x_lim_1,
        x_lim_2,
        y_lim_1,
        y_lim_2,
        freq_scale,
        amp_scale,
        existing_time_plot,
        existing_env_plot,
    ):
        if not existing_time_plot or not existing_env_plot:
            return no_update, no_update

        ff_hz = _to_float(ff_hz, 0.0)
        n_harmonics = int(_to_float(n_harmonics, 0.0))
        f_sb_hz = _to_float(f_sb_hz, 0.0)
        t_cursor_s = _to_float(t_cursor_s, 0.5)
        freq_scale = freq_scale or "linear"
        amp_scale = amp_scale or "linear"

        upper_plot = copy.deepcopy(existing_time_plot)
        env_plot = copy.deepcopy(existing_env_plot)

        _remove_time_cursor_overlays(upper_plot)
        add_time_period_cursor(upper_plot, ff_hz, t_cursor_s)

        _remove_cursor_overlays(env_plot)
        env_plot["layout"]["xaxis"]["type"] = freq_scale
        env_plot["layout"]["yaxis"]["type"] = amp_scale
        add_harmonic_lines(env_plot, ff_hz, n_harmonics, None, f_sb_hz)
        update_axis_ranges(
            env_plot,
            x_lim_1,
            x_lim_2,
            y_lim_1,
            y_lim_2,
            freq_scale,
            amp_scale,
        )

        if ff_hz and n_harmonics and ff_hz > 0:
            env_plot["layout"]["showlegend"] = True
            env_plot["layout"]["legend"] = {
                "orientation": "h",
                "yanchor": "top",
                "y": -0.5,
                "xanchor": "center",
                "x": 0.5,
            }
        else:
            env_plot["layout"]["showlegend"] = False

        return upper_plot, env_plot

    @app.callback(
        [
            Output("bearing-fault-results", "children"),
            Output("bearing-freq-store", "data"),
        ],
        Input("bearing-calc-btn", "n_clicks"),
        State("bearing-speed-rpm", "value"),
        State("bearing-n-rollers", "value"),
        State("bearing-ball-d-mm", "value"),
        State("bearing-pitch-d-mm", "value"),
        State("bearing-contact-angle-deg", "value"),
        prevent_initial_call=True,
    )
    def calculate_bearing_faults(n_clicks, rpm, n, d_mm, D_mm, theta_deg):
        if (
            rpm is None
            or n is None
            or d_mm is None
            or D_mm is None
            or theta_deg is None
        ):
            return (
                dmc.Alert(
                    "Please fill all bearing inputs before calculating.",
                    color="yellow",
                    title="Missing Input",
                ),
                None,
            )

        if rpm <= 0 or n < 1 or d_mm <= 0 or D_mm <= 0:
            msg = (
                "Shaft speed, number of rolling elements, "
                "and diameters must be positive."
            )
            return (
                dmc.Alert(msg, color="yellow", title="Invalid Input"),
                None,
            )

        if d_mm >= D_mm:
            msg = (
                "Rolling element diameter d must be smaller "
                "than pitch diameter D."
            )
            return (
                dmc.Alert(msg, color="red", title="Invalid Bearing Geometry"),
                None,
            )

        fr = rpm / 60.0
        ratio = (d_mm / D_mm) * np.cos(np.radians(theta_deg))

        ftf = 0.5 * fr * (1 - ratio)
        bpfo = 0.5 * n * fr * (1 - ratio)
        bpfi = 0.5 * n * fr * (1 + ratio)
        bsf = (D_mm / (2 * d_mm)) * fr * (1 - ratio**2)

        # Store the calculated values
        freq_data = {"ftf": ftf, "bpfo": bpfo, "bpfi": bpfi, "bsf": bsf}

        display = dmc.SimpleGrid(
            cols=2,
            spacing="xs",
            children=[
                html.Div(
                    dmc.Paper(
                        [
                            dmc.Text("FTF", fw=700, size="xs"),
                            dmc.Text(f"{ftf:.3f} Hz"),
                        ],
                        p="sm",
                        withBorder=True,
                        style={"cursor": "pointer"},
                    ),
                    id="ftf-click",
                    n_clicks=0,
                ),
                html.Div(
                    dmc.Paper(
                        [
                            dmc.Text("BPFO", fw=700, size="xs"),
                            dmc.Text(f"{bpfo:.3f} Hz"),
                        ],
                        p="sm",
                        withBorder=True,
                        style={"cursor": "pointer"},
                    ),
                    id="bpfo-click",
                    n_clicks=0,
                ),
                html.Div(
                    dmc.Paper(
                        [
                            dmc.Text("BPFI", fw=700, size="xs"),
                            dmc.Text(f"{bpfi:.3f} Hz"),
                        ],
                        p="sm",
                        withBorder=True,
                        style={"cursor": "pointer"},
                    ),
                    id="bpfi-click",
                    n_clicks=0,
                ),
                html.Div(
                    dmc.Paper(
                        [
                            dmc.Text("BSF", fw=700, size="xs"),
                            dmc.Text(f"{bsf:.3f} Hz"),
                        ],
                        p="sm",
                        withBorder=True,
                        style={"cursor": "pointer"},
                    ),
                    id="bsf-click",
                    n_clicks=0,
                ),
            ],
        )

        return display, freq_data

    @app.callback(
        [
            Output("metadata-display", "children"),
            Output("parquet-freq-store", "data"),
        ],
        Input("upload-data", "contents"),
    )
    def update_metadata(contents):
        if contents is None:
            return dmc.Text("No file uploaded yet", c="dimmed"), None

        try:
            signal, fs, name, loc, unit, m_id, fault, f_freqs, rot_freq = (
                read_from_parquet(contents)
            )
            meas_id_display = m_id if m_id is not None else "N/A"
            rot_freq_display = (
                f"{float(rot_freq):.3f}"
                if rot_freq is not None
                else "N/A"
            )

            # Create clickable fault frequency items and store their values
            freq_items = []
            freq_data = {}
            idx = 0

            for bearing_loc, freqs in f_freqs.items():
                for f_type, val in freqs.items():
                    freq_id = f"parquet-freq-{idx}"
                    freq_data[freq_id] = val
                    freq_items.append(
                        html.Div(
                            dmc.Text(
                                f"{f_type}: {val:.2f}",
                                size="sm",
                                style={
                                    "cursor": "pointer",
                                    "padding": "2px 0",
                                },
                            ),
                            id={"type": "parquet-freq", "index": idx},
                            n_clicks=0,
                        )
                    )
                    idx += 1

            return (
                dmc.Table(
                    striped=True,
                    highlightOnHover=True,
                    withTableBorder=True,
                    children=[
                        dmc.TableTbody(
                            [
                                dmc.TableTr(
                                    [dmc.TableTd("Name:"), dmc.TableTd(name)]
                                ),
                                dmc.TableTr(
                                    [
                                        dmc.TableTd("MAT file number:"),
                                        dmc.TableTd(str(meas_id_display)),
                                    ]
                                ),
                                dmc.TableTr(
                                    [
                                        dmc.TableTd("Meas location:"),
                                        dmc.TableTd(loc),
                                    ]
                                ),
                                dmc.TableTr(
                                    [
                                        dmc.TableTd("Rotating frequency:"),
                                        dmc.TableTd(f"{float(rot_freq):.3f} Hz"),
                                    ]
                                ),
                                dmc.TableTr(
                                    [
                                        dmc.TableTd("Sampling Rate:"),
                                        dmc.TableTd(f"{fs} Hz"),
                                    ]
                                ),
                                dmc.TableTr(
                                    [
                                        dmc.TableTd("Fault Frequencies:"),
                                        dmc.TableTd(html.Div(freq_items)),
                                    ]
                                ),
                            ]
                        )
                    ],
                ),
                freq_data,
            )
        except Exception as e:
            return (
                dmc.Alert(
                    f"Error: {str(e)}", color="red", title="Upload Error"
                ),
                None,
            )

    # Callbacks to update FF input when clicking on fault frequency results
    @app.callback(
        Output("ff_hz", "value"),
        [
            Input("ftf-click", "n_clicks"),
            Input("bpfo-click", "n_clicks"),
            Input("bpfi-click", "n_clicks"),
            Input("bsf-click", "n_clicks"),
        ],
        [State("bearing-freq-store", "data"), State("ff_hz", "value")],
        prevent_initial_call=True,
    )
    def update_ff_from_click(
        ftf_clicks, bpfo_clicks, bpfi_clicks, bsf_clicks, freq_data, current_ff
    ):
        """Update FF input when a fault frequency is clicked."""
        from dash import ctx

        if not ctx.triggered or not freq_data:
            return current_ff

        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

        # Map the triggered component to its corresponding value from store
        value_map = {
            "ftf-click": freq_data.get("ftf"),
            "bpfo-click": freq_data.get("bpfo"),
            "bpfi-click": freq_data.get("bpfi"),
            "bsf-click": freq_data.get("bsf"),
        }

        if triggered_id in value_map and value_map[triggered_id] is not None:
            return value_map[triggered_id]

        return current_ff

    # Callback to update FF input when clicking on parquet fault frequencies
    @app.callback(
        Output("ff_hz", "value", allow_duplicate=True),
        [Input({"type": "parquet-freq", "index": ALL}, "n_clicks")],
        [State("parquet-freq-store", "data"), State("ff_hz", "value")],
        prevent_initial_call=True,
    )
    def update_ff_from_parquet_click(n_clicks_list, freq_data, current_ff):
        """Update FF input when a parquet fault frequency is clicked."""
        from dash import ctx

        if not ctx.triggered or not freq_data:
            return current_ff

        triggered_id = ctx.triggered[0]["prop_id"]

        # Parse the triggered ID to get the index
        import json

        if triggered_id and triggered_id != ".":
            try:
                id_dict = json.loads(triggered_id.split(".")[0])
                if id_dict.get("type") == "parquet-freq":
                    idx = id_dict.get("index")
                    freq_id = f"parquet-freq-{idx}"
                    if freq_id in freq_data:
                        return freq_data[freq_id]
            except json.JSONDecodeError:
                pass

        return current_ff

    @app.callback(
        Output("ff_hz", "value", allow_duplicate=True),
        Input("envelope-plot", "clickData"),
        State("ff_hz", "value"),
        prevent_initial_call=True,
    )
    def update_ff_from_spectrum_click(click_data, current_ff):
        """Update FF from spectrum click (middle mouse when available)."""
        if (
            not click_data
            or "points" not in click_data
            or not click_data["points"]
        ):
            return no_update

        point = click_data["points"][0]
        x_val = point.get("x", None)
        if x_val is None:
            return no_update

        # Plotly/Dash may not always expose
        # mouse button info in clickData.
        # If present, honor middle-button
        #  (button == 1). Otherwise, accept click.
        event_info = (
            click_data.get("event", {}) if isinstance(click_data, dict) else {}
        )
        button = event_info.get("button", None)
        if button is not None and button != 1:
            return no_update

        try:
            return float(x_val)
        except (TypeError, ValueError):
            return current_ff

    # Client-side wheel handling is now in assets/spectrum_wheel.js
    # No need for server callback - JavaScript handles bin calculation directly
