"""Home page of the app: what the app is for, how the data was prepared and the models trained.

Every number is read from the loaded app data and the model registry, so the page stays true
when the models are retrained.
"""

from __future__ import annotations

from dash import html
from dash_iconify import DashIconify
import dash_mantine_components as dmc
import numpy as np
import plotly.graph_objects as go

from src import app_data
from src.app_figures import MONO_FONT, THEMES, axis_style, base_layout, channel_color, rgba
from src.detection import MIN_DURATION_S, MIN_GAP_S, windows_to_intervals
from src.features import BANDS, TIME_FEATURES
from src.io import FS


# Recording and channels of the example shown at the top of the page.
EXAMPLE_RECORDING = "211104B_D"
EXAMPLE_CHANNELS = ["F3", "C3", "T4", "O2"]
EXAMPLE_MARGIN_S = 6.0
MODE_TEXT = {
    "unbalanced": "Every training window, as recorded: about 4 % seizure.",
    "balanced": "All seizure windows plus as many normal windows drawn at random (undersampling).",
    "augmented": "Every training window plus extra seizure windows: shifted by 0.25 to 0.75 s and mirrored left-right.",
}


def icon(name: str, size: int = 20):
    return DashIconify(icon=name, width=size)


def facts(data) -> dict:
    windows = data.predictions[data.default_model]
    registry = data.registry
    return {
        "recordings": len(data.recordings),
        "seizures": len(data.annotations),
        "hours": float(registry["recorded_hours"]),
        "channels": len(registry["channels"]),
        "features": len(registry["features"]),
        "windows": len(windows),
        "seizure_windows": int(windows["true_class"].sum()),
        "models": len(data.models),
        "modes": [mode for mode in app_data.MODES if any(model.endswith(f"_{mode}") for model in data.models)],
    }


def example_figure(data, theme: str):
    """A real seizure of one recording: a few EEG channels, then the default model's probability."""
    if EXAMPLE_RECORDING not in data.recordings:
        return go.Figure()
    tokens = THEMES[theme]
    model = data.default_model
    threshold = data.best_threshold(model)
    seizure = data.real(EXAMPLE_RECORDING).iloc[0]
    start, end = float(seizure["start_s"]) - EXAMPLE_MARGIN_S, float(seizure["end_s"]) + EXAMPLE_MARGIN_S
    frame = app_data.signal(EXAMPLE_RECORDING)
    part = frame.iloc[int(start * FS) : int(end * FS) : 2]
    time = part["Time"].to_numpy()
    scale = app_data.amplitude_scale(EXAMPLE_RECORDING)
    inside = (time >= seizure["start_s"]) & (time <= seizure["end_s"])

    figure = go.Figure()
    count = len(EXAMPLE_CHANNELS)
    for index, channel in enumerate(EXAMPLE_CHANNELS):
        offset = (count - index) * 2.0 + 1.6
        values = part[channel].to_numpy() / scale * 1.1 + offset
        figure.add_trace(go.Scatter(x=time, y=values, mode="lines", line={"color": channel_color(channel, theme), "width": 1.1}, hoverinfo="skip"))
        figure.add_trace(go.Scatter(x=time, y=np.where(inside, values, np.nan), mode="lines", line={"color": tokens["real"], "width": 1.5}, hoverinfo="skip"))
        figure.add_annotation(x=start, y=offset, text=channel, xanchor="right", xshift=-6, showarrow=False, font={"family": MONO_FONT, "size": 11, "color": tokens["muted"]})

    windows = data.windows(EXAMPLE_RECORDING, model)
    windows = windows.loc[(windows["end_s"] > start) & (windows["start_s"] < end)]
    centers = ((windows["start_s"] + windows["end_s"]) / 2).to_numpy()
    probability = windows["proba"].to_numpy() * 1.2
    figure.add_trace(go.Scatter(x=centers, y=probability, mode="lines", line={"color": tokens["text"], "width": 1.4, "shape": "hv"}, hoverinfo="skip"))
    figure.add_hline(y=threshold * 1.2, line={"color": tokens["threshold"], "width": 1.2, "dash": "dash"})
    figure.add_annotation(x=start, y=0.6, text="model", xanchor="right", xshift=-6, showarrow=False, font={"family": MONO_FONT, "size": 11, "color": tokens["muted"]})
    shapes = [
        {"type": "rect", "xref": "x", "yref": "y", "x0": seizure["start_s"], "x1": seizure["end_s"], "y0": 1.5, "y1": count * 2 + 2.8,
         "fillcolor": rgba(tokens["real"], 0.09), "line": {"width": 0}, "layer": "below"},
    ]
    detected = windows_to_intervals(data.windows(EXAMPLE_RECORDING, model)["start_s"], data.windows(EXAMPLE_RECORDING, model)["end_s"], data.windows(EXAMPLE_RECORDING, model)["proba"], threshold)
    for row in detected.itertuples(index=False):
        if row.end_s > start and row.start_s < end:
            shapes.append({"type": "rect", "xref": "x", "yref": "y", "x0": max(row.start_s, start), "x1": min(row.end_s, end), "y0": 0, "y1": 1.25,
                           "fillcolor": rgba(tokens["detected"], 0.14), "line": {"width": 1, "color": rgba(tokens["detected"], 0.8), "dash": "dot"}, "layer": "below"})
    layout = base_layout(theme)
    layout.update(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    figure.update_layout(
        **layout, height=300, margin={"l": 44, "r": 8, "t": 6, "b": 26}, showlegend=False, shapes=shapes, hovermode=False, dragmode=False,
        xaxis=axis_style(theme, range=[start, end], showgrid=False, ticksuffix=" s", fixedrange=True, tickfont={"family": MONO_FONT, "size": 10, "color": tokens["muted"]}),
        yaxis=axis_style(theme, range=[-0.1, count * 2 + 2.9], visible=False, fixedrange=True),
    )
    return figure


def legend(color: str, text: str, kind: str = "square"):
    return html.Span([html.Span(className=f"key {kind}", style={"background": color} if kind == "square" else {"borderColor": color}), text])


def step(number: int, title: str, body, detail=None, aside=None):
    return html.Div(
        [
            html.Div(str(number), className="home-step-number"),
            html.Div(
                [html.H3(title, className="home-step-title"), html.Div(body, className="home-step-body"), detail or None],
                className="home-step-text",
            ),
            html.Div(aside, className="home-step-aside") if aside is not None else None,
        ],
        className="home-step",
    )


def stat(value: str, label: str):
    return html.Div([html.Div(value, className="home-stat-value mono"), html.Div(label, className="home-stat-label")], className="home-stat")


def pill_list(items: list[str]):
    return html.Div([html.Span(item, className="home-pill mono") for item in items], className="home-pills")


def share_bar(name: str, share: float, windows: float, text: str):
    return html.Div(
        [
            html.Div([html.B(name), html.Span(f"{share:.0%} seizure · {windows:,.0f} windows", className="mono")], className="home-mode-head"),
            html.Div(html.Div(style={"width": f"{max(share, 0.01) * 100:.1f}%"}, className="home-bar-fill"), className="home-bar"),
            html.Div(text, className="home-mode-text"),
        ],
        className="home-mode",
    )


def part_card(page: str, icon_name: str, title: str, text: str):
    return html.Div(
        [
            html.Div(icon(icon_name, 22), className="home-part-icon"),
            html.H3(title, className="home-part-title"),
            html.P(text, className="home-part-text"),
            dmc.Button(f"Open {title}", id={"type": "go-page", "page": page, "place": "parts"}, variant="light", radius="md", size="sm", rightSection=icon("tabler:arrow-right", 14)),
        ],
        className="home-part",
    )


def home_page(data):
    f = facts(data)
    registry = data.registry
    default = data.default_model
    best = data.entry(default)["at_best_threshold"]
    event = best["event"]
    share = f["seizure_windows"] / f["windows"]
    best_by_mode = {
        mode: max((model for model in data.models if model.endswith(f"_{mode}")), key=lambda model: data.entry(model)["at_best_threshold"]["event"]["f1"])
        for mode in f["modes"]
    }
    preprocessing = registry["preprocessing"]
    low, high = preprocessing["band_hz"]

    hero = html.Section(
        [
            html.Div(
                [
                    html.Div("Data-mining project · EEG", className="home-eyebrow"),
                    html.H1("EEG Absence Detector", className="home-title"),
                    html.P(
                        "Finds absence seizures automatically in long EEG recordings, and shows where a trained model agrees "
                        "with the neurologist's annotations, where it misses a seizure and where it raises a false alarm.",
                        className="home-lead",
                    ),
                    dmc.Group(
                        [
                            dmc.Button("Open the app", id={"type": "go-page", "page": "visualisation", "place": "hero"}, radius="md", size="md", rightSection=icon("tabler:arrow-right", 16)),
                            html.A(["How it works ", icon("tabler:arrow-down", 15)], href="#how", className="home-link"),
                        ],
                        gap="lg", mt=8,
                    ),
                    html.Div(
                        [
                            stat(f"{f['recordings']}", "recordings"),
                            stat(f"{f['hours']:.1f} h", "of EEG"),
                            stat(f"{f['seizures']}", "annotated seizures"),
                            stat(f"{f['channels']}", "EEG channels"),
                            stat(f"{f['models']}", "trained models"),
                        ],
                        className="home-stats",
                    ),
                ],
                className="home-hero-text",
            ),
            html.Div(
                [
                    html.Div(
                        [html.Span("A real seizure", className="card-title"), html.Span(f"{EXAMPLE_RECORDING} · {data.label(default)}", className="hint mono")],
                        className="home-example-head",
                    ),
                    html.Div(id="home-example"),
                    html.Div(
                        [
                            legend("var(--app-real)", "Real seizure"),
                            legend("var(--app-detected)", "Detected by the model"),
                            legend("var(--app-text)", "Seizure probability", "line"),
                            legend("var(--app-threshold)", "Threshold", "line dashed"),
                        ],
                        className="plot-legend home-example-legend",
                    ),
                ],
                className="panel home-example",
            ),
        ],
        className="home-hero",
    )

    why = html.Section(
        [
            html.Div("Why this app", className="home-kicker"),
            html.Div(
                [
                    html.Div(
                        [html.Div(icon("tabler:clock-hour-4"), className="home-why-icon"), html.H3("The problem"),
                         html.P("Absence seizures last only a few seconds but can happen many times a day. Finding each one by eye in hours of EEG is slow and easy to miss.")],
                        className="home-why",
                    ),
                    html.Div(
                        [html.Div(icon("tabler:wave-sine"), className="home-why-icon"), html.H3("The idea"),
                         html.P("During an absence the whole brain shows a regular spike-and-wave rhythm near 3 Hz. Its shape and frequency can be measured, so a model can learn to recognise it.")],
                        className="home-why",
                    ),
                    html.Div(
                        [html.Div(icon("tabler:eye-check"), className="home-why-icon"), html.H3("What you can do here"),
                         html.P("Look at the recordings, see the seizures a model finds or misses on each patient, and compare the 4 algorithms and their training modes.")],
                        className="home-why",
                    ),
                ],
                className="home-why-grid",
            ),
            html.P("A study and teaching tool built for a university project: it is not a medical device.", className="home-note"),
        ],
        className="home-section",
    )

    time_features = ", ".join(TIME_FEATURES)
    bands = ", ".join(f"{name} {lo:g}–{hi:g} Hz" for name, (lo, hi) in BANDS.items())
    per_channel = len(TIME_FEATURES) + 2 * len(BANDS)
    how = html.Section(
        [
            html.Div("How it works", className="home-kicker", id="how"),
            html.H2("From raw EEG to detected seizures", className="home-h2"),
            html.P("Seven steps, applied the same way to every recording.", className="home-sub"),
            html.Div(
                [
                    step(
                        1, "Collect the recordings",
                        f"{f['recordings']} EEG recordings ({f['hours']:.1f} hours) sampled at {FS:g} Hz. The {f['channels']} EEG channels of the 10-20 system are used. "
                        "ECG, EMG and photic stimulation (SLI) are kept only as references and never given to the models. "
                        f"The start and end of each of the {f['seizures']} seizures come from the neurologist's annotation workbook.",
                        aside=pill_list(list(registry["channels"])),
                    ),
                    step(
                        2, "Clean the signal",
                        f"Each channel is filtered with a {preprocessing['order']}th-order Butterworth band-pass between {low:g} and {high:g} Hz, applied forwards and backwards "
                        "so that nothing is shifted in time. It removes the slow drift of the electrodes and the 50 Hz mains noise, and keeps the seizure rhythm.",
                    ),
                    step(
                        3, "Cut into windows",
                        f"The signal is cut into 2-second windows, with a new window every second (a 1 s step, so neighbouring windows overlap by half). "
                        f"A window is labelled seizure when at least half of it is inside an annotated seizure.",
                        aside=html.Div(
                            [
                                html.Div([html.Span("Windows", className="home-mini-label"), html.B(f"{f['windows']:,}", className="mono")]),
                                html.Div([html.Span("Seizure", className="home-mini-label"), html.B(f"{f['seizure_windows']:,} · {share:.1%}", className="mono")]),
                                html.Div(html.Div(style={"width": f"{share * 100:.1f}%"}, className="home-bar-fill"), className="home-bar"),
                                html.Div("Only about 1 window in 25 is a seizure: the classes are very unbalanced.", className="home-mode-text"),
                            ],
                            className="home-mini",
                        ),
                    ),
                    step(
                        4, "Describe each window with numbers",
                        f"{per_channel} features per channel: {len(TIME_FEATURES)} in time ({time_features}) and, from the spectrum, "
                        f"the absolute and relative power of 4 bands ({bands}). With the average over channels, each window becomes "
                        f"{f['features']} numbers.",
                    ),
                    step(
                        5, "Train 4 algorithms in several ways",
                        [
                            html.Span("Decision Tree, Random Forest, KNN and SVM, each trained with "),
                            html.Span(f"{len(f['modes'])} ways of handling the imbalance" if len(f["modes"]) > 1 else "the real proportions"),
                            html.Span(f" ({f['models']} models)."),
                        ],
                        aside=html.Div(
                            [
                                html.Div("Training data of one fold", className="section-label"),
                                *[
                                    share_bar(app_data.MODES[mode], data.entry(f"rf_{mode}")["training"]["class_1_share"], data.entry(f"rf_{mode}")["training"]["windows"], MODE_TEXT[mode])
                                    for mode in f["modes"]
                                ],
                            ],
                            className="home-modes",
                        ),
                    ),
                    step(
                        6, "Test on patients the model has never seen",
                        "The recordings are split into 5 groups. Each group is predicted by a model trained on the 4 others, so every probability shown in the app "
                        "comes from a model that never saw that patient. Balancing and augmentation change only the training data: "
                        "the test always keeps the real windows and the real 4 %.",
                    ),
                    step(
                        7, "Turn probabilities into seizures",
                        f"Windows above a threshold are joined into detections; detections less than {MIN_GAP_S:g} s apart are merged and those shorter than "
                        f"{MIN_DURATION_S:g} s are dropped. A real seizure touched by a detection is found, one with no detection is missed, and a detection "
                        "away from every seizure is a false alarm. Each model keeps the threshold that gives its best event F1.",
                    ),
                ],
                className="home-steps",
            ),
        ],
        className="home-section",
    )

    results = html.Section(
        [
            html.Div("Result", className="home-kicker"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Best model", className="section-label"),
                            html.Div(data.label(default), className="home-best-name"),
                            html.Div(f"{event['f1']:.3f}", className="hero-value mono"),
                            html.Div("event F1 on all patients", className="hint"),
                            html.Div(
                                [
                                    html.Div([html.Span("Seizures found"), html.B(f"{event['tp']} of {event['tp'] + event['fn']}", className="mono")]),
                                    html.Div([html.Span("False alarms"), html.B(f"{event['fp']} · {event['false_alarms_per_hour']:.1f} / h", className="mono")]),
                                    html.Div([html.Span("Threshold"), html.B(f"{best['threshold']:.2f}", className="mono")]),
                                ],
                                className="kv home-kv",
                            ),
                        ],
                        className="panel home-best",
                    ),
                    html.Div(
                        [
                            html.Div("Best model of each training mode", className="section-label"),
                            *[
                                html.Div(
                                    [
                                        html.Span(app_data.MODES[mode], className="home-row-name"),
                                        html.Span(app_data.ALGORITHM_SHORT[model.split("_")[0]], className="hint mono"),
                                        html.Div(html.Div(style={"width": f"{data.entry(model)['at_best_threshold']['event']['f1'] * 100:.1f}%"},
                                                          className="home-bar-fill " + ("best" if model == default else "score")), className="home-bar"),
                                        html.B(f"{data.entry(model)['at_best_threshold']['event']['f1']:.3f}", className="mono"),
                                    ],
                                    className="home-row",
                                )
                                for mode, model in best_by_mode.items()
                            ],
                            html.P(
                                "Event F1 combines how many seizures are found (recall) and how many detections are real (precision). "
                                "Accuracy is not used: a model that never detects anything is already 96 % accurate here.",
                                className="home-mode-text", style={"marginTop": 12},
                            ),
                        ],
                        className="panel home-by-mode",
                    ),
                ],
                className="home-results",
            ),
        ],
        className="home-section",
    )

    parts = html.Section(
        [
            html.Div("Explore", className="home-kicker"),
            html.H2("Three parts", className="home-h2"),
            html.Div(
                [
                    part_card("visualisation", "tabler:wave-square", "Visualisation",
                              "The data only: choose a patient and channels, page through the EEG, real seizures drawn in green on the curves."),
                    part_card("prediction", "tabler:target-arrow", "Prediction",
                              "One model on one patient: its probability, its detections, the threshold, and what it found, missed or raised by mistake."),
                    part_card("comparison", "tabler:chart-bar", "Comparison",
                              f"The {f['models']} models side by side: who wins and why, threshold and precision-recall curves, results per patient."),
                ],
                className="home-parts",
            ),
        ],
        className="home-section",
    )

    return html.Div([hero, why, how, results, parts], id="page-home", className="home")
