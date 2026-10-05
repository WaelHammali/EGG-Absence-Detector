"""EEG Absence Detector — interactive review of recordings, annotations and detections.

Run with ``python app.py``. Only precomputed files are read; nothing is trained here.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from urllib.parse import parse_qs
import sys
import uuid

import dash
from dash import Input, Output, Patch, State, clientside_callback, ctx, dcc, html, no_update
from dash_iconify import DashIconify
import dash_mantine_components as dmc
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import app_data
from src.app_figures import (
    DETECTED, EVENT, REAL, THRESHOLD, event_shapes, message_figure, overview_figure, recording_figure,
    resampled_dict, spectrum_figure, view_relayout,
)
from src.io import CHANNELS


try:
    DATA = app_data.load()
    LOAD_ERROR = None
except app_data.MissingData as error:
    DATA, LOAD_ERROR = None, str(error)

GREEK = {"Delta": "δ", "Theta": "θ", "Alpha": "α", "Beta": "β"}
# URL values for ?page=…, including the names used before the app was split in three parts.
PAGES = {
    "visualisation": "visualisation", "prediction": "prediction", "comparison": "comparison",
    "recording": "prediction", "overview": "comparison",
}
PAGE_LENGTHS = [("10 s", "10"), ("30 s", "30"), ("60 s", "60"), ("Full", "full")]
# Server-side full-resolution figures, one per browser session.
FIGURES: OrderedDict[str, object] = OrderedDict()
MAX_SESSIONS = 8


# ----------------------------------------------------------------------------- small UI helpers


def icon(name: str, size: int = 18):
    return DashIconify(icon=name, width=size)


def tip(label: str, child, block: bool = False, **kwargs):
    """Every control carries a one-sentence explanation.

    ``block`` makes the tooltip wrapper fill its row, which full-width controls need.
    """
    if block:
        kwargs["boxWrapperProps"] = {"w": "100%", "display": "block"}
    return dmc.Tooltip(label=label, children=child, multiline=True, w=260, withArrow=True, openDelay=350, **kwargs)


def wide_tip(label: str, child):
    return tip(label, child, block=True)


def section(text: str):
    return dmc.Text(text, className="section-label", c="dimmed")


def panel(children, **kwargs):
    return dmc.Paper(children, className="panel", p="md", **kwargs)


def swatch(color: str, text: str):
    return html.Span([html.Span(className="swatch", style={"background": color}), text])


def kpi(title: str, value: str, subtitle: str, icon_name: str, color: str):
    return panel(
        dmc.Group(
            [
                dmc.ThemeIcon(icon(icon_name, 22), size=44, radius="md", variant="light", color=color),
                dmc.Stack(
                    [
                        dmc.Text(title, size="xs", c="dimmed", fw=600, tt="uppercase"),
                        dmc.Text(value, className="mono kpi-value"),
                        dmc.Text(subtitle, size="xs", c="dimmed"),
                    ],
                    gap=2,
                ),
            ],
            align="flex-start", wrap="nowrap", gap="md",
        )
    )


def seconds(value) -> str:
    return "—" if pd.isna(value) else f"{value:.1f}"


def signed(value) -> str:
    return "—" if pd.isna(value) else f"{value:+.1f}"


STATUS = {
    "TP": ("Found", "teal", "tabler:check"),
    "FN": ("Missed", "gray", "tabler:eye-off"),
    "FP": ("False alarm", "red", "tabler:alert-triangle"),
}


def status_badge(status: str):
    label, color, icon_name = STATUS[status]
    return dmc.Badge(label, color=color, variant="light", leftSection=icon(icon_name, 12), radius="sm", miw=112, style={"flexShrink": 0})


def interval(start, end) -> str:
    return "—" if pd.isna(start) else f"{start:.1f} → {end:.1f}"


def intervals_table(matches: pd.DataFrame):
    if matches.empty:
        return dmc.Text("No annotated seizure and no detection at this threshold.", c="dimmed", size="sm", py="lg", ta="center")
    head = ["#", "Status", "Real (s)", "Detected (s)", "Onset err.", "Offset err.", "Max p"]
    rows = []
    for number, row in enumerate(matches.itertuples(index=False), 1):
        cells = [
            str(number), status_badge(row.status), interval(row.real_start_s, row.real_end_s),
            interval(row.detected_start_s, row.detected_end_s), signed(row.onset_error_s), signed(row.offset_error_s),
            "—" if pd.isna(row.max_proba) else f"{row.max_proba:.2f}",
        ]
        rows.append(dmc.TableTr([dmc.TableTd(cell, className="" if index == 1 else "mono") for index, cell in enumerate(cells)]))
    return dmc.TableScrollContainer(
        dmc.Table(
            [dmc.TableThead(dmc.TableTr([dmc.TableTh(name, style={"whiteSpace": "nowrap"}) for name in head])), dmc.TableTbody(rows)],
            striped=True, highlightOnHover=True, verticalSpacing=6, horizontalSpacing=6, fz="sm", stickyHeader=True,
        ),
        minWidth=500, mah=330, type="native",
    )


# ----------------------------------------------------------------------------- layout


def header():
    return dmc.AppShellHeader(
        dmc.Group(
            [
                dmc.Group(
                    [
                        dmc.ThemeIcon(icon("mdi:brain", 22), size=38, radius="md", variant="light", color="teal"),
                        dmc.Stack(
                            [
                                dmc.Text("EEG Absence Detector", fw=700, size="lg", lh=1.1),
                                dmc.Text("Visualise the EEG, test a model on a patient, compare models", size="xs", c="dimmed", visibleFrom="md"),
                            ],
                            gap=0,
                        ),
                    ],
                    gap="sm", wrap="nowrap",
                ),
                dmc.Group(
                    [
                        tip(
                            "Visualisation shows the data and the real seizures; Prediction adds a model's detections; Comparison puts the models side by side.",
                            dmc.SegmentedControl(
                                id="page", value="visualisation", radius="md",
                                data=[
                                    {"value": "visualisation", "label": "Visualisation"},
                                    {"value": "prediction", "label": "Prediction"},
                                    {"value": "comparison", "label": "Comparison"},
                                ],
                            ),
                        ),
                        tip(
                            "Toggle between the dark monitor theme and a light theme suited to report screenshots.",
                            dmc.ActionIcon(icon("tabler:sun", 20), id="theme-toggle", variant="default", size="lg", radius="md", **{"aria-label": "Toggle color scheme"}),
                        ),
                    ],
                    gap="sm", wrap="nowrap",
                ),
            ],
            justify="space-between", h="100%", px="md", wrap="nowrap",
        ),
        className="shell-surface",
    )


def sidebar():
    recordings = [
        {"value": row.recording, "label": f"{row.recording} · {row.sex}, {row.age} y"} for row in DATA.metadata.itertuples(index=False)
    ]
    return dmc.AppShellNavbar(
        dmc.ScrollArea(
            dmc.Stack(
                [
                    section("Patient"),
                    wide_tip(
                        "Choose the patient (recording) to look at; the label shows sex and age.",
                        html.Div(
                            dmc.Select(
                                id="recording", data=recordings, value=recordings[0]["value"], searchable=True, allowDeselect=False,
                                leftSection=icon("tabler:user", 16), comboboxProps={"withinPortal": True},
                            )
                        ),
                    ),
                    html.Div(
                        dmc.Stack(
                            [
                                dmc.Divider(),
                                section("Prediction"),
                                wide_tip(
                                    "Model used to predict the seizures. Every model is tested on patients it never saw during training.",
                                    html.Div(
                                        dmc.Select(
                                            id="model", data=DATA.models, value=app_data.DEFAULT_MODEL, allowDeselect=False,
                                            leftSection=icon("tabler:cpu", 16), comboboxProps={"withinPortal": True},
                                        )
                                    ),
                                ),
                            dmc.Stack(
                                [
                                    dmc.Group(
                                        [
                                            dmc.Text("Probability threshold", size="sm", fw=500),
                                            dmc.Group(
                                                [
                                                    dmc.Text(id="threshold-value", className="mono", size="sm", c="violet"),
                                                    tip(
                                                        "Reset the threshold to the best value for the selected model (the one giving the highest event-level F1).",
                                                        dmc.ActionIcon(icon("tabler:restore", 16), id="threshold-reset", variant="subtle", color="gray", size="sm", **{"aria-label": "Reset threshold"}),
                                                    ),
                                                ],
                                                gap=4,
                                            ),
                                        ],
                                        justify="space-between",
                                    ),
                                    wide_tip(
                                        "A window counts as seizure when its probability reaches this value; detections and metrics update live.",                                html.Div(
                                            dmc.Slider(
                                                id="threshold", min=0.05, max=0.95, step=0.01, value=DATA.best_threshold(app_data.DEFAULT_MODEL), color="violet",
                                                updatemode="mouseup", precision=2,
                                                marks=[{"value": 0.25, "label": "0.25"}, {"value": 0.5, "label": "0.50"}, {"value": 0.75, "label": "0.75"}],
                                            ),
                                            style={"paddingBottom": 18},
                                        ),
                                    ),
                                ],
                                gap=6,
                            ),
                            ],
                            gap="md",
                        ),
                        id="prediction-controls", style={"display": "none"},
                    ),
                    html.Div(
                        dmc.Stack(
                            [
                                dmc.Divider(),
                                section("Display"),
                                wide_tip(
                                    "Stacked shows one row per channel like an EEG reader; Overlay draws the selected channels on one axis.",                                    dmc.SegmentedControl(
                                        id="mode", value="stacked", fullWidth=True,
                                        data=[{"value": "stacked", "label": "All stacked"}, {"value": "overlay", "label": "Overlay"}],
                                    ),
                                ),
                                dmc.Group(
                                    [
                                        dmc.Text("Channels", size="sm", fw=500),
                                        tip(
                                            "Show all eight channels again.",
                                            dmc.Button("Show all", id="all-channels", variant="subtle", size="compact-xs", leftSection=icon("tabler:list-check", 12)),
                                        ),
                                    ],
                                    justify="space-between", mb=-8,
                                ),
                                wide_tip(
                                    "Curves to display: remove a channel with its ×, or pick one from the list to add it back.",                                    html.Div(
                                        dmc.MultiSelect(
                                            id="channels", data=list(CHANNELS), value=list(CHANNELS), clearable=False,
                                            leftSection=icon("tabler:wave-sine", 16), comboboxProps={"withinPortal": True},
                                        )
                                    ),
                                ),
                                dmc.Stack(
                                    [
                                        dmc.Text("Page length", size="sm", fw=500),
                                        wide_tip(
                                            "Duration shown on screen; use the page buttons or the ← → keys to move by one page.",                                            dmc.SegmentedControl(
                                                id="page-length", value="30", fullWidth=True,
                                                data=[{"value": value, "label": label} for label, value in PAGE_LENGTHS],
                                            ),
                                        ),
                                    ],
                                    gap=6,
                                ),
                                dmc.Stack(
                                    [
                                        dmc.Text("Gain", size="sm", fw=500),
                                        wide_tip(
                                            "Amplitude scale: higher gain magnifies the traces, the same for every channel.",                                            html.Div(
                                                dmc.Slider(
                                                    id="gain", min=0, max=4, step=1, value=1, updatemode="mouseup", showLabelOnHover=False, label=None,
                                                    marks=[{"value": index, "label": label} for index, label in enumerate(["×0.5", "×1", "×2", "×4", "×8"])],
                                                ),
                                                style={"paddingBottom": 18},
                                            ),
                                        ),
                                    ],
                                    gap=6,
                                ),
                                tip(
                                    "Show the technician's notes from the recording header (HPN, SLI, eyes open/closed, artifacts, movements).",
                                    html.Div(dmc.Switch(id="events", checked=True, label="Technician events", color="yellow")),
                                ),
                            ],
                            gap="md",
                        ),
                        id="recording-controls",
                    ),
                    dmc.Divider(),
                    section("Legend"),
                    dmc.Stack(
                        [
                            dmc.Text(swatch(REAL, "Real seizure (annotation), also drawn on the curves"), size="sm"),
                            dmc.Text(swatch(DETECTED, "Detection by the model"), size="sm"),
                            dmc.Text(swatch(EVENT, "Technician event"), size="sm"),
                            dmc.Text(swatch(THRESHOLD, "Threshold"), size="sm"),
                        ],
                        gap=4,
                    ),
                ],
                gap="md", p="md",
            ),
            type="auto", h="100%",
        ),
        className="shell-surface",
    )


def overview_page():
    return html.Div(
        dmc.Stack(
            [
                panel(
                    [
                        dmc.Group(
                            [
                                dmc.Stack(
                                    [
                                        dmc.Text("Model comparison", fw=600),
                                        dmc.Text(
                                            "Each model at its best threshold, tested on patients it never saw. Choose a model in the sidebar to see its results below.",
                                            size="sm", c="dimmed",
                                        ),
                                    ],
                                    gap=0,
                                ),
                                dmc.Badge("Out-of-fold predictions", variant="light", color="gray", leftSection=icon("tabler:shield-check", 12)),
                            ],
                            justify="space-between", align="flex-start", mb="xs",
                        ),
                        html.Div(id="model-table"),
                    ]
                ),
                html.Div(id="kpi-title"),
                dmc.SimpleGrid(id="kpis", cols={"base": 2, "md": 3, "xl": 5}, spacing="md"),
                panel(
                    [
                        dmc.Group(
                            [
                                dmc.Stack(
                                    [
                                        dmc.Text("Seizure timeline of the selected model", fw=600),
                                        dmc.Text(
                                            "One row per patient: real seizures above, detections below. Click a row to open that patient in Prediction.",
                                            size="sm", c="dimmed",
                                        ),
                                    ],
                                    gap=0,
                                ),
                            ],
                            justify="space-between", align="flex-start", mb="xs",
                        ),
                        dcc.Loading(
                            dcc.Graph(id="timeline", config={"displaylogo": False, "modeBarButtonsToRemove": ["select2d", "lasso2d"]}),
                            delay_show=300, overlay_style={"visibility": "visible", "opacity": 0.5}, type="dot",
                        ),
                    ]
                ),
            ],
            gap="md",
        ),
        id="page-overview", style={"display": "none"},
    )


def recording_page():
    button = {"variant": "default", "size": "compact-sm", "radius": "md"}
    return html.Div(
        dmc.Stack(
            [
                panel([html.Div(id="recording-info"), html.Div(id="verdict")]),
                panel(
                    [
                        dmc.Group(
                            [
                                dmc.Group(
                                    [
                                        tip("Go back one page (← key).", dmc.Button("Page", id="prev-page", leftSection=icon("tabler:chevron-left", 14), **button)),
                                        tip("Go forward one page (→ key).", dmc.Button("Page", id="next-page", rightSection=icon("tabler:chevron-right", 14), **button)),
                                        dmc.Divider(orientation="vertical"),
                                        tip("Center the view on the previous annotated seizure.", dmc.Button("Seizure", id="prev-seizure", leftSection=icon("tabler:player-skip-back", 14), **button)),
                                        tip("Center the view on the next annotated seizure.", dmc.Button("Seizure", id="next-seizure", rightSection=icon("tabler:player-skip-forward", 14), **button)),
                                        dmc.Group([dmc.Kbd("←"), dmc.Kbd("→"), dmc.Text("page", size="xs", c="dimmed")], gap=4, visibleFrom="lg"),
                                    ],
                                    gap="xs",
                                ),
                                dmc.Group(
                                    [
                                        dmc.Text(id="view-label", className="mono", size="sm", c="dimmed"),
                                        tip("Save the current view as a high-resolution PNG for the report.", dmc.Button("Save view as PNG", id="save-png", leftSection=icon("tabler:camera", 14), **button)),
                                        tip("Download the seizure intervals of this patient as a CSV file (with the detections in Prediction).", dmc.Button("Download intervals CSV", id="save-csv", leftSection=icon("tabler:download", 14), **button)),
                                    ],
                                    gap="xs",
                                ),
                            ],
                            justify="space-between", mb="xs",
                        ),
                        dcc.Loading(
                            dcc.Graph(
                                id="eeg", config={"displaylogo": False, "scrollZoom": False, "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"]},
                            ),
                            delay_show=400, overlay_style={"visibility": "visible", "opacity": 0.5}, type="dot", target_components={"eeg": "figure"},
                        ),
                        dmc.Text(
                            "Curves turn green during a real seizure and a diamond marks each seizure start. Drag to zoom in time, double-click to reset, drag the slider under the bottom plot to navigate. Click a curve to analyze that 2 s window.",
                            size="xs", c="dimmed", mt=4,
                        ),
                    ]
                ),
                dmc.Grid(
                    [
                        dmc.GridCol(
                            panel(
                                [
                                    dmc.Text("Spectrum of the selected window", fw=600),
                                    dmc.Text(id="spectrum-caption", size="sm", c="dimmed"),
                                    dcc.Graph(id="spectrum", config={"displaylogo": False, "displayModeBar": False}),
                                    html.Div(id="spectrum-bands"),
                                ],
                                h="100%",
                            ),
                            span={"base": 12, "xl": 5},
                        ),
                        dmc.GridCol(
                            panel(
                                [
                                    html.Div(id="intervals"),
                                ],
                                h="100%",
                            ),
                            span={"base": 12, "xl": 7},
                        ),
                    ],
                    gutter="md",
                ),
            ],
            gap="md",
        ),
        id="page-recording",
    )


def layout():
    theme = {
        "fontFamily": "Inter, system-ui, -apple-system, Segoe UI, sans-serif",
        "fontFamilyMonospace": "JetBrains Mono, ui-monospace, monospace",
        "primaryColor": "teal",
        "defaultRadius": "md",
        "colors": {
            "dark": ["#E5E9F0", "#C3CCDB", "#94A3B8", "#64748B", "#2A3756", "#1E2A44", "#111A2E", "#0B1220", "#080E1A", "#050912"],
        },
    }
    if LOAD_ERROR:
        body = dmc.Center(
            dmc.Alert(LOAD_ERROR, title="The app data is not ready yet", color="yellow", icon=icon("tabler:database-off"), maw=720),
            h="100vh", p="xl",
        )
        return dmc.MantineProvider(body, forceColorScheme="dark", theme=theme, id="provider")
    return dmc.MantineProvider(
        [
            dcc.Location(id="url"),
            dcc.Store(id="session", data=uuid.uuid4().hex),
            dcc.Store(id="theme", data="dark"),
            dcc.Store(id="view"),
            dcc.Store(id="figure-meta"),
            dcc.Store(id="spectrum-point"),
            dcc.Store(id="keys-ready"),
            dcc.Download(id="download"),
            dmc.AppShell(
                [
                    header(),
                    sidebar(),
                    dmc.AppShellMain(
                        [
                            dmc.Alert(id="error", title="Something went wrong", color="red", icon=icon("tabler:alert-circle"), hide=True, withCloseButton=True, mb="md"),
                            overview_page(),
                            recording_page(),
                        ]
                    ),
                ],
                header={"height": 60}, navbar={"width": 300, "breakpoint": "sm"}, padding="md",
            ),
        ],
        id="provider", forceColorScheme="dark", theme=theme,
    )


def on_error(error: Exception):
    """Show a friendly message instead of a traceback."""
    message = str(error) if isinstance(error, app_data.MissingData) else "The last action could not be completed. Check the terminal for details and try again."
    dash.set_props("error", {"children": message, "hide": False})


app = dash.Dash(__name__, title="EEG Absence Detector", update_title=None, on_error=on_error, suppress_callback_exceptions=True)
app.layout = layout


# ----------------------------------------------------------------------------- shell callbacks


@app.callback(
    Output("theme", "data"), Output("page", "value"), Output("recording", "value"),
    Input("url", "search"), Input("theme-toggle", "n_clicks"), State("theme", "data"),
)
def shell_state(search, _, theme):
    """Initial state from the URL (?page=…&theme=…&recording=…), then the theme toggle."""
    if ctx.triggered_id == "theme-toggle":
        return ("light" if theme == "dark" else "dark"), no_update, no_update
    query = {key: values[0] for key, values in parse_qs((search or "").lstrip("?")).items()}
    recording = query.get("recording")
    return (
        query["theme"] if query.get("theme") in ("dark", "light") else no_update,
        PAGES.get(query.get("page"), no_update),
        recording if recording in DATA.recordings else no_update,
    )


@app.callback(Output("provider", "forceColorScheme"), Output("theme-toggle", "children"), Input("theme", "data"))
def apply_theme(theme):
    return theme, icon("tabler:sun" if theme == "dark" else "tabler:moon", 20)


@app.callback(
    Output("page-overview", "style"), Output("page-recording", "style"),
    Output("recording-controls", "style"), Output("prediction-controls", "style"),
    Input("page", "value"),
)
def show_page(page):
    shown, hidden = {"display": "block"}, {"display": "none"}
    if page == "comparison":
        return shown, hidden, hidden, shown
    return hidden, shown, shown, (shown if page == "prediction" else hidden)


@app.callback(Output("threshold-value", "children"), Input("threshold", "value"))
def threshold_label(threshold):
    return f"{threshold:.2f}"


@app.callback(
    Output("threshold", "value"), Input("model", "value"), Input("threshold-reset", "n_clicks"),
    prevent_initial_call=True,
)
def model_threshold(model, _):
    """Each model starts at its own best threshold; the reset button returns to it."""
    return DATA.best_threshold(model)


@app.callback(Output("channels", "value"), Input("all-channels", "n_clicks"), prevent_initial_call=True)
def show_all_channels(_):
    return list(CHANNELS)


# ----------------------------------------------------------------------------- overview


def model_table(selected: str, recording: str):
    """All models side by side: overall scores and the result on the selected patient."""
    head = ["Model", "Threshold", "Event F1", "Recall", "Precision", "Found / missed", "False alarms", "Window F1", f"On {recording}"]
    best = DATA.comparison["event_f1"].max()
    rows = []
    for row in DATA.comparison.itertuples(index=False):
        _, _, patient = DATA.detect(recording, row.threshold, row.model)
        name = [dmc.Text(row.model, fw=600, size="sm", span=True)]
        if row.model == selected:
            name.append(dmc.Badge("Selected", size="xs", variant="light", color="teal", ml=6))
        if row.event_f1 == best:
            name.append(dmc.Badge("Best F1", size="xs", variant="light", color="violet", ml=6))
        cells = [
            name, f"{row.threshold:.2f}", f"{row.event_f1:.2f}", f"{row.event_recall:.0%}", f"{row.event_precision:.0%}",
            f"{row.found} / {row.missed}", f"{row.false_alarms} ({row.false_alarms_per_hour:.1f}/h)", f"{row.window_f1:.2f}",
            f"{patient['tp']}/{patient['tp'] + patient['fn']} found · {patient['fp']} false",
        ]
        rows.append(dmc.TableTr([dmc.TableTd(cell, className="" if index == 0 else "mono") for index, cell in enumerate(cells)]))
    return dmc.TableScrollContainer(
        dmc.Table(
            [dmc.TableThead(dmc.TableTr([dmc.TableTh(name, style={"whiteSpace": "nowrap"}) for name in head])), dmc.TableTbody(rows)],
            striped=True, highlightOnHover=True, verticalSpacing=8, fz="sm",
        ),
        minWidth=860, type="native",
    )


@app.callback(
    Output("model-table", "children"), Output("kpi-title", "children"), Output("kpis", "children"), Output("timeline", "figure"),
    Input("threshold", "value"), Input("theme", "data"), Input("model", "value"), Input("recording", "value"),
)
def comparison(threshold, theme, model, recording):
    detected, matches, metrics = DATA.detect_everything(threshold, model)
    total = metrics["tp"] + metrics["fn"]
    cards = [
        kpi("Event F1", f"{metrics['f1']:.2f}", "balance of recall and precision", "tabler:target-arrow", "teal"),
        kpi("Recall", f"{metrics['recall']:.0%}", f"{metrics['tp']} of {total} seizures found", "tabler:radar-2", "teal"),
        kpi("Precision", f"{metrics['precision']:.0%}", f"of {metrics['detections']} detections", "tabler:focus-2", "blue"),
        kpi("Found / missed", f"{metrics['tp']} / {metrics['fn']}", "annotated seizures", "tabler:checks", "green"),
        kpi("False alarms", f"{metrics['fp']}", f"{metrics['false_alarms_per_hour']:.1f} per recorded hour", "tabler:alert-triangle", "red"),
    ]
    title = dmc.Text(
        ["Selected model: ", dmc.Text(model, fw=700, span=True), " at threshold ", dmc.Text(f"{threshold:.2f}", className="mono", span=True), ", all patients"],
        size="sm", c="dimmed",
    )
    return model_table(model, recording), title, cards, overview_figure(DATA.metadata, DATA.annotations, detected, matches, theme)


@app.callback(
    Output("recording", "value", allow_duplicate=True), Output("page", "value", allow_duplicate=True),
    Input("timeline", "clickData"), prevent_initial_call=True,
)
def open_recording(click):
    if not click:
        return no_update, no_update
    custom = click["points"][0].get("customdata")
    recording = custom if isinstance(custom, str) else (custom[0] if custom else None)
    if recording not in DATA.recordings:
        return no_update, no_update
    return recording, "prediction"


# ----------------------------------------------------------------------------- recording page


def card_title(title: str, note: str):
    return dmc.Group([dmc.Text(title, fw=600), dmc.Text(note, size="xs", c="dimmed")], justify="space-between", mb="xs")


def real_table(real: pd.DataFrame):
    if real.empty:
        return dmc.Text("No seizure is annotated for this patient.", c="dimmed", size="sm", py="lg", ta="center")
    rows = [
        dmc.TableTr([dmc.TableTd(cell, className="mono") for cell in (str(number), f"{row.start_s:.1f}", f"{row.end_s:.1f}", f"{row.end_s - row.start_s:.1f}")])
        for number, row in enumerate(real.itertuples(index=False), 1)
    ]
    return dmc.TableScrollContainer(
        dmc.Table(
            [dmc.TableThead(dmc.TableTr([dmc.TableTh(name) for name in ("Seizure", "Start (s)", "End (s)", "Duration (s)")])), dmc.TableTbody(rows)],
            striped=True, highlightOnHover=True, verticalSpacing=6, fz="sm", stickyHeader=True,
        ),
        minWidth=360, mah=330, type="native",
    )


def verdict(model: str, threshold: float, metrics: dict):
    """Plain statement of how the model did on this patient, with the rule used for the label."""
    total = metrics["tp"] + metrics["fn"]
    if total and metrics["fn"] == 0 and metrics["fp"] == 0:
        label, color, icon_name = "Predicts well", "teal", "tabler:circle-check"
    elif total and (metrics["tp"] < total / 2 or metrics["fp"] > 2 * total):
        label, color, icon_name = "Predicts poorly", "red", "tabler:circle-x"
    else:
        label, color, icon_name = "Partly correct", "yellow", "tabler:alert-circle"
    window = metrics["window"]
    share = lambda value: "—" if pd.isna(value) else f"{value:.0%}"
    sentence = (
        f"{model} at threshold {threshold:.2f} found {metrics['tp']} of {total} real seizures, missed {metrics['fn']} "
        f"and raised {metrics['fp']} false alarm{'s' if metrics['fp'] != 1 else ''}."
    )
    return dmc.Stack(
        [
            dmc.Divider(my="sm"),
            dmc.Group(
                [
                    dmc.Group(
                        [
                            tip(
                                "Well: every real seizure found and no false alarm. Poorly: fewer than half found, or more than two false alarms per real seizure. Otherwise partly correct.",
                                dmc.Badge(label, color=color, size="lg", variant="light", leftSection=icon(icon_name, 14)),
                            ),
                            dmc.Text(sentence, size="sm"),
                        ],
                        gap="sm",
                    ),
                    dmc.Group(
                        [
                            dmc.Badge(f"{metrics['tp']} found", color="teal", variant="light", leftSection=icon("tabler:check", 12)),
                            dmc.Badge(f"{metrics['fn']} missed", color="gray", variant="light", leftSection=icon("tabler:eye-off", 12)),
                            dmc.Badge(f"{metrics['fp']} false alarms", color="red", variant="light", leftSection=icon("tabler:alert-triangle", 12)),
                        ],
                        gap="xs",
                    ),
                ],
                justify="space-between",
            ),
            dmc.Text(
                f"Per 2 s window: {window['tp']} seizure windows detected out of {window['tp'] + window['fn']} (recall {share(window['recall'])}); "
                f"{window['fp']} normal windows flagged by mistake (precision {share(window['precision'])}). "
                "The model never saw this patient during training."
                + (" A false alarm is a detection outside the annotated seizures; it can also be a real seizure missing from the annotation file." if metrics["fp"] else ""),
                size="xs", c="dimmed",
            ),
        ],
        gap=6,
    )


@app.callback(
    Output("recording-info", "children"), Output("verdict", "children"), Output("intervals", "children"),
    Input("recording", "value"), Input("threshold", "value"), Input("model", "value"), Input("page", "value"),
)
def recording_summary(recording, threshold, model, page):
    info = DATA.info(recording)
    minutes, rest = divmod(int(info["duration_s"]), 60)
    facts = [
        ("tabler:user", "Patient", recording),
        ("tabler:gender-bigender", "Sex", {"M": "Male", "F": "Female"}.get(info["sex"], str(info["sex"]))),
        ("tabler:cake", "Age", f"{info['age']} y"),
        ("tabler:clock", "Duration", f"{minutes} min {rest:02d} s"),
        ("tabler:activity", "Real seizures", f"{info['seizures']} ({info['seizure_seconds']:.0f} s)"),
    ]
    left = dmc.Group(
        [
            dmc.Group(
                [
                    dmc.ThemeIcon(icon(icon_name, 16), variant="light", color="gray", size=30),
                    dmc.Stack([dmc.Text(label, size="xs", c="dimmed"), dmc.Text(value, className="mono", size="sm", fw=500)], gap=0),
                ],
                gap="xs", wrap="nowrap",
            )
            for icon_name, label, value in facts
        ],
        gap="xl",
    )
    badges = []
    if info["annotation_source"] == "fallback":
        badges.append(
            tip(
                "This recording is missing from the updated annotation file; its seizure times come from the earlier signal-based alignment.",
                dmc.Badge("Fallback annotations", color="yellow", variant="light", leftSection=icon("tabler:info-circle", 12)),
            )
        )
    if info["absence_type"]:
        badges.append(dmc.Badge(f"Type: {info['absence_type']}", color="gray", variant="outline"))
    header_row = dmc.Group([left, dmc.Group(badges, gap="xs")], justify="space-between")
    if page != "prediction":
        table = [card_title("Real seizures of this patient", "from the annotations; times in seconds"), real_table(DATA.real(recording))]
        return header_row, None, table
    _, matches, metrics = DATA.detect(recording, threshold, model)
    table = [card_title(f"Real vs predicted — {model}", "times in seconds; errors are detected − real"), intervals_table(matches)]
    return header_row, verdict(model, threshold, metrics), table


def page_length(value: str, duration: float) -> float:
    return duration if value == "full" else min(float(value), duration)


def clamp(start: float, length: float, duration: float) -> list[float]:
    start = max(0.0, min(start, duration - length))
    return [round(start, 3), round(start + length, 3)]


@app.callback(
    Output("eeg", "figure"), Output("figure-meta", "data"), Output("view", "data"),
    Input("recording", "value"), Input("channels", "value"), Input("mode", "value"), Input("gain", "value"),
    Input("events", "checked"), Input("theme", "data"), Input("page", "value"), Input("model", "value"),
    State("threshold", "value"), State("view", "data"), State("page-length", "value"), State("session", "data"),
)
def build_eeg(recording, channels, mode, gain, show_events, theme, page, model, threshold, view, length, session):
    if page == "comparison":
        # The EEG is hidden on this page; keep whatever is drawn.
        return no_update, no_update, no_update
    prediction = page == "prediction"
    if ctx.triggered_id == "model":
        # The slider is moving to this model's best threshold at the same time.
        threshold = DATA.best_threshold(model)
    try:
        frame = app_data.signal(recording)
    except app_data.MissingData as error:
        return message_figure(str(error), theme, 420), None, no_update
    channels = [channel for channel in CHANNELS if channel in (channels or [])]
    if not channels:
        return message_figure("Select at least one channel to display the EEG.", theme, 420), None, no_update
    duration = float(DATA.info(recording)["duration_s"])
    if view is None or view.get("recording") != recording:
        view_range = clamp(0.0, page_length(length, duration), duration)
    else:
        view_range = view["range"]
    detected = DATA.detect(recording, threshold, model)[0] if prediction else None
    figure, description = recording_figure(
        frame["Time"].to_numpy(), {channel: frame[channel].to_numpy() for channel in channels},
        DATA.windows(recording, model) if prediction else None,
        DATA.real(recording), detected, DATA.technician(recording) if show_events else None, threshold if prediction else None,
        stacked=mode == "stacked", full_scale=app_data.amplitude_scale(recording), gain=0.5 * 2 ** gain,
        view=tuple(view_range), theme=theme,
    )
    FIGURES[session] = figure
    FIGURES.move_to_end(session)
    while len(FIGURES) > MAX_SESSIONS:
        FIGURES.popitem(last=False)
    description.update({"recording": recording, "channels": channels, "model": model})
    return resampled_dict(figure, tuple(view_range)), description, {"recording": recording, "range": view_range}


def relayout_range(relayout: dict | None, duration: float) -> list[float] | None:
    """Time range requested by a zoom, pan, range-slider drag or axis reset."""
    if not relayout:
        return None
    for axis in ("xaxis", "xaxis2"):
        if f"{axis}.range[0]" in relayout and f"{axis}.range[1]" in relayout:
            return [float(relayout[f"{axis}.range[0]"]), float(relayout[f"{axis}.range[1]"])]
        if f"{axis}.range" in relayout:
            return [float(value) for value in relayout[f"{axis}.range"]]
    if any(relayout.get(f"{axis}.autorange") for axis in ("xaxis", "xaxis2")):
        return [0.0, duration]
    return None


@app.callback(
    Output("eeg", "figure", allow_duplicate=True), Output("view", "data", allow_duplicate=True),
    Input("prev-page", "n_clicks"), Input("next-page", "n_clicks"), Input("prev-seizure", "n_clicks"), Input("next-seizure", "n_clicks"),
    Input("page-length", "value"), Input("eeg", "relayoutData"),
    State("view", "data"), State("session", "data"),
    prevent_initial_call=True,
)
def navigate(_a, _b, _c, _d, length, relayout, view, session):
    figure = FIGURES.get(session)
    if view is None or figure is None:
        return no_update, no_update
    recording = view["recording"]
    duration = float(DATA.info(recording)["duration_s"])
    start, end = view["range"]
    span = end - start
    trigger = ctx.triggered_id
    if trigger == "eeg":
        target = relayout_range(relayout, duration)
        if target is None:
            return no_update, no_update
        target = [max(0.0, target[0]), min(duration, target[1])]
        if target[1] - target[0] < 0.05:
            return no_update, no_update
    elif trigger == "page-length":
        target = clamp(start, page_length(length, duration), duration)
    elif trigger in ("prev-page", "next-page"):
        target = clamp(start + (span if trigger == "next-page" else -span), span, duration)
    else:
        real = DATA.real(recording)
        middle = (start + end) / 2
        centers = ((real["start_s"] + real["end_s"]) / 2).to_numpy()
        candidates = centers[centers > middle + 0.5] if trigger == "next-seizure" else centers[centers < middle - 0.5][::-1]
        if not len(candidates):
            return no_update, no_update
        # A full-recording view cannot be centered: fall back to a 30 s page.
        width = span if span < duration else min(30.0, duration)
        target = clamp(float(candidates[0]) - width / 2, width, duration)
    patch = figure.construct_update_data_patch(view_relayout(tuple(target)))
    if patch is no_update:
        patch = Patch()
    if trigger != "eeg":
        for axis in ("xaxis", "xaxis2"):
            patch["layout"][axis]["range"] = target
            patch["layout"][axis]["autorange"] = False
    elif relayout_range(relayout, duration) == [0.0, duration] and "xaxis.range[0]" not in (relayout or {}):
        # Axis reset: make the explicit range match the server-side view.
        for axis in ("xaxis", "xaxis2"):
            patch["layout"][axis]["range"] = target
            patch["layout"][axis]["autorange"] = False
    return patch, {"recording": recording, "range": [round(target[0], 3), round(target[1], 3)]}


@app.callback(Output("view-label", "children"), Input("view", "data"))
def view_label(view):
    if not view:
        return ""
    start, end = view["range"]
    return f"{start:.1f} s – {end:.1f} s"


@app.callback(
    Output("eeg", "figure", allow_duplicate=True),
    Input("threshold", "value"), State("figure-meta", "data"), State("theme", "data"),
    prevent_initial_call=True,
)
def update_detections(threshold, meta, theme):
    """Threshold changes only redraw the shading and the threshold line."""
    if not meta or not meta.get("prediction"):
        return no_update
    detected, _, _ = DATA.detect(meta["recording"], threshold, meta["model"])
    patch = Patch()
    patch["layout"]["shapes"] = event_shapes(DATA.real(meta["recording"]), detected, threshold, meta["probability_ref"], theme)
    return patch


@app.callback(
    Output("spectrum-point", "data"),
    Input("eeg", "clickData"), Input("recording", "value"), State("figure-meta", "data"),
    prevent_initial_call=True,
)
def select_window(click, recording, meta):
    if ctx.triggered_id == "recording" or not click or not meta:
        return None
    point = click["points"][0]
    channel = meta["curve_channels"].get(str(point.get("curveNumber")))
    return {"recording": meta["recording"], "time": float(point["x"]), "channel": channel}


@app.callback(
    Output("spectrum", "figure"), Output("spectrum-caption", "children"), Output("spectrum-bands", "children"),
    Input("spectrum-point", "data"), Input("theme", "data"), Input("channels", "value"),
)
def show_spectrum(point, theme, channels):
    channels = [channel for channel in CHANNELS if channel in (channels or [])]
    if not point or not channels:
        return (
            message_figure("Click a trace in the EEG plot<br>to see the spectrum of that 2 s window.", theme, 300),
            "Shaded bands: δ Delta 0.5–4 Hz, θ Theta 4–8 Hz, α Alpha 8–13 Hz, β Beta 13–30 Hz.", None,
        )
    result = app_data.spectrum(point["recording"], point["time"], channels)
    highlighted = point["channel"] if point["channel"] in channels else None
    reference = highlighted or channels[0]
    shares = result["relative"][reference]
    dominant = max(shares, key=shares.get)
    bands = dmc.Group(
        [dmc.Text(f"{reference} relative power", size="xs", c="dimmed")]
        + [
            dmc.Badge(f"{GREEK[band]} {band} {share:.0%}", variant="filled" if band == dominant else "light", color="gray", className="mono", tt="none")
            for band, share in shares.items()
        ],
        gap="xs", mt=4,
    )
    caption = f"{point['recording']} · window {result['start_s']:.2f} s – {result['end_s']:.2f} s" + (f" · clicked channel {highlighted}" if highlighted else "")
    return spectrum_figure(result, channels, highlighted, theme), caption, bands


@app.callback(
    Output("download", "data"), Input("save-csv", "n_clicks"),
    State("recording", "value"), State("threshold", "value"), State("model", "value"), State("page", "value"),
    prevent_initial_call=True,
)
def download_intervals(_, recording, threshold, model, page):
    if page != "prediction":
        real = DATA.real(recording)
        table = real.assign(duration_s=real["end_s"] - real["start_s"])
        table.insert(0, "recording", recording)
        table.insert(1, "seizure", range(1, len(table) + 1))
        return dcc.send_data_frame(table.to_csv, f"{recording}_real_seizures.csv", index=False)
    _, matches, _ = DATA.detect(recording, threshold, model)
    table = matches.rename(columns={"status": "status_code"})
    table.insert(0, "recording", recording)
    table.insert(1, "model", model)
    table.insert(2, "threshold", threshold)
    table.insert(3, "status", table["status_code"].map({code: label for code, (label, _, _) in STATUS.items()}))
    table = table.drop(columns="status_code").round({"max_proba": 2})
    name = f"{recording}_{model.lower().replace(' ', '_')}_threshold_{threshold:.2f}.csv"
    return dcc.send_data_frame(table.to_csv, name, index=False)


# High-resolution PNG of the current view on the theme's panel color.
clientside_callback(
    """
    function(clicks, view) {
        if (!clicks) { return window.dash_clientside.no_update; }
        const plot = document.querySelector('#eeg .js-plotly-plot');
        if (!plot || !window.Plotly) { return window.dash_clientside.no_update; }
        const dark = document.documentElement.getAttribute('data-mantine-color-scheme') === 'dark';
        const background = dark ? '#111A2E' : '#FFFFFF';
        const layout = Object.assign({}, plot.layout, {paper_bgcolor: background, plot_bgcolor: background});
        const name = view ? `${view.recording}_${view.range[0].toFixed(0)}-${view.range[1].toFixed(0)}s` : 'eeg_view';
        window.Plotly.toImage({data: plot.data, layout: layout}, {format: 'png', scale: 3, width: plot.offsetWidth, height: plot.offsetHeight})
            .then(function(url) {
                const link = document.createElement('a');
                link.href = url; link.download = name + '.png';
                document.body.appendChild(link); link.click(); link.remove();
            });
        return window.dash_clientside.no_update;
    }
    """,
    Output("save-png", "n_clicks"), Input("save-png", "n_clicks"), State("view", "data"),
    prevent_initial_call=True,
)

# Left/right arrows turn pages unless a form control has focus.
clientside_callback(
    """
    function(_) {
        if (!window.eegKeysBound) {
            window.eegKeysBound = true;
            document.addEventListener('keydown', function(event) {
                const target = event.target;
                const tag = target.tagName;
                if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || target.isContentEditable || target.getAttribute('role') === 'slider') { return; }
                if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') { return; }
                const page = document.getElementById('page-recording');
                if (!page || page.style.display === 'none') { return; }
                const button = document.getElementById(event.key === 'ArrowLeft' ? 'prev-page' : 'next-page');
                if (button) { button.click(); event.preventDefault(); }
            });
        }
        return true;
    }
    """,
    Output("keys-ready", "data"), Input("session", "data"),
)


if __name__ == "__main__":
    print("EEG Absence Detector: http://127.0.0.1:8050")
    app.run(debug=False, host="127.0.0.1", port=8050)
