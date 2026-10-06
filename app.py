"""EEG Absence Detector — interactive review of recordings, annotations and detections.

Run with ``python app.py``. Only precomputed files are read; nothing is trained here.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from urllib.parse import parse_qs
import sys
import traceback
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
    DETECTED, EVENT, REAL, THRESHOLD, confusion_figure, curves_figure, event_shapes, message_figure, model_style,
    patient_comparison_figure, recording_figure, resampled_dict, spectrum_figure, training_distribution_figure, view_relayout,
)
from src.io import CHANNELS


try:
    DATA = app_data.load()
    LOAD_ERROR = None
except app_data.MissingData as error:
    DATA, LOAD_ERROR = None, str(error)

# Channel selector grouped by scalp region; the curves themselves stay in 10-20 order.
REGION_NAMES = {"Fp": "Frontal", "F": "Frontal", "C": "Central", "T": "Temporal", "P": "Parietal", "O": "Occipital"}
CHANNEL_GROUPS = [
    {"group": region, "items": [channel for channel in CHANNELS if REGION_NAMES[channel.rstrip("0123456789z")] == region]}
    for region in ("Frontal", "Central", "Temporal", "Parietal", "Occipital")
]
GREEK = {"Delta": "δ", "Theta": "θ", "Alpha": "α", "Beta": "β"}
# URL values for ?page=…, including the names used before the app was split in three parts.
# Ranking columns: label, whether higher is better, formatter.
RANKING = {
    "event_f1": ("Event F1", True, lambda value: f"{value:.3f}"),
    "event_recall": ("Event recall", True, lambda value: f"{value:.0%}"),
    "event_precision": ("Event precision", True, lambda value: f"{value:.0%}"),
    "found": ("Found", True, lambda value: f"{int(value)}"),
    "missed": ("Missed", False, lambda value: f"{int(value)}"),
    "false_alarms": ("False alarms", False, lambda value: f"{int(value)}"),
    "false_alarms_per_hour": ("FA / hour", False, lambda value: f"{value:.1f}"),
    "window_f1": ("Window F1", True, lambda value: f"{value:.3f}"),
    "window_recall": ("Window recall", True, lambda value: f"{value:.0%}"),
    "window_precision": ("Window precision", True, lambda value: f"{value:.0%}"),
    "onset_error": ("Onset error (s)", False, lambda value: f"{value:+.2f}"),
    "offset_error": ("Offset error (s)", False, lambda value: f"{value:+.2f}"),
    "fit_seconds": ("Training (s/fold)", False, lambda value: f"{value:.2f}"),
}
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
    return dmc.Badge(label, color=color, variant="light", leftSection=icon(icon_name, 12), radius="sm", miw=112, style={"flexShrink": 0}, styles={"label": {"overflow": "visible"}})


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
                                dmc.Text("Visualise the EEG, test a model on a patient, compare models", size="xs", c="dimmed", visibleFrom="xl"),
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
                            "Algorithm of the model used in Prediction.",
                            dmc.SegmentedControl(
                                id="algo", value=DATA.default_model.split("_")[0], radius="md", size="xs",
                                data=[{"value": key, "label": label} for key, label in app_data.ALGORITHM_SHORT.items()],
                            ),
                        ),
                        tip(
                            "Training mode: Unbalanced trains on the real class proportions (about 96 % / 4 %); Balanced trains on as many normal windows as seizure windows (50 / 50).",
                            dmc.SegmentedControl(
                                id="train-mode", value=DATA.default_model.split("_")[1], radius="md", size="xs",
                                data=[{"value": key, "label": label} for key, label in app_data.MODES.items()],
                            ),
                        ),
                        html.Div(id="active-model"),
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
                                                id="threshold", min=0.05, max=0.99, step=0.01, value=DATA.best_threshold(DATA.default_model), color="violet",
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
                                            "Show every channel again.",
                                            dmc.Button("Show all", id="all-channels", variant="subtle", size="compact-xs", leftSection=icon("tabler:list-check", 12)),
                                        ),
                                    ],
                                    justify="space-between", mb=-8,
                                ),
                                wide_tip(
                                    "Curves to display, listed by scalp region: remove a channel with its ×, or pick one from the list to add it back.",                                    html.Div(
                                        dmc.MultiSelect(
                                            id="channels", data=CHANNEL_GROUPS, value=list(CHANNELS), clearable=False, maxDropdownHeight=320,
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
                                tip(
                                    "Show ECG, EMG and SLI in gray under the EEG. They are reference traces only: no model uses them. EMG exists in 6 recordings; a constant SLI is not drawn.",
                                    html.Div(dmc.Switch(id="reference", checked=False, label="Reference channels (ECG, EMG, SLI)", color="gray")),
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


def comparison_page():
    model_options = [
        {"group": name, "items": [{"value": f"{key}_{mode}", "label": DATA.label(f"{key}_{mode}", short=True)} for mode in app_data.MODES]}
        for key, name in app_data.ALGORITHMS.items()
    ]
    graph = lambda identifier: dcc.Loading(
        dcc.Graph(id=identifier, config={"displaylogo": False, "displayModeBar": False}),
        delay_show=300, overlay_style={"visibility": "visible", "opacity": 0.5}, type="dot",
    )
    return html.Div(
        dmc.Stack(
            [
                panel(
                    [
                        dmc.Group(
                            [
                                dmc.Stack(
                                    [
                                        dmc.Text("Models to compare", fw=600),
                                        dmc.Text(
                                            "Each model at its own best threshold, tested on patients it never saw during training. One color per algorithm; solid = unbalanced, dashed or hatched = balanced.",
                                            size="sm", c="dimmed",
                                        ),
                                    ],
                                    gap=0,
                                ),
                                dmc.Badge("Out-of-fold predictions", variant="light", color="gray", leftSection=icon("tabler:shield-check", 12)),
                            ],
                            justify="space-between", align="flex-start", mb="sm",
                        ),
                        dmc.Group(
                            [
                                tip(
                                    "Same training mode compares the 4 algorithms; Same algorithm compares unbalanced with balanced; All models shows the 8; Custom keeps the list you edit.",
                                    dmc.SegmentedControl(
                                        id="cmp-preset", value="mode", radius="md",
                                        data=[
                                            {"value": "mode", "label": "Same training mode"},
                                            {"value": "algorithm", "label": "Same algorithm, both modes"},
                                            {"value": "all", "label": "All models"},
                                            {"value": "custom", "label": "Custom"},
                                        ],
                                    ),
                                ),
                                html.Div(
                                    tip(
                                        "Training mode whose 4 algorithms are compared.",
                                        dmc.SegmentedControl(
                                            id="cmp-training", value="unbalanced", radius="md",
                                            data=[{"value": key, "label": label} for key, label in app_data.MODES.items()],
                                        ),
                                    ),
                                    id="cmp-training-box",
                                ),
                                html.Div(
                                    tip(
                                        "Algorithm whose two training modes are compared.",
                                        dmc.SegmentedControl(
                                            id="cmp-algo", value="rf", radius="md",
                                            data=[{"value": key, "label": label} for key, label in app_data.ALGORITHM_SHORT.items()],
                                        ),
                                    ),
                                    id="cmp-algo-box", style={"display": "none"},
                                ),
                            ],
                            gap="sm",
                        ),
                        wide_tip(
                            "Models shown below. Add or remove any model to build your own comparison.",
                            html.Div(
                                dmc.MultiSelect(
                                    id="cmp-models", data=model_options, value=[f"{key}_unbalanced" for key in app_data.ALGORITHMS],
                                    clearable=False, mt="sm", leftSection=icon("tabler:cpu", 16), comboboxProps={"withinPortal": True}, maxDropdownHeight=360,
                                )
                            ),
                        ),
                    ]
                ),
                html.Div(id="cmp-empty"),
                html.Div(
                    dmc.Stack(
                        [
                            dmc.SimpleGrid(id="cmp-cards", cols={"base": 1, "sm": 2, "xl": 4}, spacing="md"),
                            panel(
                                [
                                    dmc.Group(
                                        [
                                            dmc.Stack(
                                                [
                                                    dmc.Text("Ranking", fw=600),
                                                    dmc.Text("All patients together. The best value of each column is highlighted.", size="sm", c="dimmed"),
                                                ],
                                                gap=0,
                                            ),
                                            tip(
                                                "Column used to sort the ranking; scores sort from highest, errors and false alarms from lowest.",
                                                html.Div(
                                                    dmc.Select(
                                                        id="cmp-sort", value="event_f1", allowDeselect=False, w=230, leftSection=icon("tabler:arrows-sort", 16),
                                                        data=[{"value": key, "label": f"Sort by {label}"} for key, (label, *_rest) in RANKING.items()],
                                                        comboboxProps={"withinPortal": True},
                                                    )
                                                ),
                                            ),
                                        ],
                                        justify="space-between", align="flex-start", mb="xs",
                                    ),
                                    html.Div(id="cmp-ranking"),
                                ]
                            ),
                            dmc.Grid(
                                [
                                    dmc.GridCol(
                                        panel(
                                            [card_title("Event F1 against the threshold", "dot = best threshold of each model"), graph("cmp-f1")], h="100%",
                                        ),
                                        span={"base": 12, "lg": 6},
                                    ),
                                    dmc.GridCol(
                                        panel(
                                            [card_title("Precision–recall curves", "per 2 s window; dot = best threshold"), graph("cmp-pr")], h="100%",
                                        ),
                                        span={"base": 12, "lg": 6},
                                    ),
                                ],
                                gutter="md",
                            ),
                            dmc.Grid(
                                [
                                    dmc.GridCol(
                                        panel(
                                            [card_title("Training data of each mode", "average of the 5 training folds"), graph("cmp-distribution")], h="100%",
                                        ),
                                        span={"base": 12, "lg": 5},
                                    ),
                                    dmc.GridCol(
                                        panel(
                                            [card_title("Confusion matrices", "2 s windows at each model's best threshold; % of the true class"), graph("cmp-confusion")], h="100%",
                                        ),
                                        span={"base": 12, "lg": 7},
                                    ),
                                ],
                                gutter="md",
                            ),
                            panel([html.Div(id="cmp-timeline-title"), graph("cmp-timeline")]),
                            panel(
                                [
                                    card_title("Event F1 per patient", "found / real seizures · false alarms; best of each row highlighted; selected patient marked"),
                                    html.Div(id="cmp-patients"),
                                ]
                            ),
                        ],
                        gap="md",
                    ),
                    id="cmp-body",
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
                html.Div(id="model-overview"),
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
            dcc.Store(id="model", data=DATA.default_model),
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
                            comparison_page(),
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
    """Show a friendly message in the page; the full traceback goes to the terminal."""
    traceback.print_exception(error)
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
        # No model to choose here: the four are always shown together.
        return shown, hidden, hidden, hidden
    return hidden, shown, shown, (shown if page == "prediction" else hidden)


@app.callback(Output("model", "data"), Output("active-model", "children"), Input("algo", "value"), Input("train-mode", "value"), Input("theme", "data"))
def choose_model(algorithm, mode, theme):
    model = f"{algorithm}_{mode}"
    style = model_style(model, theme)
    badge = tip(
        "Model currently used in Prediction. Line style in Comparison: solid = unbalanced, dashed = balanced.",
        dmc.Badge(
            DATA.label(model), variant="light", color="gray", radius="md", size="lg", tt="none", visibleFrom="lg",
            leftSection=html.Span(style={"display": "inline-block", "width": 14, "borderTop": f"3px {'dashed' if mode == 'balanced' else 'solid'} {style['color']}"}),
        ),
    )
    return model, badge


@app.callback(Output("threshold-value", "children"), Input("threshold", "value"))
def threshold_label(threshold):
    return f"{threshold:.2f}"


@app.callback(
    Output("threshold", "value"), Input("model", "data"), Input("threshold-reset", "n_clicks"),
    prevent_initial_call=True,
)
def model_threshold(model, _):
    """Each model starts at its own best threshold; the reset button returns to it."""
    return DATA.best_threshold(model)


@app.callback(Output("channels", "value"), Input("all-channels", "n_clicks"), prevent_initial_call=True)
def show_all_channels(_):
    return list(CHANNELS)


# ----------------------------------------------------------------------------- overview


def percent(value) -> str:
    return "—" if pd.isna(value) else f"{value:.0%}"


def verdict_label(metrics: dict) -> tuple[str, str, str]:
    """Label, color and icon for one model on one patient."""
    total = metrics["tp"] + metrics["fn"]
    if total and metrics["fn"] == 0 and metrics["fp"] == 0:
        return "Predicts well", "teal", "tabler:circle-check"
    if total and (metrics["tp"] < total / 2 or metrics["fp"] > 2 * total):
        return "Predicts poorly", "red", "tabler:circle-x"
    return "Partly correct", "yellow", "tabler:alert-circle"


def model_swatch(model: str, theme: str, width: int = 22):
    """Small line in the model's color and style, so identity never relies on the name alone."""
    style = model_style(model, theme)
    line = "dashed" if style["dash"] == "dash" else "solid"
    return html.Span(style={"display": "inline-block", "width": width, "borderTop": f"3px {line} {style['color']}", "verticalAlign": "middle", "marginRight": 8})


def ranking_values(model: str) -> dict:
    entry = DATA.entry(model)
    best = entry["at_best_threshold"]
    event, window = best["event"], best["window"]
    return {
        "event_f1": event["f1"], "event_recall": event["recall"], "event_precision": event["precision"],
        "found": event["tp"], "missed": event["fn"], "false_alarms": event["fp"], "false_alarms_per_hour": event["false_alarms_per_hour"],
        "window_f1": window["f1"], "window_recall": window["recall"], "window_precision": window["precision"],
        "onset_error": abs(event["onset_error_mean_s"]) if event["tp"] else float("nan"),
        "offset_error": abs(event["offset_error_mean_s"]) if event["tp"] else float("nan"),
        "fit_seconds": entry["fit_seconds_per_fold"], "threshold": best["threshold"],
        "onset_signed": event["onset_error_mean_s"] if event["tp"] else float("nan"),
        "offset_signed": event["offset_error_mean_s"] if event["tp"] else float("nan"),
    }


@app.callback(
    Output("cmp-models", "value"), Output("cmp-preset", "value"), Output("cmp-training-box", "style"), Output("cmp-algo-box", "style"),
    Input("cmp-preset", "value"), Input("cmp-training", "value"), Input("cmp-algo", "value"), Input("cmp-models", "value"),
)
def choose_comparison(preset, training, algorithm, models):
    """Presets fill the model list; editing the list by hand switches to Custom."""
    shown, hidden = {"display": "block"}, {"display": "none"}
    presets = {
        "mode": [f"{key}_{training}" for key in app_data.ALGORITHMS],
        "algorithm": [f"{algorithm}_{mode}" for mode in app_data.MODES],
        "all": DATA.models,
    }
    if ctx.triggered_id == "cmp-models":
        ordered = [model for model in DATA.models if model in (models or [])]
        matching = next((name for name in (preset, "mode", "algorithm", "all") if presets.get(name) == ordered), "custom")
        preset = matching
    elif preset != "custom":
        ordered = presets[preset]
    else:
        ordered = [model for model in DATA.models if model in (models or [])]
    return ordered, preset, shown if preset == "mode" else hidden, shown if preset == "algorithm" else hidden


@app.callback(
    Output("cmp-empty", "children"), Output("cmp-body", "style"), Output("cmp-cards", "children"), Output("cmp-ranking", "children"),
    Input("cmp-models", "value"), Input("cmp-sort", "value"), Input("theme", "data"),
)
def comparison_summary(models, sort, theme):
    models = [model for model in DATA.models if model in (models or [])]
    if not models:
        message = dmc.Alert("Choose at least one model in the list above to see the comparison.", color="gray", icon=icon("tabler:info-circle"))
        return message, {"display": "none"}, [], None
    values = {model: ranking_values(model) for model in models}
    cards = []
    for model in models:
        value = values[model]
        cards.append(
            panel(
                dmc.Stack(
                    [
                        dmc.Group([html.Span([model_swatch(model, theme), dmc.Text(DATA.label(model), fw=600, size="sm", span=True)]), dmc.Badge(f"threshold {value['threshold']:.2f}", size="sm", variant="light", color="violet", tt="none", className="mono")], justify="space-between"),
                        dmc.Group(
                            [
                                dmc.Stack([dmc.Text("Event F1", size="xs", c="dimmed", fw=600, tt="uppercase"), dmc.Text(f"{value['event_f1']:.2f}", className="mono kpi-value")], gap=0),
                                dmc.Stack(
                                    [
                                        dmc.Text(f"recall {value['event_recall']:.0%} · precision {value['event_precision']:.0%}", size="xs", className="mono"),
                                        dmc.Text(f"{value['found']} found · {value['missed']} missed", size="xs", className="mono", c="dimmed"),
                                        dmc.Text(f"{value['false_alarms']} false alarms ({value['false_alarms_per_hour']:.1f}/h)", size="xs", className="mono", c="dimmed"),
                                    ],
                                    gap=0,
                                ),
                            ],
                            justify="space-between", align="flex-end",
                        ),
                    ],
                    gap="xs",
                )
            )
        )
    _, higher, _ = RANKING[sort]
    ordered = sorted(models, key=lambda model: (pd.isna(values[model][sort]), -values[model][sort] if higher else values[model][sort]))
    best = {}
    for key, (_, better_high, _) in RANKING.items():
        column = [values[model][key] for model in models if not pd.isna(values[model][key])]
        best[key] = (max(column) if better_high else min(column)) if column else None
    head = ["#", "Model", "Threshold", *[label for label, *_rest in RANKING.values()]]
    rows = []
    for rank, model in enumerate(ordered, 1):
        value = values[model]
        cells = [dmc.TableTd(str(rank), className="mono"), dmc.TableTd(html.Span([model_swatch(model, theme), dmc.Text(DATA.label(model), fw=600, size="sm", span=True)]), style={"whiteSpace": "nowrap"}), dmc.TableTd(f"{value['threshold']:.2f}", className="mono")]
        for key, (_, _, formatter) in RANKING.items():
            shown = value[{"onset_error": "onset_signed", "offset_error": "offset_signed"}.get(key, key)]
            is_best = best[key] is not None and not pd.isna(value[key]) and value[key] == best[key] and len(models) > 1
            cells.append(
                dmc.TableTd(
                    "—" if pd.isna(shown) else formatter(shown), className="mono",
                    style={"whiteSpace": "nowrap", **({"fontWeight": 700, "background": "var(--mantine-color-teal-light)"} if is_best else {})},
                )
            )
        rows.append(dmc.TableTr(cells))
    table = dmc.TableScrollContainer(
        dmc.Table(
            [dmc.TableThead(dmc.TableTr([dmc.TableTh(name, style={"whiteSpace": "nowrap", **({"textDecoration": "underline"} if name == RANKING[sort][0] else {})}) for name in head])), dmc.TableTbody(rows)],
            highlightOnHover=True, verticalSpacing=8, horizontalSpacing=10, fz="sm",
        ),
        minWidth=1250, type="native",
    )
    return None, {"display": "block"}, cards, table


@app.callback(
    Output("cmp-f1", "figure"), Output("cmp-pr", "figure"), Output("cmp-distribution", "figure"), Output("cmp-confusion", "figure"),
    Input("cmp-models", "value"), Input("theme", "data"),
)
def comparison_charts(models, theme):
    models = [model for model in DATA.models if model in (models or [])]
    if not models:
        empty = message_figure("No model selected.", theme, 200)
        return empty, empty, empty, empty
    f1_curves, pr_curves, confusion = {}, {}, []
    for model in models:
        entry = DATA.entry(model)
        best = entry["at_best_threshold"]
        label = DATA.label(model, short=True)
        f1_curves[model] = (label, DATA.sweeps.loc[DATA.sweeps["model"] == model], (best["threshold"], best["event"]["f1"]))
        pr_curves[model] = (label, DATA.pr_curves.loc[DATA.pr_curves["model"] == model], (best["window"]["recall"], best["window"]["precision"]))
        confusion.append((model, label, best["window"]))
    modes = [mode for mode in app_data.MODES if any(model.endswith(mode) for model in models)]
    distribution = [(app_data.MODES[mode], DATA.entry(f"rf_{mode}")["training"]) for mode in modes]
    return (
        curves_figure(f1_curves, "threshold", "f1", "Probability threshold", "Event F1", theme),
        curves_figure(pr_curves, "recall", "precision", "Recall", "Precision", theme),
        training_distribution_figure(distribution, theme),
        confusion_figure(confusion, theme),
    )


@app.callback(
    Output("cmp-timeline-title", "children"), Output("cmp-timeline", "figure"), Output("cmp-patients", "children"),
    Input("cmp-models", "value"), Input("recording", "value"), Input("theme", "data"),
)
def comparison_patients(models, recording, theme):
    models = [model for model in DATA.models if model in (models or [])]
    title = card_title(f"Timeline of patient {recording}", "real seizures, then one row per model; faded = false alarm")
    if not models:
        return title, message_figure("No model selected.", theme, 200), None
    results = {}
    for model in models:
        detected, matches, metrics = DATA.detect(recording, DATA.best_threshold(model), model)
        results[model] = (DATA.label(model, short=True), detected, matches, metrics)
    figure = patient_comparison_figure(recording, float(DATA.info(recording)["duration_s"]), DATA.real(recording), results, theme)

    table = DATA.per_recording.loc[DATA.per_recording["model"].isin(models)].set_index(["recording", "model"])
    head = [dmc.TableTh("Patient"), dmc.TableTh("Real seizures")] + [
        dmc.TableTh(html.Span([model_swatch(model, theme, 16), DATA.label(model, short=True)]), style={"whiteSpace": "nowrap"}) for model in models
    ]
    rows = []
    for patient in DATA.recordings:
        scores = {model: table.loc[(patient, model)] for model in models}
        top = max(score.event_f1 for score in scores.values())
        cells = [
            dmc.TableTd([dmc.Text(patient, className="mono", size="sm", span=True)] + ([dmc.Badge("Selected", size="xs", variant="light", color="teal", ml=6)] if patient == recording else [])),
            dmc.TableTd(str(int(DATA.info(patient)["seizures"])), className="mono"),
        ]
        for model in models:
            score = scores[model]
            is_best = score.event_f1 == top and top > 0 and len(models) > 1
            cells.append(
                dmc.TableTd(
                    [dmc.Text(f"{score.event_f1:.2f}", span=True, fw=700 if is_best else 400), dmc.Text(f"  {int(score.found)}/{int(score.found + score.missed)} · {int(score.false_alarms)}", span=True, c="dimmed", size="xs")],
                    className="mono", style={"whiteSpace": "nowrap", **({"background": "var(--mantine-color-teal-light)"} if is_best else {})},
                )
            )
        rows.append(dmc.TableTr(cells))
    patients = dmc.TableScrollContainer(
        dmc.Table([dmc.TableThead(dmc.TableTr(head)), dmc.TableTbody(rows)], highlightOnHover=True, verticalSpacing=6, horizontalSpacing=10, fz="sm", stickyHeader=True),
        minWidth=260 + 150 * len(models), mah=560, type="native",
    )
    return title, figure, patients


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
    label, color, icon_name = verdict_label(metrics)
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
                            dmc.Badge(f"{metrics['fp']} false alarm{'s' if metrics['fp'] != 1 else ''}", color="red", variant="light", leftSection=icon("tabler:alert-triangle", 12)),
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
    Input("recording", "value"), Input("threshold", "value"), Input("model", "data"), Input("page", "value"),
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
    table = [card_title(f"Real vs predicted — {DATA.label(model)}", "times in seconds; errors are detected − real"), intervals_table(matches)]
    return header_row, verdict(DATA.label(model), threshold, metrics), table


@app.callback(Output("model-overview", "children"), Input("model", "data"), Input("threshold", "value"), Input("page", "value"))
def model_overview(model, threshold, page):
    """How the active model does on all patients at the current threshold."""
    if page != "prediction":
        return None
    _, _, metrics = DATA.detect_everything(threshold, model)
    total = metrics["tp"] + metrics["fn"]
    facts = [
        ("Event F1", f"{metrics['f1']:.2f}"),
        ("Recall", f"{metrics['recall']:.0%} ({metrics['tp']} of {total} seizures)"),
        ("Precision", f"{metrics['precision']:.0%} of {metrics['detections']} detections"),
        ("Missed", str(metrics["fn"])),
        ("False alarms", f"{metrics['fp']} ({metrics['false_alarms_per_hour']:.1f} per hour)"),
    ]
    return panel(
        dmc.Group(
            [
                dmc.Stack(
                    [dmc.Text("This model on all 21 patients", fw=600, size="sm"), dmc.Text(f"{DATA.label(model)} at threshold {threshold:.2f}", size="xs", c="dimmed")],
                    gap=0,
                ),
                dmc.Group(
                    [dmc.Stack([dmc.Text(label, size="xs", c="dimmed"), dmc.Text(value, className="mono", size="sm", fw=500)], gap=0) for label, value in facts],
                    gap="xl",
                ),
            ],
            justify="space-between",
        ),
        py="sm",
    )


def page_length(value: str, duration: float) -> float:
    return duration if value == "full" else min(float(value), duration)


def clamp(start: float, length: float, duration: float) -> list[float]:
    start = max(0.0, min(start, duration - length))
    return [round(start, 3), round(start + length, 3)]


@app.callback(
    Output("eeg", "figure"), Output("figure-meta", "data"), Output("view", "data"),
    Input("recording", "value"), Input("channels", "value"), Input("mode", "value"), Input("gain", "value"),
    Input("events", "checked"), Input("reference", "checked"), Input("theme", "data"), Input("page", "value"), Input("model", "data"),
    State("threshold", "value"), State("view", "data"), State("page-length", "value"), State("session", "data"),
)
def build_eeg(recording, channels, mode, gain, show_events, show_reference, theme, page, model, threshold, view, length, session):
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
    references = None
    if show_reference:
        table = app_data.reference(recording)
        references = {name: table[name].to_numpy() for name in ("ECG", "EMG1", "EMG2", "SLI") if name in table}
    figure, description = recording_figure(
        frame["Time"].to_numpy(), {channel: frame[channel].to_numpy() for channel in channels},
        DATA.windows(recording, model) if prediction else None,
        DATA.real(recording), detected, DATA.technician(recording) if show_events else None, threshold if prediction else None,
        stacked=mode == "stacked", full_scale=app_data.amplitude_scale(recording), gain=0.5 * 2 ** gain,
        view=tuple(view_range), theme=theme, references=references,
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
    State("recording", "value"), State("threshold", "value"), State("model", "data"), State("page", "value"),
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
    table.insert(1, "model", DATA.label(model))
    table.insert(2, "threshold", threshold)
    table.insert(3, "status", table["status_code"].map({code: label for code, (label, _, _) in STATUS.items()}))
    table = table.drop(columns="status_code").round({"max_proba": 2})
    name = f"{recording}_{model}_threshold_{threshold:.2f}.csv"
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
