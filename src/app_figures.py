"""Themed Plotly figures for app.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly_resampler import FigureResampler

from src.app_data import BANDS
from src.io import CHANNELS


# Semantic colors, identical in plots, tables, badges and legends.
REAL = "#34D399"
DETECTED = "#F87171"
EVENT = "#FBBF24"
THRESHOLD = "#A78BFA"
UI_FONT = "Inter, system-ui, -apple-system, Segoe UI, sans-serif"
MONO_FONT = "JetBrains Mono, ui-monospace, SFMono-Regular, Menlo, monospace"

THEMES = {
    "dark": {
        "background": "#0B1220", "panel": "#111A2E", "text": "#E5E9F0", "muted": "#94A3B8",
        "grid": "rgba(148,163,184,0.13)", "axis": "rgba(148,163,184,0.35)", "track": "rgba(148,163,184,0.10)",
        "probability": "#CBD5E1",
        # Region hues (frontal, central, temporal, posterior); channels of a region share a hue.
        "regions": {"frontal": "#4FD1C5", "central": "#3B82F6", "temporal": "#E05599", "posterior": "#4698CF"},
    },
    "light": {
        "background": "#F6F8FB", "panel": "#FFFFFF", "text": "#0F172A", "muted": "#64748B",
        "grid": "rgba(15,23,42,0.08)", "axis": "rgba(15,23,42,0.30)", "track": "rgba(15,23,42,0.05)",
        "probability": "#475569",
        "regions": {"frontal": "#0D9488", "central": "#2563EB", "temporal": "#DB2777", "posterior": "#0284C7"},
    },
}


def rgba(color: str, alpha: float) -> str:
    red, green, blue = (int(color[index : index + 2], 16) for index in (1, 3, 5))
    return f"rgba({red},{green},{blue},{alpha})"


# Scalp region of each electrode family; parietal and occipital form the posterior region.
REGIONS = {"Fp": "frontal", "F": "frontal", "C": "central", "T": "temporal", "P": "posterior", "O": "posterior"}


def channel_color(channel: str, theme: str) -> str:
    family = channel.rstrip("0123456789z")
    return THEMES[theme]["regions"][REGIONS[family]]


def base_layout(theme: str) -> dict:
    tokens = THEMES[theme]
    return {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": UI_FONT, "color": tokens["text"], "size": 12},
        "hoverlabel": {"font": {"family": MONO_FONT, "size": 12}, "bgcolor": tokens["panel"], "bordercolor": tokens["axis"]},
        "legend": {
            "orientation": "h", "x": 0, "y": 1.0, "xanchor": "left", "yanchor": "bottom",
            "font": {"size": 12, "color": tokens["text"]}, "bgcolor": "rgba(0,0,0,0)",
        },
        "modebar": {"bgcolor": "rgba(0,0,0,0)", "color": tokens["muted"], "activecolor": tokens["text"]},
    }


def axis_style(theme: str, **overrides) -> dict:
    tokens = THEMES[theme]
    style = {
        "gridcolor": tokens["grid"], "zerolinecolor": tokens["grid"], "linecolor": tokens["axis"],
        "tickfont": {"family": MONO_FONT, "size": 11, "color": tokens["muted"]},
        "title": {"font": {"size": 12, "color": tokens["muted"]}},
    }
    style.update(overrides)
    return style


def message_figure(text: str, theme: str, height: int = 320) -> go.Figure:
    """Friendly empty or error state."""
    figure = go.Figure()
    figure.update_layout(
        **base_layout(theme), height=height, margin={"l": 20, "r": 20, "t": 20, "b": 20},
        xaxis={"visible": False}, yaxis={"visible": False},
        annotations=[{"text": text, "showarrow": False, "font": {"size": 14, "color": THEMES[theme]["muted"]}, "xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5}],
    )
    return figure


# ----------------------------------------------------------------------------- recording view


def legend_key(name: str, color: str, symbol: str = "square", dash: str | None = None, **axes) -> go.Scatter:
    """Legend-only entry explaining a semantic color."""
    if dash:
        return go.Scatter(x=[None], y=[None], mode="lines", name=name, line={"color": color, "dash": dash, "width": 2}, hoverinfo="skip", **axes)
    return go.Scatter(x=[None], y=[None], mode="markers", name=name, marker={"color": color, "size": 11, "symbol": symbol}, hoverinfo="skip", **axes)


def event_shapes(
    real: pd.DataFrame, detected: pd.DataFrame | None, threshold: float | None, probability_axis: str, theme: str
) -> list[dict]:
    """Seizure shading, plus detection shading and the threshold line when a model is shown."""
    if detected is None:
        detected = pd.DataFrame(columns=["start_s", "end_s"])
    labels = len(real) + len(detected) <= 40
    tokens = THEMES[theme]
    shapes = []
    for number, row in enumerate(real.itertuples(index=False), 1):
        shape = {
            "type": "rect", "xref": "x", "yref": "paper", "x0": row.start_s, "x1": row.end_s, "y0": 0, "y1": 1,
            "fillcolor": rgba(REAL, 0.20), "line": {"width": 0}, "layer": "below",
        }
        if labels:
            shape["label"] = {"text": f"Seizure {number}", "textposition": "top left", "font": {"size": 10, "color": tokens["text"]}, "padding": 3}
        shapes.append(shape)
    for number, row in enumerate(detected.itertuples(index=False), 1):
        shape = {
            "type": "rect", "xref": "x", "yref": "paper", "x0": row.start_s, "x1": row.end_s, "y0": 0, "y1": 1,
            "fillcolor": rgba(DETECTED, 0.16), "line": {"width": 1, "color": rgba(DETECTED, 0.9), "dash": "dot"}, "layer": "below",
        }
        if labels:
            shape["label"] = {"text": f"Detected {number}", "textposition": "bottom left", "font": {"size": 10, "color": tokens["text"]}, "padding": 3}
        shapes.append(shape)
    if threshold is not None:
        shapes.append(
            {
                "type": "line", "xref": "x2 domain", "yref": probability_axis, "x0": 0, "x1": 1, "y0": threshold, "y1": threshold,
                "line": {"color": THRESHOLD, "width": 1.6, "dash": "dash"},
                "label": {"text": f"Threshold {threshold:.2f}", "textposition": "end", "yanchor": "bottom", "font": {"size": 10, "color": tokens["text"]}},
            }
        )
    return shapes


def label_curve(real: pd.DataFrame, duration: float) -> tuple[list[float], list[float]]:
    """0/1 step curve of the annotations: it rises to 1 for the duration of each real seizure."""
    x, y = [0.0], [0.0]
    for row in real.itertuples(index=False):
        x += [row.start_s, row.start_s, row.end_s, row.end_s]
        y += [0.0, 1.0, 1.0, 0.0]
    return x + [duration], y + [0.0]


def recording_figure(
    time: np.ndarray,
    signals: dict[str, np.ndarray],
    windows: pd.DataFrame | None,
    real: pd.DataFrame,
    detected: pd.DataFrame | None,
    events: pd.DataFrame | None,
    threshold: float | None,
    stacked: bool,
    full_scale: float,
    gain: float,
    view: tuple[float, float],
    theme: str,
    references: dict[str, np.ndarray] | None = None,
) -> tuple[FigureResampler, dict]:
    """EEG rows + a bottom row sharing the time axis.

    ``references`` are non-EEG channels (ECG, EMG, SLI) drawn in gray under the EEG rows for
    orientation only: they are never model inputs.

    Without ``windows`` (visualisation) the bottom row is the real seizure curve only. With
    ``windows`` (prediction) it also shows the model probability, the threshold and detections.

    Returns the resampler (it keeps the full-resolution data server-side) and a small
    description of the axes used by callbacks.
    """
    tokens = THEMES[theme]
    channels = list(signals)
    rows = channels if stacked else ["EEG"]
    prediction = windows is not None
    has_events = events is not None and len(events) > 0
    has_strip = has_events or len(real) > 0
    duration = float(time[-1])
    probability_height, event_height = 140, 34
    # Rows get shorter as channels are added, so the full montage still fits on a screen.
    row_pixels = 78 if len(rows) <= 8 else 60 if len(rows) <= 12 else 46
    eeg_height = (row_pixels * len(rows) if stacked else 360)
    eeg_height = max(eeg_height, 200)
    gap = 26
    references = references or {}
    reference_pixels, reference_gap = 46, 14
    reference_height = reference_pixels * len(references) + (reference_gap if references else 0)
    plot_height = eeg_height + reference_height + probability_height + gap + (event_height + 6 if has_strip else 0)
    # The top margin holds the toolbar and, below it, up to two legend rows.
    margin = {"l": 64, "r": 24, "t": 76, "b": 36}
    height = plot_height + margin["t"] + margin["b"]

    def fraction(pixels: float) -> float:
        return pixels / plot_height

    # Keep the channel names clean in the legend: no "[R]" prefix or aggregation-size suffix.
    figure = FigureResampler(
        go.Figure(), default_n_shown_samples=2000, resampled_trace_prefix_suffix=("", ""), show_mean_aggregation_size=False,
    )
    layout: dict = {}
    # Axis 1 is reserved for the technician-event strip, 2.. for EEG rows, the last for probability.
    top = 1.0
    if has_strip:
        layout["yaxis"] = axis_style(
            theme, domain=[top - fraction(event_height), top], range=[-1, 1], showgrid=False, zeroline=False,
            tickvals=[0], ticktext=["Marks"], fixedrange=True, anchor="x",
        )
        top -= fraction(event_height + 6)
    row_height = fraction(eeg_height) / len(rows)
    limit = full_scale / gain
    eeg_axes = []
    for index, row in enumerate(rows):
        number = index + 2
        key = f"yaxis{number}"
        domain = [top - (index + 1) * row_height + fraction(3), top - index * row_height]
        layout[key] = axis_style(
            theme, domain=domain, range=[-limit, limit], fixedrange=True, anchor="x", zeroline=False,
            showgrid=not stacked, nticks=3 if stacked else 7,
            **(
                {"tickvals": [0], "ticktext": [row], "tickfont": {"family": MONO_FONT, "size": 12, "color": channel_color(row, theme)}}
                if stacked else {"title": {"text": "Amplitude (unit not provided)", "font": {"size": 12, "color": tokens["muted"]}}}
            ),
        )
        eeg_axes.append(key)
    reference_axes = {}
    reference_top = top - fraction(eeg_height) - fraction(reference_gap)
    for index, (name, values) in enumerate(references.items()):
        number = len(rows) + index + 2
        # Each reference channel has its own scale: their units differ and are not comparable to EEG.
        scale = max(float(np.percentile(np.abs(values[::8]), 99.5)) * 1.3, 1e-6)
        layout[f"yaxis{number}"] = axis_style(
            theme, anchor="x", fixedrange=True, zeroline=False, showgrid=False, range=[-scale, scale],
            domain=[reference_top - (index + 1) * fraction(reference_pixels) + fraction(3), reference_top - index * fraction(reference_pixels)],
            tickvals=[0], ticktext=[name], tickfont={"family": MONO_FONT, "size": 11, "color": tokens["muted"]},
        )
        reference_axes[name] = f"y{number}"
    probability_number = len(rows) + len(references) + 2
    probability_key = f"yaxis{probability_number}"
    probability_ref = f"y{probability_number}"
    layout[probability_key] = axis_style(
        theme, domain=[0, fraction(probability_height)], range=[0, 1.05], fixedrange=True, anchor="x2",
        **(
            {"tickvals": [0, 0.5, 1], "title": {"text": "Seizure<br>probability", "font": {"size": 11, "color": tokens["muted"]}}}
            if prediction else
            {"tickvals": [0, 1], "ticktext": ["no", "yes"], "title": {"text": "Seizure<br>(real)", "font": {"size": 11, "color": tokens["muted"]}}}
        ),
    )
    spikes = {"showspikes": True, "spikemode": "across", "spikesnap": "cursor", "spikethickness": 1, "spikedash": "dot", "spikecolor": tokens["muted"]}
    layout["xaxis"] = axis_style(
        theme, range=list(view), matches="x2", showticklabels=False, anchor=f"y{len(rows) + len(references) + 1}", showgrid=True, **spikes,
    )
    layout["xaxis2"] = axis_style(
        theme, range=list(view), anchor=probability_ref, ticksuffix=" s", showgrid=True, **spikes,
        title={"text": "Time since recording start", "font": {"size": 12, "color": tokens["muted"]}},
        rangeslider={"visible": True, "thickness": 0.07, "range": [0.0, duration], "autorange": False, "bgcolor": tokens["track"], "bordercolor": tokens["axis"], "borderwidth": 1},
    )

    if has_events:
        figure.add_trace(
            go.Scatter(
                x=events["time_s"], y=np.zeros(len(events)), mode="markers", name="Technician note", xaxis="x", yaxis="y",
                marker={"color": EVENT, "symbol": "triangle-down", "size": 10, "line": {"color": tokens["panel"], "width": 1}},
                customdata=np.stack([events["label"], events["category"]], axis=1),
                hovertemplate="<b>%{customdata[1]}</b><br>%{customdata[0]}<br>%{x:.0f} s<extra></extra>",
            )
        )
    if len(real):
        # A flag at the start of every real seizure.
        figure.add_trace(
            go.Scatter(
                x=real["start_s"], y=np.zeros(len(real)), mode="markers", name="Seizure start", xaxis="x", yaxis="y",
                marker={"color": REAL, "symbol": "diamond", "size": 11, "line": {"color": tokens["panel"], "width": 1}},
                customdata=np.stack([np.arange(1, len(real) + 1), real["start_s"], real["end_s"]], axis=1),
                hovertemplate="<b>Seizure %{customdata[0]}</b><br>%{customdata[1]:.1f} s → %{customdata[2]:.1f} s<extra></extra>",
            )
        )
    # Samples inside a real seizure, to redraw those parts of each curve in the seizure color.
    inside = np.zeros(len(time), dtype=bool)
    for row in real.itertuples(index=False):
        inside[np.searchsorted(time, row.start_s) : np.searchsorted(time, row.end_s)] = True
    curve_channels: dict[int, str] = {}
    for index, channel in enumerate(channels):
        axis = f"y{(index if stacked else 0) + 2}"
        figure.add_trace(
            go.Scatter(
                name=channel, mode="lines", xaxis="x", yaxis=axis, showlegend=not stacked,
                line={"color": channel_color(channel, theme), "width": 1},
                hovertemplate=f"<b>{channel}</b><br>%{{x:.3f}} s<br>%{{y:.1f}}<extra></extra>",
            ),
            hf_x=time, hf_y=signals[channel],
        )
        curve_channels[len(figure.data) - 1] = channel
        if inside.any():
            # The resampler breaks the line at the gaps between seizures.
            figure.add_trace(
                go.Scatter(
                    name="Seizure on the curve", mode="lines", xaxis="x", yaxis=axis, legendgroup="seizure-curve",
                    showlegend=index == 0, line={"color": REAL, "width": 1.6}, hoverinfo="skip",
                ),
                hf_x=time[inside], hf_y=signals[channel][inside],
            )
    for index, (name, values) in enumerate(references.items()):
        figure.add_trace(
            go.Scatter(
                name="Reference channel (not used by the model)", mode="lines", xaxis="x", yaxis=reference_axes[name],
                legendgroup="reference", showlegend=index == 0, line={"color": tokens["muted"], "width": 1},
                hovertemplate=f"<b>{name}</b> (reference)<br>%{{x:.3f}} s<br>%{{y:.1f}}<extra></extra>",
            ),
            hf_x=time, hf_y=values,
        )
    label_x, label_y = label_curve(real, duration)
    figure.add_trace(
        go.Scatter(
            x=label_x, y=label_y, mode="lines", name="Real seizure", xaxis="x2", yaxis=probability_ref,
            line={"color": REAL, "width": 1.6 if prediction else 2}, fill="tozeroy", fillcolor=rgba(REAL, 0.10 if prediction else 0.25),
            hovertemplate="%{x:.1f} s<br>real seizure: %{y:.0f}<extra></extra>",
        )
    )
    if prediction:
        centers = ((windows["start_s"] + windows["end_s"]) / 2).to_numpy()
        figure.add_trace(
            go.Scatter(
                x=centers, y=windows["proba"].to_numpy(), mode="lines", name="Model probability", xaxis="x2", yaxis=probability_ref,
                line={"color": tokens["probability"], "width": 1.3, "shape": "hvh"}, fill="tozeroy", fillcolor=rgba("#94A3B8", 0.18),
                hovertemplate="window center %{x:.1f} s<br>p = %{y:.2f}<extra></extra>",
            )
        )
        keys = {"xaxis": "x2", "yaxis": probability_ref}
        figure.add_trace(legend_key("Detection", DETECTED, **keys))
        figure.add_trace(legend_key("Threshold", THRESHOLD, dash="dash", **keys))
    figure.update_layout(
        **base_layout(theme), **layout, height=height, margin=margin, hovermode="closest", dragmode="zoom",
        shapes=event_shapes(real, detected, threshold, probability_ref, theme), uirevision="eeg",
    )
    if references:
        figure.add_annotation(
            x=1, xref="paper", xanchor="right", y=reference_top, yref="paper", yanchor="bottom", showarrow=False,
            text="reference — not used by the model", font={"size": 10, "color": tokens["muted"]},
        )
    description = {
        "probability_ref": probability_ref, "curve_channels": curve_channels, "height": height,
        "eeg_axes": eeg_axes, "full_scale": full_scale, "prediction": prediction,
    }
    return figure, description


def view_relayout(view: tuple[float, float]) -> dict:
    """Relayout event for the resampler: every high-frequency trace is on the first x-axis."""
    return {"xaxis.range[0]": view[0], "xaxis.range[1]": view[1]}


def resampled_dict(figure: FigureResampler, view: tuple[float, float]) -> dict:
    """Figure as a dict whose high-frequency traces are already aggregated for ``view``."""
    result = figure.to_dict()
    update = figure._construct_update_data(view_relayout(view))
    if isinstance(update, list):
        for trace in update[1:]:
            target = result["data"][trace["index"]]
            target.update({key: value for key, value in trace.items() if key != "index"})
    return result


# ----------------------------------------------------------------------------- comparison

# One hue per algorithm; green (real seizures) and violet (threshold) stay reserved.
ALGORITHM_COLORS = {
    "dark": {"dt": "#C98500", "rf": "#3987E5", "knn": "#D55181", "svm": "#D95926"},
    "light": {"dt": "#B7791F", "rf": "#2A78D6", "knn": "#D55181", "svm": "#D95926"},
}


def model_style(model: str, theme: str) -> dict:
    """Color by algorithm; solid for unbalanced training, dashed / hatched for balanced."""
    algorithm, mode = model.split("_")
    balanced = mode == "balanced"
    return {"color": ALGORITHM_COLORS[theme][algorithm], "dash": "dash" if balanced else "solid", "pattern": "/" if balanced else ""}


def training_distribution_figure(rows: list[tuple[str, dict]], theme: str) -> go.Figure:
    """Class shares of the training data for each training mode shown."""
    tokens = THEMES[theme]
    names = [name for name, _ in rows]
    figure = go.Figure()
    for column, label, color in [("class_0", "Class 0 — no seizure", rgba("#94A3B8", 0.55)), ("class_1", "Class 1 — seizure", REAL)]:
        shares = [training[column] / training["windows"] for _, training in rows]
        counts = [training[column] for _, training in rows]
        figure.add_trace(
            go.Bar(
                y=names, x=shares, orientation="h", name=label, marker={"color": color, "line": {"width": 0}}, width=0.5,
                text=[f"{share:.0%} · {count:,.0f} windows" for share, count in zip(shares, counts)], textposition="inside", insidetextanchor="middle",
                textfont={"family": MONO_FONT, "size": 11, "color": tokens["text"]},
                hovertemplate="<b>%{y}</b><br>" + label + ": %{x:.1%}<extra></extra>",
            )
        )
    figure.update_layout(
        **base_layout(theme), barmode="stack", height=70 * len(rows) + 90, margin={"l": 96, "r": 16, "t": 34, "b": 34}, hovermode="closest",
        xaxis=axis_style(theme, tickformat=".0%", range=[0, 1], showgrid=False),
        yaxis=axis_style(theme, showgrid=False, autorange="reversed", tickfont={"size": 12, "color": tokens["text"]}),
    )
    return figure


def confusion_figure(models: list[tuple[str, str, dict]], theme: str) -> go.Figure:
    """Window-level confusion matrix of each model at its best threshold: counts and share of the true class."""
    tokens = THEMES[theme]
    columns = min(4, len(models))
    rows = int(np.ceil(len(models) / columns))
    figure = go.Figure()
    layout, annotations = {}, []
    gap_x, gap_y = 0.05, 0.36 / rows
    width = (1 - gap_x * (columns - 1)) / columns
    height = (1 - gap_y * (rows - 1)) / rows
    scale = [[0, rgba("#3B82F6", 0.06)], [1, rgba("#3B82F6", 0.85 if theme == "dark" else 0.7)]]
    for index, (key, label, window) in enumerate(models):
        row, column = divmod(index, columns)
        suffix = "" if index == 0 else str(index + 1)
        left = column * (width + gap_x)
        top = 1 - row * (height + gap_y)
        matrix = np.array([[window["tn"], window["fp"]], [window["fn"], window["tp"]]], dtype=float)
        share = matrix / matrix.sum(axis=1, keepdims=True)
        layout[f"xaxis{suffix}"] = axis_style(theme, domain=[left, min(1.0, left + width)], anchor=f"y{suffix}", showgrid=False, side="bottom", fixedrange=True, tickangle=0)
        layout[f"yaxis{suffix}"] = axis_style(theme, domain=[max(0.0, top - height), top], anchor=f"x{suffix}", showgrid=False, autorange="reversed", fixedrange=True, showticklabels=column == 0)
        figure.add_trace(
            go.Heatmap(
                z=share, x=["Pred. 0", "Pred. 1"], y=["True 0", "True 1"], xaxis=f"x{suffix}", yaxis=f"y{suffix}",
                colorscale=scale, zmin=0, zmax=1, showscale=False, xgap=3, ygap=3,
                text=[[f"{int(matrix[i, j]):,}<br>{share[i, j]:.1%}" for j in range(2)] for i in range(2)], texttemplate="%{text}",
                textfont={"family": MONO_FONT, "size": 12, "color": tokens["text"]},
                hovertemplate=f"<b>{label}</b><br>%{{y}}, %{{x}}<br>%{{text}}<extra></extra>",
            )
        )
        annotations.append(
            {
                "x": left + width / 2, "xref": "paper", "y": top, "yref": "paper", "yanchor": "bottom", "showarrow": False,
                # Algorithm and training mode on two lines, so neighbouring titles never touch.
                "text": label.replace(" · ", "<br>"), "font": {"size": 11, "color": tokens["text"]},
            }
        )
    figure.update_layout(**base_layout(theme), **layout, annotations=annotations, height=250 * rows + 50, margin={"l": 56, "r": 12, "t": 46, "b": 34})
    return figure


def curves_figure(curves: dict[str, tuple[str, pd.DataFrame, tuple[float, float]]], x: str, y: str, x_title: str, y_title: str, theme: str) -> go.Figure:
    """Shared chart with one line per model and a marker at each model's best threshold.

    ``curves`` maps a model key to (label, data frame, (x, y) of the best-threshold point).
    """
    tokens = THEMES[theme]
    figure = go.Figure()
    for key, (label, frame, point) in curves.items():
        style = model_style(key, theme)
        figure.add_trace(
            go.Scatter(
                x=frame[x], y=frame[y], mode="lines", name=label, legendgroup=key,
                line={"color": style["color"], "width": 2, "dash": style["dash"]}, customdata=frame["threshold"],
                hovertemplate=f"<b>{label}</b><br>threshold %{{customdata:.2f}}<br>{x_title}: %{{x:.2f}}<br>{y_title}: %{{y:.2f}}<extra></extra>",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[point[0]], y=[point[1]], mode="markers", legendgroup=key, showlegend=False,
                marker={"color": style["color"], "size": 10, "symbol": "circle", "line": {"color": tokens["panel"], "width": 2}},
                hovertemplate=f"<b>{label}</b><br>best threshold<br>{x_title}: %{{x:.2f}}<br>{y_title}: %{{y:.2f}}<extra></extra>",
            )
        )
    layout = base_layout(theme)
    layout["legend"] = {**layout["legend"], "orientation": "h", "y": -0.22, "yanchor": "top", "font": {"size": 11, "color": tokens["text"]}}
    figure.update_layout(
        **layout, height=380, margin={"l": 56, "r": 16, "t": 16, "b": 40}, hovermode="closest",
        xaxis=axis_style(theme, title={"text": x_title, "font": {"size": 12, "color": tokens["muted"]}}, range=[0, 1.02]),
        yaxis=axis_style(theme, title={"text": y_title, "font": {"size": 12, "color": tokens["muted"]}}, range=[0, 1.05]),
    )
    return figure


def patient_comparison_figure(recording: str, duration: float, real: pd.DataFrame, results: dict[str, tuple], theme: str) -> go.Figure:
    """One patient: the real seizures on the first row, then one row of detections per model.

    ``results`` maps a model key to (row label, detected intervals, event matches, metrics).
    Detections outside every real seizure (false alarms) are drawn faded.
    """
    tokens = THEMES[theme]
    labels = {key: value[0] for key, value in results.items()}
    rows = ["Real seizures", *labels.values()]
    position = {name: len(rows) - 1 - index for index, name in enumerate(rows)}
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            y=list(position.values()), x=[duration] * len(rows), base=0, orientation="h", width=0.7,
            marker={"color": tokens["track"], "line": {"width": 0}}, hoverinfo="skip", showlegend=False,
        )
    )

    def lane(frame: pd.DataFrame, row: str, name: str, color: str, pattern: str = "", opacity: float = 1.0) -> None:
        if frame.empty:
            return
        custom = np.stack([frame["start_s"], frame["end_s"]], axis=1)
        hover = f"<b>{row}</b><br>{name}<br>%{{customdata[0]:.1f}} s → %{{customdata[1]:.1f}} s<extra></extra>"
        y = [position[row]] * len(frame)
        figure.add_trace(
            go.Bar(
                y=y, x=(frame["end_s"] - frame["start_s"]).to_numpy(), base=frame["start_s"], orientation="h", width=0.5,
                showlegend=False, customdata=custom, hovertemplate=hover, opacity=opacity,
                marker={"color": color, "line": {"width": 0}, "pattern": {"shape": pattern, "fgcolor": tokens["panel"], "size": 5}},
            )
        )
        # Tick markers keep short events visible when the bar is narrower than a pixel.
        figure.add_trace(
            go.Scatter(
                x=(frame["start_s"] + frame["end_s"]) / 2, y=y, mode="markers", showlegend=False, opacity=opacity,
                marker={"symbol": "line-ns", "size": 16, "line": {"color": color, "width": 2}}, customdata=custom, hovertemplate=hover,
            )
        )

    lane(real, "Real seizures", "Real seizure (annotation)", REAL)
    notes = [("Real seizures", f"{len(real)} annotated")]
    for key, (label, detected, matches, metrics) in results.items():
        style = model_style(key, theme)
        false_starts = set(matches.loc[matches["status"] == "FP", "detected_start_s"])
        is_false = detected["start_s"].isin(false_starts) if len(detected) else pd.Series(dtype=bool)
        lane(detected.loc[~is_false], label, "Detection on a real seizure", style["color"], style["pattern"])
        lane(detected.loc[is_false], label, "False alarm", style["color"], style["pattern"], opacity=0.4)
        notes.append((label, f"{metrics['tp']}/{metrics['tp'] + metrics['fn']} found · {metrics['fp']} false"))
    figure.update_layout(
        **base_layout(theme), barmode="overlay", height=46 * len(rows) + 100, margin={"l": 150, "r": 150, "t": 16, "b": 48},
        hovermode="closest", bargap=0, showlegend=False,
        annotations=[
            {
                "x": 1.0, "xref": "paper", "xanchor": "left", "y": position[row], "yref": "y", "showarrow": False, "xshift": 8,
                "text": text, "font": {"family": MONO_FONT, "size": 11, "color": tokens["muted"]},
            }
            for row, text in notes
        ],
        xaxis=axis_style(theme, title={"text": f"Time since recording start — {recording}", "font": {"size": 12, "color": tokens["muted"]}}, ticksuffix=" s", range=[0, duration]),
        yaxis=axis_style(
            theme, tickvals=list(position.values()), ticktext=list(position.keys()), showgrid=False, zeroline=False,
            range=[-0.6, len(rows) - 0.4], fixedrange=True, tickfont={"size": 12, "color": tokens["text"]},
        ),
    )
    return figure


# ----------------------------------------------------------------------------- spectrum


def spectrum_figure(result: dict, channels: list[str], highlighted: str | None, theme: str) -> go.Figure:
    tokens = THEMES[theme]
    figure = go.Figure()
    frequencies = result["frequencies"]
    keep = (frequencies >= 0.5) & (frequencies <= 40)
    for channel in channels:
        emphasis = highlighted is None or channel == highlighted
        figure.add_trace(
            go.Scatter(
                x=frequencies[keep], y=result["psd"][channel][keep], mode="lines", name=channel,
                line={"color": channel_color(channel, theme), "width": 2.2 if channel == highlighted else 1.2},
                opacity=1.0 if emphasis else 0.45,
                hovertemplate=f"<b>{channel}</b><br>%{{x:.1f}} Hz<br>%{{y:.3g}}<extra></extra>",
            )
        )
    shapes, labels = [], []
    for index, (band, (low, high)) in enumerate(BANDS.items()):
        shapes.append(
            {
                "type": "rect", "xref": "x", "yref": "paper", "x0": low, "x1": high, "y0": 0, "y1": 1, "layer": "below",
                "fillcolor": rgba("#94A3B8", 0.16 if index % 2 == 0 else 0.07), "line": {"width": 0},
            }
        )
        labels.append(
            {
                "x": (low + high) / 2, "xref": "x", "y": 1, "yref": "paper", "yanchor": "bottom", "showarrow": False,
                "text": {"Delta": "δ", "Theta": "θ", "Alpha": "α", "Beta": "β"}[band], "font": {"size": 13, "color": tokens["muted"]},
            }
        )
    figure.update_layout(
        **base_layout(theme), height=300, margin={"l": 64, "r": 64, "t": 30, "b": 44}, shapes=shapes, annotations=labels, hovermode="closest",
        xaxis=axis_style(theme, title={"text": "Frequency", "font": {"size": 12, "color": tokens["muted"]}}, ticksuffix=" Hz", range=[0, 40]),
        yaxis=axis_style(theme, type="log", title={"text": "Power spectral density", "font": {"size": 12, "color": tokens["muted"]}}, exponentformat="power"),
        showlegend=True,
    )
    figure.update_layout(legend={"orientation": "v", "x": 1.02, "xanchor": "left", "y": 1, "yanchor": "top"})
    return figure
