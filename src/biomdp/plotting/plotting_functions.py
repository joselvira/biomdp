# %% -*- coding: utf-8 -*-

"""
Created on Fri Oct 23 13:23:15 2026

@author: Jose L. L. Elvira

Creates PDF reports for data blocks.
Version using multiple CPU cores in parallel for faster execution.

--------------------------------------------------------------------------------
Fast grid plot types (rows x columns, one page per block).

Page splitting, multiprocessing, layout, and PDF merging are handled by the engine
(``facet_engine.facet_pdf``). This module contains ONLY the plot-specific logic.

How to add a new plot type
--------------------------
Each plot type consists of three parts, in this order within its section:

1. ``_extract_xxx(...)`` -> returns an ``extract(da_panel) -> dict`` function.
   Runs in the main process: uses xarray to select panel-specific data from
   auxiliary DataArrays (``select_facet``) and returns a dict containing ONLY
   NumPy/float/str/None values.

2. ``draw_xxx(ax, p, cfg)`` -> MODULE-LEVEL function (so it can be pickled and
   passed to worker processes). Draws onto ``ax`` using the dict ``p`` (without
   xarray). Typically starts with ``plot_lines(ax, p, cfg)``. For vertical events,
   use ``mark_events``.

3. ``xxx_plots_fast(...)`` -> public function. Prepares data (trimming, axis
   selection...), sets plot-specific options in ``cfg``, and calls
   ``facet_pdf(daGraph, extract, draw, save_file, ..., cfg=cfg)``.

Minimal template::

    def _extract_my_graph(daAux):
        def extract(da_p):
            aux = select_facet(da_p, daAux)
            return {"value": None if aux is None else float(aux.values)}
        return extract

    def draw_my_graph(ax, p, cfg):
        plot_lines(ax, p, cfg)
        if p["value"] is not None:
            ax.axhline(p["value"], color="C1", ls="--")

    def my_graph_plots_fast(daGraph, daAux, save_file, **engine):
        return facet_pdf(
            daGraph, _extract_my_graph(daAux), draw_my_graph, save_file, **engine
        )

Important: this module is also imported by worker processes, so DO NOT import
heavy libraries (xarray, biomdp, seaborn, etc.) at top level (slow startup). If
a public function needs them, import them INSIDE the function (see
``weight_plots_fast``).
"""

import numpy as np

try:
    from .facet_engine import (
        decimate_minmax,
        facet_pdf,
        mark_events,
        plot_lines,
    )
except:
    from src.biomdp.plotting.facet_engine import (
        decimate_minmax,
        facet_pdf,
        mark_events,
        plot_lines,
    )


__author__ = "Jose L. L. Elvira"
__version__ = "1.0.0"
__date__ = "12/11/2026"


"""
TODO: to make the cuts interpolating so that the phases start right at the criterion?

Updates:
    12/11/2026, v1.0.0
        - First implementation of the fast plotting functions for biomdp.
"""
# =============================================================================
# %% Common Helpers & Constants
# =============================================================================
colr = {
    "iniAnalisis": "grey",
    "finAnalisis": "grey",
    "iniMov": "C0",
    "finMov": "C0",
    "preactiv": "dodgerblue",
    "iniPeso": "deepskyblue",
    "finPeso": "deepskyblue",
    #'iniPeso2':'deepskyblue', 'finPeso2':'deepskyblue', # for when start and end weight are set simultaneously
    "iniImpPos": "orange",
    "finImpPos": "orange",
    "despegue": "r",
    "aterrizaje": "r",
    "maxFz": "brown",
    "minFz": "cyan",
    "maxFlex": "g",
}


def select_facet(da_subset, da_aux):
    """
    Helper for completing facets in xarray.
    Selects the portion corresponding to the current facet from the auxiliary
    DataArray/Dataset, based on the scalar coordinates of the subset.

    Works regardless of the dimensions present in the DataArray.
    Returns None if the selection fails.
    """
    selector = {}
    for dim in da_aux.dims:
        if dim in da_subset.coords and dim not in da_subset.dims:
            selector[dim] = da_subset.coords[dim].values
    try:
        return da_aux.sel(selector) if selector else da_aux
    except Exception:
        return None


def _labeled_to_dict(obj, dim=None):
    """
    Converts labeled values into ``{label: float (nan if missing)}``, whatever the
    name of the label coordinate.

    - Dataset: one entry per data variable.
    - DataArray: the labels are the coordinate values of ``dim``. If ``dim`` is
      None, the only non-singleton dimension is used (a 0-d array is not valid).
    """
    if obj is None:
        return {}
    if hasattr(obj, "data_vars"):  # Dataset
        return {str(k): float(np.squeeze(obj[k].values)) for k in obj.data_vars}
    obj = obj.squeeze()
    if dim is None:
        if obj.ndim != 1:
            raise ValueError(
                f"Cannot infer the label dimension of {obj.name!r} (dims={obj.dims}); "
                "specify it with events_dim / hlines_dim."
            )
        dim = obj.dims[0]
    elif dim not in obj.dims:  # squeezed to a single label
        return {}
    # Leftover dimensions (e.g. several series in the same panel): each label
    # gets an array with one value per series.
    vals = obj.transpose(dim, ...).values.astype(float)
    vals = vals.reshape(vals.shape[0], -1)
    if vals.shape[1] == 1:
        vals = vals[:, 0]
    return dict(zip(obj[dim].values.astype(str), vals))


def _events_to_dict(daevs, dim=None):
    """Backward-compatible wrapper of ``_labeled_to_dict``."""
    return _labeled_to_dict(daevs, dim)


def _as_dict_arg(value, names):
    """``None | scalar | list | dict`` -> dict {name: value} (list is cycled)."""
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    if isinstance(value, str) or not hasattr(value, "__iter__"):
        return {n: value for n in names}
    value = list(value)
    return {n: value[i % len(value)] for i, n in enumerate(names)} if value else {}


# =============================================================================
# %% Type 1: Time series with events (vertical lines), horizontal lines and areas
# =============================================================================


def _series_values(da_p, da_aux):
    """
    One value per data line of the panel ``da_p`` (same order and labels as the
    lines drawn by ``plot_lines``) taken from ``da_aux``, which has the dims of the
    data except the x axis. Returns {line label: value}.
    """
    import itertools

    import xarray as xr

    x = "time" if "time" in da_p.dims else da_p.dims[-1]
    others = [d for d in da_p.dims if d != x]
    sel = select_facet(da_p, da_aux)
    if sel is None:
        return None
    if not others:
        return {str(da_p.name): float(np.squeeze(sel.values))}
    sel, _ = xr.broadcast(sel, da_p.isel({x: 0}, drop=True))
    vals = sel.transpose(*others).values.ravel().astype(float)
    labels = [
        ", ".join(str(v) for v in c)
        for c in itertools.product(*[da_p[d].values for d in others])
    ]
    return dict(zip(labels, vals))


def _extract_events(
    daEvents,
    daHlines=None,
    events_dim=None,
    hlines_dim=None,
    hlines_per_series=False,
):
    def extract(da_p):
        hlines = None
        if daHlines is not None:
            if hlines_per_series:
                hlines = _series_values(da_p, daHlines)
            else:
                hl = select_facet(da_p, daHlines)
                if hl is not None:
                    hlines = _labeled_to_dict(hl, hlines_dim)

        events, fr = None, None
        if daEvents is not None:
            fr = float(da_p.freq)
            ev = select_facet(da_p, daEvents)
            events = _labeled_to_dict(ev, events_dim) if ev is not None else {}

        return {
            "freq": fr,
            "events": events,
            "hlines": hlines,
        }

    return extract


def draw_hlines(ax, hlines, colors, fmt, fontsize="x-small"):
    """Horizontal dashed lines with a label at the left edge of the axes."""
    for n, (name, vals) in enumerate(hlines.items()):
        c = colors.get(name, f"C{n}")
        for w in np.atleast_1d(vals):
            if np.isnan(w):
                continue
            ax.axhline(w, color=c, lw=1, ls="--", dash_capstyle="round", alpha=0.7)
            ax.text(
                0.0,
                w,
                fmt.format(name=name, value=w),
                ha="left",
                va="bottom",
                c=c,
                alpha=0.7,
                fontsize=fontsize,
                transform=ax.get_yaxis_transform(),
            )


def draw_events(ax, p, cfg):
    """Panel displaying data lines, horizontal lines, and event lines."""
    plot_lines(ax, p, cfg)

    if p["hlines"]:
        draw_hlines(
            ax,
            p["hlines"],
            cfg["hlines_colors"],
            cfg["hlines_fmt"],
        )

    evs = p["events"]  # dict name -> sample/time (float, nan if non-existent)
    fr = p["freq"]
    if evs is None:
        return
    scale = 1.0 if cfg["events_in_time"] else fr

    # Vertical event lines and labels
    names, xs = [], []
    for n, vals in evs.items():
        for v in np.atleast_1d(vals):
            if not np.isnan(v):
                names.append(n)
                xs.append(v / scale)

    ymax = ax.get_ylim()[1]
    shade = cfg["events_shade"]
    if shade:  # shaded window of ``shade`` samples (or s) starting at each event
        width = shade if cfg["events_in_time"] else shade / fr
        for x in xs:
            ax.axvspan(x, x + width, color="grey", alpha=0.15, lw=0)
    frac = cfg["events_label_pos"]
    colors = cfg["events_colors"]
    mark_events(
        ax,
        names,
        xs,
        [colors.get(n, "k") for n in names],
        [ymax * frac.get(n, 0.97) for n in names],
    )

    if cfg["fit_start_end"]:
        e_ini, e_fin = cfg["fit_events"]
        if e_ini in evs and e_fin in evs:
            eini, efin = e_ini, e_fin
        elif len(evs) > 1:
            nom = list(evs)
            eini, efin = nom[0], nom[-1]
        else:
            eini = efin = None
        if eini is not None:
            with np.errstate(all="ignore"):
                a = np.nanmin(evs[eini]) / scale
                b = np.nanmax(evs[efin]) / scale
            if np.isfinite(a) and np.isfinite(b) and a != b:
                ax.set_xlim(a, b)


def block_plots_fast(
    daGraph,
    daEvents=None,
    daHlines=None,
    num_per_block=4,
    n_rows="ID",
    n_cols=None,
    fit_start_end=False,
    events_dim=None,
    hlines_dim=None,
    axis="z",
    events_in_time=False,
    exclude_events=("offset_ini",),
    fit_events=("iniMov", "finMov"),
    events_colors=None,
    events_label_pos=None,
    events_shade=None,
    hlines_colors=None,
    hlines_fmt="{name}={value:.1f}",
    save_file=None,
    show_in_console=False,
    sharex=False,
    sharey=False,
    alpha=1.0,
    aspect=1.62,
    n_workers=None,
    max_points=1500,
    rasterized=False,
    dpi=150,
    legend=True,
    spacing=(0.4, 0.5),
    low_priority=True,
    **kwargs,
):
    """
    Fast version of ``block_plots``. Generates a multi-page PDF with the same
    panel logic and events, but using ``facet_engine``
    (NumPy instead of xarray during drawing, no FacetGrid, series decimation,
    fixed margins, and one process per page).

    Parameters
    ----------
    daGraph, num_per_block, n_rows, n_cols, fit_start_end,
    sharex, sharey, aspect : Same as in ``block_plots``.
    daEvents : DataArray, optional. Events drawn as vertical lines with a label.
        The label dimension can have any name (``events_dim``; if None it is the
        only dimension of ``daEvents`` that is not in ``daGraph``). Values are in
        samples (divided by ``daGraph.freq``) unless ``events_in_time=True``.
        NaN values are not drawn.
    daHlines : DataArray or Dataset, optional. Any number of values drawn as
        labeled horizontal lines. DataArray: label dimension ``hlines_dim``
        (same inference as ``events_dim``). Dataset: one line per data variable.
        If the DataArray has no label dimension (same dims as ``daGraph`` without
        time, e.g. ``daGraph.mean("time")``), one line is drawn for each data line
        of the panel, with its same color.
    axis : Selects this ``axis`` coordinate in the auxiliary arrays if they have
        an ``axis`` dimension (None to disable).
    exclude_events : Event labels not to draw.
    fit_events : (start, end) labels used to trim the x-axis if ``fit_start_end``.
    events_colors : dict {event label: color}. Default: module ``colr`` (else black).
    events_label_pos : dict {event label: fraction of ymax} for the label height.
    events_shade : float, optional. Shades a window of this length (samples, or
        seconds if ``events_in_time``) starting at each event.
    hlines_colors : dict {label: color}, or list of colors (cycled). Default C0, C1...
    hlines_fmt : Label format with fields ``{name}`` and ``{value}``.
    save_file : Path or str. Output PDF path. Required unless ``show_in_console``.
    show_in_console : bool. If True, also shows each page in the interactive
        window / Jupyter console. The PDF is only saved if ``save_file`` is given.
    alpha : Transparency of data lines.
    n_workers, max_points, rasterized, dpi, legend, spacing, low_priority :
        See ``facet_engine.facet_pdf``.
    **kwargs : Extra arguments for ``ax.plot()``.

    Note (Windows)
    --------------
    Child processes do not require the script to be protected with
    ``if __name__ == "__main__":`` (``__main__`` is hidden while active).
    """

    # --- Data preparation (main process, xarray) ---
    def _sel_axis(da):
        if da is not None and axis is not None and "axis" in da.dims:
            return da.sel(axis=axis)
        return da

    def _label_dim(da, dim):
        """Label dimension: the given one, or the only one not in daGraph."""
        if dim is not None:
            return dim
        extra = [
            d for d in da.dims if d not in daGraph.dims and d not in daGraph.coords
        ]
        return extra[0] if len(extra) == 1 else None

    daEvents, daHlines = _sel_axis(daEvents), _sel_axis(daHlines)

    if daEvents is not None and hasattr(daEvents, "data_vars"):
        raise TypeError("daEvents must be a DataArray with a label dimension.")

    if daEvents is not None:
        events_dim = _label_dim(daEvents, events_dim)
        if exclude_events and events_dim is not None:
            labels = daEvents[events_dim].astype(str)
            daEvents = daEvents.sel({events_dim: ~labels.isin(list(exclude_events))})
        if fit_start_end:
            e_ini, e_fin = fit_events
            lab = daEvents[events_dim]
            if e_ini in lab and e_fin in lab:
                daEvents = daEvents.where(
                    (daEvents >= daEvents.sel({events_dim: e_ini}))
                    & (daEvents <= daEvents.sel({events_dim: e_fin})),
                    drop=True,
                )
    # Without a label dimension, daHlines holds one value per data line
    # (same dims as the data without time), e.g. ``daGraph.mean("time")``.
    hlines_per_series = False
    if daHlines is not None and not hasattr(daHlines, "data_vars"):
        hlines_dim = _label_dim(daHlines, hlines_dim)
        hlines_per_series = hlines_dim is None

    names_ev = [] if daEvents is None else list(daEvents[events_dim].values.astype(str))
    if daHlines is None or hlines_per_series:
        names_hl = []
    elif hasattr(daHlines, "data_vars"):
        names_hl = [str(k) for k in daHlines.data_vars]
    else:
        names_hl = list(daHlines[hlines_dim].values.astype(str))

    events_colors = {**colr, **_as_dict_arg(events_colors, names_ev)}
    events_label_pos = {
        "iniImpPos": 0.7,
        "finImpPos": 0.7,
        "minFz": 0.8,
        "maxFlex": 0.8,
        **(events_label_pos or {}),
    }

    return facet_pdf(
        daGraph,
        _extract_events(
            daEvents,
            daHlines,
            events_dim,
            hlines_dim,
            hlines_per_series,
        ),
        draw_events,
        save_file,
        show_in_console=show_in_console,
        n_rows=n_rows,
        n_cols=n_cols,
        num_per_block=num_per_block,
        sharex=sharex,
        sharey=sharey,
        aspect=aspect,
        n_workers=n_workers,
        max_points=max_points,
        rasterized=rasterized,
        alpha=alpha,
        dpi=dpi,
        legend=legend,
        spacing=spacing,
        low_priority=low_priority,
        plot_kwargs=kwargs,
        cfg={
            "fit_start_end": fit_start_end,
            "fit_events": tuple(fit_events),
            "events_in_time": events_in_time,
            "events_colors": events_colors,
            "events_label_pos": events_label_pos,
            "events_shade": events_shade,
            "hlines_colors": _as_dict_arg(hlines_colors, names_hl)
            if not isinstance(hlines_colors, dict)
            else hlines_colors,
            "hlines_fmt": hlines_fmt,
        },
    )


# =============================================================================
# %% Type 2: Weight Verification (Fast version of jfu.graphs_weight_check)
# =============================================================================


def _extract_weights(dsWeights, daEvents):
    def extract(da_p):
        weights = None
        if dsWeights is not None:
            ds = select_facet(da_p, dsWeights)
            if ds is not None:
                weights = {
                    str(k): float(np.squeeze(ds[k].values)) for k in ds.data_vars
                }

        events = None
        if daEvents is not None:
            ev = select_facet(da_p, daEvents)
            if ev is not None:
                events = _events_to_dict(ev.sel(n_event=["iniPeso", "finPeso"]))

        return {"freq": float(da_p.freq), "weights": weights, "events": events}

    return extract


def draw_weights(ax, p, cfg):
    """Panel displaying the signal in grey, one line per weight type, and the weight window."""
    plot_lines(ax, p, cfg)

    weights = p["weights"]
    if weights:
        ref = weights.get(cfg["ref_weight"])
        for n, (name, w) in enumerate(weights.items()):
            c = f"C{n}"
            ax.axhline(w, color=c, lw=1, ls="--", dash_capstyle="round", alpha=0.7)
            ax.text(
                0.0,
                w,
                f"weight {name}={w:.1f} N",
                ha="left",
                va="bottom",
                c=c,
                alpha=0.7,
                fontsize="x-small",
            )
            # Trigger warning if the difference from the reference exceeds the threshold
            if n > 0 and ref is not None and abs(w - ref) > cfg["threshold"]:
                ax.text(
                    0.5,
                    0.9,
                    f"REVIEW (delta={w - ref:.1f} N)",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    c="r",
                    alpha=0.7,
                    fontsize="large",
                    bbox={
                        "facecolor": "lightgrey",
                        "alpha": 0.3,
                        "edgecolor": "r",
                        "boxstyle": "round,pad=0.3,rounding_size=.5",
                    },
                )

    evs = p["events"]
    if evs:
        names = [n for n, v in evs.items() if not np.isnan(v)]
        xs = [evs[n] / p["freq"] for n in names]
        y0, y1 = ax.get_ylim()
        mark_events(
            ax,
            names,
            xs,
            [colr.get(n, "k") for n in names],
            [(y1 - y0) * 0.97 + y0] * len(names),
            fontsize="small",
        )


def weight_plots_fast(
    daData,
    daEvents,
    dsWeights=None,
    threshold_weight_diff=20,
    allowed_window=0.3,
    ref_weight="media",
    num_per_block=4,
    save_file=None,
    **engine,
):
    """
    Fast version of ``biomdp.jump_forces_utils.graphs_weight_check``.

    Displays the signal trimmed to the weight calculation window (``iniPeso``-``finPeso``
    ± ``allowed_window``), with a line for each variable in ``dsWeights`` (weight
    calculation types), and warns with "REVIEW" if any differs from ``ref_weight``
    by more than ``threshold_weight_diff``.

    Parameters
    ----------
    allowed_window : float (s) or int (samples) before iniPeso and after finPeso.
    ref_weight : Variable in ``dsWeights`` to which others are compared.
    save_file : Path or str. Required (unlike original, single path instead of
        ``work_path`` + ``n_file_graph_global``).
    **engine : Options for ``facet_engine.facet_pdf`` (n_workers, max_points, ...).

    Does not include legacy ``daWeight``/``daWeight_mean`` (dead code in original).
    """
    # Heavy imports placed only here: worker processes import this module and
    # should not load xarray/biomdp at top level.
    import biomdp.jump_forces_utils as jfu
    import xarray as xr

    if isinstance(allowed_window, float):
        allowed_window = allowed_window * daData.freq

    # In case data with an axis dimension is passed
    if "axis" in daData.dims:
        daData = daData.sel(axis="z")
    if daEvents is not None and "axis" in daEvents.dims:
        daEvents = daEvents.sel(axis="z")
    if dsWeights is not None and "axis" in dsWeights.dims:
        dsWeights = dsWeights.sel(axis="z")

    # Trimming window around iniPeso-finPeso (staying within data boundaries)
    daWindow = daEvents.sel(n_event=["iniPeso", "finPeso"]).copy()
    daWindow.loc[{"n_event": "iniPeso"}] = xr.where(
        daWindow.loc[{"n_event": "iniPeso"}] - allowed_window > 0,
        daWindow.loc[{"n_event": "iniPeso"}] - allowed_window,
        0,
    )
    daWindow.loc[{"n_event": "finPeso"}] = xr.where(
        daWindow.loc[{"n_event": "finPeso"}] + allowed_window < daData.time.size,
        daWindow.loc[{"n_event": "finPeso"}] + allowed_window,
        daData.time.size,
    )
    daDat = jfu.trim_analysis_window(daData, daWindow)
    daEvents = daEvents - daWindow.loc[{"n_event": "iniPeso"}]

    return facet_pdf(
        daDat,
        _extract_weights(dsWeights, daEvents),
        draw_weights,
        save_file,
        n_rows="ID",
        n_cols="repe" if "repe" in daDat.dims else None,
        num_per_block=num_per_block,
        alpha=engine.pop("alpha", 0.8),
        plot_kwargs=engine.pop("plot_kwargs", {"color": "lightgrey", "lw": 1}),
        cfg={"threshold": threshold_weight_diff, "ref_weight": ref_weight},
        **engine,
    )


# =============================================================================
# %% EXAMPLES
# =============================================================================
if __name__ == "__main__":
    # =============================================================================
    # %%---- Create a sample
    # =============================================================================

    import matplotlib.pyplot as plt
    import numpy as np

    # import pandas as pd
    import xarray as xr

    from biomdp.create_time_series import create_time_series_xr

    rnd_seed = np.random.seed(
        12340
    )  # fija la aleatoriedad para asegurarse la reproducibilidad
    n = 5
    duration = 5

    freq = 200.0
    Pre_a = (
        create_time_series_xr(
            rnd_seed=rnd_seed,
            num_subj=n,
            fs=freq,
            IDini=0,
            range_offset=[25, 29],
            range_amp=[40, 45],
            range_freq=[1.48, 1.52],
            range_af=[0, 30],
            amplific_noise=[0.4, 0.7],
            fc_noise=[3.0, 3.5],
            range_duration=[duration, duration],
        )
        .expand_dims({"n_var": ["a"], "moment": ["pre"]})
        .transpose("ID", "moment", "n_var", "time")
    )
    Post_a = (
        create_time_series_xr(
            rnd_seed=rnd_seed,
            num_subj=n,
            fs=freq,
            IDini=0,
            range_offset=[22, 26],
            range_amp=[36, 40],
            range_freq=[1.48, 1.52],
            range_af=[0, 30],
            amplific_noise=[0.4, 0.7],
            fc_noise=[3.0, 3.5],
            range_duration=[duration, duration],
        )
        .expand_dims({"n_var": ["a"], "moment": ["post"]})
        .transpose("ID", "moment", "n_var", "time")
    )
    var_a = xr.concat([Pre_a, Post_a], dim="moment")

    Pre_b = (
        create_time_series_xr(
            rnd_seed=rnd_seed,
            num_subj=n,
            fs=freq,
            IDini=0,
            range_offset=[35, 39],
            range_amp=[50, 55],
            range_freq=[1.48, 1.52],
            range_af=[0, 30],
            amplific_noise=[0.4, 0.7],
            fc_noise=[3.0, 3.5],
            range_duration=[duration, duration],
        )
        .expand_dims({"n_var": ["b"], "moment": ["pre"]})
        .transpose("ID", "moment", "n_var", "time")
    )
    Post_b = (
        create_time_series_xr(
            rnd_seed=rnd_seed,
            num_subj=n,
            fs=freq,
            IDini=0,
            range_offset=[32, 36],
            range_amp=[32, 45],
            range_freq=[1.48, 1.52],
            range_af=[0, 30],
            amplific_noise=[0.4, 0.7],
            fc_noise=[3.0, 3.5],
            range_duration=[duration, duration],
        )
        .expand_dims({"n_var": ["b"], "moment": ["post"]})
        .transpose("ID", "moment", "n_var", "time")
    )
    var_b = xr.concat([Pre_b, Post_b], dim="moment")

    Pre_c = (
        create_time_series_xr(
            rnd_seed=rnd_seed,
            num_subj=n,
            fs=freq,
            IDini=0,
            range_offset=[35, 39],
            range_amp=[10, 15],
            range_freq=[1.48, 1.52],
            range_af=[0, 30],
            amplific_noise=[0.4, 0.7],
            fc_noise=[3.0, 3.5],
            range_duration=[duration, duration],
        )
        .expand_dims({"n_var": ["c"], "moment": ["pre"]})
        .transpose("ID", "moment", "n_var", "time")
    )
    Post_c = (
        create_time_series_xr(
            rnd_seed=rnd_seed,
            num_subj=n,
            fs=freq,
            IDini=0,
            range_offset=[32, 36],
            range_amp=[12, 16],
            range_freq=[1.48, 1.52],
            range_af=[0, 30],
            amplific_noise=[0.4, 0.7],
            fc_noise=[3.0, 3.5],
            range_duration=[duration, duration],
        )
        .expand_dims({"n_var": ["c"], "moment": ["post"]})
        .transpose("ID", "moment", "n_var", "time")
    )
    var_c = xr.concat([Pre_c, Post_c], dim="moment")

    # Concatenate all subjects
    daTotal = xr.concat([var_a, var_b, var_c], dim="n_var")
    daTotal.name = "Angle"
    daTotal.attrs["freq"] = 1 / (
        daTotal.time[1].values - daTotal.time[0].values
    )  # incluimos la frequencia como atributo
    daTotal.attrs["units"] = "deg"
    daTotal.time.attrs["units"] = "s"

    # Plot
    daTotal.plot.line(x="time", col="moment", hue="ID", row="n_var")

    # =============================================================================
    # %% Test the functions
    # =============================================================================
    from biomdp.plotting.plotting_functions import block_plots_fast

    block_plots_fast(daTotal, num_per_block=3, show_in_console=True)
    block_plots_fast(daTotal, num_per_block=3, save_file="test_block_plots_fast.pdf")

    block_plots_fast(daTotal, n_rows="ID", n_cols="n_var", show_in_console=True)
    block_plots_fast(daTotal, n_rows="ID", n_cols="n_var", show_in_console=True)

    block_plots_fast(
        daTotal,
        n_rows="ID",
        n_cols="moment",
        num_per_block=3,
        show_in_console=True,
        sharex=True,
        sharey=True,
    )

    block_plots_fast(
        daTotal,
        n_rows="ID",
        n_cols="n_var",
        max_points=500,
        show_in_console=True,
        marker=".",
        alpha=0.5,
    )

    # ------------------------------------------------------------------------------
    from biomdp.slice_time_series_phases import detect_events_onset

    evts = detect_events_onset(
        daTotal, reference_var={"moment": "pre", "n_var": "b"}, threshold=45
    )
    mean = daTotal.mean(dim="time")
    evts = evts.rename({"num_event": "n_event"})
    block_plots_fast(
        daTotal,
        daEvents=evts,
        daHlines=mean,
        hlines_dim="n_var",
        n_rows="ID",
        n_cols="moment",
        show_in_console=True,
    )

    thres = xr.full_like(mean, 45)
    block_plots_fast(
        daTotal,
        daEvents=evts,
        daHlines=thres,
        hlines_dim="n_var",
        n_rows="ID",
        n_cols="moment",
        show_in_console=True,
    )
