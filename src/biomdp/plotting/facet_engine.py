# -*- coding: utf-8 -*-
"""
Generic engine for rapidly creating multi-page grid PDFs of plots (facets by
rows and columns). It isagnostic to events, weights, etc.: that logic resides
in ``facet_plots.py``.

Engine Overview
---------------
1. Splits ``daGraph`` into blocks (one PDF page per block).
2. Converts each panel to pure NumPy (``t``, ``y``, labels) and appends whatever
   is returned by the plot type's ``extract`` function (main process).
3. Draws each page in a separate process using the plot type's ``draw`` function,
   directly using ``Figure`` (without pyplot) and fixed margins.
4. Merges the pages in order using ``pypdf``.

Contract with each plot type (see ``facet_plots.py``)
----------------------------------------------------
``extract(da_panel) -> dict``   (main process; can use xarray and closures)
    Receives the panel (DataArray with row/column scalar coordinates) and
    returns a dict containing ONLY numpy/float/str/None (sent to child process).

``draw(ax, p, cfg) -> None``   (child process; module-level function)
    Draws onto axis ``ax``. ``p`` is the panel dict: contains ``extract`` output
    plus ``t``, ``y`` (n_lines, n_time), ``labels``, ``title``, ``xlabel``, ...
    ``cfg`` is the options dict. Free of xarray: child processes do not import it.
    Typically starts with ``plot_lines(ax, p, cfg)``.

This module only imports NumPy and Matplotlib upon loading so that child processes
(via ``spawn`` on Windows) start up quickly. Heavy dependencies (pypdf, tqdm, loguru,
concurrent.futures) are imported inside ``facet_pdf``, which only runs in the main process.
"""

import io
import os

import matplotlib
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

# ``Figure`` is used directly (without pyplot): it doesn't touch the backend or
# the global state of the caller's session (e.g., inline plots in Jupyter).
# plot_settings is not imported (which drags in seaborn/pandas/scipy, ~6s per process):
# the main process passes the already applied rcParams in ``cfg["rc"]``.

_RC_KEYS = (
    "axes.prop_cycle",
    "lines.linewidth",
    "lines.dash_capstyle",
    "lines.solid_capstyle",
    "axes.grid",
    "grid.linestyle",
    "grid.linewidth",
    "grid.color",
    "font.size",
)


# =============================================================================
# %% Reusable Drawing Utilities for Any Plot Type
# =============================================================================


def decimate_minmax(t, y, max_points):
    """
    Reduces a series to ~``max_points`` while preserving peaks (min and max per
    segment). ``y`` must be 1D. Returns decimated (t, y).
    """
    n = len(y)
    if max_points is None or n <= max_points:
        return t, y
    k = int(np.ceil(n / (max_points / 2)))  # number of samples per segment
    n_full = (n // k) * k
    yy = y[:n_full].reshape(-1, k)
    base = np.arange(0, n_full, k)[:, None]
    imin = np.argmin(np.where(np.isnan(yy), np.inf, yy), axis=1)[:, None]
    imax = np.argmax(np.where(np.isnan(yy), -np.inf, yy), axis=1)[:, None]
    idx = np.sort(np.concatenate([base + imin, base + imax], axis=1).ravel())
    if n_full < n:  # tail
        idx = np.append(idx, n - 1)
    return t[idx], y[idx]


def plot_lines(ax, p, cfg):
    """
    Draws panel data lines (decimated). If there are multiple variables, each
    is assigned a fixed color ``C{i}`` consistent with the legend.
    """
    ys = p["y"]  # (n_lines, n_time)
    for i, y in enumerate(ys):
        kws = dict(cfg["plot_kwargs"])
        if len(ys) > 1 and "color" not in kws and "c" not in kws:
            kws["color"] = f"C{i}"
        td, yd = decimate_minmax(p["t"], y, cfg["max_points"])
        ax.plot(td, yd, alpha=cfg["alpha"], rasterized=cfg["rasterized"], **kws)


def mark_events(ax, names, xs, colors, ys_text, fontsize="xx-small"):
    """
    Dashed vertical lines at ``xs`` (single ``vlines`` call) with a vertical label
    per event at height ``ys_text[i]`` (in data coordinates).
    """
    if not names:
        return
    ax.vlines(
        xs,
        *ax.get_ylim(),
        colors=colors,
        lw=0.8,
        linestyles="--",  # (a custom pattern, e.g. (5, (10, 10)), also works)
        capstyle="round",
        alpha=0.6,
    )
    for n_ev, x, c, yt in zip(names, xs, colors, ys_text):
        ax.text(
            x,
            yt,
            n_ev,
            ha="right",
            va="top",
            rotation="vertical",
            c="k",
            alpha=0.7,
            fontsize=fontsize,
            bbox={
                "facecolor": c,
                "alpha": 0.3,
                "edgecolor": "none",
                "boxstyle": "round,pad=0.3,rounding_size=.5",
            },
        )


# =============================================================================
# %% Worker Process Side: Single Page Rendering
# =============================================================================


def init_worker(low_priority=True):
    """Worker process initializer: lowers process priority to avoid maxing out CPU."""
    if not low_priority:
        return
    try:
        if os.name == "nt":
            import ctypes

            BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
            ctypes.windll.kernel32.SetPriorityClass(
                ctypes.windll.kernel32.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS
            )
        else:
            os.nice(10)
    except Exception:
        pass


def _default_n_workers():
    """
    Default number of processes. On Linux, respects CPU cores assigned to process
    (cgroups, taskset, SLURM) and uses all minus one; on Windows/macOS estimates
    physical cores as half of logical cores minus one.
    """
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)) - 1)
    return max(1, (os.cpu_count() or 2) // 2 - 1)


def _mp_context():
    """
    ``fork`` on Linux: workers start instantly and inherit already imported modules
    (in Python >= 3.14 the default shifts to ``forkserver``, which is slower).
    On Windows/macOS, ``spawn``.
    """
    import multiprocessing as mp
    import sys

    return mp.get_context("fork" if sys.platform.startswith("linux") else "spawn")


def _legend_handles(panels, cfg):
    """
    One element per line (variable) in panels, with its assigned color. Labels are
    derived from extra dimension coordinates of ``daGraph`` (shared across all panels).
    Without extra dimensions, no legend is generated.
    """
    labels = panels[0][2]["labels"] if panels else None
    if not labels:
        return [], None
    kws = cfg["plot_kwargs"]
    fixed_color = kws.get("color", kws.get("c"))
    handles = [
        Line2D([], [], color=fixed_color or f"C{i}", lw=2, label=lab)
        for i, lab in enumerate(labels)
    ]
    return handles, panels[0][2]["leg_title"]


def render_block(task):
    """
    Renders one page (block) and returns ``(pdf_bytes, png_bytes)``; ``png_bytes`` is
    None unless ``cfg["show_in_console"]``.

    task: dict containing panels [(row, col, panel_dict)], n_rows, n_cols, figsize,
          title, sharex, sharey, draw, cfg.
    """
    cfg = task["cfg"]
    with matplotlib.rc_context(cfg["rc"]):
        return _render_block(task, cfg)


def _render_block(task, cfg):
    draw = task["draw"]
    fig = Figure(figsize=task["figsize"])
    axes = fig.subplots(
        task["n_rows"],
        task["n_cols"],
        sharex=task["sharex"],
        sharey=task["sharey"],
        squeeze=False,
    )
    for ax in axes.ravel():
        ax.set_visible(False)

    has_ylabel2 = False
    for fi, ci, p in task["panels"]:
        ax = axes[fi, ci]
        ax.set_visible(True)

        draw(ax, p, cfg)  # <-- plot type specific logic

        ax.set_title(p["title"])
        if p.get("xlabel"):
            ax.set_xlabel(p["xlabel"])
        if p.get("ylabel"):
            ax.set_ylabel(p["ylabel"])
        if p.get("ylabel2") and ci == task["n_cols"] - 1:
            # Axis coordinate text (no twinx: no extra grid or memory overhead)
            ax.text(
                1.01,
                0.5,
                p["ylabel2"],
                transform=ax.transAxes,
                rotation=270,
                ha="left",
                va="center",
            )
            has_ylabel2 = True

    fig.suptitle(task["title"] or "", fontweight="bold")

    # Single legend per page on the right side (page expands to fit it)
    w, h = task["figsize"]
    m_left, m_right, m_bottom, m_top = 0.9, 0.2, 0.6, 0.7  # margins in inches
    if has_ylabel2:
        m_right += 0.3  # space for text to the right of the last column
    handles, legend_title = (
        _legend_handles(task["panels"], cfg) if cfg["legend"] else ([], None)
    )
    if handles:
        n_char = max(len(hd.get_label()) for hd in handles)
        n_char = max(n_char, len(legend_title or ""))
        extra_w = 0.6 + 0.075 * n_char  # line handle + text in inches
        w += extra_w
        m_right += extra_w
        fig.set_size_inches(w, h)
        fig.legend(
            handles=handles,
            title=legend_title,
            loc="center right",
            bbox_to_anchor=(1 - 0.05 / w, 0.5),
            fontsize="small",
            title_fontsize="small",
            frameon=False,
        )

    # Gaps between panels fixed in inches (arithmetic, zero cost). Using
    # tight_layout/constrained_layout would adjust "exactly" but forces measuring
    # ticks across all axes, adding ~40% to render time.
    n_r, n_c = task["n_rows"], task["n_cols"]
    gap_w = 0.15 if task["sharey"] else cfg["spacing"][1]
    gap_h = 0.3 if task["sharex"] else cfg["spacing"][0]
    ax_w = (w - m_left - m_right - (n_c - 1) * gap_w) / n_c
    ax_h = (h - m_bottom - m_top - (n_r - 1) * gap_h) / n_r
    fig.subplots_adjust(
        left=m_left / w,
        right=1 - m_right / w,
        bottom=m_bottom / h,
        top=1 - m_top / h,
        wspace=gap_w / ax_w,
        hspace=gap_h / ax_h,
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="pdf", dpi=cfg["dpi"])
    png = None
    if cfg.get("show_in_console"):
        buf_png = io.BytesIO()
        fig.savefig(buf_png, format="png", dpi=cfg["dpi"])
        png = buf_png.getvalue()
    return buf.getvalue(), png


# =============================================================================
# %% Main Process Side: Block Distribution, Multiprocessing & PDF Merging
# =============================================================================


def _vline_labels(da_panel, x):
    """Labels for each line (same order as reshaping to (n_lines, n_time))."""
    import itertools

    others = [d for d in da_panel.dims if d != x]
    if not others:
        return None, ""
    combos = itertools.product(*[da_panel[d].values for d in others])
    return [", ".join(str(v) for v in c) for c in combos], ", ".join(others)


def _panel_base(da_panel, x, extract, **labels_kwargs):
    """Panel dict: basic data + plot type specific data + axis labels."""
    t = da_panel[x].values
    others = [d for d in da_panel.dims if d != x]
    labels, leg_title = _vline_labels(da_panel, x)
    p = {
        "t": t,
        "y": da_panel.transpose(*others, x).values.reshape(-1, len(t)),
        "labels": labels,
        "leg_title": leg_title,
        "title": None,
        "xlabel": None,
        "ylabel": None,
        "ylabel2": None,
    }
    p.update(extract(da_panel))
    p.update(labels_kwargs)
    return p


def _with_units(text, unit, sqr_brackets=False):
    if not unit:
        return str(text)
    return f"{text} [{unit}]" if sqr_brackets else f"{text} ({unit})"


def facet_pdf(
    daGraph,
    extract,
    draw,
    save_file=None,
    *,
    show_in_console=False,
    n_rows="ID",
    n_cols=None,
    num_per_block=4,
    x="time",
    sharex=False,
    sharey=False,
    aspect=1.5,
    n_workers=None,
    max_points=1500,
    rasterized=False,
    alpha=1.0,
    dpi=150,
    legend=True,
    spacing=(0.85, 0.75),
    low_priority=True,
    title=None,
    plot_kwargs=None,
    cfg=None,
):
    """
    Creates a multi-page PDF with one page per block of ``n_rows`` and a panel grid
    (``n_rows`` × ``n_cols``, or ``num_per_block`` columns if ``n_cols`` is None).
    See ``extract`` and ``draw`` contract in the module docstring.

    Engine Parameters
    -----------------
    save_file : Path or str. Output PDF path (forces ``.pdf`` extension). Required
        unless ``show_in_console=True``.
    show_in_console : If True, each page is also displayed in the interactive
        window / Jupyter console (as PNG, via ``IPython.display``) as it is
        rendered. The PDF is only written if ``save_file`` is given.
    n_workers : Number of processes. Defaults to ~ physical cores - 1.
    max_points : Maximum points per line per panel (None = no decimation).
    rasterized, alpha, dpi : See ``plot_lines``.
    legend : Displays right-side legend if ``daGraph`` has extra dimensions.
    spacing : Gap between panels in inches (vertical, horizontal).
    title : Page title. None = ``daGraph.name``; "" = no title.
    plot_kwargs : kwargs passed to ``ax.plot`` for data lines.
    cfg : Plot type options (picklable dict) passed to ``draw`` alongside engine options.
    """
    import contextlib
    import sys
    import time
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool
    from pathlib import Path

    from loguru import logger
    from pypdf import PdfReader, PdfWriter
    from tqdm import tqdm

    if save_file is None and not show_in_console:
        raise ValueError("save_file (PDF path) is required if not show_in_console")
    n_pdf = None
    if save_file is not None:
        save_file = Path(save_file)
        if not save_file.parent.is_dir():
            raise FileNotFoundError(f"Folder {save_file.parent} does not exist")
        n_pdf = save_file.with_suffix(".pdf")
    if show_in_console:
        from IPython.display import Image, display

    timer_plot = time.time()
    logger.info("Generating plots (fast mode)...")

    name = str(daGraph.name or "")
    chunk_size = num_per_block if n_cols is not None else num_per_block**2
    starts = range(0, daGraph.sizes[n_rows], chunk_size)
    n_pages = len(starts)

    if n_workers is None:
        n_workers = _default_n_workers()
    n_workers = max(1, min(n_workers, n_pages))

    total_cfg = {
        "rasterized": rasterized,
        "max_points": max_points,
        "alpha": alpha,
        "dpi": dpi,
        "show_in_console": show_in_console,
        "legend": legend,
        "spacing": spacing,
        "plot_kwargs": plot_kwargs or {},
        "rc": {k: matplotlib.rcParams[k] for k in _RC_KEYS},
        **(cfg or {}),
    }

    def _tasks():
        for start in starts:
            dax = daGraph.isel(**{n_rows: slice(start, start + chunk_size)})
            n_ids = dax.sizes[n_rows]
            panels = []

            if n_cols is not None:
                n_rows_count, n_cols_count = n_ids, daGraph.sizes[n_cols]
                for ri in range(n_ids):
                    for ci in range(n_cols_count):
                        da_p = dax.isel({n_rows: ri, n_cols: ci})
                        et = {
                            "title": f"{n_cols}={da_p[n_cols].values}"
                            if ri == 0
                            else None,
                            "xlabel": (
                                _with_units(x, da_p[x].attrs.get("units"))
                                if ri == n_ids - 1
                                else None
                            ),
                            "ylabel": (
                                _with_units(name, da_p.attrs.get("units"))
                                if ci == 0
                                else None
                            ),
                            "ylabel2": (
                                f"{n_rows}={da_p[n_rows].values}"
                                if ci == n_cols_count - 1
                                else None
                            ),
                        }
                        panels.append((ri, ci, _panel_base(da_p, x, extract, **et)))
            else:
                n_cols_count = num_per_block
                n_rows_count = -(-n_ids // num_per_block)
                for i in range(n_ids):
                    da_p = dax.isel({n_rows: i})
                    et = {
                        "title": _with_units(
                            da_p[n_rows].values,
                            da_p.attrs.get("units"),
                            sqr_brackets=True,
                        )
                    }
                    panels.append(
                        (
                            i // num_per_block,
                            i % num_per_block,
                            _panel_base(da_p, x, extract, **et),
                        )
                    )

            yield {
                "panels": panels,
                "n_rows": n_rows_count,
                "n_cols": n_cols_count,
                "figsize": (3.5 * aspect * n_cols_count, 3.5 * n_rows_count),
                "title": name if title is None else title,
                "sharex": sharex,
                "sharey": sharey,
                "draw": draw,
                "cfg": total_cfg,
            }

    @contextlib.contextmanager
    def _hide_main():
        """
        With ``spawn`` (Windows), each child process re-executes the ``__main__``
        script if it contains ``__file__`` (such as in VS Code interactive windows
        or ``ipython script.py``), causing it to crash. Children do not need anything
        from ``__main__``, so it is hidden while worker processes are active.
        """
        main = sys.modules.get("__main__")
        empty = object()
        file_ = main.__dict__.pop("__file__", empty)
        spec = main.__dict__.get("__spec__", empty)
        main.__spec__ = None  # spawn reads it directly: must exist
        try:
            yield
        finally:
            if file_ is not empty:
                main.__file__ = file_
            if spec is empty:
                del main.__spec__
            else:
                main.__spec__ = spec

    def _collect(writer, res):
        pdf_bytes, png = res
        writer.append(PdfReader(io.BytesIO(pdf_bytes)))
        if png is not None:
            display(Image(data=png))

    def _run(n_w):
        writer = PdfWriter()
        if n_w == 1:
            results = (render_block(t) for t in _tasks())
            for res in tqdm(results, total=n_pages, desc=f"Plots for {name}"):
                _collect(writer, res)
        else:
            with (
                _hide_main(),
                ProcessPoolExecutor(
                    max_workers=n_w,
                    mp_context=_mp_context(),
                    initializer=init_worker,
                    initargs=(low_priority,),
                ) as ex,
            ):
                results = ex.map(render_block, _tasks())
                for res in tqdm(results, total=n_pages, desc=f"Plots for {name}"):
                    _collect(writer, res)
        return writer

    try:
        writer = _run(n_workers)
    except BrokenProcessPool:
        logger.error(
            "A child process died during startup or rendering (the actual traceback "
            "is printed to the kernel console/terminal, not the cell output). "
            "Falling back to single-process execution."
        )
        n_workers = 1
        writer = _run(1)

    if n_pdf is not None:
        with open(n_pdf, "wb") as f:
            writer.write(f)
    writer.close()

    logger.success(
        f"Generated {n_pages} page(s) in {n_pdf.name if n_pdf else 'console'} using {n_workers} process(es) "
        f"in {time.time() - timer_plot:.3f} s"
    )
