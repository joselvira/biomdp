"""Tests for ``block_plots_fast`` (PDF output and ``show_in_console``)."""

import numpy as np
import pytest
import xarray as xr
from pypdf import PdfReader

from biomdp.create_time_series import create_time_series_xr
from biomdp.plotting.plotting_functions import block_plots_fast

N_ID = 5


def _n_pages(path):
    return len(PdfReader(path).pages)


@pytest.fixture(scope="module")
def da_total():
    """Small synthetic dataset: ID(5) x moment(2) x n_var(3) x time (2 s @ 100 Hz)."""
    params = {
        "a": ([25, 29], [40, 45]),
        "b": ([35, 39], [50, 55]),
        "c": ([35, 39], [10, 15]),
    }
    parts = []
    for var, (offset, amp) in params.items():
        moments = []
        for moment in ("pre", "post"):
            da = (
                create_time_series_xr(
                    rnd_seed=None,
                    num_subj=N_ID,
                    fs=100.0,
                    IDini=0,
                    range_offset=offset,
                    range_amp=amp,
                    range_freq=[1.48, 1.52],
                    range_af=[0, 30],
                    amplific_noise=[0.4, 0.7],
                    fc_noise=[3.0, 3.5],
                    range_duration=[2, 2],
                )
                .expand_dims({"n_var": [var], "moment": [moment]})
                .transpose("ID", "moment", "n_var", "time")
            )
            moments.append(da)
        parts.append(xr.concat(moments, dim="moment"))
    da = xr.concat(parts, dim="n_var")
    da.name = "Angle"
    da.attrs["freq"] = 1 / (da.time[1].values - da.time[0].values)
    da.attrs["units"] = "deg"
    da.time.attrs["units"] = "s"
    return da


# ----------------------------------------------------------------------------
# PDF output
# ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs, expected_pages",
    [
        # n_cols=None: num_per_block**2 IDs per page -> 5 IDs fit in 1 page
        (dict(num_per_block=3), 1),
        (dict(num_per_block=2), 2),
        # n_cols given: num_per_block IDs per page
        (dict(n_rows="ID", n_cols="n_var"), 2),  # default num_per_block=4 -> 2 pages
        (dict(n_rows="ID", n_cols="moment", num_per_block=3), 2),
        (
            dict(
                n_rows="ID",
                n_cols="moment",
                num_per_block=3,
                sharex=True,
                sharey=True,
            ),
            2,
        ),
        (
            dict(n_rows="ID", n_cols="n_var", max_points=50, marker=".", alpha=0.5),
            2,
        ),
    ],
)
def test_saves_pdf_with_expected_pages(da_total, tmp_path, kwargs, expected_pages):
    out = tmp_path / "out.pdf"
    block_plots_fast(da_total, save_file=out, n_workers=1, **kwargs)
    assert out.is_file()
    assert _n_pages(out) == expected_pages


def test_extension_forced_to_pdf(da_total, tmp_path):
    block_plots_fast(da_total, save_file=tmp_path / "out", n_workers=1)
    assert (tmp_path / "out.pdf").is_file()


def test_multiprocess_matches_single_process(da_total, tmp_path):
    p1, p2 = tmp_path / "one.pdf", tmp_path / "two.pdf"
    kw = dict(n_rows="ID", n_cols="n_var", num_per_block=2)
    block_plots_fast(da_total, save_file=p1, n_workers=1, **kw)
    block_plots_fast(da_total, save_file=p2, n_workers=2, **kw)
    assert _n_pages(p1) == _n_pages(p2) == 3


# ----------------------------------------------------------------------------
# Events and horizontal lines
# ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def evts(da_total):
    from biomdp.slice_time_series_phases import detect_events_onset

    e = detect_events_onset(
        da_total, reference_var={"moment": "pre", "n_var": "b"}, threshold=45
    )
    return e.rename({"num_event": "n_event"})


def test_events_and_hlines_from_mean(da_total, evts, tmp_path):
    out = tmp_path / "out.pdf"
    block_plots_fast(
        da_total,
        daEvents=evts,
        daHlines=da_total.mean(dim="time"),
        hlines_dim="n_var",
        n_rows="ID",
        n_cols="moment",
        save_file=out,
        n_workers=1,
    )
    assert _n_pages(out) == 2


def test_hlines_constant_threshold(da_total, evts, tmp_path):
    out = tmp_path / "out.pdf"
    block_plots_fast(
        da_total,
        daEvents=evts,
        daHlines=xr.full_like(da_total.mean(dim="time"), 45),
        hlines_dim="n_var",
        n_rows="ID",
        n_cols="moment",
        save_file=out,
        n_workers=1,
    )
    assert out.is_file()


def test_events_dataset_raises(da_total, tmp_path):
    ds = xr.Dataset({"x": ("n_event", np.arange(2.0))}, coords={"n_event": ["a", "b"]})
    with pytest.raises(TypeError):
        block_plots_fast(da_total, daEvents=ds, save_file=tmp_path / "o.pdf")


# ----------------------------------------------------------------------------
# show_in_console
# ----------------------------------------------------------------------------
@pytest.fixture
def displayed(monkeypatch):
    """Captures what is sent to ``IPython.display.display``."""
    import IPython.display as ipd

    calls = []
    monkeypatch.setattr(ipd, "display", lambda obj, *a, **k: calls.append(obj))
    return calls


PNG_MAGIC = b"\x89PNG"


def test_show_in_console_without_save_file(da_total, tmp_path, monkeypatch, displayed):
    monkeypatch.chdir(tmp_path)
    block_plots_fast(
        da_total, n_rows="ID", n_cols="n_var", show_in_console=True, n_workers=1
    )
    assert len(displayed) == 2  # one image per page
    assert all(img.data[:4] == PNG_MAGIC for img in displayed)
    assert list(tmp_path.iterdir()) == []  # nothing written


def test_show_in_console_and_save_file(da_total, tmp_path, displayed):
    out = tmp_path / "out.pdf"
    block_plots_fast(
        da_total, num_per_block=2, save_file=out, show_in_console=True, n_workers=1
    )
    assert len(displayed) == 2
    assert _n_pages(out) == 2


def test_no_display_by_default(da_total, tmp_path, displayed):
    block_plots_fast(da_total, save_file=tmp_path / "o.pdf", n_workers=1)
    assert displayed == []


# ----------------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------------
def test_requires_save_file_or_console(da_total):
    with pytest.raises(ValueError, match="save_file"):
        block_plots_fast(da_total)


def test_missing_folder_raises(da_total, tmp_path):
    with pytest.raises(FileNotFoundError):
        block_plots_fast(da_total, save_file=tmp_path / "nope" / "o.pdf")
