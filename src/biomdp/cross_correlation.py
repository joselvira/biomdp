# =============================================================================
# %% LOAD LIBRARIES
# =============================================================================

import time

import numpy as np
import xarray as xr

# =============================================================================

__author__ = "Jose L. L. Elvira"
__version__ = "0.1.0"
__date__ = "26/07/2026"

"""
Updates:
    31/07/2026, v.0.1.1
            - Included the option to normalize ftt in "auto_correlation" function.

    26/07/2026, v.0.1.0
        - Extracted from general_processing_functions module.
"""


# from scipy import stats
def _cross_correl_simple_aux(datos1, datos2, ID=None):
    """
    Simple and slow but exact function for cross correlation.
    So far, data1 has to be the longest one.
    Uses polars, faster than numpy.

    Example:
    daCrosscorr = xr.apply_ufunc(
        _cross_correl_simple_aux,
        daInstrument1,
        daInstrument2,,

        input_core_dims=[
            ["time"],
            ["time"],
        ],
        output_core_dims=[["lag"]],
        exclude_dims=set(
            (
                "lag",
                "time",
            )
        ),
        vectorize=True,
        dask="parallelized",
        keep_attrs=False,
    ).dropna(dim="lag", how="all")
    """
    import polars as pl

    if ID is not None:
        print(ID)

    # pre-creates the array where to store the correlations of each offset
    corr = np.full(max(len(datos1), len(datos2)), np.nan)

    if np.isnan(datos1).all() or np.isnan(datos2).all():
        print(f"{ID} vacío")
        return corr

    try:
        # Remove nans from the end for function stats.pearson
        dat1 = datos1[~np.isnan(datos1)]
        dat2 = datos2[~np.isnan(datos2)]

        for i in range(0, dat1.size - dat2.size):
            # Versión Polars más rápida
            df = pl.from_numpy(
                np.vstack([dat1[i : i + dat2.size], dat2]),
                schema=["a", "b"],
                orient="col",
            )
            corr[i] = df.select(pl.corr("a", "b")).item()

            # scipy version slowler
            # corr[i] = stats.pearsonr(dat1[i : i + dat2.size], dat2).statistic
        # plt.plot(corr)

    except Exception as err:
        print("Computational error, possibly empty", err)

    return corr  # si hay algún error, lo devuelve vacío


def _cross_correl_noisy_aux(datos1, datos2, ID=None):
    """
    Fast but sometimes less accurate function for cross correlation.
    Good for noisy signals.
    """
    from scipy.signal import correlate

    # if ID is not None:
    #     print(ID)

    ccorr = np.full(max(len(datos1), len(datos2)), np.nan)

    if np.isnan(datos1).all() and np.isnan(datos2).all():
        return ccorr

    # Delete Nans
    dat1 = datos1[~np.isnan(datos1)]
    dat2 = datos2[~np.isnan(datos2)]

    # Normalize
    dat1 = (dat1 - np.mean(dat1)) / np.std(dat1)
    dat2 = (dat2 - np.mean(dat2)) / np.std(dat2)

    # Path with zeros
    if len(dat1) != len(dat2):
        if len(dat1) < len(dat2):
            dat1 = np.append(dat1, np.zeros(len(dat2) - len(dat1)))
        else:
            dat2 = np.append(dat2, np.zeros(len(dat1) - len(dat2)))

    # Compute cross-correlation
    c = correlate(
        np.gradient(np.gradient(dat1)), np.gradient(np.gradient(dat2)), "full"
    )
    c = c[int(len(c) / 2) :]
    ccorr[: len(c)] = c
    # desfase = int(np.ceil(np.argmax(ccorr) - (len(ccorr)) / 2) + 1)

    return ccorr  # [int(len(ccorr) / 2) :]


def cross_correl_xr(
    da1: xr.DataArray, da2: xr.DataArray, func=_cross_correl_simple_aux
) -> xr.DataArray:
    """
    Apply the specified cross-correlation function to dataarrays
    """
    if "ID" in da1.coords:
        id = da1.ID
    else:
        id = None

    """
    datos2=da1[0,0,0,0,0].data
    datos1=da2[0,0,0,0,0].data
    ID=id[0].data
    """
    daCrosscorr = xr.apply_ufunc(
        func,
        da2,
        da1,
        id,
        input_core_dims=[
            ["time"],
            ["time"],
            [],
            # [],
            # [],
        ],
        output_core_dims=[["lag"]],  # datos que devuelve
        exclude_dims=set(
            (
                "lag",
                "time",
            )
        ),
        vectorize=True,
        dask="parallelized",
        keep_attrs=False,
        # kwargs=args_func_cortes,
    ).dropna(dim="lag", how="all")
    daCrosscorr = daCrosscorr.assign_coords(lag=range(len(daCrosscorr.lag)))
    return daCrosscorr


def _autocorrel(dat, norm, ID=None):
    """
    Autocorrelation using Polars Pearson correlation at each lag. O(N²).

    Computes the Pearson correlation coefficient between the signal and its
    shifted version for every lag from 0 to N-1. The result oscillates
    between -1 and 1 at all lags because each window is independently
    re-normalized. Slower than FFT for long signals.
    """
    import polars as pl

    # pre-creates the array where to store the correlations of each offset
    autocorr = np.full(len(dat), np.nan)
    if np.isnan(dat).all():
        # print(f"{ID} empty")
        return autocorr

    dat = dat[~np.isnan(dat)]
    df = pl.from_numpy(dat)

    corr = df.select(
        [
            pl.corr(pl.col("column_0"), pl.col("column_0").shift(lag)).alias(
                f"lag_{lag}"
            )
            for lag in range(len(df))
        ]
    ).to_numpy()[0]

    autocorr[: len(corr)] = corr

    return autocorr


def _autocorrel_fft(dat, norm, ID=None):
    """
    Autocorrelation using FFT with unbiased normalization. O(N log N).

    Computes the autocorrelation function via FFT (Wiener-Khinchin theorem)
    with an unbiased normalization that divides by (N - k) * var at each
    lag k. This produces results equivalent to Pearson correlation at each
    lag (oscillates between -1 and 1 without amplitude damping), but is
    orders of magnitude faster for long signals.

    Notes
    -----
    The unbiased normalization can exceed ±1 for very large lags (few
    overlapping samples). It is recommended to use max_lag < N // 2
    via the auto_correl wrapper to avoid this region.
    """
    from scipy.signal import correlate

    autocorr = np.full(len(dat), np.nan)
    if np.isnan(dat).all():
        return autocorr

    dat = dat[~np.isnan(dat)]
    n = len(dat)
    dat_c = dat - dat.mean()
    var = np.var(dat_c)

    if var == 0:
        # Constant signal: autocorr is 1 at all lags by convention
        autocorr[:n] = 1.0
        return autocorr

    # FFT-based cross-correlation: O(N log N)
    acf_raw = correlate(dat_c, dat_c, method="fft")
    acf_raw = acf_raw[n - 1 :]  # keep only lags >= 0

    # Unbiased normalization: divide by (N - k) * var at each lag k
    # → comparable to Pearson, oscillates between -1 and 1
    lags = np.arange(n)
    acf = acf_raw

    if norm:
        acf = acf_raw / (var * (n - lags))

    autocorr[:n] = acf
    return autocorr


def auto_correl(
    daData: xr.DataArray,
    method: str = "fft",
    norm_ftt: bool = True,
    max_lag: int | None = None,
) -> xr.DataArray:
    """
    Compute the autocorrelation function of a DataArray along the 'time' dimension.

    Parameters
    ----------
    daData : xr.DataArray
        Input data. Must have a 'time' dimension. Can contain NaNs at the
        end of each series (variable-length signals).
    method : str, optional
        Algorithm to use:
        - ``'fft'``    : FFT with unbiased normalization. O(N log N). Fast.
                         Results are equivalent to Pearson (oscillates ±1).
                         Recommended for long signals or large datasets.
        - ``'polars'`` : Pearson correlation at each lag via Polars. O(N²).
                         Slower but exact Pearson. Kept for reference.
        Default is ``'fft'``.
    norm_ftt : bool, optional
        Whether to normalize the autocorrelation values by the variance of the
        signal. Default is True.
    max_lag : int, optional
        Maximum lag to keep in the output. If None, all lags (up to N) are
        returned. For the ``'fft'`` method it is recommended to set
        ``max_lag < N // 2`` to avoid the region where the unbiased
        normalization can exceed ±1. Default is None.

    Returns
    -------
    xr.DataArray
        Autocorrelation values with a new 'lag' dimension (integer indices).

    Examples
    --------
    dat=daData[0,0,0].data
    ID=daData[0,0,0].ID.data
    """
    _func_map = {
        "fft": _autocorrel_fft,
        "polars": _autocorrel,
    }
    if method not in _func_map:
        raise ValueError(f"method must be one of {list(_func_map)}, got '{method}'")
    _func = _func_map[method]

    if "ID" in daData.coords:
        id = daData.ID
    else:
        id = None

    """
    dat=daData[0,0,0,0,0].data
    ID=id[0].data
    """
    daAutocorr = xr.apply_ufunc(
        _func,
        daData,
        norm_ftt,
        id,
        input_core_dims=[
            ["time"],
            [],
            [],
        ],
        output_core_dims=[["lag"]],
        exclude_dims=set(("lag",)),
        vectorize=True,
        dask="parallelized",
        keep_attrs=False,
    ).dropna(dim="lag", how="all")

    daAutocorr = daAutocorr.assign_coords(lag=range(len(daAutocorr.lag)))

    if max_lag is not None:
        daAutocorr = daAutocorr.isel(lag=slice(0, max_lag))

    return daAutocorr


# =============================================================================
# %% TESTS
# =============================================================================

if __name__ == "__main__":
    # =============================================================================
    # %%---- Create a sample
    # =============================================================================

    # from scipy.signal import butter, filtfilt
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np

    # import pandas as pd
    import xarray as xr

    # import seaborn as sns
    from biomdp.create_time_series import create_time_series_xr

    rnd_seed = np.random.seed(
        12340
    )  # fija la aleatoriedad para asegurarse la reproducibilidad
    n = 10
    duration = 15
    freq = 200.0
    pre_a = (
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
    post_a = (
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
    var_a = xr.concat([pre_a, post_a], dim="moment")
    var_a.sel(n_var="a").plot.line(x="time", col="ID", col_wrap=4)
    var_a.attrs["freq"] = freq

    # =============================================================================
    # %% TEST CROSSCORREL
    # =============================================================================
    daInstrument1 = var_a.sel(n_var="a", moment="pre").drop_vars("moment")
    daInstrument2 = daInstrument1.isel(
        time=slice(int(4 * var_a.freq), int(8 * var_a.freq))
    )

    daCrosscorr = xr.apply_ufunc(
        _cross_correl_simple_aux,
        daInstrument1,
        daInstrument2,  # .sel(partID=da1['partID'], tiempo='pre'),
        # daInstrument1.time.size - daInstrument2.time.size,
        # daInstrument1['partID'],
        # daInstrument1['tiempo'],
        input_core_dims=[
            ["time"],
            ["time"],
            # [],
            # [],
            # [],
        ],
        output_core_dims=[["lag"]],
        exclude_dims=set(
            (
                "lag",
                "time",
            )
        ),
        # dataset_fill_value=np.nan,
        vectorize=True,
        dask="parallelized",
        keep_attrs=False,
        # kwargs=args_func_cortes,
    ).dropna(dim="lag", how="all")
    daCrosscorr = daCrosscorr.assign_coords(lag=range(len(daCrosscorr.lag)))
    daCrosscorr.plot.line(x="lag")

    from biomdp.general_processing_functions import nanargmax_xr

    nanargmax_xr(daCrosscorr, dim="lag")

    daCrosscorr_rap = xr.apply_ufunc(
        _cross_correl_noisy_aux,
        daInstrument1,
        daInstrument2,  # .sel(partID=da1['partID'], tiempo='pre'),
        input_core_dims=[
            ["time"],
            ["time"],
            # [],
        ],
        output_core_dims=[["lag"]],
        exclude_dims=set(
            (
                "lag",
                "time",
            )
        ),
        # dataset_fill_value=np.nan,
        vectorize=True,
        dask="parallelized",
        keep_attrs=False,
        # kwargs=args_func_cortes,
    ).dropna(dim="lag", how="all")
    daCrosscorr_rap = daCrosscorr_rap.assign_coords(lag=range(len(daCrosscorr_rap.lag)))
    daCrosscorr_rap.plot.line(x="lag")
    nanargmax_xr(daCrosscorr_rap, dim="lag")

    # =============================================================================
    # %% TEST AUTOCORRELATION
    # =============================================================================
    var_b = np.log(var_a).assign_coords(n_var=["b"])
    var_c = var_b.diff(dim="time").assign_coords(n_var=["c"])
    var_d = (var_c * np.random.rand(*var_c.shape)).assign_coords(n_var=["d"])
    dataAutocorr = xr.concat([var_a, var_b, var_c, var_d], dim="n_var", join="outer")

    ac = auto_correl(daData=dataAutocorr, method="fft", norm_ftt=True)
    ac.plot.line(x="lag", row="ID", col="n_var", sharey=False)

    ac2 = auto_correl(dataAutocorr, method="polars")  # , max_lag=50)
    ac2.plot.line(x="lag", row="ID", col="n_var", sharey=False)
    (ac - ac2).plot.line(x="lag", row="ID", col="n_var", sharey=False)
