# =============================================================================
# %% LOAD LIBRARIES
# =============================================================================

__author__ = "Jose L. L. Elvira"
__version__ = "0.5.1"
__date__ = "12/03/2025"

"""
Updates:
    09/05/2025, v0.6.0
        - Added function get_list_dir to scan subdirectories, returns a list
          of subdirectories.

    12/03/2025, v0.5.1
        - Adapted to biomdp with translations.

    05/03/2025, v0.5.0
        - Removed function create_time_series_xr
          (included in file create_time_series.py).

    14/01/2025, v0.4.1
        - Añadida función round_to_nearest_even_2_decimal

    16/12/2024, v0.4.0
        - Incluida función procesaEMG, que estaba en nexus_processing_functions.
    
    13/12/2024, v0.3.2
        - Corregido nanargmax_xr, no incluía bien la dimensión.
        - Cambiado nombre _cross_correl_rapida_aux por _cross_correl_noisy_aux
    
    17/11/2024, v0.3.1
        - Corregido nanargmax_xr, no incluía bien la dimensión
        - Incluido nanargmin_xr

    04/09/2024, v0.3.0
        - Versión de crosscorrelation simple con Polars, mucho más rápida    

    29/08/2024, v0.2.0
        - Añadidas funciones auxiliares para calcular cross correlation. 

    17/08/2024, v0.1.0
        - Incluidas algunas funciones generales. 

"""

from typing import List

import numpy as np
import xarray as xr

import scipy.integrate as integrate

import time


# =============================================================================
# %% Functins
# =============================================================================


def integrate_window(
    daData: xr.DataArray,
    daWindow: xr.DataArray | None = None,
    daOffset: xr.DataArray | None = None,
    result_return: str = "continuous",
) -> xr.DataArray:
    """
    Integrates time series data within a specified window, with an optional offset.

    Parameters
    ----------
    daData : xarray.DataArray
        The data to be integrated. It should have a 'time' dimension.
    daWindow : xarray.DataArray, optional
        The window specifying the start ('ini') and end ('fin') events for integration.
        If None, defaults to the full range of daData.
    daOffset : xarray.DataArray, optional
        Offset to be subtracted from daData before integration. If None, defaults to zero.
    result_return : str, optional
        Specifies the type of result to return.
        "continuous" : returns a continuous time series with integrated signal
        "discrete" : returns discrete integrated values at the end of the window.
        Default is "continuous".

    Returns
    -------
    xarray.DataArray
        The integrated data, either as a continuous signal or discrete values depending on
        the value of result_return.

    Notes
    -----
    This function is useful for computing the integral of a time series data over a specified
    window. It allows for offset correction before integration and supports both continuous
    and discrete result types.
    """

    # If empty, fill daWindow with first and last data
    if daWindow is None:
        daWindow = (
            xr.full_like(daData.isel(time=0).drop_vars("time"), np.nan).expand_dims(
                {"event": ["ini", "fin"]}, axis=-1
            )
        ).copy()
        daWindow.loc[dict(event=["ini", "fin"])] = np.array([0, len(daData.time)])

    if daOffset is None:
        daOffset = (xr.full_like(daData.isel(time=0).drop_vars("time"), 0.0)).copy()

    if result_return == "discrete":

        def _integrate_discrete(data, t, offset, ini, fin, ID):
            if np.isnan(ini) or np.isnan(fin):
                return np.nan
            ini = int(ini)
            fin = int(fin)
            # print(ID)
            # plt.plot(data[ini:fin])
            try:
                dat = integrate.cumulative_trapezoid(
                    data[ini:fin] - offset, t[ini:fin], initial=0
                )[-1]
            except:
                # print(f'Fallo al integrar en {ID}')
                dat = np.nan
            return dat

        """
        data = daData[0,0].data
        t = daData.time.data
        ini = daWindow[0,0].isel(event=0).data
        fin = daWindow[0,0].isel(event=1).data
        offset = daOffset[0,0].data
        """
        daInt = xr.apply_ufunc(
            _integrate_discrete,
            daData,
            daData.time,
            daOffset,
            daWindow.isel(event=0),
            daWindow.isel(event=1),
            daData.ID,
            input_core_dims=[["time"], ["time"], [], [], [], []],
            # output_core_dims=[['time']],
            exclude_dims=set(("time",)),
            vectorize=True,
            # join='exact',
        )

    elif result_return == "continuous":

        def _integrate_continuous(data, time, peso, ini, fin):
            # if np.count_nonzero(~np.isnan(data))==0:
            #     return np.nan
            dat = np.full(len(data), np.nan)
            try:
                ini = int(ini)
                fin = int(fin)
                # plt.plot(data[ini:fin])
                dat[ini:fin] = integrate.cumulative_trapezoid(
                    data[ini:fin] - peso, time[ini:fin], initial=0
                )
                # plt.plot(dat)
            except:
                print("Error calculando la integral")
                pass  # dat = np.full(len(data), np.nan)
            return dat

        """
        data = daDatos[2,0].data #.sel(axis='z').data
        time = daDatos.time.data
        peso=daPeso[2,0].sel(stat='media').data
        ini = daEventos[2,0].sel(evento='iniMov').data
        fin = daEventos[2,0].sel(evento='finMov').data
        plt.plot(data[int(ini):int(fin)])
        """
        daInt = xr.apply_ufunc(
            _integrate_continuous,
            daData,
            daData.time,
            daOffset,
            daWindow.isel(event=0),
            daWindow.isel(event=1),
            input_core_dims=[["time"], ["time"], [], [], []],
            output_core_dims=[["time"]],
            # exclude_dims=set(('time',)),
            vectorize=True,
            join="exact",
        )

    else:
        raise ValueError("result_return must be 'continuous' or 'discrete'")

    daInt.attrs = daData.attrs

    return daInt


def detrend_dim(da: xr.DataArray, dim: str, deg: int = 1) -> xr.DataArray:
    """
    Detrend the signal along a single dimension.

    Parameters
    ----------
    da: xr.DataArray
        The data to be detrended.
    dim: str
        The dimension along which the detrending is performed.
    deg: int, optional
        The degree of the detrending polynomial. Default is 1 (linear).

    Returns
    -------
    daDetrended: xr.DataArray
        The detrended data.
    """

    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)

    return da - fit


def RMS(daData: xr.DataArray, daWindow: xr.DataArray | None = None) -> xr.DataArray:
    """
    Calculate RMS in dataarray with dataarray window
    """
    # If empty, fill daWindow with first and last data
    if daWindow is None:
        daWindow = (
            xr.full_like(daData.isel(time=0).drop_vars("time"), np.nan).expand_dims(
                {"event": ["ini", "fin"]}, axis=-1
            )
        ).copy()
        daWindow.loc[dict(event=["ini", "fin"])] = np.array([0, len(daData.time)])

    def _rms(data, ini, fin):
        if np.count_nonzero(~np.isnan(data)) == 0:
            return np.array(np.nan)
        data = data[int(ini) : int(fin)]
        data = data[~np.isnan(data)]
        return np.linalg.norm(data[~np.isnan(data)]) / np.sqrt(len(data))

    """
    data = daData[0,0,0].data
    ini = daWindow[0,0,0].sel(event='ini').data
    fin = daWindow[0,0,0].sel(event='fin').data
    """
    # daRecortado = recorta_ventana_analisis(daData, daWindow)
    daRMS = xr.apply_ufunc(
        _rms,
        daData,
        daWindow.isel(event=0),
        daWindow.isel(event=1),
        input_core_dims=[["time"], [], []],
        vectorize=True,
    )
    return daRMS


def calculate_distance(point1: xr.DataArray, point2: xr.DataArray):
    """
    Calculates the distance between two 3D points.
    Requires dimension with x, y, z coordinates with name 'axis'.
    """
    return np.sqrt(((point1 - point2) ** 2).sum("axis"))


# Función para detectar onsets
"""
Ref: Solnik, S., Rider, P., Steinweg, K., Devita, P., & Hortobágyi, T. (2010). Teager-Kaiser energy operator signal conditioning improves EMG onset detection. European Journal of Applied Physiology, 110(3), 489–498. https://doi.org/10.1007/s00421-010-1521-8

Función sacada de Duarte (https://nbviewer.org/github/BMClab/BMC/blob/master/notebooks/Electromyography.ipynb)
The Teager-Kaiser Energy operator to improve onset detection
The Teager-Kaiser Energy (TKE) operator has been proposed to increase the accuracy of the onset detection by improving the SNR of the EMG signal (Li et al., 2007).
"""


def tkeo(x: np.ndarray) -> np.ndarray:
    r"""Calculates the Teager-Kaiser Energy operator.

    Parameters
    ----------
    x : 1D array_like
        raw signal

    Returns
    -------
    y : 1D array_like
        signal processed by the Teager-Kaiser Energy operator

    Notes
    -----

    See this notebook [1]_.

    References
    ----------
    .. [1] https://github.com/demotu/BMC/blob/master/notebooks/Electromyography.ipynb

    """
    x = np.asarray(x)
    y = np.copy(x)
    # Teager-Kaiser Energy operator
    y[1:-1] = x[1:-1] * x[1:-1] - x[:-2] * x[2:]
    # correct the data in the extremities
    y[0], y[-1] = y[1], y[-2]

    return y


def process_EMG(
    daEMG: xr.DataArray,
    fr: float | None = None,
    fc_band: List[float] = [10, 400],
    fclow: float = 8,
    btkeo: bool = False,
) -> xr.DataArray:
    from biomdp.filter_butter import filter_butter, filter_butter_bandpass

    if fr == None:
        fr = daEMG.freq
    # Filtro band-pass
    daEMG_proces = filter_butter_bandpass(
        daEMG, fr=fr, fclow=fc_band[0], fchigh=fc_band[1]
    )
    # Centers signal, necessary?
    daEMG_proces = daEMG_proces - daEMG_proces.mean(dim="time")

    if btkeo:
        daEMG_proces = xr.apply_ufunc(
            tkeo,
            daEMG_proces,
            input_core_dims=[["time"]],
            output_core_dims=[["time"]],
            vectorize=True,
        )
    # Rectifica
    daEMG_proces = abs(daEMG_proces)
    # filtro low-pass
    daEMG_proces = filter_butter(daEMG_proces, fr=fr, fc=fclow, kind="low")

    # daEMG_proces.attrs['freq'] = daEMG.attrs['freq']
    # daEMG_proces.attrs['units'] = daEMG.attrs['units']
    # daEMG_proces.time.attrs['units'] = daEMG.time.attrs['units']
    daEMG_proces.attrs = daEMG.attrs
    daEMG_proces.name = "EMG"

    return daEMG_proces


# TODO: GENERALIZE THE NORMALIZING FUNCTION
def NormalizaBiela360_xr(
    daData, base_norm_horiz="time", graphs=False
):  # recibe da de daTodos. Versión con numpy
    if base_norm_horiz == "time":
        eje_x = daData.time
    elif base_norm_horiz in ["biela", "crank"]:
        try:
            eje_x = daData.sel(n_var="AngBiela", axis="y")
        except:
            eje_x = daData.sel(n_var="AngBiela")
    else:
        print("Normalizing base unknown")
        return

    def _normaliza_t_aux(
        data, x, base_norm_horiz
    ):  # Función auxiliar para normalizar con xarray
        # return tnorm(data, k=1, step=-361, show=False)[0]
        if np.isnan(data).all():
            data = np.full(361, np.nan)
        else:  # elimina los nan del final y se ajusta
            data = data[~np.isnan(data)]
            x = x[: len(data)]
            if base_norm_horiz in ["biela", "crank"]:
                x = np.unwrap(x)
                x = x - x[0]
            xi = np.linspace(0, x[-1], 361)
            data = np.interp(xi, x, data)  # tnorm(data, k=1, step=-361, show=False)[0]
        return data

    daNorm = xr.apply_ufunc(
        _normaliza_t_aux,
        daData,
        eje_x,
        base_norm_horiz,
        input_core_dims=[["time"], ["time"], []],
        output_core_dims=[["AngBielaInRepe"]],
        exclude_dims=set(("AngBielaInRepe",)),
        vectorize=True,
    ).assign_coords(
        dict(
            AngBielaInRepe=np.arange(
                361
            ),  # hay que meter esto a mano. Coords en grados
            AngBielaInRepe_rad=(
                "AngBielaInRepe",
                np.deg2rad(np.arange(361)),
            ),  # Coords en radianes
        )
    )
    daNorm.AngBielaInRepe.attrs["units"] = "deg"
    daNorm.AngBielaInRepe_rad.attrs["units"] = "rad"
    daNorm.name = daData.name
    daNorm.attrs["units"] = daData.attrs["units"]

    return daNorm


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
    from scipy import signal

    if ID is not None:
        print(ID)

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
    c = signal.correlate(
        np.gradient(np.gradient(dat1)), np.gradient(np.gradient(dat2)), "full"
    )
    c = c[int(len(c) / 2) :]
    ccorr[: len(c)] = c
    desfase = int(np.ceil(np.argmax(ccorr) - (len(ccorr)) / 2) + 1)

    return ccorr  # [int(len(ccorr) / 2) :]


def _nanargmax(data: np.array, ID) -> float:
    if np.count_nonzero(~np.isnan(data)) == 0:
        return np.array(np.nan)
    # if np.isnan(data).all():
    #     print('Error')
    #     return np.nan

    return float(np.nanargmax(data))


def nanargmax_xr(da: xr.DataArray, dim: str = None) -> xr.DataArray:
    """
    data = da[0,0]
    """
    if dim is None:
        raise ValueError("dim must be specified")

    daResult = xr.apply_ufunc(
        _nanargmax,
        # daiSen.sel(articulacion='rodilla', lado='L', eje='x').dropna(dim='time'),
        da,
        da["ID"],
        input_core_dims=[
            [dim],
            [],
        ],
        vectorize=True,
        dask="parallelized",
        keep_attrs=False,
        # kwargs=args_func_cortes,
    )
    return daResult


def _nanargmin(data: np.array, ID) -> float:
    if np.count_nonzero(~np.isnan(data)) == 0:
        return np.array(np.nan)
    # if np.isnan(data).all():
    #     print('Error')
    #     return np.nan

    return float(np.nanargmin(data))


def nanargmin_xr(da: xr.DataArray, dim: str = None) -> xr.DataArray:
    """
    data = da[0,0]
    """
    if dim is None:
        raise ValueError("dim must be specified")

    daResult = xr.apply_ufunc(
        _nanargmin,
        # daiSen.sel(articulacion='rodilla', lado='L', eje='x').dropna(dim='time'),
        da,
        da["ID"],
        input_core_dims=[
            [dim],
            [],
        ],
        vectorize=True,
        dask="parallelized",
        keep_attrs=False,
        # kwargs=args_func_cortes,
    )
    return daResult


def cross_correl_xr(
    da1: xr.DataArray, da2: xr.DataArray, func=_cross_correl_simple_aux
) -> xr.DataArray:
    """
    Apply the specified cross-correlation function to dataarra
    """
    daCrosscorr = xr.apply_ufunc(
        func,
        da2,
        da1,
        da1["ID"],
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


def round_to_nearest_even_2_decimal(number: float) -> float:
    """Rounds a float to the nearest even number with 2 decimal places"""
    rounded = np.round(number * 50) / 50  # Round to the nearest 0.02

    return rounded


# =============================================================================
# %% TESTS
# =============================================================================

if __name__ == "__main__":
    # =============================================================================
    # %%---- Create a sample
    # =============================================================================

    import numpy as np

    # import pandas as pd
    import xarray as xr

    # from scipy.signal import butter, filtfilt
    from pathlib import Path

    import matplotlib.pyplot as plt

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
    # %% TEST INTEGRATE
    # =============================================================================
    daWindow = (
        xr.full_like(var_a.isel(time=0).drop_vars("time"), np.nan).expand_dims(
            {"event": ["ini", "fin"]}, axis=-1
        )
    ).copy()
    daWindow.loc[dict(event=["ini", "fin"])] = np.array([100, -300])
    daWindow.loc[dict(event=["ini", "fin"], ID="00")] = np.array([0, len(var_a.time)])

    # Discrete
    integrate_window(var_a, result_return="discrete")
    integrate_window(var_a, daWindow, result_return="discrete")

    # Continuous
    integ = integrate_window(var_a, daWindow, result_return="continuous")
    integ.sel(n_var="a").plot.line(x="time", col="ID", col_wrap=4)

    integ = integrate_window(var_a, daWindow, daOffset=60, result_return="continuous")
    integ.sel(n_var="a").plot.line(x="time", col="ID", col_wrap=4)

    integ = integrate_window(
        var_a, daWindow, daOffset=var_a.mean("time"), result_return="continuous"
    )
    integ.sel(n_var="a").plot.line(x="time", col="ID", col_wrap=4)

    # =============================================================================
    # %% TEST RMS
    # =============================================================================
    daWindow = (
        xr.full_like(var_a.isel(time=0).drop_vars("time"), np.nan).expand_dims(
            {"event": ["ini", "fin"]}, axis=-1
        )
    ).copy()
    daWindow.loc[dict(event=["ini", "fin"])] = np.array([100, 300])
    daWindow.loc[dict(event=["ini", "fin"], ID="00")] = np.array([0, len(var_a.time)])

    RMS(var_a, daWindow)

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


def get_list_dir(n_path):
    """
    Recursively gets a list of all subdirectories in a given directory.
    Returns a list of Path objects.
    """
    from os import scandir
    from pathlib import Path

    subfolders = [f.path for f in scandir(n_path) if f.is_dir()]
    for n_path in list(subfolders):
        subfolders.extend(get_list_dir(n_path))
    subfolders = [Path(f) for f in subfolders]
    return subfolders
