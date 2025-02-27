# =============================================================================
# %% INICIA
# =============================================================================

__filename__ = "general_processing_functions"
__version__ = "0.4.1"
__company__ = "CIDUMH"
__date__ = "14/01/2025"
__author__ = "Jose L. L. Elvira"

"""
Modificaciones:
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


# =============================================================================
# %% Carga librerías
# =============================================================================

from typing import Optional, Union

import numpy as np
import xarray as xr

import scipy.integrate as integrate

import time


def create_time_series_xr(
    rnd_seed=None,
    num_subj=10,
    Fs=100.0,
    IDini=0,
    rango_offset=[-2.0, -0.5],
    rango_amp=[1.0, 2.2],
    rango_frec=[1.8, 2.4],
    rango_af=[0.0, 1.0],
    rango_duracion=[5.0, 5.1],
    amplific_ruido=[0.4, 0.7],
    fc_ruido=[7.0, 12.0],
) -> xr.DataArray:
    """
    Create a dummy data sample based on sine waves with noise
    """

    if rnd_seed is not None:
        np.random.seed(
            rnd_seed
        )  # para mantener la consistencia al crear los datos aleatorios
    subjects = []
    for subj in range(num_subj):
        # print(subj)
        a = np.random.uniform(rango_amp[0], rango_amp[1])
        of = np.random.uniform(rango_offset[0], rango_offset[1])
        f = np.random.uniform(rango_frec[0], rango_frec[1])
        af = np.deg2rad(
            np.random.uniform(rango_af[0], rango_af[1])
        )  # lo pasa a radianes
        err = a * np.random.uniform(amplific_ruido[0], amplific_ruido[1])
        fc_err = np.random.uniform(fc_ruido[0], fc_ruido[1])
        duracion = np.random.uniform(rango_duracion[0], rango_duracion[1])

        Ts = 1.0 / Fs  # intervalo de tiempo entre datos en segundos
        t = np.arange(0, duracion, Ts)

        senal = np.array(of + a * np.sin(2 * np.pi * f * t + af))

        # Crea un ruido aleatorio controlado
        pasadas = 2.0  # nº de pasadas del filtro adelante y atrás
        orden = 2
        Cf = (2 ** (1 / pasadas) - 1) ** (
            1 / (2 * orden)
        )  # correction factor. Para 2nd order = 0.802
        Wn = 2 * fc_err / Fs / Cf
        b1, a1 = butter(orden, Wn, btype="low")
        ruido = filtfilt(b1, a1, np.random.uniform(a - err, a + err, len(t)))

        #################################
        subjects.append(senal + ruido)
        # subjects.append(np.expand_dims(senal + ruido, axis=0))
        # sujeto.append(pd.DataFrame(senal + ruido, columns=['value']).assign(**{'ID':'{0:02d}'.format(subj+IDini), 'time':np.arange(0, len(senal)/Fs, 1/Fs)}))

    # Pad data to last the same
    import itertools

    data = np.array(list(itertools.zip_longest(*subjects, fillvalue=np.nan)))

    data = xr.DataArray(
        data=data,
        coords={
            "time": np.arange(data.shape[0]) / Fs,
            "ID": [
                f"{i:0>2}" for i in range(num_subj)
            ],  # rellena ceros a la izq. f'{i:0>2}' vale para int y str, f'{i:02}' vale solo para int
        },
    )
    return data


def integrate_window(daData, daWindow=None, daOffset=None, result_return="continuous"):
    """
    result_return: "continuous" or "discrete"
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


def detrend_dim(da, dim, deg=1):
    """
    Detrend the signal along a single dimension
    """

    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit


def RMS(daData, daWindow=None):
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


def calculate_distance(point1, point2):
    """
    Calcula la distancia entre dos puntos.
    Requiere dimensión con coordenadas x, y, z con nombre 'axis'
    """
    return np.sqrt(((point1 - point2) ** 2).sum("axis"))


# Función para detectar onsets
"""
Ref: Solnik, S., Rider, P., Steinweg, K., Devita, P., & Hortobágyi, T. (2010). Teager-Kaiser energy operator signal conditioning improves EMG onset detection. European Journal of Applied Physiology, 110(3), 489–498. https://doi.org/10.1007/s00421-010-1521-8

Función sacada de Duarte (https://nbviewer.org/github/BMClab/BMC/blob/master/notebooks/Electromyography.ipynb)
The Teager-Kaiser Energy operator to improve onset detection
The Teager-Kaiser Energy (TKE) operator has been proposed to increase the accuracy of the onset detection by improving the SNR of the EMG signal (Li et al., 2007).
"""


def tkeo(x):
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


def procesaEMG(
    daEMG: xr.DataArray, fr=None, fc_band=[10, 400], fclow=8, btkeo=False
) -> xr.DataArray:
    from biomdp.filtrar_Butter import filtrar_Butter, filtrar_Butter_bandpass

    if fr == None:
        fr = daEMG.freq
    # Filtro band-pass
    daEMG_proces = filtrar_Butter_bandpass(
        daEMG, fr=fr, fclow=fc_band[0], fchigh=fc_band[1]
    )
    # Centra, ¿es necesario?
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
    daEMG_proces = filtrar_Butter(daEMG_proces, fr=fr, fc=fclow, kind="low")

    # daEMG_proces.attrs['freq'] = daEMG.attrs['freq']
    # daEMG_proces.attrs['units'] = daEMG.attrs['units']
    # daEMG_proces.time.attrs['units'] = daEMG.time.attrs['units']
    daEMG_proces.attrs = daEMG.attrs
    daEMG_proces.name = "EMG"

    return daEMG_proces


# TODO: GENERALIZAR LA FUNCIÓN DE NORMALIZAR
def NormalizaBiela360_xr(
    daData, base_norm_horiz="time", graficas=False
):  # recibe da de daTodos. Versión con numpy
    if base_norm_horiz == "time":
        eje_x = daData.time
    elif base_norm_horiz == "biela":
        try:
            eje_x = daData.sel(n_var="AngBiela", axis="y")
        except:
            eje_x = daData.sel(n_var="AngBiela")
    else:
        print("Base de normalización no reconocida")
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
            if base_norm_horiz == "biela":
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


import polars as pl


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
    if ID is not None:
        print(ID)

    # pre-crea el array donde guardará las correlaciones de cada desfase
    corr = np.full(max(len(datos1), len(datos2)), np.nan)

    if np.isnan(datos1).all() or np.isnan(datos2).all():
        print(f"{ID} vacío")
        return corr

    try:
        # quita nans del final para función stats.pearson
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

            # Versión scipy más lenta
            # corr[i] = stats.pearsonr(dat1[i : i + dat2.size], dat2).statistic
        # plt.plot(corr)

    except Exception as err:
        print("Error de cálculo, posiblemente vacío", err)

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

    # Quita Nans
    dat1 = datos1[~np.isnan(datos1)]
    dat2 = datos2[~np.isnan(datos2)]

    # Normaliza
    dat1 = (dat1 - np.mean(dat1)) / np.std(dat1)
    dat2 = (dat2 - np.mean(dat2)) / np.std(dat2)

    # Rellena con ceros
    if len(dat1) != len(dat2):
        if len(dat1) < len(dat2):
            dat1 = np.append(dat1, np.zeros(len(dat2) - len(dat1)))
        else:
            dat2 = np.append(dat2, np.zeros(len(dat1) - len(dat2)))

    # Calcula la correlación cruzada
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
        _nanargmax,  # nombre de la función
        # daiSen.sel(articulacion='rodilla', lado='L', eje='x').dropna(dim='time'),
        da,
        da["ID"],
        input_core_dims=[
            [dim],
            [],
        ],  # lista con una entrada por cada argumento
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
        _nanargmin,  # nombre de la función
        # daiSen.sel(articulacion='rodilla', lado='L', eje='x').dropna(dim='time'),
        da,
        da["ID"],
        input_core_dims=[
            [dim],
            [],
        ],  # lista con una entrada por cada argumento
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
    Aplica la función de cross correlation que se especifique a dataarray
    """
    daCrosscorr = xr.apply_ufunc(
        func,  # nombre de la función
        # daiSen.sel(articulacion='rodilla', lado='L', eje='x').dropna(dim='time'),
        da2,
        da1,
        da1["ID"],  # .sel(partID=da1['partID'], tiempo='pre'),
        # da1.time.size - daInstrumento2.time.size,
        # da1['partID'],
        # da1['tiempo'],
        input_core_dims=[
            ["time"],
            ["time"],
            [],
            # [],
            # [],
        ],  # lista con una entrada por cada argumento
        output_core_dims=[["lag"]],  # datos que devuelve
        exclude_dims=set(
            (
                "lag",
                "time",
            )
        ),  # dimensiones que se permite que cambien (tiene que ser un set)
        # dataset_fill_value=np.nan,
        vectorize=True,
        dask="parallelized",
        keep_attrs=False,
        # kwargs=args_func_cortes,
    ).dropna(dim="lag", how="all")
    daCrosscorr = daCrosscorr.assign_coords(lag=range(len(daCrosscorr.lag)))
    return daCrosscorr


def round_to_nearest_even_2_decimal(number):
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
    from scipy.signal import butter, filtfilt
    from pathlib import Path

    import matplotlib.pyplot as plt
    import seaborn as sns

    rnd_seed = np.random.seed(
        12340
    )  # fija la aleatoriedad para asegurarse la reproducibilidad
    n = 10
    duracion = 15
    freq = 200.0
    Pre_a = (
        create_time_series_xr(
            rnd_seed=rnd_seed,
            num_subj=n,
            Fs=freq,
            IDini=0,
            rango_offset=[25, 29],
            rango_amp=[40, 45],
            rango_frec=[1.48, 1.52],
            rango_af=[0, 30],
            amplific_ruido=[0.4, 0.7],
            fc_ruido=[3.0, 3.5],
            rango_duracion=[duracion, duracion],
        )
        .expand_dims({"n_var": ["a"], "momento": ["pre"]})
        .transpose("ID", "momento", "n_var", "time")
    )
    Post_a = (
        create_time_series_xr(
            rnd_seed=rnd_seed,
            num_subj=n,
            Fs=freq,
            IDini=0,
            rango_offset=[22, 26],
            rango_amp=[36, 40],
            rango_frec=[1.48, 1.52],
            rango_af=[0, 30],
            amplific_ruido=[0.4, 0.7],
            fc_ruido=[3.0, 3.5],
            rango_duracion=[duracion, duracion],
        )
        .expand_dims({"n_var": ["a"], "momento": ["post"]})
        .transpose("ID", "momento", "n_var", "time")
    )
    var_a = xr.concat([Pre_a, Post_a], dim="momento")
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
    daInstrumento1 = var_a.sel(n_var="a", momento="pre").drop_vars("momento")
    daInstrumento2 = daInstrumento1.isel(
        time=slice(int(4 * var_a.freq), int(8 * var_a.freq))
    )

    daCrosscorr = xr.apply_ufunc(
        _cross_correl_simple_aux,
        # daiSen.sel(articulacion='rodilla', lado='L', eje='x').dropna(dim='time'),
        daInstrumento1,
        daInstrumento2,  # .sel(partID=da1['partID'], tiempo='pre'),
        # daInstrumento1.time.size - daInstrumento2.time.size,
        # daInstrumento1['partID'],
        # daInstrumento1['tiempo'],
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
        # daiSen.sel(articulacion='rodilla', lado='L', eje='x').dropna(dim='time'),
        daInstrumento1,
        daInstrumento2,  # .sel(partID=da1['partID'], tiempo='pre'),
        # daInstrumento1.time.size - daInstrumento2.time.size,
        # daInstrumento1["ID"],
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
