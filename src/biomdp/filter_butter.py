# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Aplica filtro paso Butterworth. Se puede pasar array 1D, pandas DataFrame 2D
o dataarray de xarray.
Función de paso bajo o alto y función de bandpass.
Applies Butterworth pass filter. You can pass 1D array, pandas 2D DataFrame
or xarray dataarray.
Low or high pass function and bandpass function.
"""


# =============================================================================
# %% LOAD MODULES
# =============================================================================

from typing import Optional, Union, Any
import numpy as np
import pandas as pd
import xarray as xr
import scipy.signal


__author__ = "Jose L. L. Elvira"
__version__ = "v.1.6.2"
__date__ = "17/02/2025"


"""
Modificaciones:
    05/03/2025, v1.6.2
        - Name translations.
    
    17/02/2025, v1.6.1
        - Si se pasa un dataarray con atributo freq y sin parámetro fr,
          lo utiliza.
        - Incluidos tipos en parámetros funciones
    
    23/02/2024, v1.6.0
        - Introducida función plot con xarray.
    
    22/03/2023, v1.5.4
        - En filtro band pass corregido de lfilt a filtfilt.
        
    08/01/2023, v1.5.3
        - Ahora cuando se pasa un dataarray conserva el tipo de datos original.
    
    03/08/2022, v1.5.2
        - Ahora cuando se pasa un dataarray conserva los atributos.
        
    24/02/2022, v1.5.1
        - Cambiado el color del gráfico del original para que no se confundan
        
    08/05/2021, v1.5.0
        - Arreglado con xarray. Si tiene nan los rellena interpolando y después los elimina
        - Si no se pide el RMS o hacer la gráfica, no lo calcula.
        - Cambiados nombres de argumentos a más pythonics.
"""


# =============================================================================
# %% Functions
# =============================================================================
def filter_butter(
    dat_orig: Union[np.ndarray, pd.DataFrame, xr.DataArray],
    fr: Optional[Union[float, int]] = None,
    fc: Optional[Union[float, int]] = None,
    order: Optional[Union[float, int]] = 2.0,
    kind: Optional[str] = "low",
    returnRMS: Optional[bool] = False,
    show: Optional[bool] = False,
    ax: Optional[Any] = None,
) -> Union[np.ndarray, pd.DataFrame, xr.DataArray]:
    """
    Applies a Butterworth pass filter to data.

    Parameters
    ----------
    dat_orig : array_like, numpy, pandas or dataarray
        Original data to filter.
    fr : float, optional
        Sample frequency. Defaults to None.
    fc : float, optional
        Cut-off frequency. Defaults to None.
    order : float, optional
        Filter order. Defaults to 2.0.
    kind : str, optional
        Filter type. 'low' or 'high'. Defaults to 'low'.
    returnRMS : bool, optional
        Whether to return the root mean square (RMS) of the difference between
        the original and filtered data. Defaults to False.
    show : bool, optional
        Whether to display a plot of the original and filtered data. Defaults to False.
    ax : matplotlib axes, optional
        Axes on which to plot the data. Defaults to None.

    Returns
    -------
    filtData : array_like
        Filtered data.
    RMS : float, optional
        Root mean square of the difference between the original and filtered data.

    Notes
    -----
    Describe 2nd order, 2-pass filter as "double 2nd order Butterworth filter"
    (van den Bogert) http://biomch-l.isbweb.org/threads/26625-Matlab-Code-for-EMG-processing-(negative-deflection-after-normalization!!!)?p=32073#post32073


    Examples
    --------
    >>> import numpy as np
    >>> from biomdp.filter_butter import filter_butter
    >>> y = np.cumsum(np.random.randn(1000))
    >>> fy = filter_butter(y, fr=1000, fc=10, order=2, show=True)
    >>>
    >>> dfWalks = pd.DataFrame((np.random.random([100, 4])-0.5).cumsum(axis=0), columns=['A','B','C','D'])
    >>> dfWalks_Filt, RMS = filter_butter(dfWalks, 1000, 50, 2, show=True, returnRMS=True)

    """
    RMS = []

    # order = 2 #order 2 so that when doing the double pass it is 4th order
    passes = 2.0  # number of passes, forward and backward

    if fr is None:
        if isinstance(dat_orig, xr.DataArray) and "freq" in dat_orig.attrs:
            fr = dat_orig.freq
        else:
            if not dat_orig.isnull().all():
                fr = (
                    np.round(
                        1 / (dat_orig["time"][1] - dat_orig["time"][0]),
                        1,
                    )
                ).data
            else:
                raise RuntimeError("The sampling frequency (fr) must be specified.")

    # fc = 15
    Cf = (2 ** (1 / passes) - 1) ** (
        1 / (2 * order)
    )  # correction factor. Para 2nd order = 0.802 (Winter, 2009, p69)
    Wn = 2 * fc / fr / Cf

    b, a = scipy.signal.butter(order, Wn, btype=kind)

    # Pandas dataframe
    if isinstance(dat_orig, pd.DataFrame):
        dat_filt = pd.DataFrame()

        for i in range(dat_orig.shape[1]):
            dat_filt[dat_orig.columns[i]] = scipy.signal.filtfilt(
                b, a, dat_orig.iloc[:, i]
            )
        dat_filt.index = (
            dat_orig.index
        )  # esto es necesario por si se pasa un slice del dataframe

        if returnRMS or show == True:
            RMS = pd.DataFrame()
            for i in range(dat_orig.shape[1]):
                RMS.at[0, dat_orig.columns[i]] = np.linalg.norm(
                    dat_filt.iloc[:, i].values - dat_orig.iloc[:, i].values
                ) / np.sqrt(len(dat_orig.iloc[:, i]))

    # Pandas series
    elif isinstance(dat_orig, pd.Series):
        dat_filt = pd.Series(
            scipy.signal.filtfilt(b, a, dat_orig),
            index=dat_orig.index,
            name=dat_orig.name,
        )

        if returnRMS or show == True:
            RMS = np.linalg.norm(dat_filt - dat_orig) / np.sqrt(len(dat_orig))

    # Xarray dataarray
    elif isinstance(dat_orig, xr.DataArray):
        # dat_filt = xr.apply_ufunc(scipy.signal.filtfilt, b, a, dat_orig.dropna(dim='time')) #se asume que hay una dimensión tiempo
        dat_filt = xr.apply_ufunc(
            scipy.signal.filtfilt,
            b,
            a,
            dat_orig.interpolate_na(
                dim="time", method="linear", fill_value="extrapolate"
            ),
        )  # pad with nan the interpolated data
        # Retrieves the original data number by filling in with nan the endings as the original one
        dat_filt = dat_filt.where(xr.where(np.isnan(dat_orig), False, True), np.nan)
        dat_filt.attrs = dat_orig.attrs
        dat_filt = dat_filt.astype(dat_orig.dtype)

        if returnRMS or show == True:
            RMS = pd.DataFrame()
            for i in range(dat_orig.shape[0]):
                RMS.at[0, i] = np.linalg.norm(
                    dat_filt[i, :] - dat_orig[i, :]
                ) / np.sqrt(len(dat_orig[i, :]))
                # xr.apply_ufunc(np.linalg.norm, dat_filt[0,:], dat_orig[0,:])

    # Other data types
    else:
        dat_filt = scipy.signal.filtfilt(b, a, dat_orig)

        if returnRMS or show == True:
            RMS = np.linalg.norm(dat_filt - dat_orig) / np.sqrt(len(dat_orig))

    if show:
        _plot(dat_orig, dat_filt, RMS, fc, ax)

    if returnRMS:
        return dat_filt, RMS
    else:
        return dat_filt


# =============================================================================


# =============================================================================
# Shows plot
# =============================================================================
def _plot(
    dat_orig: Union[np.ndarray, pd.DataFrame, xr.DataArray],
    dat_filt: Union[np.ndarray, pd.DataFrame, xr.DataArray],
    RMS,
    fc,
    ax,
):
    import matplotlib.pyplot as plt

    if isinstance(dat_orig, xr.DataArray):

        def _xr_plot(orig, filt):
            plt.plot(filt, "b-", lw=0.8, label="Filt")
            plt.plot(orig, "r:", lw=1, label="Original")
            plt.legend()
            plt.show()

        _ = xr.apply_ufunc(
            _xr_plot,
            dat_orig,
            dat_filt,
            input_core_dims=[["time"], ["time"]],
            vectorize=True,
        )

    else:
        bNecesarioCerrarFigura = False

        if ax is None:
            bNecesarioCerrarFigura = True
            fig, ax = plt.subplots(1, 1, figsize=(10, 4))

        # Pandas dataframe
        if isinstance(dat_orig, pd.DataFrame):
            import seaborn as sns

            cmap = sns.color_palette("bright", n_colors=dat_orig.shape[1])
            dat_filt.plot(color=cmap, legend=False, ax=ax)
            dat_orig.plot(color=cmap, alpha=0.6, linestyle=":", legend=False, ax=ax)
            labels = [
                dat_orig.columns[x] + ", RMSE=" + "{:.3f}".format(RMS.iloc[0, x])
                for x in range(dat_orig.shape[1])
            ]
            plt.legend(labels)

        # Non pandas dataframe
        else:
            ax.plot(dat_filt, "b-", label="Filt (RMSE={:.3f})".format(RMS))
            ax.plot(dat_orig, "r:", alpha=0.8, label="Original")
            plt.legend(loc="best")

        ax.set_xlabel("Num. datos")
        ax.set_ylabel("Variable")
        ax.set_title("Filtrado Butterworth {0:3g} Hz".format(fc))

        if bNecesarioCerrarFigura:
            plt.show()


def filter_butter_bandpass(
    dat_orig: Union[np.ndarray, pd.DataFrame, xr.DataArray],
    fr: Optional[Union[float, int]] = None,
    fclow: Optional[Union[float, int]] = None,
    fchigh: Optional[Union[float, int]] = None,
    order: Optional[Union[float, int]] = 2.0,
    show: Optional[bool] = False,
    ax: Optional[Any] = None,
) -> Union[np.ndarray, pd.DataFrame, xr.DataArray]:
    """
    Applies a Butterworth bandpass filter to data.

    Parameters
    ----------
    dat_orig : array_like, numpy, pandas or dataarray
        Original data to filter.
    fr : float, optional
        Sample frequency. Defaults to None.
    fclow, fchigh : float, optional
        Cut-off frequencies. Defaults to None.
    order : float, optional
        Filter order. Defaults to 2.0.
    show : bool, optional
        Whether to display a plot of the original and filtered data. Defaults to False.
    ax : matplotlib axes, optional
        Axes on which to plot the data. Defaults to None.

    Returns
    -------
    filtData : array_like
        Filtered data.

    Notes
    -----
    Información about bandpass Butterworth filter in:
    https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html

    Examples
    --------
    >>> import numpy as np
    >>> from biomdp.filter_butter import filter_butter
    >>> y = np.cumsum(np.random.randn(1000))
    >>> fy = filter_butter(y, fr=1000, fc=10, order=2, show=True)
    >>>
    >>> dfWalks = pd.DataFrame((np.random.random([100, 4])-0.5).cumsum(axis=0), columns=['A','B','C','D'])
    >>> dfWalks_filt, RMS = filter_butter(dfWalks, 1000, 50, 2, show=True, returnRMS=True)

    """

    nyq = 0.5 * fr
    low = fclow / nyq
    high = fchigh / nyq
    b, a = scipy.signal.butter(order, [low, high], btype="band")

    # Pandas dataframe
    if isinstance(dat_orig, pd.DataFrame):
        dat_filt = pd.DataFrame()
        RMS = pd.DataFrame()
        for i in range(dat_orig.shape[1]):
            dat_filt[dat_orig.columns[i]] = scipy.signal.filtfilt(
                b, a, dat_orig.iloc[:, i]
            )
        dat_filt.index = (
            dat_orig.index
        )  # this is necessary in case a slice of the dataframe is passed

    # Pandas series
    elif isinstance(dat_orig, pd.Series):
        dat_filt = pd.Series(
            scipy.signal.filtfilt(b, a, dat_orig),
            index=dat_orig.index,
            name=dat_orig.name,
        )

    # Xarray dataarray
    elif isinstance(dat_orig, xr.DataArray):
        # dat_filt = xr.apply_ufunc(scipy.signal.filtfilt, b, a, dat_orig.dropna(dim='time')) #se asume que hay una dimensión tiempo
        dat_filt = xr.apply_ufunc(
            scipy.signal.filtfilt,
            b,
            a,
            dat_orig.interpolate_na(
                dim="time", method="linear", fill_value="extrapolate"
            ),
        )  # rellena los nan con datos interpolados
        dat_filt = dat_filt.where(
            xr.where(np.isnan(dat_orig), False, True), np.nan
        )  # recupera el nº de datos original rellenando con nan los finales como el original

    else:  # si los datos no son pandas dataframe
        dat_filt = scipy.signal.filtfilt(b, a, dat_orig)

    if show:
        _plot(dat_orig, dat_filt, RMS, fclow, ax)

    return dat_filt


# =============================================================================


# =============================================================================
# %% TESTS
# =============================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    np.random.seed(2)
    y = np.cumsum(np.random.randn(1000))
    fy, rms = filter_butter(y, 1000, 10, 2, show=True, returnRMS=True)
    fy2, rms2 = filter_butter(y[100:300], 1000, 10, 2, show=True, returnRMS=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.suptitle("Título grande", y=1.03)

    fy = filter_butter(y, 1000, 50, 2, show=True, ax=ax)
    ax.set_title("Superpone el Título pequeno", y=1.0)
    plt.show()

    # Con dataframe de varias columnas
    num = 1000
    colNames = ["A", "B", "C", "D"]
    dfWalks = pd.DataFrame(
        (np.random.random([num, 4]) - 0.5).cumsum(axis=0), columns=colNames
    )

    dfWalks_filt = filter_butter(dfWalks, 1000, 5, 2, show=True)
    dfWalks_filt, RMS = filter_butter(dfWalks, 1000, 50, 2, show=True, returnRMS=True)

    # con pd series
    dfWalks_filt, RMS = filter_butter(
        dfWalks.iloc[:, 0], 1000, 5, 2, show=True, returnRMS=True
    )
    dfWalks_filt, RMS = filter_butter(
        dfWalks["A"], 1000, 50, 2, show=True, returnRMS=True
    )

    # %% Noisy wave
    t = np.arange(0, 2, 1 / 1000)
    # offset vertical
    of = [0, 0, 0, 0]
    # ampitudes
    a = [3, 0.5, 5, 0.3]
    # frecuencias
    f = [1, 60, 3, 40]
    # phase angle
    pa = [0, 0, 0, 0]
    waves = pd.DataFrame(
        np.array(
            [of[i] + a[i] * np.sin(2 * np.pi * f[i] * t + pa[i]) for i in range(len(a))]
        ).T
    )

    Wave = pd.DataFrame({"Wave1": waves[0] + waves[1], "Wave2": waves[2] + waves[3]})

    dfWave_filt = filter_butter(Wave, 1000, 10, 2, show=True)

    # With index change
    dfWave_filtCacho = filter_butter(Wave[100:300], 1000, 20, 2, show=True)

    dfWave_filtCacho, RMS = filter_butter(
        Wave.iloc[100:300, 0], 1000, 20, 2, show=True, returnRMS=True
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    fy = filter_butter(Wave.iloc[400:600, 0], 1000, 50, 2, show=True, ax=ax)
    ax.set_title("(Sup little title)", y=1.0)
    plt.suptitle("Big title", y=1.03)
    plt.show()

    # %% Bandpass tests
    # Filter a noisy signal.
    fs = 5000.0
    lowcut = 500.0
    highcut = 1250.0

    T = 0.05
    nsamples = T * fs
    t = np.linspace(0, T, int(nsamples), endpoint=False)
    a = 0.02
    f0 = 600.0  # principal frequency to extract
    x = 0.1 * np.sin(2 * np.pi * 1.2 * np.sqrt(t))
    x += 0.01 * np.cos(2 * np.pi * 312 * t + 0.1)
    x += a * np.cos(2 * np.pi * f0 * t + 0.11)
    x += 0.03 * np.cos(2 * np.pi * 2000 * t)

    xFiltBand = filter_butter_bandpass(
        x, fs, lowcut, highcut, order=6, show=False, ax=None
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, x, "b--")
    ax.plot(t, xFiltBand, "r")
    plt.hlines([-a, a], 0, T, "r", linestyles="--")
    plt.title("Bandpass Filter")
    plt.show()

    ###############################
    # %%xarray
    t = np.arange(0, 2, 1 / 1000)
    # vertical offset
    of = [0, 0, 0, 0]
    # ampitudes
    a = [3, 0.5, 5, 0.3]
    # frequencies
    f = [1, 60, 3, 40]
    # phase angle
    pa = [0, 0, 0, 0]
    waves = pd.DataFrame(
        np.array(
            [of[i] + a[i] * np.sin(2 * np.pi * f[i] * t + pa[i]) for i in range(len(a))]
        ).T
    )

    Wave = pd.DataFrame({"Wave1": waves[0] + waves[1], "Wave2": waves[2] + waves[3]})

    da = xr.DataArray(
        data=np.array(Wave).T,
        dims=["channel", "time"],
        coords={
            "channel": Wave.columns,
            "time": np.arange(0, len(Wave) / 1000, 1 / 1000),
        },
    )
    o = da.isel(channel=-1)
    da.plot.line(x="time")  # unfiltered
    da.isel(channel=1).plot()
    plt.show()

    np.linalg.norm(da.isel(channel=1) - da.isel(channel=0)) / np.sqrt(
        len(da.isel(channel=0))
    )

    oFilt, RMSEda = filter_butter(
        da, fr=1000, fc=10, order=2, returnRMS=True, show=False
    )
    da.plot.line(x="time")  # unfiltered
    oFilt.plot.line(x="time")  # filtered
    plt.show()

    # Compared with pandas is the same
    dfWave_filt, RMSEdf = filter_butter(
        Wave, fr=1000, fc=10, order=2, returnRMS=True, show=True
    )

    da.attrs["freq"] = 1000
    oFilt = filter_butter(da, fc=10, order=2, returnRMS=False, show=True)
