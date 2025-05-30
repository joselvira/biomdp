# %% -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 18:01:08 2021
Functions to perform slices on cyclic signals using an internal or external criterion.
Based on xarray.

@author: josel
"""


# =============================================================================
# %% LOAD LIBRARIES
# =============================================================================

__author__ = "Jose L. L. Elvira"
__version__ = "v.4.2.2"
__date__ = "11/03/2025"


"""
TODO: to make the cuts interpolating so that the phases start right at the criterion?

Updates:
    11/03/2025, v4.2.2
        - Adapted to biomdp with translations. 

    13/12/2024, v4.2.0
        - Ahora indica error en función cortes auxiliar.
    
    16/06/2024, v4.2.0
        - Introducido argumento add_to_ini y add_to_end en función slice_time_series (de
          momento sólo funciona en versión numpy).
        - Optimizada la función numpy, más rapida.
            
    23/04/2024, v4.1.0
        - Actualizado versión slice con Polars, puede que sea más rapido.
        - TODO: Probar versión detect_onset que use find_peaks con la señal derivada
    
    22/02/2024, v4.0.0
        - Cambio a funciones independientes para usar con el xarray accessor.
        - Se mantiene la versión con clase por retrocompatibilidad.
        - Versión inicial basada en slice_time_series_phases.py v3.1.2

    10/11/2023, v3.1.2
        - Ahora mantiene las unidades en el dataframe cortado.
    
    11/03/2023, v3.1.1
        - Solucionado error con función find_peaks_aux. Hace copia antes de buscar
        cortes.
        - Incluido parámetro show en función detect_onset_detecta_aux.
    
    13/02/2023, v3.1.0
        - Las funciones find_peaks_aux y detect_onset_detecta_aux admiten un
        argumento para buscar cortes a partir de la media + x veces SD.
    
    09/02/2023, v3.0.0
        - Metido todo dentro de la clase SliceTimeSeriesPhases.
        - Cambiada nomenclatura num_repes, max_repes, descarta_rep_ini y
        descarta_rep_fin a num_cuts, max_cuts, discard_cuts_end y
        discard_cuts_end.
    
    28/01/2023, v2.1.0
        - Función común corta_repes que distribuye según los datos sean Pandas
        o xarray.
        - Función en xarray que devuelve sólo los nº de índice de los events.
        - Cambiada terminología, de repe a corte (para no confundir con repe de
        repeticiones/series).

    26/03/2022, v2.0.1
        - Como variable de referencia (var_referencia) ahora se pasa un dict
        con varias dimensiones y sus coordenadas.
        - Incluida una versión en pruebas para tratar en bloques de dataarray.

    11/12/2021, v2.0.0
        - Incluida una versión con xarray, mucho más rápida.

    24/11/2021, v1.2.0
        - Incluida opción de que incluya al final de cada repetición el primer dato de la siguiente. De esta forma tienen más continuidad.

    08/11/2021, v1.1.1
        - A la función auxiliar detect_onset_detecta_aux se le puede pasar como argumento corte_ini=1 para que coja el final de la ventana encontrada. Por defecto coge el inicio.
        - Además a la misma función cuando se pide que corte con el final de la ventana, le suma 1 para que coja cuando ha superado el umbral.
        - También si el corte_ini=0 quita el primer corte y si es =1 quita el último, porque suelen quedar cortados.
    
    13/10/2021, v1.1.0
        - Incluidos argumentos para eliminar repeticiones iniciales o finales.
        - Falta poder elegir eliminar repeticiones intermedias
    
    30/09/2021, v1.0.0
        - Versión inicial
"""


from typing import Optional, Union, Any

import numpy as np
import xarray as xr

import itertools

import matplotlib.pyplot as plt


def detect_onset_detecta_aux(
    data,  #: Optional[np.array],
    event_ini: int = 0,
    xSD: str | dict | None = None,
    show: bool | None = False,
    **args_func_events,
) -> np.array:
    """
    Custom function to adapt from detect_onset from detecta
    (https://pypi.org/project/detecta/)

    Parameters
    ----------
    data : np.array
        Array of data where to detect onsets
    event_ini: int
        Keeps the first =0) or second (=1) value of each data pair
    xSD : str or dict
        If str, the threshold is defined by the mean + x times the standar
        deviation. If dict, the threshold is defined by the mean + x times
        the standar deviation for each dimension of the data (e.g. {'x': 1,
        'y': 2})
    show : bool
        If True, plots the detected onsets
    **args_func_events : dict
        Additional arguments to pass to the find_peaks function

    Returns
    -------
    events : np.array
        Array of indices of the detected onsets
    """

    # If event_ini=1 is passed as an argument, it takes the cut at the end of each window.
    try:
        from detecta import detect_onset
    except:
        raise ImportError(
            "This function needs Detecta to be installed (https://pypi.org/project/detecta/)"
        )

    # try: #detect_onset returns 2 indexes. If not specified, select the first
    #     event_ini=args_func_events['event_ini']
    #     args_func_events.pop('event_ini', None)
    # except:
    #     event_ini=0
    if xSD is not None:
        # the threshold is defined by the mean + x times the standard deviation
        if "threshold" in args_func_events:
            args_func_events.pop("threshold", None)
        args_func_events["threshold"] = (
            np.mean(data, where=~np.isnan(data))
            + np.std(data, where=~np.isnan(data)) * xSD
        )
        # print(args_func_events, np.mean(data, where=~np.isnan(data)), np.std(data, where=~np.isnan(data)), xSD)

    events = detect_onset(data, **args_func_events)

    if event_ini == 1:
        events = (
            events[:, event_ini] + 1
        )  # if the end of the window is chosen, 1 is added to start when the threshold has already been exceeded
        events = events[:-1]  # removes the last one because it is usually incomplete
    else:
        events = events[
            :, event_ini
        ]  # keeps the first or second value of each data pair
        events = events[1:]  # removes the last one because it is usually incomplete

    if show:
        SliceTimeSeriesPhases.show_events(
            data, events, threshold=args_func_events["threshold"]
        )

    return events


# =============================================================================
# Custom function to adapt from scipy.signal find_peaks
# =============================================================================
def find_peaks_aux(
    data,
    xSD: str | dict | None = None,
    show: bool | None = False,
    **args_func_events,
) -> np.array:
    """
    Custom function to detect onsets based on find peaks from scipy.signal

    Parameters
    ----------
    data : np.array
        Array of data where to detect onsets
    xSD : str or dict
        If str, the threshold is defined by the mean + x times the standar
        deviation. If dict, the threshold is defined by the mean + x times
        the standar deviation for each dimension of the data (e.g. {'x': 1,
        'y': 2})
    show : bool
        If True, plots the detected onsets
    **args_func_events : dict
        Additional arguments to pass to the find_peaks function

    Returns
    -------
    events : np.array
        Array of indices of the detected onsets
    """

    try:
        from scipy.signal import find_peaks
    except ImportError:
        raise Exception("This function needs scipy.signal to be installed")

    # The threshold is defined by the mean + x times the standar deviation
    if xSD is not None:
        if isinstance(xSD, list):
            args_func_events["height"] = [
                np.mean(data[~np.isnan(data)]) + xSD[0] * np.std(data[~np.isnan(data)]),
                np.mean(data[~np.isnan(data)]) + xSD[1] * np.std(data[~np.isnan(data)]),
            ]
        else:
            args_func_events["height"] = np.mean(data[~np.isnan(data)]) + xSD * np.std(
                data[~np.isnan(data)]
            )  # , where=~np.isnan(data)) + xSD * np.std(data, where=~np.isnan(data))

    data = data.copy()

    # Deal with nans
    data[np.isnan(data)] = -np.inf

    events, _ = find_peaks(data, **args_func_events)

    if show:
        SliceTimeSeriesPhases.show_events(
            data, events, threshold=args_func_events["height"]
        )

    return events  # keeps the first value of each data pair


def find_onset_aux(
    data,
    xSD: str | dict | None = None,
    show: bool | None = False,
    **args_func_events,
) -> xr.DataArray:
    """
    Custom function to detect onsets based on find peaks from scipy.signal
    UNDER CONSTRUCTION
    """

    try:
        from scipy.signal import find_peaks, detrend
        from scipy import integrate
    except:
        raise ImportError("This function needs scipy.signal to be installed")

    # -------------------------------------
    # ---- Detecta onset a partir de detect peaks e integral
    threshold = 80
    daSliced = detect_events(
        data=daTotal,
        func_events=SliceTimeSeriesPhases.detect_onset_detecta_aux,
        reference_var=dict(momento="pre", n_var="b"),
        discard_phases_ini=0,
        n_phases=None,
        discard_phases_end=0,
        **dict(threshold=threshold, show=False),
    )

    data = daTotal[0, 0, 0].values
    # do = detect_onset(data, **dict(threshold=threshold))

    integ = integrate.cumulative_trapezoid(data - threshold, daTotal.time, initial=0)
    # integDetr = integ
    integDetr = detrend(integ)
    # plt.plot(integDetr)
    # plt.plot(daIntegDetr[0, 0, 0])

    fp = find_peaks(integDetr, **dict(height=0))[0]
    # do[:7, 1] - fp[:7]

    from biomdp.general_processing_functions import integrate_window, detrend_dim

    # Integrate the signal
    daInteg = integrate_window(daTotal - threshold, daOffset=0)  # daTotal.isel(time=0))
    # daIntegDetr = daInteg

    # Eliminates the tendency
    daIntegDetr = detrend_dim(daInteg, "time")
    (daIntegDetr[2, 0, 0] * 5 + threshold).plot.line(x="time")
    daTotal[2, 0, 0].plot.line(x="time")

    # Finds cuts
    daSlicedIdx = detect_events(
        data=daIntegDetr,
        func_events=SliceTimeSeriesPhases.find_peaks_aux,
        # reference_var=dict(momento="pre", n_var="b"),
        discard_phases_ini=0,
        n_phases=None,
        discard_phases_end=0,
        **dict(height=-200, show=True),
    )
    daTotal[2, 0, 0].plot.line(x="time")
    plt.plot(
        daSlicedIdx[2, 0, 0] / daTotal.freq,
        [threshold] * len(daSlicedIdx[2, 0, 0]),
        "ro",
    )
    (daSlicedIdx[2, 0, 0] / daTotal.freq).plot(marker="o")

    daSliced[2, 0, 0]
    daSlicedIdx[2, 0, 0]

    # -------------------------------------
    if (
        xSD is not None
    ):  # the threshold is defined by the mean + x times the standar deviation
        if isinstance(xSD, list):
            args_func_events["height"] = [
                np.mean(data[~np.isnan(data)]) + xSD[0] * np.std(data[~np.isnan(data)]),
                np.mean(data[~np.isnan(data)]) + xSD[1] * np.std(data[~np.isnan(data)]),
            ]
        else:
            args_func_events["height"] = np.mean(data[~np.isnan(data)]) + xSD * np.std(
                data[~np.isnan(data)]
            )  # , where=~np.isnan(data)) + xSD * np.std(data, where=~np.isnan(data))

    data = data.copy()

    # Deal with nans
    data[np.isnan(data)] = -np.inf

    events, _ = find_peaks(data, **args_func_events)

    if show:
        SliceTimeSeriesPhases.show_events(
            data, events, threshold=args_func_events["height"]
        )

    return events  # keeps the first value of each data pair


def detect_events(
    data: xr.DataArray = xr.DataArray(),
    freq: float | None = None,
    n_dim_time: str = "time",
    reference_var: str | dict | None = None,
    discard_phases_ini: int = 0,
    n_phases: int | None = None,
    discard_phases_end: int = 0,
    # include_first_next_last: Optional[bool] = False,
    max_phases: int = 100,
    func_events: Any | None = detect_onset_detecta_aux,
    **kwargs_func_events: Optional[dict],
) -> xr.DataArray:
    """
    Detects events in a DataArray and returns an array with the indexes of the events.

    Parameters
    ----------
    data : xr.DataArray
        The data in which to detect events.
    freq : float, optional
        Frequency of the sampling. If not specified, it is inferred from the DataArray.
    n_dim_time : str, optional
        The name of the dimension of time. By default it is "time".
    reference_var : str or dict, optional
        If specified, the events are detected only in this variable. If it is a dictionary,
        it should have the keys "momento" and "n_var". "momento" can be "pre" or "post"
        and "n_var" is the variable name.
    discard_phases_ini : int, optional
        How many initial phases to discard. By default it is 0.
    n_phases : int, optional
        How many phases to keep. If not specified, all are kept.
    discard_phases_end : int, optional
        How many final phases to discard. By default it is 0.
    max_phases : int, optional
        Maximum number of phases to detect. By default it is 100.
    func_events : callable, optional
        The function to use to detect the events. It must receive a 1D array as input and
        return an array of indexes of the events.
    **kwargs_func_events : dict, optional
        Additional arguments to pass to the function func_events.

    Returns
    -------
    xr.DataArray
        An array with the indexes of the events.
    """

    # TODO: ADJUST THE FUNCTION TO SUPPORT TRIAL-SPECIFIC THRESHOLDS
    # TODO: OPTIMIZE WHEN THERE IS reference_var THAT LOOKS FOR SLICES ONLY ON THAT VARIABLE
    if func_events == None:
        raise Exception("A function to detect the events must be specified")

    if freq is None:
        if "freq" in data.attrs:
            freq = data.attrs["freq"]
        else:
            if not data.isnull().all():
                freq = (
                    np.round(
                        1 / (data[n_dim_time][1] - data[n_dim_time][0]),
                        1,
                    )
                ).data

    def _detect_aux_idx(
        dat,
        data_reference_var=None,
        func_events=None,
        max_phases=100,
        discard_phases_ini=0,
        n_phases=None,
        discard_phases_end=0,
        **kwargs_func_events,
    ):
        events = np.full(max_phases, np.nan)

        if (
            np.count_nonzero(~np.isnan(dat)) == 0
            or np.count_nonzero(~np.isnan(data_reference_var)) == 0
        ):
            return events

        try:
            evts = func_events(data_reference_var, **kwargs_func_events)
        except Exception as exc:
            print(exc)
            return events

        # If necessary, adjust initial an final events
        evts = evts[discard_phases_ini:]

        if n_phases == None:
            evts = evts[: len(evts) - discard_phases_end]
        else:  # if a specific number of phases from the first event is required
            if len(evts) >= n_phases:
                evts = evts[: n_phases + 1]
            else:  # not enought number of events in the block, trunkated to the end
                pass
        events[: len(evts)] = evts
        return events

    """
    dat = data[0,0,0,0,0,0].values
    data_reference_var = data.sel(reference_var)[0,0,0,0].values
    """
    da = xr.apply_ufunc(
        _detect_aux_idx,
        data,
        data.sel(reference_var),
        func_events,
        max_phases,
        discard_phases_ini,
        n_phases,
        discard_phases_end,
        input_core_dims=[
            [n_dim_time],
            [n_dim_time],
            [],
            [],
            [],
            [],
            [],
        ],
        output_core_dims=[["n_event"]],
        exclude_dims=set(("n_event", n_dim_time)),
        dataset_fill_value=np.nan,
        vectorize=True,
        dask="parallelized",
        # keep_attrs=True,
        kwargs=kwargs_func_events,
    )
    da = (
        da.assign_coords(n_event=range(len(da.n_event)))
        .dropna(dim="n_event", how="all")
        .dropna(dim="n_event", how="all")
    )
    da.name = "Events"
    return da


def slice_time_series(
    data: xr.DataArray = xr.DataArray(),
    events: xr.DataArray | None = None,
    freq: float | None = None,
    n_dim_time: str = "time",
    reference_var: str | dict | None = None,
    discard_phases_ini: int = 0,
    n_phases: int | None = None,
    discard_phases_end: int = 0,
    add_to_ini: int | None = None,
    add_to_end: int | None = None,
    include_first_next_last: bool | None = False,
    max_phases: int | None = 100,
    func_events: Any | None = None,
    split_version_function: str = "numpy",  # "polars" or "numpy"
    **kwargs_func_events,
) -> xr.DataArray:
    """
    Slice a DataArray into phases defined by events.

    Parameters
    ----------
    data : xr.DataArray, optional
        The data to be sliced. By default None.
    events : xr.DataArray, optional
        Events that define the phases. By default None.
    freq : float, optional
        Sample frequency of the data. If None, it is inferred from the data.
        By default None.
    n_dim_time : str, optional
        Name of the time dimension. By default "time".
    reference_var : str or dict, optional
        Name of the variable to which the events refer. If a dict, it contains
        the name of the variable as the key and a function that takes the data
        and returns the value to which the events refer. By default None.
    discard_phases_ini : int, optional
        Discard the first `discard_phases_ini` phases. By default 0.
    n_phases : int, optional
        Number of phases to keep. By default None, which means all phases are
        kept.
    discard_phases_end : int, optional
        Discard the last `discard_phases_end` phases. By default 0.
    add_to_ini : int, optional
        Add the specified number of points to the beginning from the end of the the previous phase
    add_to_end : int, optional
        Add the specified number of points to the end from the beginning of the the next phase
    include_first_next_last : bool, optional
        Whether to include the first value of the next phase as the last value
        of the present phase. Deprecating. Similar to add_to_end=1. By default False.
    max_phases : int, optional
        Maximum number of phases. By default 100.
    func_events : callable, optional
        A function that takes the data and returns the events. By default None.
    split_version_function : str, optional
        A string indicating the version of the function to use to split the data.
        By default "numpy". Can be "polars".
    **kwargs_func_events
        Additional keyword arguments to pass to the function `func_events`.

    Returns
    -------
    DataArray
        A DataArray with the sliced phases.

    """

    # If the events are not specified, detect them
    if events is None:
        events = detect_events(
            data,
            freq,
            n_dim_time,
            reference_var,
            discard_phases_ini,
            n_phases,
            discard_phases_end,
            # include_first_next_last,
            max_phases,
            func_events,
            **kwargs_func_events,
        )

    # Numpy version
    def _slice_aux(
        dat,
        evts,
        max_phases,
        max_time,
        ID,
        var,
        add_ini,
        add_end,
        # include_first_next_last,
    ):
        phases = np.full((max_phases, max_time), np.nan)
        # print(ID, var)
        if (
            np.count_nonzero(~np.isnan(dat)) == 0
            or np.count_nonzero(~np.isnan(evts)) == 0
        ):
            return phases

        evts = evts[~np.isnan(evts)].astype(int)

        if add_ini is not None and add_ini != 0:
            evts -= add_ini
            add_end += add_ini

        for sl in range(len(evts) - 1):
            d = dat[evts[sl] : evts[sl + 1] + add_end]
            phases[sl, : len(d)] = d

        """
        # Whith np.split much slower
        tpo=time.perf_counter()        
        for i in range(1000):
            t = np.split(dat, evts)[1:-1]
            try:
                t = np.array(list(itertools.zip_longest(*t, fillvalue=np.nan))).T
                phases[: t.shape[0], : t.shape[1]] = t
            except:
                pass
        print(f'{time.perf_counter()-tpo:.3f} s')
        """

        # To include the first value of the next slice as the last of the
        # present. Usefull when graphing cycles
        # TODO: improve vectorizing
        # TODO: CHECK TO INCLUDE ONE LAST FRAME IN EACH PHASE

        # if add_ini is not None or add_end is not None:
        #     for sl in range(len(evts) - 1):
        #         phases[sl, evts[sl + 1] - evts[sl] : evts[sl + 1] - evts[sl] + add_end] = dat[
        #             evts[sl + 1] + add_end
        #         ]
        #         # phases[0, evts[1] - evts[0]-4:evts[1] - evts[0]+2]

        #         # phases[1, evts[2] - evts[1]-4:evts[2] - evts[1]+2]

        return phases

    # Polars version
    if split_version_function in ["polars", "polarspiv"]:
        import polars as pl
    elif split_version_function == "pandas":
        import pandas as pd

    def _slice_aux_pl(
        dat,
        evts,
        max_phases,
        max_time,
        ID,
        var,
        add_ini,
        add_end,
        # include_first_next_last=True,
    ):
        # print(ID, var)
        phases = np.full((max_time, max_phases), np.nan)
        try:
            if (
                np.count_nonzero(~np.isnan(dat)) == 0
                or np.count_nonzero(~np.isnan(evts)) == 0
            ):
                return phases
        except Exception as inst:
            print(inst)

        # evts = np.array([0] + evts.tolist() + [len(dat)]).astype(int)
        # ind = np.repeat(range(len(evts) - 1), np.diff(evts))

        try:
            evts = evts[~np.isnan(evts)].astype(int)
            order = np.repeat(range(len(evts) - 1), np.diff(evts))

            df = pl.DataFrame(
                {
                    "data": dat[evts[0] : evts[-1]],
                    "order": order,
                }
            )

            df = pl.DataFrame(
                {"data": dat[evts[0] : evts[-1]], "idx": order}
            ).partition_by(by="idx", as_dict=False, include_key=False)

            # Rename each block to allow concatenate
            df = [df.rename({"data": f"data{n}"}) for n, df in enumerate(df)]
            dfph = pl.concat(df, how="horizontal")

            phases[: dfph.shape[0], : dfph.shape[1]] = dfph.to_numpy()

            # ph = pl.concat(df, how="horizontal").to_numpy().T
            # phases[: ph.shape[0], : ph.shape[1]] = ph

            # To include the first value of the next slice as the last of the
            # present. Usefull when graphing cycles
            # TODO: improve vectorizing
            if add_end > 0:  # include_first_next_last:

                # dfph[np.diff(evts)[:-1]-1,:-1]
                # dfph[135,:-1]

                for sl in range(len(evts) - 2):
                    phases[evts[sl + 1] - evts[sl], sl] = phases[
                        0, sl + 1
                    ]  # dat[evts[sl + 1]]
                    # phases[evts[1] - evts[0]-4:evts[1] - evts[0]+2, 0]
                    # phases[0:4, 1] #evts[2] - evts[1]-4:evts[2] - evts[1]+2, 1]
                    # phases[2, evts[2] - evts[1]-4:evts[2] - evts[1]+2]
        except:
            print(f"Error in {ID}, {var}")
        return phases.T

    def _slice_aux_pl_pivot(
        dat,
        evts,
        max_phases,
        max_time,
        ID,
        var,
        add_ini,
        add_end,
        # include_first_next_last=True,
    ):
        # print(ID, var)
        phases = np.full((max_time, max_phases), np.nan)
        try:
            if (
                np.count_nonzero(~np.isnan(dat)) == 0
                or np.count_nonzero(~np.isnan(evts)) == 0
            ):
                return phases
        except Exception as inst:
            print(inst)

        # evts = np.array([0] + evts.tolist() + [len(dat)]).astype(int)
        # ind = np.repeat(range(len(evts) - 1), np.diff(evts))

        try:
            evts = evts[~np.isnan(evts)].astype(int)
            phase = np.repeat(range(len(evts) - 1), np.diff(evts))

            repp = []
            for rep in np.unique(phase, return_counts=True)[1]:
                repp.append(np.arange(rep))
            ind = np.concatenate(repp)

            df = pl.DataFrame(
                {"data": dat[evts[0] : evts[-1]], "phase": phase, "ind": ind}
            )

            df = df.pivot(values="data", index="ind", on="phase")

            phases[: df.shape[0], : df.shape[1]] = df[:, 1:].to_numpy()

            # ph = pl.concat(df, how="horizontal").to_numpy().T
            # phases[: ph.shape[0], : ph.shape[1]] = ph

            # To include the first value of the next slice as the last of the
            # present. Usefull when graphing cycles
            # TODO: improve vectorizing

            if add_end > 0:  # include_first_next_last:
                # dfph[np.diff(evts)[:-1]-1,:-1]
                # dfph[135,:-1]

                for sl in range(len(evts) - 2):
                    phases[evts[sl + 1] - evts[sl], sl] = phases[
                        0, sl + 1
                    ]  # dat[evts[sl + 1]]
                    # phases[evts[1] - evts[0]-4:evts[1] - evts[0]+2, 0]
                    # phases[0:4, 1] #evts[2] - evts[1]-4:evts[2] - evts[1]+2, 1]
                    # phases[2, evts[2] - evts[1]-4:evts[2] - evts[1]+2]
        except Exception as inst:
            print(f"Error en {ID}, {var}, {inst}")
        return phases.T

    def _slice_aux_pd(
        dat,
        evts,
        max_phases,
        max_time,
        ID,
        var,
        add_ini,
        add_end,
        # include_first_next_last=True,
    ):
        # print(ID, var)
        phases = np.full((max_time, max_phases), np.nan)

        try:
            if (
                np.count_nonzero(~np.isnan(dat)) == 0
                or np.count_nonzero(~np.isnan(evts)) == 0
            ):
                return phases
        except Exception as inst:
            print(inst)

        # evts = np.array([0] + evts.tolist() + [len(dat)]).astype(int)
        # ind = np.repeat(range(len(evts) - 1), np.diff(evts))

        try:

            evts = evts[~np.isnan(evts)].astype(int)
            ind = np.repeat(range(len(evts) - 1), np.diff(evts))

            df = pd.Series(dat[evts[0] : evts[-1]], index=ind)
            df = pd.concat(
                [x.rename(n).reset_index(drop=True) for n, x in df.groupby(df.index)],
                axis=1,
            )
            """
            t = time.perf_counter()
            for i in range(1000):
                ind = np.repeat(range(len(evts) - 1), np.diff(evts))
                df = pd.Series(dat[evts[0] : evts[-1]], index=ind).reset_index()
                df.pivot(columns='index', values=0)                
                
            print(time.perf_counter() - t)
            
            t = time.perf_counter()
            for i in range(1000):
                pdind = pd.Series(range(len(evts) - 1)).repeat(np.diff(evts))
                df = pd.Series(dat[evts[0] : evts[-1]], index=pdind.index)
                pd.concat(
                    [
                        x.rename(n).reset_index(drop=True)
                        for n, x in df.groupby(df.index)
                    ],
                    axis=1,
                )
            print(time.perf_counter() - t)

            t = time.perf_counter()
            for i in range(1000):
                ind = np.repeat(range(len(evts) - 1), np.diff(evts))
                df = pd.Series(dat[evts[0] : evts[-1]], index=ind)
                pd.concat(
                    [
                        x.rename(n).reset_index(drop=True)
                        for n, x in df.groupby(df.index)
                    ],
                    axis=1,
                )
            print(time.perf_counter() - t)

            t = time.perf_counter()
            for i in range(1000):
                ind = np.repeat(range(len(evts) - 1), np.diff(evts))
                df = pd.Series(dat[evts[0] : evts[-1]], index=ind)
                # df.to_frame().reset_index().unstack()
                df.to_frame().reset_index().pivot(
                    columns="index", values=[0]
                )  # , index=range(len(df)))
            print(time.perf_counter() - t)
            """

            phases[: df.shape[0], : df.shape[1]] = df.to_numpy()

            # ph = pl.concat(df, how="horizontal").to_numpy().T
            # phases[: ph.shape[0], : ph.shape[1]] = ph

            # To include the first value of the next slice as the last of the
            # present. Usefull when graphing cycles
            # TODO: improve vectorizing

            if add_end > 0:  # include_first_next_last:
                # dfph[np.diff(evts)[:-1]-1,:-1]
                # dfph[135,:-1]

                for sl in range(len(evts) - 2):
                    phases[evts[sl + 1] - evts[sl], sl] = phases[
                        0, sl + 1
                    ]  # dat[evts[sl + 1]]
                    # phases[evts[1] - evts[0]-4:evts[1] - evts[0]+2, 0]
                    # phases[0:4, 1] #evts[2] - evts[1]-4:evts[2] - evts[1]+2, 1]
                    # phases[2, evts[2] - evts[1]-4:evts[2] - evts[1]+2]
        except:
            print(f"Error in {ID}, {var}")
        return phases.T

    if split_version_function == "numpy":
        func_slice = _slice_aux
    elif split_version_function == "polars":
        func_slice = _slice_aux_pl
    elif split_version_function == "polarspiv":
        func_slice = _slice_aux_pl_pivot
    elif split_version_function == "pandas":
        func_slice = _slice_aux_pd
    else:
        raise ValueError(f"Unknown split_version_function: {split_version_function}")

    """
    dat=data[0,0,0].values
    evts=events[0,0,0].values
    add_ini=add_to_ini
    add_end=add_to_end
    """
    if include_first_next_last and (add_to_end == 0 or add_to_end is None):
        add_to_end = 1
    if add_to_ini is None:
        add_to_ini = 0
    if add_to_end is None:
        add_to_end = 0

    max_phases = int(events.n_event[-1])
    max_time = int(events.diff("n_event").max()) + add_to_ini + add_to_end

    da = xr.apply_ufunc(
        func_slice,
        data,
        events,
        max_phases,
        max_time,
        data.ID,
        data.n_var,
        add_to_ini,
        add_to_end,
        input_core_dims=[[n_dim_time], ["n_event"], [], [], [], [], [], []],
        output_core_dims=[["n_event", n_dim_time]],
        exclude_dims=set(("n_event", n_dim_time)),
        dataset_fill_value=np.nan,
        vectorize=True,
        dask="parallelized",
        keep_attrs=True,
        # kwargs=args_func_events,
    )
    da = (
        da.assign_coords(n_event=range(len(da.n_event)))
        .assign_coords(time=np.arange(0, len(da.time)) / data.freq)
        .dropna(dim="n_event", how="all")
        .dropna(dim=n_dim_time, how="all")
        .rename({"n_event": "phase"})
    )
    # da[0,0,0].plot.line(x="time")
    da.attrs = data.attrs
    try:
        da.time.attrs["units"] = data.time.attrs["units"]
    except:
        pass

    return da


if False:  # TEST WITH POLARS
    # TRYIIIING
    cortes_idx = detect_events(
        data=daTotal,
        func_events=detect_onset_detecta_aux,
        **(dict(threshold=60, show=True)),
    )
    dat = data[0, 0, 0].values
    events = cortes_idx[0, 0, 0].values

    da = xr.apply_ufunc(
        slice_aux_PRUEBAAS_pl,
        data,
        events,
        max_phases,
        include_first_next_last,
        input_core_dims=[[n_dim_time], ["n_event"], [], []],
        output_core_dims=[["n_event", n_dim_time]],
        exclude_dims=set(("n_event", n_dim_time)),
        dataset_fill_value=np.nan,
        vectorize=True,
        dask="parallelized",
        keep_attrs=True,
        # kwargs=args_func_events,
    )
    da = (
        da.assign_coords(n_event=range(len(da.n_event)))
        .assign_coords(time=np.arange(0, len(da.time)) / data.freq)
        .dropna(dim="n_event", how="all")
        .dropna(dim=n_dim_time, how="all")
        .rename({"n_event": "phase"})
    )


# TEST BY ANALYZING EVERYTHING WITH POLARS. UNDER CONSTRUCTION
def slice_time_series_pl(
    data: xr.DataArray = xr.DataArray(),
    events: xr.DataArray | None = None,
    freq: float | None = None,
    n_dim_time: str = "time",
    reference_var: str | dict | None = None,
    discard_phases_ini: int = 0,
    n_phases: int | None = None,
    discard_phases_end: int = 0,
    include_first_next_last: bool = False,
    max_phases: int = 100,
    func_events: Any | None = None,
    # split_version_function: str = "numpy",  # "polars" or "numpy"
    **kwargs_func_events,
) -> xr.DataArray:
    """
    Slice a DataArray into phases defined by events.
    Polars version.

    Parameters
    ----------
    data : xr.DataArray, optional
        The data to be sliced. By default None.
    events : xr.DataArray, optional
        Events that define the phases. By default None.
    freq : float, optional
        Sample frequency of the data. If None, it is inferred from the data.
        By default None.
    n_dim_time : str, optional
        Name of the time dimension. By default "time".
    reference_var : str or dict, optional
        Name of the variable to which the events refer. If a dict, it contains
        the name of the variable as the key and a function that takes the data
        and returns the value to which the events refer. By default None.
    discard_phases_ini : int, optional
        Discard the first `discard_phases_ini` phases. By default 0.
    n_phases : int, optional
        Number of phases to keep. By default None, which means all phases are
        kept.
    discard_phases_end : int, optional
        Discard the last `discard_phases_end` phases. By default 0.
    include_first_next_last : bool, optional
        Whether to include the first value of the next phase as the last value
        of the present phase. By default False.
    max_phases : int, optional
        Maximum number of phases. By default 100.
    func_events : callable, optional
        A function that takes the data and returns the events. By default None.
    **kwargs_func_events
        Keyword arguments to be passed to `func_events`.

    Returns
    -------
    sliced_data : xr.DataArray
        The sliced data.
    """

    import polars as pl

    if events is None:  # if the events are not detected yet, detect them
        events = detect_events(
            data,
            freq,
            n_dim_time,
            reference_var,
            discard_phases_ini,
            n_phases,
            discard_phases_end,
            # include_first_next_last,
            max_phases,
            func_events,
            **kwargs_func_events,
        )

    df = pl.from_pandas(data.to_dataframe().reset_index())
    evts = pl.from_pandas(events.to_dataframe().reset_index())

    # TODO: TO ADJUST TO VARIABLE NUMBER OF DIMENSIONS
    for n, d in df.group_by(evts.columns[:-2], maintain_order=True):
        print(n)

        evt = evts.filter(pl.col(d.columns[0]) == d[0, 0])
        evt = evt.select(pl.col("Events")).cast(pl.Int32)  # evt[:,-1].astype(int)
        ind = np.repeat(range(len(evt) - 1), np.diff(evt[:, -1]))
        df = pl.DataFrame({"data": dat[evt[0] : evt[-1]], "idx": ind})

        df2 = df.partition_by(by="idx", as_dict=False, include_key=False)
        df3 = [df.rename({"data": f"data{n}"}) for n, df in enumerate(df2)]

        df4 = pl.concat(df3, how="horizontal").to_numpy().T
        # To include the first value of the next slice as the last of the
        # present. Usefull when graphing cycles
        # TODO: improve vectorizing
        if include_first_next_last:
            for sl in range(len(evts) - 2):
                df4[sl, evts[sl + 1] - evts[sl]] = data[evts[sl + 1]]
                # df4[0,-4:]
                # df4[1,:4]
        return df4

    """
    TEST BY CREATING COORDINATES WITH PHASE NUM. AND THEN SECTIONING
    MAKE DIFFERENT IF THERE IS SELECTOR VARIABLE FOR ALL OR IF THE CRITERION IS ONE BY ONEdef slice_time_series2(
    """


def slice_time_series2(
    data: xr.DataArray | None = xr.DataArray(),
    events: xr.DataArray | None = None,
    freq: float | None = None,
    n_dim_time: str = "time",
    reference_var: str | dict | None = None,
    discard_phases_ini: int = 0,
    n_phases: int | None = None,
    discard_phases_end: int = 0,
    include_first_next_last: bool = False,
    max_phases: int = 100,
    func_events: Any | None = None,
    **kwargs_func_events,
) -> xr.DataArray:
    # If the events are not detected yet, detect them
    if events is None:
        events = detect_events(
            data,
            freq,
            n_dim_time,
            reference_var,
            discard_phases_ini,
            n_phases,
            discard_phases_end,
            include_first_next_last,
            max_phases,
            func_events,
            **kwargs_func_events,
        )

    # TRYIIIIIING
    cortes_idx = detect_events(
        data=daTotal,
        func_events=detect_onset_detecta_aux,
        **(dict(threshold=60, show=True)),
    )
    dat = data[0, 0, 0].values
    events = cortes_idx[0, 0, 0].values

    def _slice_aux_PRUEBAAS(dat, events, max_phases=50, include_first_next_last=True):
        events = np.array([0] + events.tolist() + [len(dat)]).astype(int)
        ind = np.repeat(range(len(events) - 1), np.diff(events))
        # da2=da.assign_coords(time=ind)
        da2 = da.assign_coords(time2=("time", ind))
        # da2 = da2.assign_coords(time2=('time', da.time.values))
        for n, gr in da2.groupby("time2"):
            gr.plot.line(x="time")  # , col="ID")

    def _slice_aux(dat, events, max_phases=50, include_first_next_last=True):
        if (
            np.count_nonzero(~np.isnan(dat)) == 0
            or np.count_nonzero(~np.isnan(events)) == 0
        ):
            return np.full((max_phases, len(dat)), np.nan)

        events = events[~np.isnan(events)].astype(int)
        phases = np.full((max_phases, len(dat)), np.nan)
        t = np.split(dat, events)[1:-1]
        try:
            t = np.array(list(itertools.zip_longest(*t, fillvalue=np.nan))).T
            phases[: t.shape[0], : t.shape[1]] = t
        except:
            pass

        # To include the first value of the next slice as the last of the
        # present. Usefull when graphing cycles
        # TODO: improve vectorizing
        if include_first_next_last:
            for sl in range(len(events) - 1):
                phases[sl, events[sl + 1] - events[sl]] = dat[events[sl + 1]]
        return phases

    da = xr.apply_ufunc(
        _slice_aux,
        data,
        events,
        max_phases,
        include_first_next_last,
        input_core_dims=[[n_dim_time], ["n_event"], [], []],
        output_core_dims=[["n_event", n_dim_time]],
        exclude_dims=set(("n_event", n_dim_time)),
        dataset_fill_value=np.nan,
        vectorize=True,
        dask="parallelized",
        keep_attrs=True,
        # kwargs=args_func_events,
    )
    da = (
        da.assign_coords(n_event=range(len(da.n_event)))
        .assign_coords(time=np.arange(0, len(da.time)) / data.freq)
        .dropna(dim="n_event", how="all")
        .dropna(dim=n_dim_time, how="all")
        .rename({"n_event": "phase"})
    )
    da.attrs = data.attrs
    try:
        da.time.attrs["units"] = data.time.attrs["units"]
    except:
        pass

    return da


def trim_window(
    daData: xr.DataArray,
    daEvents: xr.DataArray | None = None,
    window: xr.DataArray | None = None,
) -> xr.DataArray:
    """
    Trim a DataArray given a window of time from the beginning of an event.

    Parameters
    ----------
    daData : xr.DataArray
        The data to be trimmed.
    daEvents : xr.DataArray, optional
        The events that define the beginning of the window. If None, the window
        is calculated from the first event.
    window : xr.DataArray, optional
        The window of time to trim from the beginning of the event.
        If None, daEvents is expected to have 2 events (ini to end).
        If not None, only one event should be passed to daEvents (ini to ini+window).

    Returns
    -------
    trimmed : xr.DataArray
        The trimmed DataArray.
    """

    # TODO: TRY with da.pad

    def __trim_window(datos, ini, fin):
        # print(datos.shape, ini,fin)
        d2 = np.full(datos.shape, np.nan)
        try:
            ini = int(ini)
            fin = int(fin)
            if ini < 0:
                ini = 0
            if fin > len(datos):
                fin = len(datos)
            d2[: fin - ini] = datos[ini:fin]
        except:
            pass
        return d2

    if window is not None:
        if window > 0:
            daIni = daEvents
            daFin = daEvents + window * daData.freq
        else:
            daIni = daEvents + window * daData.freq
            daFin = daEvents

    else:
        daIni = daEvents.isel(event=0)
        daFin = daEvents.isel(event=1)

    daSliced = (
        xr.apply_ufunc(
            __trim_window,
            daData,
            daIni,
            daFin,
            input_core_dims=[["time"], [], []],
            output_core_dims=[["time"]],
            exclude_dims=set(("time",)),
            vectorize=True,
        )
        .assign_coords({"time": daData.time})
        .dropna(dim="time", how="all")
    )
    daSliced.attrs = daData.attrs

    if not isinstance(daSliced, xr.Dataset):
        daSliced.name = daData.name
        daSliced = daSliced.astype(daData.dtype)
    else:
        for var in list(daSliced.data_vars):
            daSliced[var].attrs = daData[var].attrs

    # daSliced.plot.line(x='time', row='ID', col='axis')
    return daSliced


#################################################################
# TEST WITH CLASS
class SliceTimeSeriesPhases:
    def __init__(
        self,
        data: Optional[xr.DataArray] = xr.DataArray(),
        freq: Optional[float] = None,
        n_dim_time: Optional[str] = "time",
        reference_var: Optional[Union[str, dict]] = None,
        discard_phases_ini: int = 0,
        n_phases: Optional[int] = None,
        discard_phases_end: int = 0,
        include_first_next_last: Optional[bool] = False,
        max_phases: int = 100,
        func_events: Optional[Any] = None,
        **kwargs_func_events,
    ):
        self.data = data
        self.events = None
        self.n_dim_time = n_dim_time
        self.reference_var = reference_var
        self.discard_phases_ini = discard_phases_ini
        self.n_phases = n_phases
        self.discard_phases_end = discard_phases_end
        self.include_first_next_last = include_first_next_last
        self.func_events = func_events
        self.max_phases = max_phases
        self.kwargs_func_events = kwargs_func_events

        if freq == None and not data.isnull().all():
            self.freq = (
                np.round(
                    1 / (self.data[self.n_dim_time][1] - self.data[self.n_dim_time][0]),
                    1,
                )
            ).data
        else:
            self.freq = freq

    def detect_events(self) -> xr.DataArray:
        # TODO: ADJUST THE FUNCTION TO ALLOW SPECIFIC THRESHOLDS FOR EACH TEST
        def detect_aux_idx(
            data,
            data_reference_var=None,
            func_events=None,
            max_phases=50,
            discard_phases_ini=0,
            n_phases=None,
            discard_phases_end=0,
            **kwargs_func_events,
        ):
            events = np.full(max_phases, np.nan)
            if (
                np.count_nonzero(~np.isnan(data)) == 0
                or np.count_nonzero(~np.isnan(data_reference_var)) == 0
            ):
                return events
            try:
                evts = func_events(data_reference_var, **kwargs_func_events)
            except:
                return events

            # If necessary, adjust initial an final events
            evts = evts[discard_phases_ini:]

            if n_phases == None:
                evts = evts[: len(evts) - discard_phases_end]
            else:  # if a specific number of phases from the first event is required
                if len(evts) >= n_phases:
                    evts = evts[: n_phases + 1]
                else:  # not enought number of events in the block, trunkated to the end
                    pass
            events[: len(evts)] = evts
            return events

        if self.func_events == None:
            raise Exception("A function to detect the events must be specified")

        da = xr.apply_ufunc(
            detect_aux_idx,
            self.data,
            self.data.sel(self.reference_var),
            self.func_events,
            self.max_phases,
            self.discard_phases_ini,
            self.n_phases,
            self.discard_phases_end,
            input_core_dims=[
                [self.n_dim_time],
                [self.n_dim_time],
                [],
                [],
                [],
                [],
                [],
            ],  # list with one entry for each argument
            output_core_dims=[["n_event"]],
            exclude_dims=set(("n_event", self.n_dim_time)),
            dataset_fill_value=np.nan,
            vectorize=True,
            dask="parallelized",
            # keep_attrs=True,
            kwargs=self.kwargs_func_events,
        )
        da = (
            da.assign_coords(n_event=range(len(da.n_event)))
            .dropna(dim="n_event", how="all")
            .dropna(dim="n_event", how="all")
        )
        self.events = da
        return da

    def slice_time_series(self, events: Optional[xr.DataArray] = None) -> xr.DataArray:
        if events is not None:  # the events are passed manually
            self.events = events
        elif self.events is None:  # if the events are not detected yet, detect them
            self.detect_events()

        def slice_aux(data, events, max_phases=50, include_first_next_last=True):
            if (
                np.count_nonzero(~np.isnan(data)) == 0
                or np.count_nonzero(~np.isnan(events)) == 0
            ):
                return np.full((max_phases, len(data)), np.nan)

            events = events[~np.isnan(events)].astype(int)
            phases = np.full((max_phases, len(data)), np.nan)
            t = np.split(data, events)[1:-1]
            try:
                t = np.array(list(itertools.zip_longest(*t, fillvalue=np.nan))).T
                phases[: t.shape[0], : t.shape[1]] = t
            except:
                pass

            # To include the first value of the next slice as the last of the
            # present. Usefull when graphing cycles
            # TODO: improve vectorizing
            if include_first_next_last:
                for sl in range(len(events) - 1):
                    phases[sl, events[sl + 1] - events[sl]] = data[events[sl + 1]]
            return phases

        da = xr.apply_ufunc(
            slice_aux,
            self.data,
            self.events,
            self.max_phases,
            self.include_first_next_last,
            input_core_dims=[[self.n_dim_time], ["n_event"], [], []],
            output_core_dims=[["n_event", self.n_dim_time]],
            exclude_dims=set(("n_event", self.n_dim_time)),
            dataset_fill_value=np.nan,
            vectorize=True,
            dask="parallelized",
            keep_attrs=True,
            # kwargs=args_func_events,
        )
        da = (
            da.assign_coords(n_event=range(len(da.n_event)))
            .assign_coords(time=np.arange(0, len(da.time)) / self.freq)
            .dropna(dim="n_event", how="all")
            .dropna(dim=self.n_dim_time, how="all")
            .rename({"n_event": "phase"})
        )
        da.attrs = self.data.attrs
        try:
            da.time.attrs["units"] = self.data.time.attrs["units"]
        except:
            pass

        return da

    # =============================================================================
    # Custom function to adapt from Detecta detect_onset
    # =============================================================================
    def detect_onset_detecta_aux(
        data, event_ini=0, xSD=None, show=False, **args_func_events
    ):
        # If event_ini=1 is passed as an argument, it takes the cut at the end of each window.
        try:
            from detecta import detect_onset
        except ImportError:
            raise Exception(
                "This function needs Detecta to be installed (https://pypi.org/project/detecta/)"
            )

        # try: #detect_onset returns 2 indexes. If not specified, select the first
        #     event_ini=args_func_events['event_ini']
        #     args_func_events.pop('event_ini', None)
        # except:
        #     event_ini=0
        if (
            xSD is not None
        ):  # the threshold is defined by the mean + x times the standard deviation
            if "threshold" in args_func_events:
                args_func_events.pop("threshold", None)
            args_func_events["threshold"] = (
                np.mean(data, where=~np.isnan(data))
                + np.std(data, where=~np.isnan(data)) * xSD
            )
            # print(args_func_events, np.mean(data, where=~np.isnan(data)), np.std(data, where=~np.isnan(data)), xSD)

        events = detect_onset(data, **args_func_events)

        if event_ini == 1:
            events = (
                events[:, event_ini] + 1
            )  # if the end of the window is chosen, 1 is added to start when the threshold has already been exceeded
            events = events[
                :-1
            ]  # removes the last one because it is usually incomplete
        else:
            events = events[
                :, event_ini
            ]  # keeps the first or second value of each data pair
            events = events[1:]  # removes the last one because it is usually incomplete

        if show:
            SliceTimeSeriesPhases.show_events(
                data, events, threshold=args_func_events["threshold"]
            )

        return events

    # =============================================================================
    # Custom function to adapt from scipy.signal find_peaks
    # =============================================================================
    def find_peaks_aux(data, xSD=None, show=False, **args_func_events):
        try:
            from scipy.signal import find_peaks
        except ImportError:
            raise Exception("This function needs scipy.signal to be installed")
        if (
            xSD is not None
        ):  # the threshold is defined by the mean + x times the standar deviation
            if isinstance(xSD, list):
                args_func_events["height"] = [
                    np.mean(data[~np.isnan(data)])
                    + xSD[0] * np.std(data[~np.isnan(data)]),
                    np.mean(data[~np.isnan(data)])
                    + xSD[1] * np.std(data[~np.isnan(data)]),
                ]
            else:
                args_func_events["height"] = np.mean(
                    data[~np.isnan(data)]
                ) + xSD * np.std(
                    data[~np.isnan(data)]
                )  # , where=~np.isnan(data)) + xSD * np.std(data, where=~np.isnan(data))

        data = data.copy()

        # Deal with nans
        data[np.isnan(data)] = -np.inf

        events, _ = find_peaks(data, **args_func_events)

        if show:
            SliceTimeSeriesPhases.show_events(
                data, events, threshold=args_func_events["height"]
            )

        return events  # keeps the first value of each data pair

    def show_events(data, events, threshold=None):
        plt.plot(data, c="b")
        plt.plot(events, data[events], "ro")
        if threshold is not None:
            plt.hlines(y=threshold, xmin=0, xmax=len(data), color="C1", ls="--", lw=1)
        plt.show()


# =============================================================================


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

    r"""
    #Example importing
    from biomdp.slice_time_series_phases import SliceTimeSeriesPhases as stsp
    """
    from detecta import detect_peaks

    # Find indexes and and then slice
    daEvents = detect_events(data=daTotal, func_events=detect_peaks)
    dacuts = slice_time_series(daTotal, daEvents)  # slice with the previous indexes
    dacuts.sel(n_var="a").plot.line(x="time", col="moment", hue="phase", row="ID")

    # Slice directly
    dacuts = slice_time_series(data=daTotal, func_events=detect_peaks, max_phases=100)
    dacuts.sel(n_var="a").plot.line(x="time", col="moment", hue="phase", row="ID")

    # Specifying one of the variables as slice reference
    dacuts = slice_time_series(
        data=daTotal, func_events=detect_peaks, reference_var=dict(n_var="b")
    )
    dacuts.stack(var_moment=("n_var", "moment")).plot.line(
        x="time", col="var_moment", hue="phase", row="ID"
    )

    # Slice using the indexes already found
    cortes_idx = detect_events(
        data=daTotal,
        func_events=detect_peaks,
        reference_var=dict(n_var="a"),
        max_phases=100,
    )
    cortes_retocados = cortes_idx.isel(n_event=slice(3, 20, 2))

    dacor = slice_time_series(data=daTotal, events=cortes_retocados)
    dacor.isel(ID=slice(None, 6)).sel(n_var="a").plot.line(
        x="time", col="moment", hue="phase", row="ID"
    )

    cortes_idx = detect_events(
        data=daTotal,
        func_events=detect_peaks,
        reference_var=dict(n_var="a"),
        max_phases=100,
    )
    cortes_retocados = cortes_idx.isel(n_event=slice(5, 20))

    dacor = slice_time_series(data=daTotal, events=cortes_retocados)
    dacor.isel(ID=slice(None, 6)).sel(n_var="c").plot.line(
        x="time", col="moment", hue="phase", row="ID"
    )

    daSliced = slice_time_series(
        data=daTotal,
        func_events=SliceTimeSeriesPhases.detect_onset_detecta_aux,
        reference_var=dict(moment="pre", n_var="b"),
        discard_phases_ini=0,
        n_phases=None,
        discard_phases_end=0,
        include_first_next_last=False,
        **dict(threshold=60, show=False),
    )
    daSliced.sel(n_var="b").plot.line(x="time", col="moment", hue="phase", row="ID")

    daSliced = slice_time_series(
        data=daTotal,
        func_events=SliceTimeSeriesPhases.find_peaks_aux,
        reference_var=dict(moment="pre", n_var="b"),
        discard_phases_ini=0,
        n_phases=None,
        discard_phases_end=0,
        include_first_next_last=True,
        **dict(height=60, distance=10),
    )
    daSliced.sel(n_var="b").plot.line(x="time", col="moment", hue="phase", row="ID")

    daSliced = SliceTimeSeriesPhases(
        data=daTotal,
        func_events=SliceTimeSeriesPhases.find_peaks_aux,
        reference_var=dict(moment="pre", n_var="b"),
        discard_phases_ini=0,
        n_phases=None,
        discard_phases_end=0,
        include_first_next_last=True,
        **dict(height=140, distance=1),
    ).slice_time_series()
    daSliced.sel(n_var="b").plot.line(x="time", col="moment", hue="phase", row="ID")

    daSliced = slice_time_series(
        data=daTotal,
        func_events=detect_peaks,
        reference_var=dict(moment="pre", n_var="b"),
        # max_phases=100,
        **dict(mph=140),
    )
    daSliced.sel(n_var="b").plot.line(x="time", col="moment", hue="phase", row="ID")

    # Find_peaks with xSD
    daSliced = slice_time_series(
        data=daTotal,
        func_events=SliceTimeSeriesPhases.find_peaks_aux,
        # reference_var=dict(moment="pre", n_var="b"),
        discard_phases_ini=0,
        n_phases=None,
        discard_phases_end=0,
        include_first_next_last=True,
        **dict(xSD=0.8, distance=1),
        show=True,
    )
    daSliced.sel(n_var="b").plot.line(x="time", col="moment", hue="phase", row="ID")

    # Onset by xSD
    daSliced = slice_time_series(
        daTotal,
        func_events=SliceTimeSeriesPhases.detect_onset_detecta_aux,
        # reference_var=dict(moment='pre', n_var='b'),
        discard_phases_ini=0,
        n_phases=None,
        discard_phases_end=0,
        include_first_next_last=True,
        **dict(xSD=-1.2),
    )
    daSliced.sel(n_var="b").plot.line(x="time", col="moment", hue="phase", row="ID")

    # Trim data, slice ini and end events
    daEvents = detect_events(
        data=daTotal,
        func_events=SliceTimeSeriesPhases.find_peaks_aux,
        discard_phases_ini=0,
        n_phases=None,
        discard_phases_end=0,
        **dict(height=0, distance=10),
    )

    # ---- Detect onset from detect peaks y derivative
    daSliced = detect_events(
        data=daTotal,
        func_events=SliceTimeSeriesPhases.detect_onset_detecta_aux,
        reference_var=dict(moment="pre", n_var="b"),
        discard_phases_ini=0,
        n_phases=None,
        discard_phases_end=0,
        **dict(threshold=60, show=False),
    )
    daSliced.sel(n_var="b").plot.line(
        x="n_event", col="moment", hue="phase", row="ID", marker="o"
    )

    from biomdp.general_processing_functions import integrate_window

    daInteg = integrate_window(daTotal, daOffset=daTotal.isel(time=0))
    daInteg[2, 0].plot.line(x="time")
    daSliced = detect_events(
        data=daInteg,
        func_events=SliceTimeSeriesPhases.find_peaks_aux,
        reference_var=dict(moment="pre", n_var="b"),
        discard_phases_ini=0,
        n_phases=None,
        discard_phases_end=0,
        **dict(height=0, show=True),
    )
    daSliced.sel(n_var="b").plot.line(x="time", col="moment", hue="phase", row="ID")

    window = xr.concat(
        [daEvents.min("n_event"), daEvents.max("n_event")], dim="n_event"
    )
    # window = daEvents.isel(n_event=[5,7]) #xr.concat([daEvents.isel(n_event=5), daEvents.isel(n_event=7)], dim='n_event')

    daTrimed = slice_time_series(daTotal, events=window)
    daTrimed.stack(var_moment=("n_var", "moment")).plot.line(
        x="time", col="var_moment", hue="phase", row="ID"
    )

    # ---- Increasing the numbere of data in the begining and end
    daSliced = slice_time_series(
        data=daTotal,
        func_events=SliceTimeSeriesPhases.detect_onset_detecta_aux,
        reference_var=dict(moment="pre", n_var="b"),
        discard_phases_ini=0,
        n_phases=None,
        discard_phases_end=0,
        add_to_ini=1,
        add_to_end=1,
        include_first_next_last=True,
        **dict(threshold=60, show=False),
    )
    daSliced[0, 0, 0, :, :2]
    daSliced[0, 0, 0, 0][~np.isnan(daSliced[0, 0, 0, 0])][-2:]  # final de una fase
    daSliced[0, 0, 0, 1][~np.isnan(daSliced[0, 0, 0, 1])][-2:]  # final de una fase
    daSliced.sel(n_var="b").plot.line(x="time", col="moment", hue="phase", row="ID")

    # =============================================================================
    # %% Test functions to slice (polars, numpy, ...)
    # =============================================================================
    import timeit

    def test_performance():
        result = slice_time_series(
            data=daTotal,
            func_events=SliceTimeSeriesPhases.find_peaks_aux,
            reference_var=dict(moment="pre", n_var="b"),
            discard_phases_ini=0,
            n_phases=None,
            discard_phases_end=0,
            include_first_next_last=False,
            split_version_function="numpy",
            **dict(xSD=0.8, distance=1),
            show=False,
        )
        return result

    print(
        f'{timeit.timeit("test_performance()", setup="from __main__ import test_performance", number=50):.4f} s'
    )

    def test_performance():
        result = slice_time_series(
            data=daTotal,
            func_events=SliceTimeSeriesPhases.find_peaks_aux,
            reference_var=dict(moment="pre", n_var="b"),
            discard_phases_ini=0,
            n_phases=None,
            discard_phases_end=0,
            include_first_next_last=False,
            split_version_function="polars",
            **dict(xSD=0.8, distance=1),
            show=False,
        )
        return result

    print(
        f'{timeit.timeit("test_performance()", setup="from __main__ import test_performance", number=50):.4f} s'
    )
    slices = test_performance()

    def test_performance():
        result = slice_time_series(
            data=daTotal,
            func_events=SliceTimeSeriesPhases.find_peaks_aux,
            reference_var=dict(moment="pre", n_var="b"),
            discard_phases_ini=0,
            n_phases=None,
            discard_phases_end=0,
            include_first_next_last=False,
            split_version_function="polarspiv",
            **dict(xSD=0.8, distance=1),
            show=False,
        )
        return result

    print(
        f'{timeit.timeit("test_performance()", setup="from __main__ import test_performance", number=50):.4f} s'
    )
    slices2 = test_performance()

    def test_performance():
        result = slice_time_series(
            data=daTotal,
            func_events=SliceTimeSeriesPhases.find_peaks_aux,
            reference_var=dict(moment="pre", n_var="b"),
            discard_phases_ini=0,
            n_phases=None,
            discard_phases_end=0,
            include_first_next_last=False,
            split_version_function="pandas",
            **dict(xSD=0.8, distance=1),
            show=False,
        )
        return result

    print(
        f'{timeit.timeit("test_performance()", setup="from __main__ import test_performance", number=50):.4f} s'
    )
    slices3 = test_performance()

    slices[0, 0, 0, :, -50:-6].plot.line(x="time", hue="phase")
    slices2[0, 0, 0, :, -50:-6].plot.line(x="time", hue="phase")
    slices3[0, 0, 0, :, -50:-6].plot.line(x="time", hue="phase")

    ###################################
    # With the class version
    def test_performance():
        result = SliceTimeSeriesPhases(
            data=daTotal,
            func_events=SliceTimeSeriesPhases.find_peaks_aux,
            # reference_var=dict(moment="pre", n_var="b"),
            discard_phases_ini=0,
            n_phases=None,
            discard_phases_end=0,
            include_first_next_last=False,
            # split_version_function="polars",
            **dict(xSD=0.8, distance=1),
            show=False,
        ).slice_time_series()
        return result

    print(
        f'{timeit.timeit("test_performance()", setup="from __main__ import test_performance", number=50):.4f} s'
    )
    daSliced = test_performance()
    daSliced.sel(n_var="b").plot.line(x="time", col="moment", hue="phase", row="ID")

    # %%Performance tests class / functions
    import timeit

    def test_performance():
        result = SliceTimeSeriesPhases(
            data=daTotal,
            func_events=SliceTimeSeriesPhases.find_peaks_aux,
            reference_var=dict(moment="pre", n_var="b"),
            discard_phases_ini=0,
            n_phases=None,
            discard_phases_end=0,
            include_first_next_last=True,
            **dict(height=60, distance=10),
        ).slice_time_series()
        return result

    print(
        f'{timeit.timeit("test_performance()", setup="from __main__ import test_performance", number=50):.4f} s'
    )

    def test_performance():
        result = slice_time_series(
            data=daTotal,
            func_events=SliceTimeSeriesPhases.find_peaks_aux,
            reference_var=dict(moment="pre", n_var="b"),
            discard_phases_ini=0,
            n_phases=None,
            discard_phases_end=0,
            include_first_next_last=True,
            **dict(height=60, distance=10),
        )
        return result

    print(
        f'{timeit.timeit("test_performance()", setup="from __main__ import test_performance", number=50):.4f} s'
    )

    # %%Performance tests trim
    # Quicker with trim function
    import timeit

    def test_performance():
        daEvents = detect_events(
            data=daTotal,
            func_events=SliceTimeSeriesPhases.find_peaks_aux,
            discard_phases_ini=0,
            n_phases=None,
            discard_phases_end=0,
            **dict(height=10, distance=10),
        )
        window = xr.concat(
            [daEvents.min("n_event"), daEvents.max("n_event")], dim="n_event"
        )
        # window = daEvents.isel(n_event=[5,7]) #xr.concat([daEvents.isel(n_event=5), daEvents.isel(n_event=7)], dim='n_event')
        result = slice_time_series(daTotal, events=window)
        return result

    print(
        f'{timeit.timeit("test_performance()", setup="from __main__ import test_performance", number=50):.4f} s'
    )

    def test_performance():
        daEvents = detect_events(
            data=daTotal,
            func_events=SliceTimeSeriesPhases.find_peaks_aux,
            discard_phases_ini=0,
            n_phases=None,
            discard_phases_end=0,
            **dict(height=0, distance=10),
        )
        window = xr.concat(
            [daEvents.min("n_event"), daEvents.max("n_event")], dim="n_event"
        )
        result = trim_window(daTotal, window)
        return result

    print(
        f'{timeit.timeit("test_performance()", setup="from __main__ import test_performance", number=50):.4f} s'
    )
