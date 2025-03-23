# -*- coding: utf-8 -*-
"""
Created on Mon Oct 02 16:36:37 2023

@author: Jose L. L. Elvira

Bioware's .c3d exports only sensor data separately,
8 per platform.
"""


# =============================================================================
# %% LOAD LIBRARIES
# =============================================================================

__author__ = "Jose L. L. Elvira"
__version__ = "0.1.1"
__date__ = "11/03/2025"

"""
Updates:
    11/03/2025, v0.1.1
            - Adapted to biomdp with translations.
    
    10/12/2024, v0.1.0
            - Comprueba si está instalado c3d y ezc3d, si no, avisa con mensaje.
            - Incluida función con ezc3d, pero no carga la serie temporal completa.
    
    29/12/2023, v0.0.2
            - Calcula las fuerzas en los 3 ejes a partir de las raw de los sensores.

    02/10/2023, v0.0.1
            - Empezado a partir de read_vicon.c3d.
            
"""

from typing import List
import numpy as np

# import pandas as pd
import xarray as xr


import time
from pathlib import Path


# =============================================================================
# %% Functions
# =============================================================================


def read_kistler_c3d(
    file: str | Path,
    n_vars_load: List[str] | None = None,
    # coincidence: str = "similar",
    engine: str = "ezc3d",
    raw: bool = False,
) -> xr.DataArray:

    if not file.exists():
        raise FileNotFoundError(f"File {file} not found")

    if engine == "c3d":

        da = read_kistler_c3d_c3d(file)

    elif engine == "ezc3d":

        da = read_kistler_ezc3d(file)

    else:
        print(f"Engine {engine} not implemented. Try 'c3d' or 'ezc3d'")

    if raw:
        if n_vars_load:
            da = da.sel(n_var=n_vars_load)

    else:
        da = split_plataforms(da)
        # da = split_dim_axis(da)
        da = compute_forces_axes(da)
        if n_vars_load:
            da = da.sel(n_var=n_vars_load)

    return da


def read_kistler_c3d_c3d(
    file: str | Path,
) -> xr.DataArray:
    try:
        import c3d
    except:
        raise ImportError("Module c3d not installed.\nInstall with pip install c3d")
    import warnings  # to remove 'no points found' warnings

    timer = time.perf_counter()

    # se asegura de que la extensión es c3d
    file = file.with_suffix(".c3d")

    try:
        timerSub = time.perf_counter()
        # print(f'Loading file: {file.name}')

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open(file, "rb") as handle:
                reader = c3d.Reader(handle)

                # freq = reader.point_rate
                freq_analog = reader.analog_rate

                points = []
                analog = []
                for i, (_, p, a) in enumerate(reader.read_frames()):
                    # points.append(p)
                    analog.append(a)
                    if not i % 10000 and i:
                        print(f"Extracted {len(points)} point frames")

        labels_analog = [s.split(".")[0].replace(" ", "") for s in reader.analog_labels]
        data_analog = np.concatenate(analog, axis=1)
        # import matplotlib.pyplot as plt
        # plt.plot(data_analog.T)
        # data_analog.shape
        coords = {
            "n_var": labels_analog,
            "time": np.arange(data_analog.shape[1]) / freq_analog,
        }
        da_analog = xr.DataArray(
            data=data_analog,
            dims=coords.keys(),
            coords=coords,
            attrs={"freq": freq_analog},
        )

        da_analog.attrs["units"] = "N"
        da_analog.time.attrs["units"] = "s"

        print(f"Time {time.perf_counter() - timerSub:.3f} s \n")

    except Exception as err:
        print(f"\nATTENTION. Unable to process {file.name}, {err}, \n")

    return da_analog


def read_kistler_ezc3d(
    file: str | Path,
):
    """
    DOES NOT SEEM TO LOAD THE COMPLETE TIME SERIES
    """
    try:
        import ezc3d
    except:
        raise ImportError(
            "Module ezc3d not installed.\nInstall with pip install ezc3d or conda install -c conda-forge ezc3d"
        )

    timer = time.perf_counter()

    # se asegura de que la extensión es c3d
    file = file.with_suffix(".c3d")

    try:
        timerSub = time.perf_counter()
        # print(f'Loading file: {file.name}')

        acq = ezc3d.c3d(file.as_posix())  # , extract_forceplat_data=True)

        freq = acq["parameters"]["ANALOG"]["RATE"]["value"][0]
        labels = acq["parameters"]["ANALOG"]["LABELS"]["value"]
        unit_force = acq["parameters"]["ANALOG"]["UNITS"]["value"][0]
        num_plats = acq["parameters"]["ANALOG"]["USED"]["type"]

        if False:
            acq["parameters"]["FORCE_PLATFORM"]["CORNERS"]["value"]

            plats = acq["data"]["platform"]
            num_plats = len(plats)
            unit_force = plats[0]["unit_force"]

            plat1 = plats[0]["force"][:, :]
            plat1.shape
            plt.plot(data.T)

        data = acq["data"]["analogs"][0]  # analogs
        coords = {
            "n_var": labels,
            "time": np.arange(data.shape[-1]) / freq,
        }
        da = xr.DataArray(
            data=data,
            dims=coords.keys(),
            coords=coords,
            attrs={"freq": freq},
        )

        da.attrs["units"] = unit_force
        da.time.attrs["units"] = "s"

        print(f"Time {time.perf_counter() - timerSub:.3f} s \n")

    except Exception as err:
        print(f"\nATTENTION. Unable to process {file.name}, {err}\n")

    # if raw:
    #     if n_vars_load:
    #         da = da.sel(n_var=n_vars_load)
    #     return da
    # else:
    #     da = split_plataforms(da)
    #     # da = split_dim_axis(da)
    #     da = compute_forces_axes(da)
    #     if n_vars_load:
    #         da = da.sel(n_var=n_vars_load)

    # if n_vars_load:
    #     da = da.sel(n_var=n_vars_load)

    return da


def split_plataforms(da: xr.DataArray) -> xr.DataArray:
    if len(da.n_var) == 8:  # 1 plate
        da = da.expand_dims("plate").assign_coords(plate=[1])

    elif len(da.n_var) == 16:
        plat1 = da.sel(n_var=da.n_var.str.startswith("F1"))
        plat1 = plat1.assign_coords(n_var=plat1.n_var.str.lstrip("F1"))

        plat2 = da.sel(n_var=da.n_var.str.startswith("F2"))
        plat2 = plat2.assign_coords(n_var=plat2.n_var.str.lstrip("F2"))

        da = xr.concat([plat1, plat2], dim="plate").assign_coords(plate=[1, 2])

    # if plat2.n_var.size == 0:
    #     print("There is only one platform.")
    #     return da.sel(plate=[1])
    # else:
    return da


def split_dim_axis(da: xr.DataArray) -> xr.DataArray:
    # NOT NECESSARY WITH COMPUTE_FORCES_AXES???
    # TODO: The letter of the axis in the name must be removed.
    x = da.sel(n_var=da.n_var.str.contains("x"))
    y = da.sel(n_var=da.n_var.str.contains("y"))
    z = da.sel(n_var=da.n_var.str.contains("z"))
    da = (
        xr.concat([x, y, z], dim="axis")
        # .assign_coords(n_var='plat1')
        .assign_coords(axis=["x", "y", "z"])
        # .expand_dims({'n_var':1})
    )
    return da


def compute_forces_axes(da: xr.DataArray) -> xr.DataArray:
    # da=daForce

    if "plate" not in da.coords:
        da = split_plataforms(da)

    Fx = da.sel(n_var=da.n_var.str.contains("x")).sum(dim="n_var")
    Fy = da.sel(n_var=da.n_var.str.contains("y")).sum(dim="n_var")
    Fz = da.sel(n_var=da.n_var.str.contains("z")).sum(dim="n_var")

    daReturn = xr.concat([Fx, Fy, Fz], dim="axis").assign_coords(axis=["x", "y", "z"])
    # daReturn.plot.line(x='time', col='plate')

    return daReturn


def compute_moments_axes(da: xr.DataArray) -> xr.DataArray:
    # da=daForce
    raise Exception("Not implemented yet.")
    """
    if 'plate' not in da.coords:
        da = split_plataforms(da)

    Fx = da.sel(n_var=da.n_var.str.contains('x')).sum(dim='n_var')
    Fy = da.sel(n_var=da.n_var.str.contains('y')).sum(dim='n_var')
    Fz = da.sel(n_var=da.n_var.str.contains('z')).sum(dim='n_var')
        
    daReturn = (xr.concat([Fx, Fy, Fz], dim='axis')
                .assign_coords(axis=['x', 'y', 'z'])
                )
    #daReturn.plot.line(x='time', col='plate')
    """
    return daReturn


# =============================================================================
# %% TESTS
# =============================================================================
if __name__ == "__main__":

    # from biomdp.io.read_kistler_c3d import read_kistler_c3d_xr

    work_path = Path(r"src\datasets")
    file = work_path / "kistler_CMJ_1plate.c3d"
    daForce = read_kistler_c3d(file, engine="c3d")
    daForce.isel(plate=0).plot.line(x="time")  # , col="plate")

    work_path = Path(r"src\biomdp\datasets")
    file = work_path / "kistler_DJ_2plates.c3d"
    daForce = read_kistler_c3d(file, engine="c3d")
    daForce.plot.line(x="time", col="plate")

    daForce = read_kistler_c3d(file, engine="c3d", raw=True)
    daForce.plot.line(x="time")
    daForce = split_plataforms(daForce)
    daForce = compute_forces_axes(daForce)
    daForce.plot.line(x="time", col="plate")

    # With ezc3d does not read the entire file???
    daForce2 = read_kistler_c3d(file, engine="ezc3d", raw=True)
    daForce2.plot.line(x="time")

    # Compare with c3d and ezc3d
    work_path = Path(r"src\biomdp\datasets")
    file = work_path / "kistler_DJ_2plates.c3d"
    daForce = read_kistler_c3d(file, engine="c3d")
    daForce.plot.line(x="time", col="plate")

    daForce2 = read_kistler_c3d(file, engine="ezc3d")
    daForce2.plot.line(x="time", col="plate")

    daForce2.equals(daForce.isel(time=slice(None, daForce2.time.size)))
