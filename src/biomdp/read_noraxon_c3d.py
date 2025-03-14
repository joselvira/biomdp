# -*- coding: utf-8 -*-
"""
Created on Fry Mar 08 12:17:37 2024

@author: josel
"""

# =============================================================================
# %% LOAD LIBRARIES
# =============================================================================

__author__ = "Jose L. L. Elvira"
__version__ = "0.1.0"
__date__ = "11/03/2025"

"""
Updates:    
    11/03/2025, v0.1.0
            - Adapted to biomdp with translations.
    
    08/03/2024, v0.0.1
            - Empezado tomando trozos sueltos.
            
"""

from typing import List
import warnings  # para quitar warnings de no encontrar points

import numpy as np

# import pandas as pd
import xarray as xr

import time
from pathlib import Path


# import sys
# sys.path.append('F:\Programacion\Python\Mios\Functions')
# #sys.path.append('G:\Mi unidad\Programacion\Python\Mios\Functions')


# =============================================================================
# %% Carga trayectorias desde c3d
# =============================================================================
def read_noraxon_c3d_xr(
    file: str | Path,
    section: str | None = None,
    n_vars_load: List[str] | None = None,
    coincidence: str = "similar",
    engine: str = "ezc3d",
) -> xr.DataArray:
    if engine == "c3d":
        return read_noraxon_c3d_c3d_xr(file, section, n_vars_load, coincidence)

    elif engine == "ezc3d":
        return read_noraxon_ezc3d_xr(file, section, n_vars_load, coincidence)

    else:
        print("Engine {engine} not implemented. Try 'c3d' or 'ezc3d'")


def read_noraxon_c3d_c3d_xr(
    file: str | Path,
    section: str | None = None,
    n_vars_load: List[str] | None = None,
    coincidence: str = "similar",
) -> xr.DataArray:

    try:
        import c3d
    except:
        raise ImportError("Module c3d not installed.\nInstall with pip install c3d")

    if section not in [
        "Trajectories",
        "Model Outputs",
        "EMG",
    ]:  # not ('Trajectories' in section or 'Model Outputs'in section or 'Forces' in section or 'EMG'in section):
        raise Exception(
            'Section header not found, try "Trajectories", "Model outputs" or "EMG"'
        )
        return

    timer = time.perf_counter()  # inicia el contador de tiempo

    # se asegura de que la extensión es c3d
    file = file.with_suffix(".c3d")

    try:
        timerSub = time.perf_counter()  # inicia el contador de tiempo
        # print(f'Loading section {section}, file: {file.name}')

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open(file, "rb") as handle:
                reader = c3d.Reader(handle)

                freq = reader.point_rate
                freq_analog = reader.analog_rate

                points = []
                analog = []
                # for i, (_, p, a) in enumerate(reader.read_frames()):
                for _, p, a in reader.read_frames():
                    points.append(p)
                    # analog.append(a)
                    # if not i % 10000 and i:
                    #     print("Extracted %d point frames", len(points))

        # Trajectiories and Modeled outputs
        if "Trajectories" in section or "Model Outputs" in section:
            labels = [s.replace(" ", "") for s in reader.point_labels]
            data = np.asarray(points)[:, :, :3]

            coords = {
                "time": np.arange(data.shape[0]) / freq,
                "n_var": labels,
                "axis": ["x", "y", "z"],
            }
            da = xr.DataArray(
                data,  # =np.expand_dims(data, axis=0),
                dims=coords.keys(),
                coords=coords,
                name="Trajectories",
                attrs={
                    "freq": freq,
                    "units": "mm",
                },
            ).transpose("n_var", "axis", "time")
            if "Trajectories" in section:
                # Delete unnamed trajectories and modeled outputs
                da = da.sel(
                    n_var=(
                        ~da.n_var.str.startswith("*") & ~da.n_var.str.contains("USERMO")
                    )
                )
            if "Model Outputs" in section:
                da = da.sel(n_var=da.n_var.str.contains("USERMO"))

        # Analogs
        elif section in ["Forces", "EMG"]:  # ('Forces' in section or 'EMG' in section):
            labels_analog = [
                s.split(".")[1].replace(" ", "") for s in reader.analog_labels
            ]
            data_analog = np.concatenate(analog, axis=1)

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

            # EMG
            if section == "EMG":
                if da_analog.n_var.str.contains("EMG").any():
                    da = da_analog.sel(n_var=da_analog.n_var.str.contains("EMG"))
                    da.attrs["units"] = "mV"
                    # da.n_var.sortby('n_var')
                    # da.plot.line(x='time', col='n_var', col_wrap=3)
                else:
                    da = xr.DataArray()
                    raise Exception("No EMG data in file")

        da.time.attrs["units"] = "s"
        da.name = section

        # print('Tiempo {0:.3f} s \n'.format(time.perf_counter()-timerSub))

    except Exception as err:
        da = xr.DataArray()
        print(f"\nATENCIÓN. No se ha podido procesar {file.name}, {err}\n")

    if n_vars_load is not None and "n_var" in da.coords:
        da = da.sel(n_var=n_vars_load)

    return da  # daTrajec, daModels, daForce, daEMG


def read_noraxon_ezc3d_xr(
    file: str | Path,
    section: str | None = None,
    n_vars_load: List[str] | None = None,
    coincidence: str = "similar",
) -> xr.DataArray:

    print("This function has not been tested yet.")

    try:
        import ezc3d
    except:
        raise ImportError(
            "Module ezc3d not installed.\nInstall with pip install ezc3d or conda install -c conda-forge ezc3d"
        )

    if section not in [
        "Trajectories",
        "Model Outputs",
        "EMG",
    ]:  # not ('Trajectories' in section or 'Model Outputs'in section or 'Forces' in section or 'EMG'in section):
        raise Exception(
            'Section header not found, try "Trajectories", "Model outputs" or "EMG"'
        )
        return

    timer = time.perf_counter()  # inicia el contador de tiempo

    # se asegura de que la extensión es c3d
    file = file.with_suffix(".c3d")

    try:
        timerSub = time.perf_counter()  # inicia el contador de tiempo
        # print(f'Loading section {section}, file: {file.name}')

        acq = ezc3d.c3d(file.as_posix())

        # Trajectiories and Modeled outputs
        if section in ["Trajectories", "Model Outputs"]:
            freq = acq["parameters"]["POINT"]["RATE"]["value"][0]

            labels = acq["parameters"]["POINT"]["LABELS"]["value"]
            data = acq["data"]["points"][:3, :, :]

            coords = {
                "axis": ["x", "y", "z"],
                "n_var": labels,
                "time": np.arange(data.shape[-1]) / freq,
            }
            da = (
                xr.DataArray(
                    data,  # =np.expand_dims(data, axis=0),
                    dims=coords.keys(),
                    coords=coords,
                    name="Trajectories",
                    attrs={
                        "freq": freq,
                        "units": "mm",
                    },
                )
                .transpose("n_var", "axis", "time")
                .dropna("time", how="all")  # para quitar los ceros cuano no hay dato
            )

            if "Trajectories" in section:
                # Delete unnamed trajectories and modeled outputs
                da = da.sel(
                    n_var=(
                        ~da.n_var.str.startswith("*") & ~da.n_var.str.contains("USERMO")
                    )
                )
            elif "Model Outputs" in section:
                da = da.sel(n_var=da.n_var.str.contains("USERMO"))

        # Analogs
        elif section in ["Forces", "EMG"]:  # ('Forces' in section or 'EMG' in section):
            freq = acq["parameters"]["ANALOG"]["RATE"]["value"][0]

            labels = acq["parameters"]["ANALOG"]["LABELS"]["value"]

            data = acq["data"]["analogs"][0]

            # data_analog.shape
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

            # Forces
            # Sometimes contains 'Force' and others 'Fx', Fy', 'Fz'
            # Get only force variables
            if section == "Forces":
                if da.n_var.str.contains("Force").any():  # new versions of Nexus?
                    da = da.sel(n_var=da.n_var.str.contains("Force"))
                elif da.n_var.str.contains("Fz").any():  # old versions of Nexus?
                    da = da.sel(n_var=da.n_var.str.startswith("F"))
                else:
                    da = xr.DataArray()
                    raise Exception("Apparently no force data in file")
                # if da.n_var.str.startswith('F').any():#da.n_var.str.contains('Force').any(): #'Force' in da.n_var:
                try:
                    # da = da.sel(n_var=da.n_var.str.startswith('F')) #'Force') #.sel(n_var=da_analog.n_var.str.contains('Force'))
                    if len(da.n_var) == 3:  # 1 platform
                        x = da.isel(n_var=0)
                        y = da.isel(n_var=1)
                        z = da.isel(n_var=2)
                        da = (
                            xr.concat([x, y, z], dim="axis")
                            .assign_coords(n_var="plat1")
                            .assign_coords(axis=["x", "y", "z"])
                            .expand_dims({"n_var": 1})
                        )
                    elif len(da.n_var) == 6:  # 2 platforms
                        x = da.isel(n_var=[0, 3]).assign_coords(
                            n_var=["plat1", "plat2"]
                        )
                        y = da.isel(n_var=[1, 4]).assign_coords(
                            n_var=["plat1", "plat2"]
                        )
                        z = da.isel(n_var=[2, 5]).assign_coords(
                            n_var=["plat1", "plat2"]
                        )
                        da = (
                            xr.concat([x, y, z], dim="axis")
                            # .assign_coords(n_var=['plat1', 'plat2'])
                            .assign_coords(axis=["x", "y", "z"])
                        )
                    elif len(da.n_var) == 8:  # Bioware 1 plat
                        da.attrs["description"] = "Data from Bioware 1 plat"
                    else:
                        raise Exception("The number of Force variables is not 3 or 6")

                    da.attrs["units"] = "N"
                    # da.time.attrs['units']='s'
                    # da.plot.line(x='time', col='axis', hue='n_var')
                except:
                    da = xr.DataArray()
                    raise Exception("Not available force data in file")

            # EMG
            elif section == "EMG":
                if da.n_var.str.contains("EMG").any():
                    da = da.sel(n_var=da.n_var.str.contains("EMG"))
                    da.attrs["units"] = "mV"
                    # da.n_var.sortby('n_var')
                    # da.plot.line(x='time', col='n_var', col_wrap=3)
                else:
                    da = xr.DataArray()
                    raise Exception("No EMG data in file")

        da.time.attrs["units"] = "s"
        da.name = section

        # print('Tiempo {0:.3f} s \n'.format(time.perf_counter()-timerSub))

    except Exception as err:
        da = xr.DataArray()
        raise Exception(f"\nATTENTION. Unable to process {file.name}, {err}\n")

    if n_vars_load is not None and "n_var" in da.coords:
        da = da.sel(n_var=n_vars_load)

    return da  # daTrajec, daModels, daForce, daEMG


# =============================================================================
# %% MAIN
# =============================================================================
if __name__ == "__main__":

    file = Path(
        r"F:\Investigacion\Proyectos\Tesis\TesisCoralPodologa\Registros\PilotoNoraxon\S000\2024-03-08-10-46_PO_S000_MA_001.c3d"
    )
    daTrajec = read_noraxon_c3d_xr(file, section="EMG", engine="c3d")
    daTrajec = read_noraxon_ezc3d_xr(file, section="EMG")

    ruta_archivo = Path(
        r"F:\Investigacion\Proyectos\Tesis\TesisCoralPodologa\Registros\PilotoNoraxon\S000"
    )
    file = ruta_archivo / "2024-03-08-10-43_PO_S000_carrera_001.c3d"

    daTrajec = read_noraxon_c3d_xr(file, section="EMG")
    daTrajec.isel(n_var=slice(6)).plot.line(x="time", col="n_var", col_wrap=3)
