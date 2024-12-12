# -*- coding: utf-8 -*-
"""
Created on Mon Oct 02 16:36:37 2023

@author: Jose L. L. Elvira

Los .c3d del Bioware exportan solo los datos de los sensores por separado,
8 por plataforma.
"""
from __future__ import division, print_function

import numpy as np
import pandas as pd
import xarray as xr

try:
    import c3d
except:
    print("Module c3d not installed")

try:
    import ezc3d
except:
    print("Module ezc3d not installed")

import time

import warnings  # para quitar warnings de no encontrar points

# warnings.filterwarnings("ignore")

__author__ = "Jose Luis Lopez Elvira"
__version__ = "0.1.0"
__date__ = "10/12/2024"

"""
Modificaciones:
    10/12/2024, v0.1.0
            - Comprueba si está instalado c3d y ezc3d, si no, avisa con mensaje.
            - Incluida función con ezc3d, pero no carga la serie temporal completa.
    
    29/12/2023, v0.0.2
            - Calcula las fuerzas en los 3 ejes a partir de las raw de los sensores.

    02/10/2023, v0.0.1
            - Empezado a partir de read_vicon.c3d.
            
"""


# =============================================================================
# %% Carga trayectorias desde c3d
# =============================================================================
def read_kistler_c3d_xr(file, n_vars_load=None, coincidence="similar"):

    timer = time.perf_counter()  # inicia el contador de tiempo

    # se asegura de que la extensión es c3d
    file = file.with_suffix(".c3d")

    try:
        timerSub = time.perf_counter()  # inicia el contador de tiempo
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
                        print("Extracted %d point frames", len(points))

        labels_analog = [s.split(".")[0].replace(" ", "") for s in reader.analog_labels]
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

        da_analog.attrs["units"] = "N"
        da_analog.time.attrs["units"] = "s"

        # print("Tiempo {0:.3f} s \n".format(time.perf_counter() - timerSub))

    except Exception as err:
        print("\nATENCIÓN. No se ha podido procesar " + file.name, err, "\n")

    if n_vars_load:
        da_analog = da_analog.sel(n_var=n_vars_load)

    return da_analog

def read_kistler_ezc3d_xr(file, n_vars_load=None, coincidence="similar"):
    """
    PARECE QUE NO CARGA LA SERIE TEMPORAL COMPLETA
    """
    timer = time.perf_counter()  # inicia el contador de tiempo

    # se asegura de que la extensión es c3d
    file = file.with_suffix(".c3d")

    try:
        timerSub = time.perf_counter()  # inicia el contador de tiempo
        # print(f'Loading file: {file.name}')

        acq = ezc3d.c3d(file.as_posix()) #, extract_forceplat_data=True)

        freq = acq['parameters']['ANALOG']['RATE']['value'][0]
        
        labels = acq["parameters"]["ANALOG"]["LABELS"]["value"]
        
        data = acq["data"]['analogs'][0]
        data.shape
        
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
       
        da.attrs["units"] = "N"
        da.time.attrs["units"] = "s"

        print("Tiempo {0:.3f} s \n".format(time.perf_counter() - timerSub))

    except Exception as err:
        print("\nATENCIÓN. No se ha podido procesar " + file.name, err, "\n")

    if n_vars_load:
        da = da.sel(n_var=n_vars_load)

    return da



def split_plataforms(da):
    plat1 = da.sel(n_var=da.n_var.str.startswith("F1"))
    plat1 = plat1.assign_coords(n_var=plat1.n_var.str.lstrip("F1"))

    plat2 = da.sel(n_var=da.n_var.str.startswith("F2"))
    plat2 = plat2.assign_coords(n_var=plat2.n_var.str.lstrip("F2"))

    da = xr.concat([plat1, plat2], dim="plat").assign_coords(plat=[1, 2])

    return da


def separa_ejes(da):
    # NO ES NECESARIO CON COMPUTE_FORCES_AXES???
    # TODO: Falta quitarles la letra del eje en el nombre
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


def compute_forces_axes(da):
    # da=daForce

    if "plat" not in da.coords:
        da = split_plataforms(da)

    Fx = da.sel(n_var=da.n_var.str.contains("x")).sum(dim="n_var")
    Fy = da.sel(n_var=da.n_var.str.contains("y")).sum(dim="n_var")
    Fz = da.sel(n_var=da.n_var.str.contains("z")).sum(dim="n_var")

    daReturn = xr.concat([Fx, Fy, Fz], dim="axis").assign_coords(axis=["x", "y", "z"])
    # daReturn.plot.line(x='time', col='plat')

    return daReturn


def compute_moments_axes(da):
    # da=daForce
    raise Exception("Not implemented yet")
    """
    if 'plat' not in da.coords:
        da = split_plataforms(da)

    Fx = da.sel(n_var=da.n_var.str.contains('x')).sum(dim='n_var')
    Fy = da.sel(n_var=da.n_var.str.contains('y')).sum(dim='n_var')
    Fz = da.sel(n_var=da.n_var.str.contains('z')).sum(dim='n_var')
        
    daReturn = (xr.concat([Fx, Fy, Fz], dim='axis')
                .assign_coords(axis=['x', 'y', 'z'])
                )
    #daReturn.plot.line(x='time', col='plat')
    """
    return daReturn


# =============================================================================
# %% MAIN
# =============================================================================
if __name__ == "__main__":
    from pathlib import Path
    import sys

    sys.path.append(r"F:\Programacion\Python\Mios\Functions")
    from read_kistler_c3d import read_kistler_c3d_xr

    ruta_archivo = Path(
        r"F:\Investigacion\Proyectos\Saltos\2023PreactivacionSJ\DataCollection\S01\FeedbackFuerza"
    )
    file = ruta_archivo / "S01_CMJ_000.c3d"
    daForce = read_kistler_c3d_xr(file)
    daForce = split_plataforms(daForce)
    daForce = separa_ejes(daForce)
    daForce.sel(axis='z').plot.line(x="time", col="plat")

    daForce2 = read_kistler_ezc3d_xr(file)

    ruta_archivo = Path(
        r"F:\Investigacion\Proyectos\Saltos\PotenciaDJ\Registros\2023PotenciaDJ\S02"
    )
    file = ruta_archivo / "S02_DJ_30_002.c3d"
    daForce = read_kistler_c3d_xr(file)
    # daForce = split_plataforms(daForce)
    daForce = compute_forces_axes(daForce)
    daForce.plot.line(x="time", col="plat")

    daForce2 = read_kistler_ezc3d_xr(file) # CARGA SÓLO UNA PARTE INICIAL
    # daForce = split_plataforms(daForce)
    daForce2 = compute_forces_axes(daForce2)
    daForce2.plot.line(x="time", col="plat")

    daForce2==daForce
    daForce2.plot.line(x="time")

