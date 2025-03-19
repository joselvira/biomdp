# -*- coding: utf-8 -*-
"""
Created on Sun Mar 09 11:12:37 2025

@author: Jose L. L. Elvira

Read data from Kistler Bioware .txt exported files.
"""


# =============================================================================
# %% LOAD LIBRARIES
# =============================================================================

__author__ = "Jose L. L. Elvira"
__version__ = "0.1.0"
__date__ = "09/03/2025"

"""
Updates:
    09/03/2025, v0.1.0
        - Version imported from jump_forces_utils.

"""

from typing import List
import numpy as np
import pandas as pd
import xarray as xr
import polars as pl

import time
from pathlib import Path


# =============================================================================
# %% Functions
# =============================================================================


# Carga un archivo de Bioware como dataframe de Polars
def read_bioware_pl(
    file: str | Path,
    lin_header: int = 17,
    n_vars_load: List[str] | None = None,
    to_dataarray: bool = False,
) -> pl.DataFrame | xr.DataArray:
    df = (
        pl.read_csv(
            file,
            has_header=True,
            skip_rows=lin_header,
            skip_rows_after_header=1,
            columns=n_vars_load,
            separator="\t",
        )  # , columns=nom_vars_cargar)
        # .slice(1, None) #quita la fila de unidades (N) #no hace falta con skip_rows_after_header=1
        # .select(pl.col(n_vars_load))
        # .rename({'abs time (s)':'time'}) #'Fx':'x', 'Fy':'y', 'Fz':'z',
        #          #'Fx_duplicated_0':'x_duplicated_0', 'Fy_duplicated_0':'y_duplicated_0', 'Fz_duplicated_0':'z'
        #          })
    ).with_columns(pl.all().cast(pl.Float64()))

    # ----Transform polars to xarray
    if to_dataarray:
        x = df.select(pl.col("^*Fx$")).to_numpy()
        y = df.select(pl.col("^*Fy$")).to_numpy()
        z = df.select(pl.col("^*Fz$")).to_numpy()
        data = np.stack([x, y, z])
        freq = 1 / (df[1, "abs time (s)"] - df[0, "abs time (s)"])
        ending = -3
        coords = {
            "axis": ["x", "y", "z"],
            "time": np.arange(data.shape[1]) / freq,
            "n_var": ["Force"],  # [x[:ending] for x in df.columns if 'x' in x[-1]],
        }
        da = (
            xr.DataArray(
                data=data,
                dims=coords.keys(),
                coords=coords,
            )
            .astype(float)
            .transpose("n_var", "axis", "time")
        )
        da.name = "Forces"
        da.attrs["freq"] = freq
        da.time.attrs["units"] = "s"
        da.attrs["units"] = "N"

        return da

    return df


def load_bioware_arrow(
    file: str | Path,
    lin_header: int = 17,
    n_vars_load: List[str] | None = None,
    to_dataarray: bool = False,
) -> pd.DataFrame:
    """In test, at the moment it does not work when there are repeated cols"""
    from pyarrow import csv

    read_options = csv.ReadOptions(
        # column_names=['Fx', 'Fy', 'Fz'],
        skip_rows=lin_header,
        skip_rows_after_names=1,
    )
    parse_options = csv.ParseOptions(delimiter="\t")
    data = csv.read_csv(file, read_options=read_options, parse_options=parse_options)
    return data.to_pandas()


def load_bioware_pd(
    file: str | Path,
    lin_header: int = 17,
    n_vars_load: List[str] | None = None,
    to_dataarray: bool = False,
) -> pd.DataFrame | xr.DataArray:

    df = (
        pd.read_csv(
            file,
            header=lin_header,
            usecols=n_vars_load,  # ['Fx', 'Fy', 'Fz', 'Fx.1', 'Fy.1', 'Fz.1'], #n_vars_load,
            # skiprows=18,
            delimiter="\t",
            # dtype=np.float64,
            engine="c",  # "pyarrow" con pyarrow no funciona bien de momento cargar columnas con nombre repetido,
        ).drop(index=0)
        # , columns=nom_vars_cargar)
        # .slice(1, None) #quita la fila de unidades (N) #no hace falta con skip_rows_after_header=1
        # .select(pl.col(n_vars_load))
        # .rename({'abs time (s)':'time'}) #'Fx':'x', 'Fy':'y', 'Fz':'z',
        #          #'Fx_duplicated_0':'x_duplicated_0', 'Fy_duplicated_0':'y_duplicated_0', 'Fz_duplicated_0':'z'
        #          })
    )
    # df.dtypes

    # ----Transform pandas to xarray
    if to_dataarray:
        x = df.filter(regex="Fx*")  # .to_numpy()
        y = df.filter(regex="Fy*")
        z = df.filter(regex="Fx*")
        data = np.stack([x, y, z])
        freq = 1 / (df.loc[2, "abs time (s)"] - df.loc[1, "abs time (s)"])
        ending = -3
        coords = {
            "axis": ["x", "y", "z"],
            "time": np.arange(data.shape[1]) / freq,
            "n_var": ["Force"],  # [x[:ending] for x in df.columns if 'x' in x[-1]],
        }
        da = (
            xr.DataArray(
                data=data,
                dims=coords.keys(),
                coords=coords,
            )
            .astype(float)
            .transpose("n_var", "axis", "time")
        )
        da.name = "Forces"
        da.attrs["freq"] = freq
        da.time.attrs["units"] = "s"
        da.attrs["units"] = "N"

        return da

    return df


def split_plataforms(da: xr.DataArray) -> xr.DataArray:
    plat1 = da.sel(n_var=da.n_var.str.startswith("F1"))
    plat1 = plat1.assign_coords(n_var=plat1.n_var.str.lstrip("F1"))

    plat2 = da.sel(n_var=da.n_var.str.startswith("F2"))
    plat2 = plat2.assign_coords(n_var=plat2.n_var.str.lstrip("F2"))

    da = xr.concat([plat1, plat2], dim="plat").assign_coords(plat=[1, 2])

    return da


def split_axis(da: xr.DataArray) -> xr.DataArray:
    # NOT NECESSARY WITH COMPUTE_FORCES_AX???
    # TODO: The letter of the axis in the name must be removed
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

    if "plat" not in da.coords:
        da = split_plataforms(da)

    Fx = da.sel(n_var=da.n_var.str.contains("x")).sum(dim="n_var")
    Fy = da.sel(n_var=da.n_var.str.contains("y")).sum(dim="n_var")
    Fz = da.sel(n_var=da.n_var.str.contains("z")).sum(dim="n_var")

    daReturn = xr.concat([Fx, Fy, Fz], dim="axis").assign_coords(axis=["x", "y", "z"])
    # daReturn.plot.line(x='time', col='plat')

    return daReturn


def compute_moments_axes(da: xr.DataArray) -> xr.DataArray:
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

    from biomdp.read_kistler_c3d import read_kistler_c3d_xr, read_kistler_ezc3d_xr

    ruta_archivo = Path(
        r"F:\Investigacion\Proyectos\Saltos\2023PreactivacionSJ\DataCollection\S01\FeedbackFuerza"
    )
    file = ruta_archivo / "S01_CMJ_000.c3d"
    daForce = read_kistler_c3d_xr(file)
    daForce = split_plataforms(daForce)
    daForce = split_axis(daForce)
    daForce.sel(axis="z").plot.line(x="time", col="plat")

    daForce2 = read_kistler_c3d_xr(file)  # read_kistler_ezc3d_xr(file)

    ruta_archivo = Path(
        r"F:\Investigacion\Proyectos\Saltos\PotenciaDJ\Registros\2023PotenciaDJ\S02"
    )
    file = ruta_archivo / "S02_DJ_30_002.c3d"
    daForce = read_kistler_c3d_xr(file)  # read_kistler_c3d_xr(file)
    # daForce = split_plataforms(daForce)
    daForce = compute_forces_axes(daForce)
    daForce.plot.line(x="time", col="plat")

    daForce2 = read_kistler_c3d_xr(
        file
    )  # read_kistler_ezc3d_xr(file)  # ezc3d CARGA SÓLO UNA PARTE INICIAL
    # daForce = split_plataforms(daForce)
    daForce2 = compute_forces_axes(daForce2)
    daForce2.plot.line(x="time", col="plat")

    daForce2 == daForce
    daForce2.plot.line(x="time")
