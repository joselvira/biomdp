# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 13:05:37 2019

@author: Jose L. L. Elvira
"""

"""Loads data files exported from the Vicon Nexus"""

# =============================================================================
# %% LOAD LIBRARIES
# =============================================================================

__author__ = "Jose L. L. Elvira"
__version__ = "v4.1.2"
__date__ = "10/03/2025"

"""
Updates:
    10/03/2025, v4.1.2
        - Adapted to biomdp with translations. 

    06/03/2025, v.4.1.1
        - Included type specifications in functions.
    
    25/11/2023, v.4.1.0
        - La función read_vicon_csv_pl admite el parámetro to_dataarray 
          para decidir si se devuelve un xr.DataArray o un polars dataframe.
    
    07/05/2023, v.4.0.4
        - Mejorada lectura con polars
    
    22/04/2023, v.4.0.3
        - Corregido que n Polars actual (0.17.2) no coindicen los saltos de
          línea cuando empieza con línea en blanco.
        - Optimizado código de lectura con Polars.
          
    17/04/2023, v.4.0.2
        - Corregido error nombre variable archivo en read_vicon_csv_pl.
        - Optimizada parte lectura con open en versión Polars.
        - Puede cargar por separado datos EMG o Forces, aunque vayan en el
          mismo archivo.
        - Cambiada nombre variables a n_var por conflicto en xarray, .var
          calcula la varianza.
        - En función con Polars, La selección de variables la hace en el df
          Polars, no en el dataarray.

    27/03/2023, v.4.0.0
        - Incluida una versión que lee con Polars y lo pasa a DataArray
          directamente (read_vicon_csv_pl), mucho más rápida.
        - La versión que lee con Polars es capaz de leer csv con encabezados
          repetidos.
          
    07/06/2022, v.3.0.0
        - Intento de corrección cuando tiene que leer csv con variables EMG
          modeladas, que se interfieren entre las xyz.
        
    09/04/2022, v.2.3.0
        - Habilitado para cargar como dataframe y dataArray EMG Noraxon en modo Devices.
        - Incluye el tiempo en una columna.

    29/03/2021, v2.1.1
        - Incluido parámetro 'header_format' para que devuelva el encabezado como 'flat' en una sola línea (variable_x, variable_y, ...) o en dos líneas ((variable,x), (variable,y), ...).
            
    28/03/2021, v2.1.1
        - Mejorada lectura con Pandas. Ahora puede cargar archivos que empiezan sin datos en las primeras líneas.

	21/03/2021, v2.1.0
        - Cambiado lector del bloque de archivos por pd.read_csv con número de columnas delimitado a los que carga en las variables (quitando los de velocidad y aceleración)
        - Solucionado fallo al leer frecuencia cuando terminaba la línea rellenando con separadores (como al exportar en Excel)

    10/01/2021, v2.0.1
        - Ajustado para que pueda devolver xArray con Model Outputs

    13/12/2020, v2.0.0
        - Con el argumento to_dataarray se puede pedir que devuelva los datos en formato xArray
"""
from typing import List, Any
import numpy as np

# import pandas as pd
import xarray as xr

from pathlib import Path


# import scipy.signal

import os


# =============================================================================
# %% Functions
# =============================================================================


def read_vicon_csv(
    file: str | Path,
    section: str = "Model Outputs",
    n_vars_load: List[str] | None = None,
    coincidence: str = "similar",
    sep: str = ",",
    to_dataarray: bool = False,
    engine="polars",
    raw: bool = False,
) -> Any:

    if not file.exists():
        raise FileNotFoundError(f"File {file} not found")

    # Ensures that file is a csv file
    file = file.with_suffix(".csv")

    if engine == "polars":
        da = read_vicon_csv_pl(
            file,
            section=section,
            n_vars_load=n_vars_load,
            coincidence=coincidence,
            sep=sep,
            to_dataarray=to_dataarray,
            raw=raw,
        )
    elif engine == "polars2":
        da = read_vicon_csv_pl2(
            file,
            section=section,
            n_vars_load=n_vars_load,
            coincidence=coincidence,
            sep=sep,
            raw=raw,
        )
    elif engine == "pandas":
        da = read_vicon_csv_pd(
            file,
            section=section,
            sep=sep,
            n_vars_load=n_vars_load,
            raw=True,
        )
    else:
        raise ValueError(f"Engine {engine} not valid\nTry with 'polars' or 'pandas'")

    return da


def read_vicon_csv_pl(
    file: str | Path,
    section: str = "Model Outputs",
    n_vars_load: List[str] | None = None,
    coincidence: str = "similar",
    sep: str = ",",
    to_dataarray: bool = True,
    raw: bool = False,
) -> xr.DataArray:
    """
    Reads data from a Vicon csv file and returns a Polars DataFrame or xarray DataArray

    Parameters
    ----------
    file : string or path of the file
        DESCRIPTION.
    section : string, optional
        Kind of data variables to load.
        Options: 'Trajectories', 'Model Outputs', 'Forces', 'EMG', 'Model Outputs EMG'
        The default is 'Model Outputs'.
    n_vars_load : list, optional
        DESCRIPTION. The default is None.
    coincidence: string
        When selecting which variables to load, allows strings containing the
        indicated or forces to be exact.
        Options: 'similar', 'exact'. The default is 'similar'.
    sep : string, optional
        Separator used in the csv file. The default is ','.
    to_dataarray : bool, optional
        Transforms data to dataarray, otherway is polars dataframe. The default is True.

    Returns
    -------
    Xarray DataArray.

    """

    try:
        import polars as pl
    except:
        raise ImportError(
            "Polars package not instaled. Install it if you want to use the accelerated version"
        )

    if section in ["Forces", "EMG"]:
        n_block = "Devices"
    elif section == "Model Outputs EMG":
        n_block = "Model Outputs"
    elif section in ["Trajectories", "Model Outputs"]:
        n_block = section
    else:
        raise ValueError(
            f"Section '{section}' not recognized\nTry with 'Trajectories', 'Model Outputs', 'Forces', 'EMG', 'Model Outputs EMG'"
        )

    # ----Check for blank lines. In current Polars (0.17.2) line breaks do not match when starting with blank line
    # Vicon Nexus writes a first empty line when export Events is selected but thre are no events
    with open(file, mode="rt") as f:
        offset_blank_ini = 0
        while f.readline() in ["\r\n", "\n", "ï»¿\n", r"\n", "\ufeff"]:
            # print("blank line")
            offset_blank_ini += 1
    # Second Check for blank lines (some times the previous one does not work)
    if offset_blank_ini == 0:
        with open(file, mode="rt") as f:
            line = f.readline()
            if not any(
                filter(
                    lambda x: x in line,
                    ["Events", "Trajectories", "Model Outputs", "Devices"],
                )
            ):
                offset_blank_ini += 1

    # ----Search section position and length
    ini_section = None
    end_section = None
    with open(file, mode="rt") as f:
        num_lin = 0

        # Scrolls through the entire file to find the start and end of the section and the number of lines
        for line in f:
            # Search for section start label
            if ini_section is None and n_block in line:
                ini_section = num_lin
                # The frequency is below the section label
                freq = int(
                    f.readline().replace(sep, "")
                )  # removes the separator for cases where the file has been saved with Excel (full line with separator)

                # #Load columns names
                # n_head = str(f.readline()[:-1]).split(sep) #variable names
                # n_subhead = str(f.readline()[:-1]).lower().split(sep) #coordinate names (x,y,z)
                # num_lin+=3
                num_lin += 1
            # When start found, search the end
            if ini_section is not None and end_section is None and line == "\n":
                end_section = num_lin - 1
                break

            num_lin += 1
        # file_end = num_lin

    if ini_section is None:
        raise Exception(f"Section header '{section}' not found")

    # ----Load data from file
    # pl.read_csv(file, truncate_ragged_lines=True)
    df = pl.read_csv(
        file,
        has_header=True,
        skip_rows=ini_section + 2 - offset_blank_ini,
        n_rows=end_section - ini_section - 2,
        truncate_ragged_lines=True,
        separator=sep,
        # columns=range(len(n_vars_merged)),
        # new_columns=n_vars_merged, separator=sep
    )

    n_head = df.columns
    n_subhead = list(np.char.lower(df.slice(0, 1).to_numpy()[0].astype(str)))

    # Remove subhead and units rows
    df = df.slice(2, None)

    if n_block in ["Trajectories", "Model Outputs", "Model Outputs EMG"]:
        n_vars_merged = ["Frame", "Sub Frame"]
        for i in range(2, len(n_subhead)):
            if (
                n_subhead[i] in "x"
                and "'" not in n_subhead[i]
                and "''" not in n_subhead[i]
            ):
                # print(n_subhead[i], n_head[i])
                n_vars_merged.append(n_head[i].split(":")[1] + "_" + n_subhead[i])  # X
                n_vars_merged.append(
                    n_head[i].split(":")[1] + "_" + n_subhead[i + 1]
                )  # Y
                n_vars_merged.append(
                    n_head[i].split(":")[1] + "_" + n_subhead[i + 2]
                )  # Z
            elif "emg" in n_subhead[i]:
                # print(n_subhead[i], n_head[i])
                n_vars_merged.append(n_head[i].split(":")[1] + "_" + n_subhead[i])

        # Rename headers
        df = df.rename(dict(zip(n_head, n_vars_merged)))

        if section == "Model Outputs EMG":
            df = df.select(pl.col("^*_emg.*$"))
            df = (
                df.rename(dict(zip(df.columns, [n[:-3] for n in df.columns]))).select(
                    pl.exclude("^*_duplicate.*$")
                )  # remove duplicates
                # .slice(2, None) #remove subhead and units rows
            )
        else:  # Trajectories and Model Outputs
            df = (
                df.select(pl.exclude(["Frame", "Sub Frame"]))
                .select(pl.exclude("^*_duplicate.*$"))  # remove duplicates
                .select(pl.exclude("^*_emg.*$"))  # remove 1D EMG variables
                # .slice(2, None) #remove subhead and units rows
            )

    elif n_block == "Devices":
        if section == "EMG":
            # n_head = n_head[:len(n_subhead)] #ajusta el número de variables
            # n_vars_merged = rename_duplicates(n_head)
            # selection = [s for s in n_vars_merged[2:] if 'EMG' in s]
            # selection2 = selection

            df = (
                df.select(pl.col("^*EMG.*$")).select(
                    pl.exclude("^*_duplicate.*$")
                )  # remove duplicates
                # .slice(2, None) #remove subhead and units rows
            )

        elif section == "Forces":
            n_vars_merged = ["Frame", "Sub Frame"]
            for i in range(2, len(n_subhead)):
                if (
                    "x" in n_subhead[i]
                    and "'" not in n_subhead[i]
                    and "''" not in n_subhead[i]
                ):
                    # print(n_subhead[i], n_head[i])
                    n_vars_merged.append(n_head[i] + "_" + n_subhead[i])  # X
                    n_vars_merged.append(n_head[i] + "_" + n_subhead[i + 1])  # Y
                    n_vars_merged.append(n_head[i] + "_" + n_subhead[i + 2])  # Z
                elif "v" in n_subhead[i]:
                    n_vars_merged.append(n_head[i] + "_" + n_subhead[i])

            df = (
                df.rename(dict(zip(df.columns, n_vars_merged)))
                .select(pl.exclude("^*EMG.*$"))
                .select(pl.exclude("^*_duplicate.*$"))  # remove duplicates
                # .slice(2, None) #remove subhead and units rows
            )

    # ----Filter variables
    if n_vars_load is not None:
        if not isinstance(n_vars_load, list):
            n_vars_load = [n_vars_load]  # in case a string is passed
        if coincidence == "similar":
            selection = [s for s in df.columns if any(xs in s for xs in n_vars_load)]
        elif coincidence == "exact":
            selection = n_vars_load
        df = df.select(pl.col(selection))

    if raw:  # keep Polars dataframe format
        # Add time column
        df = df.with_columns(
            pl.lit(np.arange(len(df)) / freq).alias("time"),
            # pl.all()
        )
        return df

    else:  # ----Transform polars to xarray
        if section in ["Trajectories", "Model Outputs", "Forces"]:
            # Decompose on its axes
            x = df.select(pl.col("^*x$")).to_numpy()
            y = df.select(pl.col("^*y$")).to_numpy()
            z = df.select(pl.col("^*z$")).to_numpy()
            data = np.stack([x, y, z])

            ending = -3 if section == "Forces" else -2
            coords = {
                "axis": ["x", "y", "z"],
                "time": np.arange(data.shape[1]) / freq,
                "n_var": [x[:ending] for x in df.columns if "x" in x[-1]],
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

        elif section in ["EMG", "Model Outputs EMG"]:
            data = df.to_numpy().T
            coords = {
                "n_var": df.columns,  # selection2,
                "time": np.arange(df.shape[0]) / freq,
            }
            da = xr.DataArray(
                data=data,
                dims=coords.keys(),
                coords=coords,
            ).astype(float)

        da.name = section
        da.attrs["freq"] = freq
        da.time.attrs["units"] = "s"
        if section == "Trajectories":
            da.attrs["units"] = "mm"
        elif "EMG" in section:
            da.attrs["units"] = "V"
        elif "Forces" in section:
            da.attrs["units"] = "N"

        return da


def rename_duplicates(names: List[str]):
    cols = pd.Series(names)
    if len(names) != len(cols.unique()):
        for dup in cols[cols.duplicated()].unique():
            cols[cols[cols == dup].index.values.tolist()] = [
                dup + "." + str(i) if i != 0 else dup for i in range(sum(cols == dup))
            ]
        names = list(cols)
    return names


# -----------------------------------------------------------------------------
# EXPÈRIMENTAL
def read_vicon_csv_pl2(
    file: str | Path,
    section: str = "Model Outputs",
    n_vars_load: List[str] | None = None,
    coincidence: str = "similar",
    sep: str = ",",
    raw: bool = False,
) -> xr.DataArray:
    """
    Reads data from a Vicon csv file and returns a Polars DataFrame or xarray DataArray

    Parameters
    ----------
    file : string or path of the file
        DESCRIPTION.
    section : string, optional
        Kind of data variables to load.
        Options: 'Trajectories', 'Model Outputs', 'Forces', 'EMG', 'Model Outputs EMG'
        The default is 'Model Outputs'.
    n_vars_load : list, optional
        DESCRIPTION. The default is None.
    coincidence: string
        When selecting which variables to load, allows strings containing the
        indicated or forces to be exact.
        Options: 'similar', 'exact'. The default is 'similar'.
    sep : string, optional
        Separator used in the csv file. The default is ','.
    to_dataarray : bool, optional
        Transforms data to dataarray, otherway is polars dataframe. The default is True.

    Returns
    -------
    Xarray DataArray.

    """
    print("alternativo")
    if section in ["Forces", "EMG"]:
        n_block = "Devices"
    elif section == "Model Outputs EMG":
        n_block = "Model Outputs"
    elif section in ["Trajectories", "Model Outputs"]:
        n_block = section
    else:
        raise ValueError(
            f"Section '{section}' not recognized\nTry with 'Trajectories', 'Model Outputs', 'Forces', 'EMG', 'Model Outputs EMG'"
        )

    # Ensures that file is a csv file
    file = file.with_suffix(".csv")

    # ----Check for blank lines. In current Polars (0.17.2) line breaks do not match when starting with blank line
    offset_blank_ini = 0
    with open(file, mode="rt") as f:
        while f.readline() in ["\r\n", "\n", "ï»¿\n", r"\n", "\ufeff"]:
            # print("blank line")
            offset_blank_ini += 1
    # Second Check for blank lines (some times the previous one does not work)
    if offset_blank_ini == 0:
        with open(file, mode="rt") as f:
            line = f.readline()
            # print(line)
            if not any(
                filter(
                    lambda x: x in line,
                    ["Events", "Trajectories", "Model Outputs", "Devices"],
                )
            ):
                offset_blank_ini += 1

    # ----Search section position and length
    ini_section = None
    end_section = None
    with open(file, mode="rt") as f:
        num_lin = 0

        # Scrolls through the entire file to find the start and end of the section and the number of lines
        for line in f:
            # Search for section start label
            if ini_section is None and n_block in line:
                ini_section = num_lin
                # The frequency is below the section label
                freq = int(
                    f.readline().replace(sep, "")
                )  # removes the separator for cases where the file has been saved with Excel (full line with separator)

                # Load columns names
                n_head = str(f.readline()[:-1]).split(sep)  # variable names
                n_subhead = (
                    str(f.readline()[:-1]).lower().split(sep)
                )  # coordinate names (x,y,z)
                num_lin += 3

            # When start found, search the end
            if ini_section is not None and end_section is None and line == "\n":
                end_section = num_lin - 1
                break
            num_lin += 1
        # file_end = num_lin

    if ini_section is None:
        raise Exception("Section header not found")

    # ----Assign header labels
    if n_block == "Devices":
        if section == "EMG":
            n_head = n_head[: len(n_subhead)]  # adjust number of variables
            n_vars_merged = rename_duplicates(n_head)
            selection = [s for s in n_vars_merged[2:] if "EMG" in s]
            selection2 = selection

        elif section == "Forces":
            n_vars_merged = [h + "_" + sh for h, sh in zip(n_head, n_subhead)]
            n_vars_merged = rename_duplicates(n_vars_merged)
            selection = [s for s in n_vars_merged[2:] if "EMG" not in s]
            selection2 = []
            for i in range(2, len(n_subhead)):
                if "x" in n_subhead[i] and "EMG" not in n_subhead[i]:
                    # print(n_subhead[i], n_head[i])
                    selection2.append(n_head[i] + "_" + n_subhead[i][-1])  # x
                    selection2.append(n_head[i] + "_" + n_subhead[i + 1][-1])  # y
                    selection2.append(n_head[i] + "_" + n_subhead[i + 2][-1])  # z

    elif n_block in ["Trajectories", "Model Outputs", "Model Outputs EMG"]:
        n_vars_merged = ["Frame", "Sub Frame"]
        for i in range(2, len(n_subhead)):
            if (
                n_subhead[i] in "xX"
                and "'" not in n_subhead[i]
                and "''" not in n_subhead[i]
            ):
                # print(n_subhead[i], n_head[i])
                n_vars_merged.append(n_head[i].split(":")[1] + "_" + n_subhead[i])  # X
                n_vars_merged.append(
                    n_head[i].split(":")[1] + "_" + n_subhead[i + 1]
                )  # Y
                n_vars_merged.append(
                    n_head[i].split(":")[1] + "_" + n_subhead[i + 2]
                )  # Z
            elif "emg" in n_subhead[i]:
                # print(n_subhead[i], n_head[i])
                n_vars_merged.append(n_head[i].split(":")[1] + "_" + n_subhead[i])
        selection = set(n_vars_merged[2:])  # remove duplicates
        if section == "Model Outputs EMG":
            selection = sorted([s for s in selection if "_emg" in s])
            selection2 = [s[:-4] for s in selection]
        else:
            selection = sorted([s for s in selection if "_emg" not in s])
            selection2 = selection
        n_vars_merged = rename_duplicates(n_vars_merged)

    # ----Load data from file
    df = (
        pl.read_csv(
            file,
            has_header=False,
            skip_rows=ini_section + 3 - offset_blank_ini,
            n_rows=end_section - ini_section - 2,
            columns=range(len(n_vars_merged)),
            new_columns=n_vars_merged,
            separator=sep,
        )
        .select(pl.col(selection))
        .rename(dict(zip(selection, selection2)))
        .slice(2, None)  # remove subhead and units rows
    )

    # ----Filter variables
    if n_vars_load:
        if not isinstance(n_vars_load, list):
            n_vars_load = [n_vars_load]  # in case a string is passed
        if coincidence == "similar":
            selection2 = [s for s in selection2 if any(xs in s for xs in n_vars_load)]
        elif coincidence == "exact":
            selection2 = n_vars_load
        df = df.select(pl.col(selection2))

    if raw:  # keep Polars dataframe format
        # Add time column
        df = df.with_columns(
            pl.lit(np.arange(len(df)) / freq).alias("time"),
            # pl.all()
        )
        return df

    else:  # ----Transform polars to xarray
        if section in ["EMG", "Model Outputs EMG"]:
            data = df.to_numpy().T
            coords = {
                "n_var": selection2,
                "time": np.arange(df.shape[0]) / freq,
            }
            da = xr.DataArray(
                data=data,
                dims=coords.keys(),
                coords=coords,
            ).astype(float)

        elif section in ["Trajectories", "Model Outputs", "Forces"]:
            # Decompose on its axes. Those that end in the coordinate or if they are repeated
            x = df.select(pl.col("^*_x|_x.$")).to_numpy()
            y = df.select(pl.col("^*_y|_y.$")).to_numpy()
            z = df.select(pl.col("^*_z|_z.*$")).to_numpy()
            data = np.stack([x, y, z])

            ending = -3 if section == "Forces" else -2
            coords = {
                "axis": ["x", "y", "z"],
                "time": np.arange(data.shape[1]) / freq,
                "n_var": [x[:ending] for x in df.columns if "_x" in x or "_X" in x],
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

        da.name = section
        da.attrs["freq"] = freq
        da.time.attrs["units"] = "s"
        if section == "Trajectories":
            da.attrs["units"] = "mm"
        elif "EMG" in section:
            da.attrs["units"] = "V"
        elif "Forces" in section:
            da.attrs["units"] = "N"
        return da


# -----------------------------------------------------------------------------


def read_vicon_csv_pd(
    file: str | Path,
    section: str = "Model Outputs",
    sep: str = ",",
    return_freq: bool = False,
    to_dataarray: bool = False,
    n_vars_load: List[str] | None = None,
    header_format: str = "flat",
    raw: bool = False,
):
    """
    Reads data from a Vicon csv file and returns a Polars DataFrame or xarray DataArray

    Parameters
    ----------
    file : str or Path
        Path to the csv file.
    section : str, optional
        Section to be read. Options are 'Trajectories', 'Model Outputs', 'Forces', 'EMG'.
        Defaults to 'Model Outputs'.
    sep : str, optional
        Separator used in the csv file. Defaults to ','.
    return_freq : bool, optional
        If True, it returns the sampling frequency of the data. Defaults to False.
    to_dataarray : bool, optional
        If True, it returns a xarray DataArray. Defaults to False.
    header_format : str, optional
        Format of the header. Options are 'flat' and 'hierarchical'. Defaults to 'flat'.

    Returns
    -------
    df : Polars DataFrame
        Data with columns as variables and rows as time.
    da : xarray DataArray
        Data with dimensions (n_var, axis, time). Only returned if `to_dataarray` is True.
    freq : float
        Sampling frequency of the data. Only returned if `return_freq` is True.

    Examples
    --------
    >>> dfData = read_vicon_csv_pd(file_name, section='Model Outputs')
    >>> dfData, frecuencia = read_vicon_csv_pd(file_name, section='Trajectories', return_freq=True)

    >>> #Con formato dataarray de xArray
    >>> daDatos = read_vicon_csv_pd(file_name, section='Trajectories', to_dataarray=True)

    """

    try:
        import pandas as pd
    except:
        raise ImportError("Pandas package not instaled")

    with open(file, mode="rt") as f:
        num_line = 0
        # Search section label
        line = f.readline()
        while section not in line:
            if line == "":
                raise Exception(f"Section header '{section}' not found")

            num_line += 1
            line = f.readline()

        ini_section = num_line

        # Frequency after the section label
        line = f.readline()
        freq = int(
            line.replace(sep, "")
        )  # quita el separador para los casos en los que el archivo ha sido guardado con Excel (completa línea con separador)

        # Load column names
        # line = f.readline()
        nomColsVar = str(f.readline()[:-1]).split(sep)  # nombreVariables
        nomCols = str(f.readline()[:-1]).split(sep)  # name coordinates X,Y,Z.
        # nomCols = [s.lower() for s in nomCols]

        # Finds the end of the section
        while line != "\n":
            if line == "":
                raise Exception(f"End of section '{section}' not found")

            num_line += 1
            # print('line '+ str(num_line))
            line = f.readline()

    end_section = num_line - 1  # removes 1 to compensate the empty line

    # Counts total line number
    with open(file, mode="rt") as f:
        file_end = len(f.readlines())

    # Column labels
    if section == "Devices":
        n_vars = ["Frame", "Sub Frame"] + nomColsVar[
            2:-1
        ]  # removes the last, which is ''
        # nomVars2=list(filter(lambda c: c!='', n_vars))

    else:  # Trajectories and Models
        # primero asigna los nombres según el propio archivo
        # n_vars=['Frame', 'Sub Frame']
        # for i in range(2,len(nomCols),3):
        #     if "'" not in nomCols[i] and "''" not in nomCols[i] and 'EMG' not in nomCols[i]: #elimina las posibles columnas de velocidad y aceleración
        #         print(nomCols[i])
        #         n_vars.append(nomColsVar[i].split(':')[1]+'_' + nomCols[i])#X
        #         n_vars.append(nomColsVar[i].split(':')[1]+'_' + nomCols[i+1])#Y
        #         n_vars.append(nomColsVar[i].split(':')[1]+'_' + nomCols[i+2])#Z

        n_vars = ["Frame", "Sub Frame"]
        for i in range(2, len(nomCols)):
            if nomCols[i] in "xX" and "'" not in nomCols[i] and "''" not in nomCols[i]:
                # print(nomCols[i], nomColsVar[i])
                n_vars.append(nomColsVar[i].split(":")[1] + "_" + nomCols[i])  # X
                n_vars.append(nomColsVar[i].split(":")[1] + "_" + nomCols[i + 1])  # Y
                n_vars.append(nomColsVar[i].split(":")[1] + "_" + nomCols[i + 2])  # Z
            elif "EMG" in nomCols[i]:
                # print(nomCols[i], nomColsVar[i])
                n_vars.append(nomColsVar[i].split(":")[1] + "_" + nomCols[i])
            # else:
            # print(nomCols[i])

    # [i for i in nomColsVar if "'" in i]
    # nomColsVar = [i for i in nomColsVar if "'" not in i]

    # carga todos los datos
    # CON GENFROMTXT FALLA SI NO EMPIEZA LA PRIMERA LÍNEA CON DATOS
    # provisional= np.genfromtxt(file_name, skip_header= ini_section+5, max_rows=end_section-ini_section-1, delimiter=sep, missing_values='', filling_values=np.nan, invalid_raise=True)
    # provisional=provisional[:, :len(n_vars)] #recorta solo hasta las variables

    # Convierte los datos en pandas dataframe. Pasa solo los que no son de velocidad o aceleración
    # dfReturn = pd.DataFrame(provisional[:, :len(n_vars)], columns=n_vars)
    # dfReturn = dfReturn.iloc[:, :len(n_vars)] #se queda solo con las columnas de las variables, quita las de velocidad si las hay

    # Con pandas directamente funciona (para evitar error si primera línea no son datos, lee la fila de las unidades y luego la quita)
    dfReturn = (
        pd.read_csv(
            file,
            delimiter=sep,
            header=None,
            skiprows=ini_section + 4,
            skipfooter=file_end - end_section - 5,
            usecols=range(len(n_vars)),
            engine="python",
        )
        .drop(index=0)
        .reset_index(drop=True)
        .astype(float)
    )

    # x=pd.read_csv(file, delimiter=sep, header=ini_section, skipfooter=file_end-end_section-5, engine='python')
    # x.columns
    # sub_nom_cols = x.iloc[0,:]

    # Nombra encabezado
    if section == "Devices":
        if "Noraxon Ultium" in n_vars[3]:
            var = [s.split("- ")[-1] for s in n_vars]
            coord = nomCols[: len(n_vars)]
            dfReturn.columns = var
            dimensions = ["n_var", "time"]
            var_name = ["n_var"]

    else:
        var = [
            "_".join(s.split("_")[:-1]) for s in n_vars[: len(n_vars)]
        ]  # gestiona si la variable tiene sep '_', lo mantiene
        coord = [
            s.split(":")[-1].lower() for s in nomCols[: len(n_vars)]
        ]  # pasa coordenadas a minúscula
        dfReturn.columns = pd.MultiIndex.from_tuples(
            list(zip(*[var, coord])), names=["n_var", "axis"]
        )
        # Elimina columnas con variables modeladas EMG si las hay
        dfReturn = dfReturn.drop(columns=dfReturn.filter(regex="emg"))
        # dfReturn.duplicated().sum()
        dimensions = ["n_var", "axis", "time"]
        var_name = ["n_var", "axis"]

    # dfReturn.columns=[var, coord]
    # dfReturn.columns.set_names(names=['Variable', 'Coord'], level=[0,1], inplace=True)

    # Incluye columna con tiempo
    dfReturn.insert(2, "time", np.arange(len(dfReturn)) / freq)

    if n_vars_load:
        dfReturn = dfReturn.loc[:, n_vars_load]

    if to_dataarray:
        # if section == 'Devices':
        #    dfReturn.columns = dfReturn.columns.droplevel(1)

        # dfReturn.iloc[:,2:].melt(id_vars='time', var_name=['n_var', 'axis']).set_index(dimensions).to_xarray().to_array()

        # TODO: FIX THIS, DOES NOT WORK
        print("to_dataArray not implemented yet.\nTry with the Polars version")
        try:
            daReturn = (
                dfReturn.iloc[:, 2:]
                # .assign(**{'time':np.arange(len(dfReturn))/freq})
                .melt(id_vars="time", var_name=var_name)
                .set_index(dimensions)
                .to_xarray()
                .to_array()
                .squeeze("variable")
                .drop_vars("variable")  # la quita de dimensions y coordenadas
            )
            daReturn.name = section
            daReturn.attrs["freq"] = freq
            daReturn.time.attrs["units"] = "s"
        except:
            daReturn = xr.DataArray()

    if header_format == "flat" and section != "Devices":
        dfReturn.columns = dfReturn.columns.map("_".join).str.strip()

    # #Elimina las columnas de velocidad y aceleración, si las hay
    # borrarColsVA = dfReturn.filter(regex='|'.join(["'", "''"])).columns
    # dfReturn = dfReturn.drop(columns=borrarColsVA)

    # Si hace falta lo pasa a xArray
    if False:  # to_dataarray:
        # prueba para hacerlo directamente desde dataframe
        # dfReturn.assign(**{'time':np.arange(len(dfReturn))/freq}).drop(columns='').melt(id_vars='time').set_index(['n_var', 'axis', 'time']).to_xarray().to_array()

        if header_format != "flat":
            dfReturn.columns = dfReturn.columns.map("_".join).str.strip()

        # transforma los datos en xarray
        x = dfReturn.filter(regex="|".join(["_x", "_X"])).to_numpy().T
        y = dfReturn.filter(regex="|".join(["_y", "_Y"])).to_numpy().T
        z = dfReturn.filter(regex="|".join(["_z", "_Z"])).to_numpy().T
        data = np.stack([x, y, z])

        # Quita el identificador de la coordenada del final
        n_vars = dfReturn.filter(regex="|".join(["_x", "_X"])).columns.str.rstrip(
            "|".join(["_x", "_X"])
        )

        time = np.arange(x.shape[1]) / freq
        coords = {}
        coords["axis"] = ["x", "y", "z"]
        coords["n_var"] = n_vars
        coords["time"] = time

        daReturn = xr.DataArray(
            data=data,
            dims=("axis", "n_var", "time"),
            coords=coords,
            name=section,
            attrs={"freq": freq},
            # **kwargs,
        )
        if header_format != "flat":  # si hace falta lo vuelve a poner como multiindex
            dfReturn.columns = pd.MultiIndex.from_tuples(
                list(zip(*[var, coord])), names=["n_var", "axis"]
            )

    if to_dataarray and return_freq:
        return dfReturn, daReturn, freq
    elif to_dataarray:
        return dfReturn, daReturn
    elif return_freq:
        return dfReturn, freq
    else:
        return dfReturn


# =============================================================================
# %% TESTS
# =============================================================================
if __name__ == "__main__":

    # from biomdp.io.read_vicon_csv import read_vicon_csv
    work_path = Path(r"src\datasets")
    file = work_path / "vicon_CMJ_kinem_kinet_emg.csv"
    daData = read_vicon_csv(file, section="Trajectories")
    daData.plot.line(x="time", col="n_var", col_wrap=3)

    daData = read_vicon_csv(file, section="Model Outputs")
    daData.plot.line(x="time", col="n_var", col_wrap=3, sharey=False)

    daData = read_vicon_csv(file, section="Forces")
    daData.plot.line(x="time", col="n_var", col_wrap=3, sharey=False)

    daData = read_vicon_csv(file, section="EMG")
    daData.plot.line(x="time", col="n_var", col_wrap=3, sharey=False)

    import timeit

    def test_performance():
        result = read_vicon_csv(
            file, section="Model Outputs", engine="polars", raw=False
        )
        return result

    print(
        f'{timeit.timeit("test_performance()", setup="from __main__ import test_performance", number=10):.4f} s'
    )

    def test_performance():
        result = read_vicon_csv(file, section="Model Outputs", engine="pandas")
        return result

    print(
        f'{timeit.timeit("test_performance()", setup="from __main__ import test_performance", number=10):.4f} s'
    )
