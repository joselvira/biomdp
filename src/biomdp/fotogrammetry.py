# %% -*- coding: utf-8 -*-
"""
Functions to perform DLT operations with markers from video.
Based on xarray.

@author: Jose L. L. Elvira
"""


# =============================================================================
# %% LOAD LIBRARIES
# =============================================================================

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

try:
    from dltx import dlt_calibrate, dlt_reconstruct
except ImportError:
    raise ImportError(
        "dltx is not installed. Please install it with 'pip install dltx'"
    )
# --------------------------------------------------------------


__author__ = "Jose L. L. Elvira"
__version__ = "1.0.0"
__date__ = "15/02/2026"


"""
Updates:
    15/02/2026, v1.0.0
        - Initial release. 

    
"""


# =============================================================================
# %% FUNCTIONS
# =============================================================================


def dlt_calib(
    xyz: pd.DataFrame | np.ndarray,
    uv: list | pd.DataFrame | np.ndarray,
    show: bool = True,
) -> tuple[np.ndarray, float]:
    """
    Camera calibration by DLT using known object points and their image points.
    At least 6 calibration points for the 3D DLT and 4
    for the 2D DLT.

    Parameters
    ----------
    xyz : pd.DataFrame | np.ndarray
        Coordinates in the object space, num_points as rows and num_dims as columns (x, y) or (x, y, z).
    uv : list | pd.DataFrame | np.ndarray
        Coordinates in the image space, num_points as rows and num_dims as columns (u, v).
        If uv is a list, it must contain the uv coordinates for each camera.

    Returns
    -------
    L : np.ndarray
        Calibration parameters (8,) or (11,).
    err_image : float
        RMS error of the DLT transformation in units of image space.
    err_real : float
        RMS error of the DLT transformation in units of real object space.
    """
    xyz = np.asarray(xyz)
    uv = np.asarray(uv)
    err_real = np.nan  # only in 2D reconstruction
    if uv.ndim == 2:
        uv = np.expand_dims(uv, axis=0)

    num_dims = xyz.shape[1]
    num_cams = uv.shape[0]
    num_points = uv.shape[1]

    L = []
    err_image = []

    for i, _uv in enumerate(uv):
        _L, _err_image = dlt_calibrate(num_dims, xyz, _uv)
        L.append(_L)
        err_image.append(_err_image)
        print(f"Calibrated Camera {i}")
    L = np.array(L)
    err_image = np.array(err_image)

    # TODO: join as a general case? uv[0,i], uv[:,i]
    if num_cams == 1:
        # L, err_image = dlt_calibrate(num_dims, xyz, uv)

        # Reconstruct original points after calibration
        xyz_recons = np.zeros((num_points, num_dims))
        for i in range(num_points):
            try:
                xyz_recons[i, :] = dlt_reconstruct(num_dims, num_cams, L, uv[0, i])
            except np.linalg.LinAlgError:
                xyz_recons[i, :] = np.nan
            except Exception as e:
                print(f"Error in dlt_reconstruct for point {i}: {e}")
                xyz_recons[i, :] = np.nan

    else:
        xyz_recons = np.zeros((num_points, num_dims))
        for i in range(num_points):
            try:
                xyz_recons[i, :] = dlt_reconstruct(num_dims, num_cams, L, uv[:, i])
            except np.linalg.LinAlgError:
                xyz_recons[i, :] = np.nan

    err_real = np.mean(np.linalg.norm((xyz_recons - xyz), axis=1))
    # err_real = np.mean(np.sqrt(np.sum((np.array(xyz_recons) - np.array(xyz)) ** 2, 1)))

    if show:
        if num_dims == 2:
            fig, ax = plt.subplots()
            ax.plot(xyz[:, 0], xyz[:, 1], "bo", alpha=0.6, label="Real")
            ax.plot(
                xyz_recons[:, 0],
                xyz_recons[:, 1],
                "ro",
                alpha=0.6,
                label="Reconst",
            )
            # ax.text(
            #     0.5,
            #     0.95,
            #     f"Error original: {err_real:.4f} (original units)",
            #     transform=ax.transAxes,
            #     fontsize=8,
            #     c="r",
            #     horizontalalignment="center",
            # )
            # ax.text(
            #     0.5,
            #     0.90,
            #     f"Error image: {err_image[cam]:.4f} (image units)",
            #     transform=ax.transAxes,
            #     fontsize=8,
            #     c="r",
            #     horizontalalignment="center",
            # )
            ax.set_aspect("equal", "box")
            ax.set_title(
                "Real vs. reconstructed\n"
                f"Error orig: {err_real:.4f} (orig units)\n"
                f"Error img: {np.mean(err_image):.4f} (img units)"
            )
            ax.legend(fontsize=8)
            plt.show()

        elif num_dims == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(
                xyz[:, 0],
                xyz[:, 1],
                xyz[:, 2],
                c="b",
                marker="o",
                alpha=0.6,
                label="Real",
            )
            ax.scatter(
                xyz_recons[:, 0],
                xyz_recons[:, 1],
                xyz_recons[:, 2],
                c="r",
                marker="o",
                alpha=0.6,
                label="Reconst",
            )

            ax.set_title(
                "Real vs. reconstructed\n"
                f"Error orig: {err_real:.4f} (orig units)\n"
                f"Error img: {np.mean(err_image):.4f} (img units)"
            )
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.legend()
            plt.show()

    return L, err_image, err_real


def dlt_recons_xr(
    da_uv: xr.DataArray | list[xr.DataArray],
    L: np.ndarray,
    num_cams: int = 1,
    num_dims: int = 2,
) -> xr.DataArray:
    """
    Vectorized DLT reconstruction for 2D/3D case with N cameras using xarray.

    Parameters
    ----------
    da_uv : xr.DataArray or list of xr.DataArray
        DataArray with coordinates.
        If list: len(da_uv) must be equal to num_cams. Each DA must have 'axis' dim.
        If DataArray: must have 'camera' and 'axis' dimensions, or be a single camera (nc=1).
    L : np.array
        Calibration parameters.
        Shape (8,) or (9,) for 2D 1-cam; (11,) or (12,) for 3D 1-cam.
        Shape (nc, 8/9) or (nc, 11/12) for multi-cam.
    num_cams : int
        Number of cameras.
    num_dims : int
        Number of dimensions (2 or 3).

    Returns
    -------
    xr.DataArray
        Reconstructed coordinates. Shape (..., num_dims).
    """

    # Standardize input to single DataArray with 'camera' dimension
    if isinstance(da_uv, list):
        num_cams = len(da_uv)

        da_input = xr.concat(da_uv, dim="camera").assign_coords(
            {"camera": np.arange(num_cams)}
        )

    elif isinstance(da_uv, xr.DataArray):
        da_input = da_uv
        if num_cams == 1 and "camera" not in da_input.dims:
            # Ensure camera dimension exists for single camera
            da_input = da_input.expand_dims("camera")

        if num_cams > 1 and "camera" not in da_input.dims:
            raise ValueError(
                "For num_cams > 1, input DataArray must have 'camera' dimension."
            )
        num_cams = da_input.sizes["camera"]
    else:
        raise TypeError("da_uv must be a DataArray or a list of DataArrays.")

    # Standardize and validate L
    L = np.asarray(L)
    if num_cams == 1 and L.ndim == 1:
        L = L.reshape(1, -1)

    expected_params = {2: [8, 9], 3: [11, 12]}
    if num_dims not in expected_params or L.shape[1] not in expected_params[num_dims]:
        raise ValueError(
            f"L shape {L.shape} invalid for {num_dims}D. Expected ({num_cams}, {expected_params.get(num_dims, '?')})"
        )

    # Fast path for 2D single camera using direct matrix inversion
    Hinv = None
    if num_cams == 1 and num_dims == 2:
        try:
            L_flat = L.flatten()
            if L_flat.size == 8:
                L_flat = np.append(L_flat, 1.0)
            Hinv = np.linalg.inv(L_flat.reshape(3, 3))
        except Exception:
            pass  # Fallback to SVD

    def _recon_svd(uv_all, L_all):
        """Core reconstruction using SVD or fast path"""

        # Fast path for 2D 1-cam
        if Hinv is not None:
            uv = uv_all.squeeze(axis=-2)
            uv1 = np.concatenate([uv, np.ones(uv.shape[:-1] + (1,))], axis=-1)
            xyz = uv1 @ Hinv.T
            with np.errstate(divide="ignore", invalid="ignore"):
                return xyz[..., :2] / xyz[..., 2:3]

        # General SVD path
        input_shape = uv_all.shape
        batch_shape = input_shape[:-2]
        n_samples = int(np.prod(batch_shape)) if batch_shape else 1

        uv_flat = uv_all.reshape(n_samples, num_cams, 2)
        valid_mask = ~np.isnan(uv_flat).any(axis=(1, 2))

        X_out = np.full((n_samples, num_dims), np.nan)
        if not valid_mask.any():
            return X_out.reshape(batch_shape + (num_dims,))

        uv_valid = uv_flat[valid_mask]
        u, v = uv_valid[..., 0], uv_valid[..., 1]

        # Build M matrix - unified logic for 2D and 3D
        n_params = L_all.shape[1]
        L_norm = 1.0 if n_params in [8, 11] else L_all[:, -1]

        if num_dims == 3:
            # Extract 3D parameters
            L_vals = L_all[:, :11].T  # (11, nc)
            row1 = np.stack([L_vals[i] - u * L_vals[i + 7] for i in range(4)], axis=-1)
            row2 = np.stack(
                [L_vals[i + 3] - v * L_vals[i + 7] for i in range(4)], axis=-1
            )
            if n_params == 12:
                row1[..., -1] = L_vals[3] - u * L_norm
                row2[..., -1] = L_vals[7] - v * L_norm
        else:  # 2D
            L_vals = L_all[:, :8].T  # (8, nc)
            row1 = np.stack([L_vals[i] - u * L_vals[i + 5] for i in range(3)], axis=-1)
            row2 = np.stack(
                [L_vals[i + 2] - v * L_vals[i + 5] for i in range(3)], axis=-1
            )
            if n_params == 9:
                row1[..., -1] = L_vals[2] - u * L_norm
                row2[..., -1] = L_vals[5] - v * L_norm

        M = np.stack([row1, row2], axis=2).reshape(valid_mask.sum(), num_cams * 2, -1)

        # Solve using SVD
        try:
            _, _, Vh = np.linalg.svd(M)
            X = Vh[..., -1, :]
            with np.errstate(divide="ignore", invalid="ignore"):
                X_out[valid_mask] = X[..., :-1] / X[..., -1:]
        except np.linalg.LinAlgError:
            pass  # Return NaNs

        return X_out.reshape(batch_shape + (num_dims,))

    coord_axis = ["x", "y"] if num_dims == 2 else ["x", "y", "z"]

    return xr.apply_ufunc(
        _recon_svd,
        da_input.sel(axis=["x", "y"]),
        kwargs={"L_all": L},
        input_core_dims=[["camera", "axis"]],
        output_core_dims=[["axis"]],
        exclude_dims={"axis"},
        vectorize=False,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={"output_sizes": {"axis": num_dims}},
    ).assign_coords({"axis": coord_axis})


# %% MAIN
if __name__ == "__main__":
    # Test with real data from numpy arrays
    # Calibration points in object space (centimetres)
    xy = np.array([[0, 0], [0, 45], [0, 90], [60, 0], [60, 45], [60, 90]])

    # Calibration points in image space (pixels)
    uv = np.array(
        [
            [195.99, 74.83],
            [193.43, 312.34],
            [188.3, 547.52],
            [512.51, 77.39],
            [510.06, 311.46],
            [508.88, 544.23],
        ]
    )

    # Calculate calibration parameters
    L, err_image, err_real = dlt_calib(xy, uv, show=True)

    # --------------------------------------
    # Test with simulated example prepared to read from csv files
    # Real object space dimensions (metres)
    xy = pd.DataFrame(
        columns=["x", "y"],
        data=[
            [0.0, 13.410],
            [6.1, 13.410],
            [6.1, 6.705],
            [6.1, 0.000],
            [0.0, 0.000],
            [0.0, 6.705],
        ],
    )
    # xy = pd.read_csv("xy.csv")

    # Digitised in image space dimensions, 3 frames of the same 6 points
    uv = pd.DataFrame(
        columns=["fot", "p", "x", "y"],
        data=[
            [0, 0, 743, 881],
            [0, 1, 1165, 894],
            [0, 2, 1230, 602],
            [0, 3, 1345, 99],
            [0, 4, 620, 76],
            [0, 5, 696, 587],
            [11, 0, 743, 881],
            [11, 1, 1165, 894],
            [11, 2, 1230, 601],
            [11, 3, 1344, 100],
            [11, 4, 620, 76],
            [11, 5, 697, 587],
            [12, 0, 744, 881],
            [12, 1, 1165, 893],
            [12, 2, 1230, 601],
            [12, 3, 1344, 100],
            [12, 4, 620, 76],
            [12, 5, 696, 587],
        ],
    )
    # uv = pd.read_csv("uv.csv")
    # Calculate the mean of each digitised point
    uv = uv.groupby("p").mean().drop(columns="fot")

    # calcula parámetros de calibración
    L, err_img, err_real = dlt_calib(xy, uv, show=True)

    # --------------------------------------
    # Test 3D with 4 cameras
    xyz = [
        [0, 0, 0],
        [0, 12.3, 0],
        [14.5, 12.3, 0],
        [14.5, 0, 0],
        [0, 0, 14.5],
        [0, 12.3, 14.5],
        [14.5, 12.3, 14.5],
        [14.5, 0, 14.5],
    ]

    uv1 = [
        [1302, 1147],
        [1110, 976],
        [1411, 863],
        [1618, 1012],
        [1324, 812],
        [1127, 658],
        [1433, 564],
        [1645, 704],
    ]
    uv2 = [
        [1094, 1187],
        [1130, 956],
        [1514, 968],
        [1532, 1187],
        [1076, 854],
        [1109, 647],
        [1514, 659],
        [1523, 860],
    ]
    uv3 = [
        [1073, 866],
        [1319, 761],
        [1580, 896],
        [1352, 1016],
        [1064, 545],
        [1304, 449],
        [1568, 557],
        [1313, 668],
    ]
    uv4 = [
        [1205, 1511],
        [1193, 1142],
        [1601, 1121],
        [1631, 1487],
        [1157, 1550],
        [1139, 1124],
        [1628, 1100],
        [1661, 1520],
    ]

    # Calibrate
    L, err_image, err_real = dlt_calib(xyz, [uv1, uv2, uv3, uv4], show=True)

    # ------------------------------------
    # Test 2D reconstruction

    xy = pd.DataFrame(
        columns=["x", "y"],
        data=[
            [0, 0],
            [0, 45],
            [0, 90],
            [60, 0],
            [60, 45],
            [60, 90],
        ],
    )

    uv = pd.DataFrame(
        columns=["x", "y"],
        data=[
            [195.99, 74.83],
            [193.43, 312.34],
            [188.3, 547.52],
            [512.51, 77.39],
            [510.06, 311.46],
            [508.88, 544.23],
        ],
    )

    # Calibrate Camera
    L, err_image, err_real = dlt_calib(xy, uv, show=True)

    # Digitised markers
    markers_uv = pd.DataFrame(
        columns=["1x", "1y", "2x", "2y", "3x", "3y", "4x", "4y"],
        index=np.arange(10) / 50,
        data=[
            [195.99, 74.83, 188.3, 547.52, 512.51, 77.39, 508.88, 544.23],
            [199.99, 89.55, 192.7, 548.45, 510.11, 77.39, 508.88, 543.86],
            [207.99, 116.94, 201.5, 549.18, 505.31, 77.39, 508.88, 543.18],
            [219.99, 153.13, 214.7, 549.60, 498.11, 77.39, 508.88, 542.27],
            [235.99, 193.05, 232.3, 549.67, 488.51, 77.39, 508.88, 541.27],
            [255.99, 231.09, 254.3, 549.36, 476.51, 77.39, 508.88, 540.32],
            [279.99, 261.91, 280.7, 548.72, 462.11, 77.39, 508.88, 539.55],
            [307.99, 286.25, 311.5, 546.85, 445.31, 77.40, 508.88, 539.07],
            [339.99, 286.25, 346.7, 545.88, 426.11, 77.41, 508.88, 538.95],
            [375.99, 276.25, 386.3, 545.88, 404.51, 77.42, 508.88, 539.20],
        ],
    )

    # Convert to dataarray
    x = markers_uv.filter(regex="x")
    y = markers_uv.filter(regex="y")
    daMarkers_uv = xr.DataArray(
        data=np.stack([x, y], axis=1),
        coords={
            "time": markers_uv.index,
            "axis": ["x", "y"],
            "marker": range(x.shape[1]),
        },
        dims=["time", "axis", "marker"],
    ).transpose("marker", "axis", "time")

    # Reconstruct in object space
    daMarkers_xy = dlt_recons_xr(
        da_uv=daMarkers_uv, L=L, num_cams=1, num_dims=2
    ).transpose("marker", "axis", "time")

    # Show reconstruction
    ax = xy.plot.scatter(x="x", y="y")
    daMarkers_xy.to_dataset(dim="axis").plot.scatter(
        x="x", y="y", marker="o", c="r", alpha=0.5, ax=ax
    )
    ax.set_aspect("equal", "box")
