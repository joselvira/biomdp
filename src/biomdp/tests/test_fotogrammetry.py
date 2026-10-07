import os
import sys

import numpy as np
import numpy.testing as npt
import xarray as xr

# Ensure `src` is on path so `biomdp` can be imported
sys.path.insert(
    # 0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
    # 0, os.path.abspath(os.path.dirname(__file__))
    0,
    "F:\Programacion\Python\Mios\biomdp\src\biomdp",
)

from biomdp.fotogrammetry import DLTrecon, dlt_calib, dlt_recons_xr, normalization


def test_normalization_and_dltcalib_2d():
    # Square corners in cm
    xy = np.array([[0, 0], [0, 12.3], [14.5, 12.3], [14.5, 0]], dtype=float)

    # Two views (only use uv1 here to test single-view calibration)
    uv1 = np.array([[1302, 1147], [1110, 976], [1411, 863], [1618, 1012]], dtype=float)

    # Test Normalization: centroid ~ 0 and mean distance ~ sqrt(2)
    T, x_norm = Normalization(2, xy)

    npt.assert_allclose(np.mean(x_norm, axis=0), [0.0, 0.0], atol=1e-12)
    mean_dist = np.mean(np.linalg.norm(x_norm, axis=1))
    npt.assert_allclose(mean_dist, np.sqrt(2), rtol=1e-6)

    # Calibrate using 2D DLT with a single view
    L, err = dlt_calib(2, xy, uv1)
    L1, err1 = DLTcalib(2, xy, uv1)
    assert np.isfinite(err) and err >= 0

    # Reconstruct points using DLTrecon (single view)
    xy_rec = np.zeros_like(xy)
    for i in range(len(uv1)):
        xy_rec[i, :] = DLTrecon(2, 1, L, uv1[i])

    # Reconstruction should match original points within numerical tolerance
    npt.assert_allclose(xy_rec, xy, rtol=1e-6, atol=1e-6)


def test_dlt_recons_xr_vectorized():
    # Same data as above
    xy = np.array([[0, 0], [0, 12.3], [14.5, 12.3], [14.5, 0]], dtype=float)
    uv1 = np.array([[1302, 1147], [1110, 976], [1411, 863], [1618, 1012]], dtype=float)

    # Calibrate
    L, _ = DLTcalib(2, xy, uv1)

    # Build xarray DataArray with axis dimension labeled 'axis' and coords ['x','y']
    da_uv = xr.DataArray(uv1, dims=("points", "axis"), coords={"axis": ["x", "y"]})

    rec = dlt_recons_xr(da_uv, L, nd=2, nc=1)

    # rec should be a DataArray with same shape and values close to original xy
    assert list(rec.dims) == ["points", "axis"]
    npt.assert_allclose(rec.values, xy, rtol=1e-6, atol=1e-6)


def test_dlt_recons_xr_3d_multi_view():
    # 3D cube points (same as module example)
    xyz = np.array(
        [
            [0, 0, 0],
            [0, 12.3, 0],
            [14.5, 12.3, 0],
            [14.5, 0, 0],
            [0, 0, 14.5],
            [0, 12.3, 14.5],
            [14.5, 12.3, 14.5],
            [14.5, 0, 14.5],
        ],
        dtype=float,
    )

    uv1 = np.array(
        [
            [1302, 1147],
            [1110, 976],
            [1411, 863],
            [1618, 1012],
            [1324, 812],
            [1127, 658],
            [1433, 564],
            [1645, 704],
        ],
        dtype=float,
    )
    uv2 = np.array(
        [
            [1094, 1187],
            [1130, 956],
            [1514, 968],
            [1532, 1187],
            [1076, 854],
            [1109, 647],
            [1514, 659],
            [1523, 860],
        ],
        dtype=float,
    )
    uv3 = np.array(
        [
            [1073, 866],
            [1319, 761],
            [1580, 896],
            [1352, 1016],
            [1064, 545],
            [1304, 449],
            [1568, 557],
            [1313, 668],
        ],
        dtype=float,
    )
    uv4 = np.array(
        [
            [1205, 1511],
            [1193, 1142],
            [1601, 1121],
            [1631, 1487],
            [1157, 1550],
            [1139, 1124],
            [1628, 1100],
            [1661, 1520],
        ],
        dtype=float,
    )

    # Calibrate each view
    L1, _ = DLTcalib(3, xyz, uv1)
    L2, _ = DLTcalib(3, xyz, uv2)
    L3, _ = DLTcalib(3, xyz, uv3)
    L4, _ = DLTcalib(3, xyz, uv4)

    Ls = np.vstack([L1, L2, L3, L4])

    # Build DataArray with dims (view, point, axis)
    uv_stack = np.stack([uv1, uv2, uv3, uv4], axis=0)  # (view, point, 2)
    da_uv = xr.DataArray(
        uv_stack, dims=("view", "point", "axis"), coords={"axis": ["x", "y"]}
    )

    rec = dlt_recons_xr(da_uv, Ls, nd=3, nc=4)

    npt.assert_allclose(rec.values, xyz, atol=1e-6, rtol=1e-6)


def test_3d_reconstruction():
    print("Setting up 3D reconstruction test...")

    # 1. Define 3D points (object space)
    # 8 corners of a cube
    xyz = np.array(
        [
            [0, 0, 0],
            [0, 10, 0],
            [10, 10, 0],
            [10, 0, 0],
            [0, 0, 10],
            [0, 10, 10],
            [10, 10, 10],
            [10, 0, 10],
        ],
        dtype=float,
    )

    # 2. Define 2 separate views (cameras)
    # Camera 1 (Front-ish)
    uv1 = np.array(
        [
            [1302, 1147],
            [1110, 976],
            [1411, 863],
            [1618, 1012],
            [1324, 812],
            [1127, 658],
            [1433, 564],
            [1645, 704],
        ],
        dtype=float,
    )

    # Camera 2 (Side-ish)
    uv2 = np.array(
        [
            [1094, 1187],
            [1130, 956],
            [1514, 968],
            [1532, 1187],
            [1076, 854],
            [1109, 647],
            [1514, 659],
            [1523, 860],
        ],
        dtype=float,
    )

    # 3. Calibrate cameras
    nd = 3
    L1, err1 = dlt_calib(nd, xyz, uv1)
    L2, err2 = dlt_calib(nd, xyz, uv2)
    print(f"Calibration Error Cam 1: {err1:.4f}")
    print(f"Calibration Error Cam 2: {err2:.4f}")

    # 4. Create DataArrays simulating time series (just 1 frame here for simplicity, or 8 frames)
    # Let's say we have 8 frames, one point per frame, or 1 frame with 8 points.
    # The user mentioned "dataarrays", probably `time`, `marker`, `axis`.

    # Create DataArray for each camera
    # Dims: time x marker x axis. Let's assume 1 marker, 8 time frames corresponding to the 8 points

    da1 = xr.DataArray(
        uv1.reshape(8, 1, 2),
        coords={"time": np.arange(8), "marker": ["pt1"], "axis": ["u", "v"]},
        dims=("time", "marker", "axis"),
    )

    da2 = xr.DataArray(
        uv2.reshape(8, 1, 2),
        coords={"time": np.arange(8), "marker": ["pt1"], "axis": ["u", "v"]},
        dims=("time", "marker", "axis"),
    )

    print("\nAttempting reconstruction with dlt_recons_xr...")

    try:
        # User wants to pass a list of DataArrays or verify if it works
        # The current implementation might only support a single DataArray with specific dims

        # Test Case 1: List of DataArrays (User's likely use case)
        print("Test 1: Passing list of DataArrays [da1, da2]")
        Ls = np.vstack([L1, L2])
        recon = dlt_recons_xr([da1, da2], Ls, nd=3, nc=2)
        print("Success! Result shape:", recon.shape)
        print(recon)

    except Exception as e:
        print(f"Caught expected error: {e}")

    try:
        # Test Case 2: Concatenated DataArray
        # This is how the current function EXPECTS valid input: (view, point, axis)
        # But our data is (time, marker, axis).
        # We need to stack them creating a 'view' dimension.
        print("\nTest 2: Concatenating input manually (view, time, marker, axis)")
        da_joined = xr.concat([da1, da2], dim="view")
        da_joined.coords["view"] = [0, 1]

        # NOTE: Current dlt_recons_xr expects specifically (view, point, axis) or similar flat structure?
        # Let's see if it handles 'time' and 'marker' dimensions correctly via broadcasting or if it fails.
        # The code I read earlier does `uv_np = da_uv.sel(axis=['x','y']).values` which completely flattens or might fail if excessive dims.

        Ls = np.vstack([L1, L2])
        recon = dlt_recons_xr(da_joined, Ls, nd=3, nc=2)
        print("Success! Result shape:", recon.shape)

    except Exception as e:
        print(f"Caught expected error in Test 2: {e}")


# %%
if __name__ == "__main__":
    import timeit

    xy = np.array([[0, 0], [0, 12.3], [14.5, 12.3], [14.5, 0]], dtype=float)
    uv1 = np.array([[1302, 1147], [1110, 976], [1411, 863], [1618, 1012]], dtype=float)

    def test_performance():
        result = normalization(2, xy)
        return result

    print(
        f"{timeit.timeit('test_performance()', setup='from __main__ import test_performance', number=5000):.4f} s"
    )
    results2 = test_performance()

    # %%-------------------------------------
    test_3d_reconstruction()
