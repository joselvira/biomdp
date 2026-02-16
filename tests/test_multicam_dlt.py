import numpy as np
import xarray as xr
import sys
from pathlib import Path

# Add the source directory to sys.path to import biomdp
sys.path.append(r"f:\Programacion\Python\Mios\biomdp\src")

try:
    from biomdp.fotogrammetry import (
        dlt_calib,
        dlt_recons_xr,
        dlt_calibrate,
        dlt_reconstruct,
    )
except ImportError as e:
    print(f"Error importing biomdp: {e}")
    sys.exit(1)


def test_multicam_dlt():
    print("Testing Multi-Camera DLT Reconstruction...")

    # 1. Define 3D Control Points (Cube)
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
        ]
    )

    # 2. Define Synthetic Cameras (Viewpoints)
    # We will use dlt_calibrate to generate L parameters from known 2D projections
    # Let's project these 3D points onto 2D planes slightly different for each camera

    # Camera 1: Looking from front-left
    uv1 = np.array(
        [
            [500, 500],
            [500, 400],
            [600, 400],
            [600, 500],
            [520, 480],
            [520, 380],
            [620, 380],
            [620, 480],
        ]
    )

    # Camera 2: Looking from front-right
    uv2 = np.array(
        [
            [400, 500],
            [400, 400],
            [300, 400],
            [300, 500],
            [420, 480],
            [420, 380],
            [320, 380],
            [320, 480],
        ]
    )

    # Create Calibration Parameters
    nd = 3
    L1, err1 = dlt_calibrate(nd, xyz, uv1)
    L2, err2 = dlt_calibrate(nd, xyz, uv2)

    Ls = np.vstack([L1, L2])
    print(f"Calibration Errors: Cam1={err1:.4f}, Cam2={err2:.4f}")

    # 3. Prepare Data for dlt_recons_xr
    # We want to reconstruct the same points we used for calibration
    # Create DataArray with dimensions (time, camera, axis) or list of DataArrays

    # Using list of DataArrays (one per camera)
    n_points = xyz.shape[0]
    times = np.arange(n_points)

    da_uv1 = xr.DataArray(
        uv1,
        dims=("time", "axis"),
        coords={"time": times, "axis": ["x", "y"]},
        name="cam1",
    )

    da_uv2 = xr.DataArray(
        uv2,
        dims=("time", "axis"),
        coords={"time": times, "axis": ["x", "y"]},
        name="cam2",
    )

    # 4. Run Reconstruction
    try:
        # Pass list of DataArrays
        recons_da = dlt_recons_xr([da_uv1, da_uv2], Ls, num_dims=3, num_cams=2)

        print("\nReconstructed DataArray:")
        print(recons_da)

        # 5. Check Accuracy
        recons_xyz = recons_da.values

        # Calculate localized error
        diff = recons_xyz - xyz
        dist_err = np.linalg.norm(diff, axis=1)
        mean_err = np.mean(dist_err)

        print(f"\nMean Reconstruction Error: {mean_err:.6f}")

        if (
            mean_err < 1e-1
        ):  # Allow some tolerance due to the fake projection/calibration loop
            print("SUCCESS: Reconstruction is accurate.")
        else:
            print("FAILURE: Reconstruction error is too high.")

    except Exception as e:
        print(f"\nFAILURE with Exception: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_multicam_dlt()
