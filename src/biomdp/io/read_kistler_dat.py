"""
Created on Mon Aug 27 17:43:37 2026

@author: Jose L. L. Elvira

Bioware's .dat file reader.,
8 signals per platform.
"""


# =============================================================================
# %% LOAD LIBRARIES
# =============================================================================

__author__ = "Jose L. L. Elvira"
__version__ = "0.1.0"
__date__ = "27/08/2026"

"""
Updates:
    27/08/2026, v0.1.0
        - First version, with the help of AI.
            
"""

import re
import struct
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


# =============================================================================
# %% Functions
# =============================================================================
_RANGE_LABELS = {"Range1": 1, "Range2": 2, "Range3": 3, "Range4": 4}
_HEADER2_SIGNATURE = b"\x01\x80\x01\x00\x00\x00\x01\x00\x00\x00\x08\x00\x00\x00\x08\x00"


@dataclass
class PlateConfig:
    """Configuration parameters for a Kistler force platform."""

    name: str
    serial: str
    start_channel: int  # ADC home channel (1-based)
    num_channels: int  # number of channels (normalmente 8)
    width_mm: float  # plataform width [mm]
    length_mm: float  # plataform length [mm]
    a_mm: float  # half-distance between sensors in X [mm]
    b_mm: float  # half-distance between sensors in Y [mm]
    azx_mm: float  # height of the Z-sensor’s origin above the surface [mm]
    range_xy: str  # Fx/Fy ampliflier range
    range_z: str  # Fz ampliflier range
    # Sensitivities by channel and by range (mV/N)
    # Shape: (num_channels, 4)  columns = SR1, SR2, SR3, SR4
    sensitivities: np.ndarray = field(default_factory=lambda: np.zeros((8, 4)))


@dataclass
class PlateData:
    """Data for a Kistler force platform, already calculated in physical units."""

    name: str
    time: np.ndarray  # time [s]
    Fx: np.ndarray | None  # resultant force in X [N] (None if raw=True)
    Fy: np.ndarray | None  # resultant force in Y [N] (None if raw=True)
    Fz: np.ndarray | None  # resultant force in Z [N] (None if raw=True)
    Mx: (
        np.ndarray | None
    )  # moment in X with respect to the center [N·m] (None if raw=True)
    My: (
        np.ndarray | None
    )  # moment in Y with respect to the center [N·m] (None if raw=True)
    Mz: np.ndarray | None  # moment in Z (free moment) [N·m] (None if raw=True)
    Ax: np.ndarray | None  # Center of pressure, X coordinate [m] (None if raw=True)
    Ay: np.ndarray | None  # Center of pressure, Y coordinate [m] (None if raw=True)
    sample_rate: float  # sampling frequency [Hz]
    raw_channels: np.ndarray | None = None  # raw signals from the sensor (if available)
    signals: dict[str, np.ndarray] | None = None

    def __repr__(self) -> str:
        num = len(self.time)
        dur = self.time[-1] - self.time[0] if num > 1 else 0.0
        fz_repr = (
            f"Fz=[{self.Fz.min():.1f}..{self.Fz.max():.1f}] N"
            if self.Fz is not None
            else "raw_channels only"
        )
        return (
            f"PlateData('{self.name}', "
            f"n={num}, fs={self.sample_rate:.1f} Hz, "
            f"dur={dur:.3f} s, "
            f"{fz_repr})"
        )


def parse_config_xml(xml_path: str | Path) -> list[PlateConfig]:
    """Read the Configuration.xml file from BioWare and return a list of PlateConfig."""
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    plates = []
    for dev in root.findall("Device"):
        active = dev.findtext("Active", "false").lower() == "true"
        if not active:
            continue

        name = dev.findtext("Name", "Unknown")
        serial = dev.findtext("SerialNumber", "")
        start_ch = int(dev.findtext("StartChannel", "1"))
        num_ch = int(dev.findtext("Channels", "8"))

        amp = dev.find("PlateInternalAmp")
        if amp is None:
            continue

        phys = amp.find("Physical")
        w = float(phys.findtext("Width_mm", "600"))
        l = float(phys.findtext("Length_mm", "900"))
        a = float(phys.findtext("a_mm", "210"))
        b = float(phys.findtext("b_mm", "350"))
        azx = float(phys.findtext("Azx_mm", "-52"))

        ctrl = amp.find("Control")
        range_xy = ctrl.findtext("RangeXY", "Range3") if ctrl is not None else "Range3"
        range_z = ctrl.findtext("RangeZ", "Range3") if ctrl is not None else "Range3"

        # Sensitivities: the XML lists SR1_mVperN, SR2_mVperN, ... for the num_ch channels
        sens_block = amp.find("Sensitivities")
        sens_arr = np.zeros((num_ch, 4))
        if sens_block is not None:
            for r_idx, sr_tag in enumerate(
                ["SR1_mVperN", "SR2_mVperN", "SR3_mVperN", "SR4_mVperN"]
            ):
                vals = [float(el.text) for el in sens_block.findall(sr_tag)]
                for ch_idx, v in enumerate(vals[:num_ch]):
                    sens_arr[ch_idx, r_idx] = v

        plates.append(
            PlateConfig(
                name=name,
                serial=serial,
                start_channel=start_ch,
                num_channels=num_ch,
                width_mm=w,
                length_mm=l,
                a_mm=a,
                b_mm=b,
                azx_mm=azx,
                range_xy=range_xy,
                range_z=range_z,
                sensitivities=sens_arr,
            )
        )
    return plates


def _extract_plate_config_from_header(
    header_bytes: bytes,
    plate_idx: int,
    fallback: PlateConfig,
) -> PlateConfig:
    """Extract a force platform configuration from the DAT header."""
    cfg = PlateConfig(**fallback.__dict__)

    def _clean_header_text(raw: bytes) -> str:
        text = raw.decode("latin-1", errors="ignore")
        text = re.sub(r"[^\x20-\x7E]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    match_full = re.search(
        rb"Kistler[^A-Za-z0-9]*Grande[^A-Za-z0-9]*(\d*)[^A-Za-z0-9]*9287([A-Z]{1,3})",
        header_bytes,
    )
    if match_full:
        plate_num = match_full.group(1).decode("latin-1", errors="ignore").strip()
        model = f"9287{match_full.group(2).decode('latin-1', errors='ignore')}"
        if plate_num:
            cfg.name = f"Kistler Grande {plate_num} {model}"
        else:
            cfg.name = f"Kistler Grande {model}"
    else:
        match_name = re.search(rb"Kistler[^\x00\r\n]{0,64}", header_bytes)
        if match_name:
            cfg.name = _clean_header_text(match_name.group(0))

    match_model = re.search(rb"9287[A-Z0-9]*", header_bytes)
    if match_model and "9287" not in cfg.name:
        model = _clean_header_text(match_model.group(0))
        cfg.name = f"{cfg.name} {model}".strip()

    # Geometry: find a sequence of pairs [width, length, a, b] and then azx.
    best_geom = None
    for off in range(max(0, len(header_bytes) - 32)):
        try:
            w = struct.unpack_from("<d", header_bytes, off)[0]
            l = struct.unpack_from("<d", header_bytes, off + 8)[0]
            a = struct.unpack_from("<d", header_bytes, off + 16)[0]
            b = struct.unpack_from("<d", header_bytes, off + 24)[0]
        except struct.error:
            continue
        if not all(np.isfinite(x) for x in (w, l, a, b)):
            continue
        if not (200.0 <= w <= 1200.0 and 200.0 <= l <= 2000.0):
            continue
        if not (50.0 <= a <= 450.0 and 50.0 <= b <= 600.0):
            continue

        score = 0.0
        if l >= w:
            score += 1.0
        if 500.0 <= w <= 700.0:
            score += 0.5
        if 800.0 <= l <= 1000.0:
            score += 0.5
        if best_geom is None or score > best_geom[0]:
            best_geom = (score, off, w, l, a, b)

    if best_geom is not None:
        _, off, w, l, a, b = best_geom
        cfg.width_mm = float(w)
        cfg.length_mm = float(l)
        cfg.a_mm = float(a)
        cfg.b_mm = float(b)

        azx = None
        azx_zero = None
        for rel in range(32, 128, 8):
            if off + rel + 8 > len(header_bytes):
                break
            try:
                cand = struct.unpack_from("<d", header_bytes, off + rel)[0]
            except struct.error:
                continue
            if not np.isfinite(cand):
                continue
            if -250.0 <= cand <= -1e-6:
                azx = float(cand)
                break
            if -1e-6 < cand <= 1e-6 and azx_zero is None:
                azx_zero = 0.0
        if azx is None:
            azx = azx_zero
        if azx is not None:
            cfg.azx_mm = azx

    # Range XY/Z if it appears textually in the header.
    ranges = [
        f"Range{m.decode('latin-1')}"
        for m in re.findall(rb"Range([1-4])", header_bytes)
    ]
    if ranges:
        cfg.range_xy = ranges[0]
    if len(ranges) > 1:
        cfg.range_z = ranges[1]

    cfg.start_channel = 1 + plate_idx * 8
    cfg.num_channels = 8
    return cfg


def _extract_plate_configs_from_dat(
    raw_bytes: bytes,
    data_offsets: list[int],
    bytes_per_plate: int,
) -> list[PlateConfig]:
    """Generate PlateConfig by platform from headers of the DAT file."""
    # num_plates = len(data_offsets)
    cfgs: list[PlateConfig] = []
    for p_idx, data_off in enumerate(data_offsets):
        if p_idx == 0:
            header_start = 0
        else:
            header_start = data_offsets[p_idx - 1] + bytes_per_plate
        header_end = data_off
        header = raw_bytes[max(0, header_start) : max(0, header_end)]
        fallback = _default_config(p_idx)
        cfg = _extract_plate_config_from_header(header, p_idx, fallback)
        cfgs.append(cfg)
    return cfgs


def compute_plate_outputs_from_forces(
    forces8: np.ndarray,
    cfg: PlateConfig,
    time: np.ndarray,
    sample_rate: float,
    selected_variables: set[str] | None = None,
    skip_moments: bool = False,
    include_raw: bool = True,
) -> PlateData:
    """
    Calculate Fx, Fy, Fz, Mx, My, Mz y COP when the 8 channels are already in newtons.

    Expected channel order in ``forces8``:
        [Fz1, Fz2, Fz3, Fz4, Fx12, Fx34, Fy14, Fy23]

    Parameters
    ----------
    skip_moments : bool
        If True (raw mode), skip the calculation of Fx, Fy, Fz, Mx, My, Mz,
        COPx, COPy (they remain as ``None`` in the resulting ``PlateData``) and
        only expose the 8 raw channels.
    include_raw : bool
        If False, omit saving forces8 in raw_channels (saves memory).
    """
    f_ch = forces8

    if skip_moments:
        # In raw mode, forces8 preserves the raw order of the file
        # [Fx12, Fx34, Fy14, Fy23, Fz1, Fz2, Fz3, Fz4], not the canonical order
        # [Fz1..Fz4, Fx12, Fx34, Fy14, Fy23] that _build_plate_signal_dict assumes.
        # For this reason, Fx/Fy/Fz/moments are not calculated here (they are left as None) and
        # the raw channels are labeled directly with the order from the file.
        Fx = Fy = Fz = Mx = My = Mz = COPx = COPy = None
        signals_all = {"time": time}
        if include_raw:
            signals_all.update(
                {
                    "Fx12": f_ch[:, 0],
                    "Fx34": f_ch[:, 1],
                    "Fy14": f_ch[:, 2],
                    "Fy23": f_ch[:, 3],
                    "Fz1": f_ch[:, 4],
                    "Fz2": f_ch[:, 5],
                    "Fz3": f_ch[:, 6],
                    "Fz4": f_ch[:, 7],
                }
            )
    else:
        Fz = f_ch[:, 0] + f_ch[:, 1] + f_ch[:, 2] + f_ch[:, 3]
        Fx = f_ch[:, 4] + f_ch[:, 5]
        Fy = f_ch[:, 6] + f_ch[:, 7]

        a = cfg.a_mm / 1000.0
        b = cfg.b_mm / 1000.0
        azx = cfg.azx_mm / 1000.0

        My = a * (-f_ch[:, 0] + f_ch[:, 1] + f_ch[:, 2] - f_ch[:, 3])
        Mx = b * (f_ch[:, 0] + f_ch[:, 1] - f_ch[:, 2] - f_ch[:, 3])
        Mz = a * (-f_ch[:, 6] + f_ch[:, 7]) + b * (f_ch[:, 4] - f_ch[:, 5])

        fz_safe = np.where(np.abs(Fz) > 1.0, Fz, np.nan)
        COPx = -(My + Fx * azx) / fz_safe
        COPy = (Mx - Fy * azx) / fz_safe

        signals_all = _build_plate_signal_dict(
            raw8=forces8 if include_raw else None,
            time=time,
            Fx=Fx,
            Fy=Fy,
            Fz=Fz,
            Mx=Mx,
            My=My,
            Mz=Mz,
            COPx=COPx,
            COPy=COPy,
        )

    if selected_variables is None:
        selected_signals = signals_all
    else:
        selected_signals = {
            k: v for k, v in signals_all.items() if k.lower() in selected_variables
        }

    return PlateData(
        name=cfg.name,
        time=time,
        Fx=Fx,
        Fy=Fy,
        Fz=Fz,
        Mx=Mx,
        My=My,
        Mz=Mz,
        Ax=COPx,
        Ay=COPy,
        sample_rate=sample_rate,
        raw_channels=forces8 if include_raw else None,
        signals=selected_signals,
    )


def _build_plate_signal_dict(
    raw8: np.ndarray,
    time: np.ndarray,
    Fx: np.ndarray | None = None,
    Fy: np.ndarray | None = None,
    Fz: np.ndarray | None = None,
    Mx: np.ndarray | None = None,
    My: np.ndarray | None = None,
    Mz: np.ndarray | None = None,
    COPx: np.ndarray | None = None,
    COPy: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Build the set of signals available for a DAT platform.

    Only includes the calculated variables if they are provided (not None).
    """
    result = {"time": time}

    if Fx is not None:
        result["Fx"] = Fx
    if Fy is not None:
        result["Fy"] = Fy
    if Fz is not None:
        result["Fz"] = Fz

    if raw8 is not None:
        result["Fx12"] = raw8[:, 4]
        result["Fx34"] = raw8[:, 5]
        result["Fy14"] = raw8[:, 6]
        result["Fy23"] = raw8[:, 7]
        result["Fz1"] = raw8[:, 0]
        result["Fz2"] = raw8[:, 1]
        result["Fz3"] = raw8[:, 2]
        result["Fz4"] = raw8[:, 3]

    if Mx is not None:
        result["Mx"] = Mx
    if My is not None:
        result["My"] = My
    if Mz is not None:
        result["Mz"] = Mz
        result["Tz"] = Mz

    if COPx is not None:
        result["Ax"] = COPx
    if COPy is not None:
        result["Ay"] = COPy

    return result


def _decode_plate_block(block: bytes, num_samples: int, endian: str) -> np.ndarray:
    """Decode a block of 8 channels x N samples as float32 channel-major."""
    arr = np.frombuffer(block, dtype=f"{endian}f4")
    return arr.reshape(8, num_samples).T


def _infer_sample_rate(raw_bytes: bytes, default_rate: float = 1000.0) -> float:
    """Infer the sampling rate (Hz) from the DAT header."""

    def _read_double(offset: int) -> float | None:
        if offset < 0 or offset + 8 > len(raw_bytes):
            return None
        try:
            val = float(struct.unpack_from("<d", raw_bytes, offset)[0])
        except struct.error:
            return None
        if not np.isfinite(val):
            return None
        if not (50.0 <= val <= 50000.0):
            return None
        return val

    # Offsets observed in various BioWare DATs.
    preferred_offsets = (82, 86, 161, 165)
    preferred = [
        v for v in (_read_double(o) for o in preferred_offsets) if v is not None
    ]
    if preferred:
        # These are often redundant values; use the median for robustness.
        return float(np.median(np.array(preferred, dtype=float)))

    # Generic fallback: brief scan of the header for plausible duplicates.
    generic: list[float] = []
    search_end = min(256, len(raw_bytes) - 8)
    for off in range(max(0, search_end)):
        v = _read_double(off)
        if v is None:
            continue
        # Avoid confusing typical non-frequency fields (e.g., 10.0 in these DATs).
        if v < 20.0:
            continue
        generic.append(v)

    if generic:
        arr = np.array(generic, dtype=float)
        rounded = np.round(arr)
        # Select the value that appears most frequently after rounding, whilst retaining the original decimal places.
        vals, counts = np.unique(rounded, return_counts=True)
        winner = vals[int(np.argmax(counts))]
        close = arr[np.isclose(rounded, winner)]
        return float(np.median(close))

    return default_rate


def _score_force_block(mat: np.ndarray) -> float:
    """Heuristic score for physical plausibility when selecting offset/endianness."""
    finite = np.isfinite(mat)
    if not finite.any():
        return -1e9

    finite_ratio = float(finite.mean())
    abs_vals = np.abs(mat[finite])
    if abs_vals.size == 0:
        return -1e9

    plausible = float((abs_vals < 1e5).mean())
    median_abs = float(np.median(abs_vals))

    # Plausibility of resultant forces with the order observed in BioWare:
    # [Fx12, Fx34, Fy14, Fy23, Fz1, Fz2, Fz3, Fz4]
    with np.errstate(invalid="ignore", over="ignore"):
        fx = mat[:, 0] + mat[:, 1]
        fy = mat[:, 2] + mat[:, 3]
        fz = mat[:, 4] + mat[:, 5] + mat[:, 6] + mat[:, 7]
    fmask = np.isfinite(fx) & np.isfinite(fy) & np.isfinite(fz)

    if not fmask.any():
        return -1e9

    p_fx = float(np.percentile(np.abs(fx[fmask]), 99))
    p_fy = float(np.percentile(np.abs(fy[fmask]), 99))
    p_fz = float(np.percentile(np.abs(fz[fmask]), 99))

    # It promotes realistic biomechanical signals and avoids numerical explosions.
    force_plaus = 0.0
    if p_fx < 2e3:
        force_plaus += 0.5
    if p_fy < 2e3:
        force_plaus += 0.5
    if p_fz < 2e4:
        force_plaus += 1.0

    median_bonus = 0.05 if 1e-4 < median_abs < 2e4 else 0.0
    return 3.0 * finite_ratio + 2.0 * plausible + force_plaus + median_bonus


def _detect_bioware_dat_layout(
    raw_bytes: bytes,
    num_samples: int,
    num_plates: int,
    endian: str,
) -> list[int]:
    """
    Returns the data offsets for each platform in a BioWare DAT file.

    Observed structure (BioWare 5.6.x):
      [Main Header][Data P1][Optional Header P2][Data P2]
    with 8 channels x N samples x float32 per platform.
    """
    bytes_per_plate = num_samples * 8 * 4
    total = len(raw_bytes)
    required = num_plates * bytes_per_plate
    header_total = total - required
    if header_total < 0:
        raise ValueError(
            "The DAT file is smaller than the expected minimum for "
            f"{num_plates} platforms and {num_samples} samples."
        )

    if num_plates == 1:
        return [header_total]

    # P2 in the observed files is at the end of the file.
    off_p2 = total - bytes_per_plate

    # Initial estimate without depending on the platform name.
    marker2 = raw_bytes.find(b"Kistler Grande2")
    sig2 = raw_bytes.find(_HEADER2_SIGNATURE)
    base = off_p2 - bytes_per_plate

    # Priority 1: structural signature at the start of header 2 (agnostic to the name).
    if sig2 >= 0:
        candidate = sig2 - bytes_per_plate
        if 0 <= candidate <= off_p2 - bytes_per_plate:
            block = raw_bytes[candidate : candidate + bytes_per_plate]
            mat = _decode_plate_block(block, num_samples, endian)
            if _score_force_block(mat) >= 6.5:
                return [candidate, off_p2]

    # Priority 2: fallback by classic text marker (BioWare 5.6.x).
    if marker2 >= 0:
        candidate = marker2 - bytes_per_plate - 67
        if 0 <= candidate <= off_p2 - bytes_per_plate:
            block = raw_bytes[candidate : candidate + bytes_per_plate]
            mat = _decode_plate_block(block, num_samples, endian)
            if _score_force_block(mat) >= 6.5:
                return [candidate, off_p2]
        base = marker2 - bytes_per_plate

    lo = max(0, min(base - 256, off_p2 - bytes_per_plate))
    hi = min(off_p2 - bytes_per_plate, max(base + 256, 0))
    if lo > hi:
        lo, hi = 0, off_p2 - bytes_per_plate

    best_off = None
    best_score = -1e9
    for off_p1 in range(lo, hi + 1):
        if off_p1 + bytes_per_plate > off_p2:
            continue
        if marker2 >= 0:
            header2_start = off_p1 + bytes_per_plate
            if not (header2_start <= marker2 < off_p2):
                continue
        if sig2 >= 0:
            header2_start = off_p1 + bytes_per_plate
            if not (header2_start <= sig2 < off_p2):
                continue
        block = raw_bytes[off_p1 : off_p1 + bytes_per_plate]
        mat = _decode_plate_block(block, num_samples, endian)
        score = _score_force_block(mat)
        if score > best_score:
            best_score = score
            best_off = off_p1

    if best_off is None:
        # Full fallback across the entire header range.
        for off_p1 in range(off_p2 - bytes_per_plate + 1):
            if marker2 >= 0:
                header2_start = off_p1 + bytes_per_plate
                if not (header2_start <= marker2 < off_p2):
                    continue
            if sig2 >= 0:
                header2_start = off_p1 + bytes_per_plate
                if not (header2_start <= sig2 < off_p2):
                    continue
            block = raw_bytes[off_p1 : off_p1 + bytes_per_plate]
            mat = _decode_plate_block(block, num_samples, endian)
            score = _score_force_block(mat)
            if score > best_score:
                best_score = score
                best_off = off_p1

    if best_off is None:
        raise ValueError("No valid data offsets could be inferred in the DAT file.")

    return [best_off, off_p2]


def read_kistler_dat(
    file_path: str | Path,
    config_xml: str | Path | None = None,
    tare_samples: int = 100,
    verbose: bool = False,
    n_vars_load: list[str] | None = None,
    raw: bool = False,
    to_dataarray: bool = True,
) -> xr.DataArray | list[PlateData]:
    """
    Reads a native BioWare .dat file (proprietary binary format).

    .. warning::
        The proprietary binary .dat format of BioWare is not publicly documented.
        This function is based on reverse engineering and may not work correctly
        with all versions of the software or platform models.

    The file has the following structure:
        [Header P1 (770 bytes)] [Data P1 (Nx8xfloat32)]
        [Header P2 (variable)] [Data P2 (Nx8xfloat32)]

    The 8 channels per platform correspond to:
        [0..3] Fz1, Fz2, Fz3, Fz4  (vertical forces at 4 corners)
        [4, 5] Fx12, Fx34           (horizontal forces in X)
        [6, 7] Fy14, Fy23           (horizontal forces in Y)

    The data is stored in amplifier units and converted to
    Newtons using the sensitivities from the Configuration.xml.

    Parameters
    ----------
    file_path : str | Path
        Path to the BioWare .dat file.
    config_xml : str | Path, optional
        Path to the Configuration.xml file. If not specified, it is searched for in the
        same directory as the .dat file.
        .
    tare_samples : int
        Number of initial samples used to subtract the resting signal (zero).
    verbose : bool
        Displays diagnostic warnings.
    n_vars_load : list[str], optional
        Variables to display or select in the DataArray. If it is ``None``,
        it returns all available ones.
    raw : bool, default False
        If ``True``, returns the raw sensor data
        ([Fz1, Fz2, Fz3, Fz4, Fx12, Fx34, Fy14, Fy23] per platform).
        If ``False``, returns the calculated variables
        (Fx, Fy, Fz, Mx, My, Mz, Ax, Ay).
    to_dataarray : bool, default True
        If ``True``, returns the result in ``xr.DataArray`` format.
        If ``False``, returns a list of ``PlateData`` objects.

    Returns
    -------
    xr.DataArray | list of PlateData
        An ``xr.DataArray`` (if ``to_dataarray=True``) or a list of ``PlateData``
        objects per platform in the file (if ``to_dataarray=False``).
    """

    file_path = Path(file_path)

    plate_cfgs: list[PlateConfig] = []
    if config_xml is not None:
        plate_cfgs = parse_config_xml(config_xml)
    if not plate_cfgs and verbose:
        warnings.warn(
            "No Configuration.xml found. Default values from 9287CA will be used.",
            UserWarning,
        )

    with open(file_path, "rb") as fh:
        raw_bytes = fh.read()

    total = len(raw_bytes)

    # --- Detect the number of samples and frequency from the header ---
    # In various DAT files observed, num_samples appears at 0x46 or 0x4A.
    num_samples = None
    num_candidates: list[int] = []
    for off in (0x46, 0x4A):
        try:
            val = int(struct.unpack_from("<I", raw_bytes, off)[0])
        except struct.error:
            continue
        if val <= 100:
            continue
        bytes_per_plate_candidate = val * 8 * 4
        if bytes_per_plate_candidate <= 0:
            continue
        max_blocks = total // bytes_per_plate_candidate
        remainder = total - max_blocks * bytes_per_plate_candidate
        if 1 <= max_blocks <= 4 and 64 <= remainder <= 65536:
            num_candidates.append(val)

    if num_candidates:
        # Usa el mayor candidato plausible para evitar valores basura (p.ej. 8).
        num_samples = max(num_candidates)
    else:
        num_samples = 8000
        warnings.warn(
            "No num_samples could be inferred from the DAT file; assuming 8000.",
            UserWarning,
        )

    # Infer the sampling frequency from the DAT header (fallback to 1000 Hz).
    sample_rate = _infer_sample_rate(raw_bytes, default_rate=1000.0)

    num_ch = 8
    bytes_per_plate = num_samples * num_ch * 4

    # Number of platforms robustly inferred from the DAT itself.
    # XML is used only as an aid, never as a requirement.
    max_plates_by_size = max(1, total // bytes_per_plate)
    markers_found = [
        raw_bytes.find(b"Kistler Grande1") >= 0,
        raw_bytes.find(b"Kistler Grande2") >= 0,
    ]
    num_plates_from_markers = max(1, int(sum(markers_found)))
    num_plates_from_cfg = len(plate_cfgs) if plate_cfgs else num_plates_from_markers
    num_plates = min(max_plates_by_size, num_plates_from_cfg)

    # If there is only one platform in the DAT but there are two configs in XML,
    # try to align the correct config by the name in the header.
    if num_plates == 1 and len(plate_cfgs) > 1:
        has_g1 = raw_bytes.find(b"Kistler Grande1") >= 0
        has_g2 = raw_bytes.find(b"Kistler Grande2") >= 0
        if has_g2 and not has_g1:
            match = [cfg for cfg in plate_cfgs if "grande2" in cfg.name.lower()]
            if match:
                plate_cfgs = [match[0]]
        elif has_g1 and not has_g2:
            match = [cfg for cfg in plate_cfgs if "grande1" in cfg.name.lower()]
            if match:
                plate_cfgs = [match[0]]

    # Select global endian from the final platform (the most stable in this format).
    off_p2 = total - bytes_per_plate
    block_p2 = raw_bytes[off_p2 : off_p2 + bytes_per_plate]
    m_le = _decode_plate_block(block_p2, num_samples, "<")
    m_be = _decode_plate_block(block_p2, num_samples, ">")
    endian = "<" if _score_force_block(m_le) >= _score_force_block(m_be) else ">"

    data_offsets = _detect_bioware_dat_layout(
        raw_bytes, num_samples, num_plates, endian
    )

    time = np.arange(num_samples) / sample_rate

    selected_variables = None
    if n_vars_load is not None:
        selected_variables = {v.lower() for v in n_vars_load}
        selected_variables.add("time")

    # Configuration by platform via the DAT itself (priority).
    dat_cfgs = _extract_plate_configs_from_dat(raw_bytes, data_offsets, bytes_per_plate)

    plates: list[PlateData] = []
    for p_idx in range(num_plates):
        off = data_offsets[p_idx] if p_idx < len(data_offsets) else None
        if off is None or off + bytes_per_plate > total:
            warnings.warn(
                f"Data offset for platform {p_idx + 1} is out of bounds; skipping.",
                UserWarning,
            )
            continue

        block = raw_bytes[off : off + bytes_per_plate]
        # DAT BioWare: bloques channel-major (8 canales x N muestras), endian variable.
        # Observed channel order: [Fx12, Fx34, Fy14, Fy23, Fz1, Fz2, Fz3, Fz4].
        raw_chmajor = _decode_plate_block(block, num_samples, endian)

        if raw:
            # Optimisation: if only raw data is requested, avoid reorganisation
            # and derived calculations. Save in the order in which it appears in the file.
            raw8 = raw_chmajor
        else:
            # Reorder to standard order: [Fz1, Fz2, Fz3, Fz4, Fx12, Fx34, Fy14, Fy23]
            raw8 = np.column_stack(
                [
                    raw_chmajor[:, 4],
                    raw_chmajor[:, 5],
                    raw_chmajor[:, 6],
                    raw_chmajor[:, 7],
                    raw_chmajor[:, 0],
                    raw_chmajor[:, 1],
                    raw_chmajor[:, 2],
                    raw_chmajor[:, 3],
                ]
            )

        cfg = dat_cfgs[p_idx] if p_idx < len(dat_cfgs) else _default_config(p_idx)

        # In the observed DAT files, the channels are already in Newtons; do not apply sensitivities.
        # Optimisation: if raw=True, skip moment calculations and include raw_channels
        #             if raw=False, avoid saving raw_channels (save memory)
        plate_data = compute_plate_outputs_from_forces(
            raw8,
            cfg,
            time,
            sample_rate,
            selected_variables=selected_variables,
            skip_moments=raw,
            include_raw=raw,
        )
        plates.append(plate_data)

    if not to_dataarray:
        return plates

    if raw:
        # Orden del archivo DAT: [Fx12, Fx34, Fy14, Fy23, Fz1, Fz2, Fz3, Fz4]
        labels_analog = [
            "Fx12",
            "Fx34",
            "Fy14",
            "Fy23",
            "Fz1",
            "Fz2",
            "Fz3",
            "Fz4",
        ]
        plate_coords = list(range(1, len(plates) + 1))
        data_analog = np.stack([p.raw_channels.T for p in plates], axis=0)

        coords = {
            "plate": plate_coords,
            "n_var": labels_analog,
            "time": time,
        }
        da_analog = xr.DataArray(
            data=data_analog,
            dims=["plate", "n_var", "time"],
            coords=coords,
            attrs={"freq": sample_rate},
        )
        da_analog.attrs["units"] = "N"
        da_analog.time.attrs["units"] = "s"

        if n_vars_load is not None:
            da_analog = da_analog.sel(n_var=n_vars_load)

        return da_analog

    else:
        var_names = ["Fx", "Fy", "Fz", "Mx", "My", "Mz", "Ax", "Ay"]
        plate_coords = list(range(1, len(plates) + 1))
        data_calc = np.stack(
            [[p.Fx, p.Fy, p.Fz, p.Mx, p.My, p.Mz, p.Ax, p.Ay] for p in plates],
            axis=0,
        )

        coords = {
            "plate": plate_coords,
            "n_var": var_names,
            "time": time,
        }
        da_calc = xr.DataArray(
            data=data_calc,
            dims=["plate", "n_var", "time"],
            coords=coords,
            attrs={"freq": sample_rate},
        )
        da_calc.attrs["units"] = "N"
        da_calc.time.attrs["units"] = "s"

        if n_vars_load is not None:
            da_calc = da_calc.sel(n_var=n_vars_load)

        return da_calc


def _default_config(plate_idx: int = 0) -> PlateConfig:
    """Default settings for a Kistler 9287CA without XML."""
    # Typical Range3 sensitivities for the 9287CA
    sr3_z = [3.8, 3.8, 3.8, 3.8, 0, 0, 0, 0]
    sr3_xy = [0, 0, 0, 0, 1.95, 1.95, 1.95, 1.95]
    sens = np.zeros((8, 4))
    for ch in range(8):
        sens[ch, 2] = sr3_z[ch] + sr3_xy[ch]
    return PlateConfig(
        name=f"Kistler Grande{plate_idx + 1}",
        serial="",
        start_channel=1 + plate_idx * 8,
        num_channels=8,
        width_mm=600.0,
        length_mm=900.0,
        a_mm=210.0,
        b_mm=350.0,
        azx_mm=-52.0,
        range_xy="Range3",
        range_z="Range3",
        sensitivities=sens,
    )


# =============================================================================
# %% TESTS
# =============================================================================
if __name__ == "__main__":
    # from biomdp.io.read_kistler_dat import read_kistler_dat

    DATASETS_DIR = Path(__file__).resolve().parents[0] / "tests" / "datasets"
    """
    DATASETS_DIR = Path(r"F:/Programacion/Python/Mios/biomdp/src/biomdp/tests/datasets")        
    """
    file = DATASETS_DIR / "kistler_DJ_2plates.dat"

    daForce = read_kistler_dat(file)
    daForce.plot.line(x="time", col="plate")

    # Load some n_vars_load
    daForce = read_kistler_dat(file, n_vars_load=["Fy", "Fz"])
    daForce.plot.line(x="time", col="plate")

    # Read raw signals
    daSignals = read_kistler_dat(file, raw=True)
    daSignals.plot.line(x="time", col="plate")

    # Return as list of dict
    plates = read_kistler_dat(file, to_dataarray=False)
    plt.plot(plates[0].Fz)
    plt.plot(plates[1].Fz)
    plt.show()

    # Return as list of dict
    plates = read_kistler_dat(file, raw=True, to_dataarray=False)
    plt.plot(plates[0].signals["Fz1"])
    plt.plot(plates[0].signals["Fz2"])
    plt.show()
