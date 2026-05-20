"""
io.py – CSV I/O helpers for SVCP angle vectors.

Angles are stored in CSV files in **degrees** (human-readable).
The read function returns a numpy array in degrees by default; pass
``to_radians=True`` to convert on load.
"""

from __future__ import annotations

import numpy as np

__all__ = ["read_angles_csv", "write_angles_csv"]


def read_angles_csv(path: str, to_radians: bool = False) -> np.ndarray:
    """Read sector angles from a CSV file.

    The file may contain a single row or a single column of numeric values.
    Values are interpreted as **degrees** unless the file contains a header
    row with the word "radians".

    Parameters
    ----------
    path : str
        Path to the CSV file.
    to_radians : bool
        If True, convert the loaded degree values to radians before returning.

    Returns
    -------
    np.ndarray
        Float64 1-D array of angles (degrees unless *to_radians* is True).
    """
    data = np.genfromtxt(path, delimiter=",", dtype=np.float64)
    alpha = data.flatten()
    alpha = alpha[~np.isnan(alpha)]
    if to_radians:
        alpha = np.radians(alpha)
    return alpha


def write_angles_csv(path: str, alpha: np.ndarray, in_radians: bool = False) -> None:
    """Write sector angles to a CSV file (stored in degrees).

    Parameters
    ----------
    path : str
        Output file path.
    alpha : np.ndarray
        1-D array of angles. If *in_radians* is True, values are converted to
        degrees before writing.
    in_radians : bool
        Set True if *alpha* is given in radians.
    """
    alpha = np.asarray(alpha, dtype=np.float64).flatten()
    if in_radians:
        alpha = np.degrees(alpha)
    np.savetxt(path, alpha.reshape(1, -1), delimiter=",", fmt="%.10f")