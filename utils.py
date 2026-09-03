"""Array-shaping helpers shared by the transforms kernels.

Extracted from :mod:`rl.math.utils` so this package stands on its own --
only the two helpers :func:`_set_dimension` and :func:`_match_depth` are
used here, and both are pure NumPy.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _set_dimension(
    data: Any,
    ndim: int = 1,
    dtype: np.ndarray = np.float64,
    reshape_matrix: bool = False,
) -> np.ndarray:
    """Sets input data to expected dimension"""
    data = np.asarray(data, dtype=dtype)

    while data.ndim < ndim:
        data = data[np.newaxis]

    # For when matrices are given as lists of 16 floats
    if reshape_matrix:
        if data.shape[-1] == 16:
            data = data.reshape(-1, 4, 4)

    return data


def _match_depth(*data) -> list:
    """Levels given data to a common length, NumPy broadcasting rules.

    It is assumed all entries are already numpy arrays whose first axis is
    the batch axis. An input is accepted when its length equals the longest
    length, or is exactly 1; a length-1 input is expanded with
    :func:`numpy.broadcast_to`, which is a zero-copy read-only view.

    Any other length raises. Padding a short input -- by repeating its last
    row, recycling it, or truncating the long one -- turns an off-by-one in
    the caller into plausible-looking numbers instead of an error, so it is
    not done here. This matches :mod:`numpy` and
    :class:`scipy.spatial.transform.Rotation`.
    """
    counts = [len(d) for d in data]
    highest = max(counts)

    if any(c not in (1, highest) for c in counts):
        raise ValueError(
            f"length mismatch: got {counts}; every input must be "
            f"length {highest} or 1"
        )

    return [
        d if len(d) == highest else np.broadcast_to(d, (highest,) + d.shape[1:])
        for d in data
    ]
