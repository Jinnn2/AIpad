from __future__ import annotations

import itertools
import math
from typing import Sequence


def cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for xa, xb in itertools.zip_longest(a, b, fillvalue=0.0):
        dot += xa * xb
        norm_a += xa * xa
        norm_b += xb * xb
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return 1.0 - (dot / math.sqrt(norm_a * norm_b))

