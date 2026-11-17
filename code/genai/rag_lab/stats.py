"""Wilson score intervals for proportions and the exact McNemar test for paired outcomes."""
from __future__ import annotations

import math


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar P from the discordant pair counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def newcombe_paired_diff(a: int, b: int, c: int, d: int, z: float = 1.959964) -> tuple[float, float, float]:
    """Difference in paired proportions (p2 - p1) with Newcombe's method 10 interval.
    a: both pass, b: first passes only, c: second passes only, d: both fail."""
    n = a + b + c + d
    if n == 0:
        return (float("nan"),) * 3
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n, z)
    l2, u2 = wilson(a + c, n, z)
    diff = p2 - p1
    denom = (a + b) * (c + d) * (a + c) * (b + d)
    if denom == 0:
        phi = 0.0
    else:
        num = a * d - b * c
        num = max(abs(num) - n / 2, 0) * (1 if num >= 0 else -1)
        phi = num / math.sqrt(denom)
    dl = math.sqrt(max(0.0, (p2 - l2) ** 2 - 2 * phi * (p2 - l2) * (u1 - p1) + (u1 - p1) ** 2))
    du = math.sqrt(max(0.0, (u2 - p2) ** 2 - 2 * phi * (u2 - p2) * (p1 - l1) + (p1 - l1) ** 2))
    return diff, max(-1.0, diff - dl), min(1.0, diff + du)
