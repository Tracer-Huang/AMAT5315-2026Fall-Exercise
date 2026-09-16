"""Statistical estimators. ACF uses biased covariance (denominator n)."""

import numpy as np


def susceptibility(m, l, t):
    m = np.asarray(m)
    return l * l * (np.mean(m * m) - np.mean(np.abs(m)) ** 2) / t


def fit_peak(t, y):
    t = np.asarray(t)
    y = np.asarray(y)
    if (
        t.ndim != 1
        or y.shape != t.shape
        or len(t) < 5
        or not np.isfinite(t).all()
        or not np.isfinite(y).all()
        or not np.all(np.diff(t) > 0)
    ):
        raise ValueError("finite increasing grid and matching finite values required")
    i = int(np.argmax(y))
    if i < 2 or i > len(t) - 3:
        raise ValueError("maximum has fewer than two neighbors")
    x = t[i - 2 : i + 3]
    center = x[2]
    coef = np.polyfit(x - center, y[i - 2 : i + 3], 2)
    if coef[0] >= 0:
        raise ValueError("peak fit does not bend downward")
    vertex = center - coef[1] / (2 * coef[0])
    if not x[0] <= vertex <= x[-1]:
        raise ValueError("peak outside fit interval")
    return float(vertex), coef, float(center), x


def peak(t, y):
    return fit_peak(t, y)[0]


def acf(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("finite one-dimensional series of length >=2 required")
    n = len(x)
    if np.ptp(x) == 0:
        return np.full(n, np.nan)
    x = x - x.mean()
    if np.dot(x, x) == 0:
        return np.full(n, np.nan)
    f = np.fft.rfft(x, n=1 << (2 * n - 1).bit_length())
    c = np.fft.irfft(f * f.conj())[:n]
    return c / c[0]


def tau_int(x):
    r = acf(x)
    if not np.isfinite(r[0]):
        return float("nan")
    total = 0.5
    for lag in range(1, len(r)):
        total += r[lag]
        if lag > 6 * total:
            return float(total) if total > 0 else float("nan")
    return float("nan")


def block_se(x, b):
    x = np.asarray(x)
    n = len(x) // b
    if n < 2:
        return float("nan")
    means = x[: n * b].reshape(n, b).mean(1)
    return means.std(ddof=1) / np.sqrt(n)


def bootstrap_moments(m, b, reps, rng):
    """Circular moving-block bootstrap; retain original n including final partial block."""
    m = np.asarray(m)
    n = len(m)
    if b > n:
        raise ValueError("block longer than chain")
    a = np.abs(m)
    sq = m * m
    whole, rem = divmod(n, b)
    starts = rng.integers(n, size=(reps, whole))

    def sums(x, length, ix):
        cs = np.r_[0.0, np.cumsum(np.r_[x, x[:length]])]
        return cs[ix + length] - cs[ix]

    am = sums(a, b, starts).sum(1)
    qm = sums(sq, b, starts).sum(1)
    if rem:
        ix = rng.integers(n, size=reps)
        am += sums(a, rem, ix)
        qm += sums(sq, rem, ix)
    return am / n, qm / n


def bootstrap_chi(m, l, t, b, reps, rng):
    a, q = bootstrap_moments(m, b, reps, rng)
    return l * l * (q - a * a) / t


def stable(errors):
    e = np.asarray(errors)
    return bool(
        e.ndim == 1
        and len(e) >= 2
        and np.isfinite(e).all()
        and np.all(e > 0)
        and (e.max() - e.min()) <= 0.1 * e.mean()
    )
