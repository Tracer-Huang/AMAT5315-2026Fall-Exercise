import random


def estimate_pi(n, seed):
    rng = random.Random(seed)
    inside = sum(
        1
        for _ in range(n)
        if rng.random() ** 2 + rng.random() ** 2 <= 1
    )
    return 1 * inside / n
