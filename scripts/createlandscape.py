"""Generate spatially autocorrelated landscapes with midpoint displacement."""

from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_output_directory(output_directory="data/raw"):
    """Create an output directory, resolving relative paths from the project root."""
    directory = Path(output_directory).expanduser()
    if not directory.is_absolute():
        directory = PROJECT_ROOT / directory
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _add_noise(values, filled, scale, rng):
    """Apply additive succession to every assigned cell."""
    values[filled] += rng.normal(0.0, scale, filled.sum())


def _next_landscape_number(output_directory):
    """Continue numbering across NumPy files and existing landscape CSVs."""
    numbers = [
        int(path.stem[9:])
        for path in Path(output_directory).glob("landscape*.*")
        if path.suffix in {".npy", ".csv"} and path.stem[9:].isdigit()
    ]
    return max(numbers, default=0) + 1


def _diamond_step(values, filled, step, scale, rng):
    """Fill square centers from their four diagonal neighbors."""
    half = step // 2
    for row in range(half, values.shape[0] - 1, step):
        for col in range(half, values.shape[1] - 1, step):
            neighbors = values[[row - half, row - half, row + half, row + half],
                               [col - half, col + half, col - half, col + half]]
            values[row, col] = neighbors.mean() + rng.normal(0.0, scale)
            filled[row, col] = True


def _square_step(values, filled, step, scale, rng):
    """Fill edge midpoints from their available orthogonal neighbors."""
    half = step // 2
    size = values.shape[0]
    for row in range(0, size, half):
        start = half if (row // half) % 2 == 0 else 0
        for col in range(start, size, step):
            neighbors = []
            for dr, dc in ((-half, 0), (half, 0), (0, -half), (0, half)):
                r, c = row + dr, col + dc
                if 0 <= r < size and 0 <= c < size and filled[r, c]:
                    neighbors.append(values[r, c])
            values[row, col] = np.mean(neighbors) + rng.normal(0.0, scale)
            filled[row, col] = True


def generate_landscape(
    dimension=65,
    fractal_dimension=2.5,
    standard_deviation=2.5,
    additive_succession=True,
    seed=None,
):
    """Return a normalized midpoint-displacement landscape."""
    levels = np.log2(dimension - 1)
    if dimension < 3 or not levels.is_integer():
        raise ValueError("dimension must equal 2^n + 1")
    if not 2 <= fractal_dimension <= 3:
        raise ValueError("fractal_dimension must be between 2 and 3")
    if standard_deviation <= 0:
        raise ValueError("standard_deviation must be positive")

    rng = np.random.default_rng(seed)
    values = np.zeros((dimension, dimension), dtype=float)
    filled = np.zeros_like(values, dtype=bool)
    corner_values = np.array([0.01, 1.225, -1.225, 0.01])
    values[0, 0], values[0, -1], values[-1, 0], values[-1, -1] = corner_values
    filled[0, 0] = filled[0, -1] = filled[-1, 0] = filled[-1, -1] = True

    hurst = 3 - fractal_dimension
    scale = np.var(corner_values, ddof=1)
    step = dimension - 1

    while step > 1:
        scale = 0.0 if fractal_dimension == 2 else scale * 0.5 ** (hurst / 2)
        _diamond_step(values, filled, step, scale, rng)
        if additive_succession:
            _add_noise(values, filled, scale, rng)

        scale *= 0.5 ** (hurst / 2)
        _square_step(values, filled, step, scale, rng)
        if additive_succession:
            _add_noise(values, filled, scale, rng)
        step //= 2

    values = (values - values.mean()) / values.std(ddof=1)
    return values * standard_deviation


if __name__ == "__main__":
    landscape = generate_landscape(seed=7)
    output_directory = resolve_output_directory()
    landscape_number = _next_landscape_number(output_directory)
    output_path = output_directory / f"landscape{landscape_number}.npy"
    np.save(output_path, landscape)
    
