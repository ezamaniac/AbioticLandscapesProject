"""Plot final species counts, total density, and local richness from a .npy snapshot."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_landscape(path):
    """Read a 2D NumPy landscape or an existing comma-delimited landscape."""
    path = project_path(path)
    values = np.loadtxt(path, delimiter=",") if path.suffix.lower() == ".csv" else np.load(path, allow_pickle=False)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("landscape must be a finite two-dimensional array")
    return values


def plot_snapshot(path):
    """Return a figure for counts[species, row, column]; rows increase upward.

    Each species has its own panel so coexistence in a microsite is visible.
    Species panels share a discrete count scale. Richness counts species with
    at least one plant in a cell. Snapshot timing is determined by the input file.
    """
    path = project_path(path)
    counts = np.load(path, allow_pickle=False)
    if counts.ndim != 3 or any(size == 0 for size in counts.shape):
        raise ValueError("snapshot must have shape (species, rows, columns)")
    if not np.issubdtype(counts.dtype, np.integer) or np.any(counts < 0):
        raise ValueError("snapshot must contain nonnegative integer plant counts")

    species_count = counts.shape[0]
    total = counts.sum(axis=0)
    richness = (counts > 0).sum(axis=0)
    panels = [counts[i] for i in range(species_count)] + [total, richness]
    titles = [f"Species {i + 1} · {counts[i].sum():,} plants" for i in range(species_count)]
    titles += ["Total plant density", "Local species richness"]
    fig, axes = plt.subplots((len(panels) + 2) // 3, 3, figsize=(15, 4.5 * ((len(panels) + 2) // 3)), squeeze=False,
                             constrained_layout=True)
    for index, (values, title, ax) in enumerate(zip(panels, titles, axes.flat)):
        maximum = max(1, int(counts.max()) if index < species_count else int(values.max()))
        cmap = plt.get_cmap("viridis", maximum + 1)
        norm = BoundaryNorm(np.arange(-0.5, maximum + 1.5), cmap.N)
        image = ax.imshow(values, origin="lower", interpolation="nearest", cmap=cmap, norm=norm)
        fig.colorbar(image, ax=ax, ticks=np.arange(maximum + 1), label="Species per cell" if index == species_count + 1 else "Plants per cell")
        ax.set(title=title, xlabel="Column (x)", ylabel="Row (y)")
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    fig.suptitle(path.name)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", help="Final adults or community .npy file")
    parser.add_argument("--output", help="Save the figure (e.g. .png) instead of opening a window")
    args = parser.parse_args()
    figure = plot_snapshot(args.snapshot)
    if args.output:
        output = project_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=180)
        plt.close(figure)
        print(f"Saved figure to {output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
