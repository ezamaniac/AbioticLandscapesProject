"""Plot final species counts, total density, and local richness from a .npy snapshot."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
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
    parser.add_argument("--view", choices=("panels", "individuals", "dominant"), default="panels")
    args = parser.parse_args()
    figure = plot_snapshot(args.snapshot) if args.view == "panels" else plot_species_map(args.snapshot, args.view)
    if args.output:
        output = project_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=180)
        plt.close(figure)
        print(f"Saved figure to {output}")
    else:
        plt.show()


def plot_species_map(path, view="individuals"):
    """Plot all species together, using consistent colors in both views.

    Individual offsets are display positions within each cell, not simulated
    subcell coordinates. Dominance is the largest count; ties are marked gray.
    """
    counts = np.load(project_path(path), allow_pickle=False)
    if counts.ndim != 3 or any(size == 0 for size in counts.shape):
        raise ValueError("snapshot must have shape (species, rows, columns)")
    if not np.issubdtype(counts.dtype, np.integer) or np.any(counts < 0):
        raise ValueError("snapshot must contain nonnegative integer counts")
    species_count, rows, cols = counts.shape
    colors = [plt.get_cmap("tab10")(i % 10) for i in range(species_count)]
    total = counts.sum(axis=0)
    fig, ax = plt.subplots(figsize=(10, 9), constrained_layout=True)
    legend = [Patch(color=color, label=f"Species {i + 1} ({counts[i].sum():,} plants)")
              for i, color in enumerate(colors)]
    if view == "individuals":
        # Allocate a distinct position for every plant, including conspecifics.
        slots = max(1, int(total.max()))
        side = int(np.ceil(np.sqrt(slots)))
        offset = (np.arange(side) + 0.5) / side - 0.5
        occupied = np.zeros((rows, cols), dtype=int)
        for species, color in enumerate(colors):
            xs, ys = [], []
            for plant in range(int(counts[species].max())):
                r, c = np.nonzero(counts[species] > plant)
                slot = occupied[r, c]
                xs.extend(c + offset[slot % side] * 0.8)
                ys.extend(r + offset[slot // side] * 0.8)
                occupied[r, c] += 1
            ax.scatter(xs, ys, color=color, s=max(2, min(24, (440 / max(rows, cols) / side)**2)),
                       linewidths=0, rasterized=True)
        title = "All plants · one dot per individual"
        note = "Offsets within microsites are for display only. White = empty space."
    elif view == "dominant":
        maximum = counts.max(axis=0)
        tied = ((counts == maximum).sum(axis=0) > 1) & (total > 0)
        labels = counts.argmax(axis=0) + 1
        labels[total == 0] = 0
        labels[tied] = species_count + 1
        cmap = ListedColormap(["white", *colors, "#777777"])
        ax.imshow(labels, origin="lower", interpolation="nearest", cmap=cmap,
                  norm=BoundaryNorm(np.arange(-0.5, species_count + 2.5), cmap.N))
        legend += [Patch(color="white", ec="#aaaaaa", label="Empty"),
                   Patch(color="#777777", label="Tied highest counts")]
        title = "Dominant species at each microsite"
        note = "Dominance uses plant counts; less abundant coexisting species are not shown."
    else:
        raise ValueError("view must be individuals or dominant")
    ax.set(xlim=(-0.5, cols - 0.5), ylim=(-0.5, rows - 0.5),
           xlabel="Column (x)", ylabel="Row (y)", title=title)
    ax.set_aspect("equal")
    ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    fig.suptitle(Path(path).name, fontsize=11)
    fig.supxlabel(note, fontsize=9)
    return fig


if __name__ == "__main__":
    main()
