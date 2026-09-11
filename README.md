# AbioticLandscapesProject
A repository for the code included in the work in spring 2026

Generate a batch with shared fractal dimension and sample standard deviation:

```bash
python scripts/create_landscape_batch.py --count 100 --fractal-dimension 2.5 --standard-deviation 2.5 --seed 7
```

Files are saved as `data/raw/landscapeN.npy`, continuing the numbering of existing
CSV and NumPy landscapes. Use `--output-directory data/raw/experiment1` to keep a
batch in its own folder. Relative output paths are resolved from the project
root, even when the script runs from another directory; missing folders are created.
Use `--dimension 129` to change the default 65 × 65 grid (must be `2^n + 1`).
Omit `--seed` for a fresh random batch, or reuse it to reproduce the same arrays.
Each landscape uses a separate random stream with the same generation parameters.
Load a saved array with `np.load("data/raw/landscape1.npy")`.

Run simulation V1 from Python with explicit species parameters. These example
values illustrate configuration; they are not fitted biological parameters:

```python
import numpy as np
from scripts.simulationV1 import load_landscape, run_simulation

landscape = load_landscape("data/raw/landscape1.npy")  # choose an existing .npy file
result = run_simulation(
    landscape, steps=100,
    optima=[-1.5, -0.5, 0.5, 1.5],
    niche_widths=1.0,
    annual=[True, True, False, False],
    h=0.8,
    fecundity=[10, 10, 5, 5],
    alpha=np.ones((4, 4)),
    carrying_capacity=3,
    dispersal_mode="adjacent",  # also "intermediate" or "universal"
    p=0.5,
    m=3,
    seed=7,
)
print(result.adults.sum(axis=(1, 2)))  # final surviving adults per species
print(result.seeds.sum(axis=(1, 2)))   # final seeds per species
```

Run the import from the project root. In a notebook launched inside
`visualizations`, first add the project root with
`import sys; sys.path.insert(0, str(Path.cwd().parent))` (after importing `Path`
from `pathlib`). Landscape input paths resolve from the project root.

V1 initializes plants, competes to capacity, then repeats germination,
competition, reproduction, and dispersal. Initial plants are adults and the
initial seed pool is empty. The same landscape, optima, and niche widths govern
all three performance functions. Germination uses binomial trials and seed
production uses Poisson sampling with mean `n_j * F_j * P_Rj`.

Competition treats `r_j` as a species' total death-event rate at a microsite:
one plant dies with species probabilities proportional to the current rates,
which are recomputed until the site reaches capacity. There are no competition
deaths below capacity. A common scalar `z` (default 1) cancels from these
probabilities; this implementation does not model elapsed competition time.
Overcrowded sites with zero total mortality raise an error.

Dispersers choose uniformly among square-neighborhood offsets, excluding their
parent's cell. Perennial maximum row/column offsets are 1, 10, or the landscape
dimension; annual offsets are twice as large. Destinations outside the landscape
are lost, including in universal mode. Seeds that do not disperse stay in the
parent cell. Annuals die after reproduction/dispersal; perennials survive with
probability `1 - 1/m`. Ungerminated seeds are discarded unless
`retain_ungerminated=True`, which retains them without additional seed mortality.

The result contains only final `adults` (after end-of-year mortality) and `seeds`
(available for the next step), each shaped `(species, rows, columns)`. No history
is retained, and results are returned in memory without automatic disk output.
Individual phases are also available as functions for experiments with different
dynamics.
