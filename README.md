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

Run simulation V1 from the project root:

```python
from scripts.simulationV1 import load_landscape, run_simulation

landscape = load_landscape("data/raw/landscape8.npy")  # choose an existing file
result = run_simulation(landscape, steps=100, seed=7)
print("Abundance:", result.community.sum(axis=(1, 2)))
print("Richness:", (result.community.sum(axis=(1, 2)) > 0).sum())
```

The baseline uses four species (annual, perennial, annual, perennial), carrying
capacity 3, maximum germination probability `h=1`, annual/perennial fecundities
20/5, and perennial mean lifespan `m=4`. `niche_widths` is a **standard deviation**:
the default `sqrt(2)` corresponds to variance 2. Default optima are evenly spaced
from `min(A) + range(A)/S` to `max(A) - range(A)/S` (the midpoint for one species).
Each species starts with `round(number_of_microsites / S)` plants: 1,056 each
on a 65 × 65 landscape with four species, without distributing a remainder.

Override `optima`, `niche_widths`, `annual`, `h`, `fecundity`, and `alpha` as needed.
Explicit optima determine species count; otherwise it is inferred from `annual`
or defaults to `number_of_species=4`. Other community sizes need explicit annual
flags. The shortcut `interspecific_strength=0.5`, `1.0` (default), or `1.5` builds
a symmetric matrix with diagonal 1; an explicit `alpha` matrix takes precedence.

Initialization competes all placed plants to capacity. In year one, survivors
reproduce, disperse, and undergo mortality. In later years, germinated seedlings
compete for the capacity left by established adults, which exert competition but
cannot be removed. Removal weights are calculated once and remain fixed during
that phase; species without remaining seedlings are excluded and the remaining
original weights normalized. `Z` is this normalization, not a model parameter.

Competition uses total pre-competition abundance `N = adults + seedlings` in
both the outer multiplier and bracket:
`w_j = N_j * (alpha_jj*N_j + sum_i!=j alpha_ji*P_Ci*N_i)`.
Only species with eligible seedlings enter removal draws. Adults are protected;
weights remain fixed, and species are excluded when their seedlings run out.
Alpha occurs once per interspecific term; the intraspecific term is independent
of performance.

Each life stage independently uses either its local Gaussian performance or its
unweighted spatial mean. The default is local for all stages. Constant baselines
average performance over **all microsites**, once before initialization; they do
not depend on occupancy or receive the paper's preferred-habitat adjustment.
`h`, `F`, and `alpha` are applied only in the corresponding lifecycle calculation.
Fractional reproduction remains `n*floor(F*P_R) + Binomial(n, frac(F*P_R))`.
Ungerminated seeds are discarded.

`p` now means the probability of **staying** in the parent cell (default 0.5).
Dispersal modes `adjacent`, `intermediate`, and `universal` use perennial radii
1, 10, and the landscape dimension. Annual radii double only in the first two
modes. Uniform square integer offsets excluding the parent cell remain an
implementation assumption; out-of-bounds seeds are lost without resampling,
including in universal mode. Annuals die after reproduction/dispersal, and
perennials survive independently with probability `1 - 1/m`.

Results contain three arrays shaped `(species, rows, columns)`:

- `community`: an independent final post-competition snapshot, before mortality;
  use this for community abundance and richness, including annuals.
- `adults`: final post-mortality survivors.
- `seeds`: final dispersed offspring available for the next year.

With `steps=0`, `community` and `adults` independently contain the initialized
post-competition plants and `seeds` is empty. No history, persistent seed bank,
or automatic disk output is included. The old `z` and `retain_ungerminated`
arguments have been removed. Results remain reproducible for a fixed seed,
although corrected biological rules change results relative to earlier versions.

Run focused model checks with `.venv/bin/python -m unittest discover -s tests -v`.

To run and save final plant locations from the terminal:

```bash
.venv/bin/python scripts/run_simulation.py --landscape data/raw/landscape8.npy --generations 100 --seed 7
```

This saves `data/simulations/landscape8_g100_seed7_adults.npy`. Each entry
`plants[species, row, column]` gives the number of surviving adults at that
microsite, preserving multiple plants per cell. For new runs, species indices 0 and 2 are
annuals and 1 and 3 are perennials. Older saved snapshots retain their original
species ordering and are not regenerated by this change. Annual adults
are absent after end-of-year mortality; use `--snapshot community` for the final
pre-mortality community including annuals. Zero generations saves initialization
after competition. No history or seeds are written by this runner.

Use `--output data/simulations/my_run.npy` to choose a filename. Relative input
and output paths resolve from the project root; existing output files are not
overwritten. Optional `--dispersal-mode` and `--interspecific-strength` select
the model's dispersal and competition treatments. Other parameters use V1 defaults.

```python
import numpy as np

plants = np.load("data/simulations/landscape8_g100_seed7_adults.npy")
species, rows, columns = np.nonzero(plants)  # occupied species/microsite combinations
counts = plants[species, rows, columns]
```

Open `visualizations/LandscapeModelV1.ipynb` and run its cells to view the abiotic
landscape and final plant structure. Set `landscape_path` and `snapshot_path` in
the settings cell to matching files. Landscape loading supports both `.npy`
and legacy `.csv` files; the optional Catella CSV comparison is preserved.
The final-state figure shows counts by species, total density, and local richness.
Use a `community` snapshot to include annuals before mortality.

You can also export the final-state figure directly:

```bash
.venv/bin/python visualizations/plot_simulation.py data/simulations/landscape8_g100_seed7_adults.npy --output data/simulations/final_structure.png
```

Compare all-constant performance with reproduction-only abiotic dependence:

```python
common = dict(steps=100, seed=7, abiotic_germination=False, abiotic_competition=False)
constant = run_simulation(landscape, abiotic_reproduction=False, **common)
reproduction_only = run_simulation(landscape, abiotic_reproduction=True, **common)
```

Stage-specific `germination_optima`, `competition_optima`, and
`reproduction_optima` override shared `optima`. Likewise `germination_widths`,
`competition_widths`, and `reproduction_widths` override `niche_widths` (standard
deviation, default `sqrt(2)`). Missing stage optima use shared optima, or the
landscape-derived defaults when shared optima are absent. For example,
`competition_optima=[mu[2], mu[1], mu[0], mu[3]]` swaps the competition preferences
of species 1 and 3 without changing their identities. All vectors must agree on
species count; explicit annual flags remain unchanged.

Inspect `result.metadata["germination"]` (or `"competition"`, `"reproduction"`)
for resolved `optima`, `niche_widths`, `baseline`, and `abiotic`. Annual flags are
in `result.metadata["annual"]`. Metadata stays in memory; this change introduces
no metadata export or time-series storage. All stages still use the same landscape.
