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
in `result.metadata["annual"]`. The Python function returns metadata in memory; the treatment batch runner below
saves it separately from classifier inputs. No time-series storage is included.

## Separate stage landscapes and saved treatment batches

The current model is `scripts/simulationV1.py`. Pass `germination_landscape=G`,
`competition_landscape=C`, and `reproduction_landscape=R` to `run_simulation`.
Stage arrays override the shared `landscape` argument. Without a shared array,
all three stage arrays are required. All resolved landscapes must be square and
have identical dimensions. Default optima use each stage's own landscape range;
explicit stage optima override shared optima. The existing lifecycle is unchanged.

The single-run CLI now exposes independent switches (all default on). For example,
this uses constant germination/competition and locally varying reproduction:

```bash
.venv/bin/python scripts/run_simulation.py --landscape data/raw/landscape8.npy --no-abiotic-germination --no-abiotic-competition --abiotic-reproduction --generations 100 --carrying-capacity 4 --snapshot community --output data/simulations/reproduction_only_new.npy
```

Use `--germination-optima`, `--competition-optima`, `--reproduction-optima` for
species vectors, and corresponding `--*-widths` options for standard deviations
(one shared value or one per species). For full, inspectable datasets run:

```bash
.venv/bin/python scripts/run_treatment_batch.py \
  --germination-landscape data/raw/landscape8.npy \
  --competition-landscape data/raw/landscape9.npy \
  --reproduction-landscape data/raw/landscape10.npy \
  --generations 100 --replicates 5 --seed 7 --carrying-capacity 4
```

For a shared G=C=R landscape, replace the three landscape options with
`--landscape data/raw/landscape8.npy`. A stage-specific option can override just
one stage. The batch always runs all eight control combinations, overriding any
individual stage switches. It creates a unique timestamp/UUID directory under
`data/treatment_batches/`; existing results are never overwritten.

The demonstration uses the project's existing 100-generation setting and five
replicates (no prior replicate count was configured). Seeds are 7–11 for each
treatment. All other parameters are held fixed: capacity 4 (the latest requested
setting), adjacent dispersal, staying probability 0.5, lifespan 4, h=1,
fecundities [20,5,20,5], all competition coefficients 1, widths sqrt(2), and
annual flags [true,false,true,false]. Stage default optima and baselines depend
on their input landscapes but do not change between treatments.

G=landscape8, C=landscape9, R=landscape10 are distinct stage inputs **within each
run**. All five replicates reuse this one assignment; they are demographic seed
replicates, not independent landscape replicates. Matching seeds across treatments
do not guarantee matching random events once simulation trajectories diverge.

Every run directory contains:

- `inputs/community.npy`: the **only classifier input**, integer species counts
  with axes `(species, row, column)`, after competition and before mortality.
- `continuation.npz`: final post-mortality `adults` and dispersed `seeds`.
- `metadata.json`: labels `flags_GCR` in germination/competition/reproduction
  order, run/replicate/seed/generation identifiers, stage source paths and hashes,
  resolved optima, widths, unweighted performance baselines, demographic parameters,
  species ordering, annual flags, and code provenance.

The batch-level `manifest.json` records planned and completed runs, assignment,
NumPy version, seed policy, source file/array hashes, commit and dirty status.
The exact simulation/runner source files are copied into `code/`, with SHA-256
hashes. A manifest entry is added only after a run's files are saved. No abiotic
arrays, performance maps, labels, or parameters are inserted into input channels.

For future train/test splits, keep all runs sharing **any** source landscape hash
in the same split, including reuse in different stages or filename aliases. This
single-assignment demonstration cannot support a landscape-independent split;
additional independent landscape sets are needed. Source hashes establish file
and array identity, not whether different generated arrays share an upstream seed.

Generate plots from saved results without rerunning any simulations:

```bash
.venv/bin/python visualizations/plot_treatment_batch.py \
  data/treatment_batches/20260925T152420Z_a4dc5fdc \
  --replicate 1 \
  --output-directory data/treatment_batches/20260925T152420Z_a4dc5fdc/figures
```

For a new batch, substitute the directory printed by the runner. The plotter
saves individual community point maps, separate species abundance panels, and
an eight-treatment comparison for one replicate and assignment. Optional repeated
`--run-id` arguments select individual runs; comparison requires all eight.
Colors, axes, and abundance scales are shared across the selected runs. Every
plant is drawn: within-cell offsets are deterministic display positions, not
simulated subcell coordinates. Figures use the pre-mortality community and contain
no abiotic landscapes. CLI batch/output paths for this plotter are relative to
its working directory; run these commands from the project root.
