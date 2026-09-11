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
