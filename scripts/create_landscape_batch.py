"""Save a batch of landscapes generated with shared parameters as NumPy files."""

import argparse

import numpy as np

if __package__:
    from .createlandscape import (
        _next_landscape_number, generate_landscape, resolve_output_directory,
    )
else:
    from createlandscape import (
        _next_landscape_number, generate_landscape, resolve_output_directory,
    )


def generate_landscape_batch(
    count,
    dimension=65,
    fractal_dimension=2.5,
    standard_deviation=2.5,
    output_directory="data/raw",
    seed=None,
):
    """Save count landscapes; a seed reproduces the batch with distinct RNG streams."""
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError("count must be a positive integer")

    directory = resolve_output_directory(output_directory)
    first_number = _next_landscape_number(directory)
    seeds = np.random.SeedSequence(seed).spawn(count)
    paths = []
    for number, landscape_seed in enumerate(seeds, start=first_number):
        landscape = generate_landscape(
            dimension=dimension,
            fractal_dimension=fractal_dimension,
            standard_deviation=standard_deviation,
            seed=landscape_seed,
        )
        path = directory / f"landscape{number}.npy"
        with path.open("xb") as output:
            np.save(output, landscape)
        paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, required=True, help="Number of landscapes")
    parser.add_argument("--dimension", type=int, default=65, help="Grid size: 2^n + 1")
    parser.add_argument("--fractal-dimension", type=float, default=2.5)
    parser.add_argument("--standard-deviation", type=float, default=2.5)
    parser.add_argument(
        "--output-directory", default="data/raw",
        help="Output folder; relative paths start at the project root",
    )
    parser.add_argument("--seed", type=int, help="Optional seed to reproduce the batch")
    args = parser.parse_args()
    try:
        paths = generate_landscape_batch(**vars(args))
    except ValueError as error:
        parser.error(str(error))
    print(f"Saved {len(paths)} landscapes to {paths[0].parent}")


if __name__ == "__main__":
    main()
