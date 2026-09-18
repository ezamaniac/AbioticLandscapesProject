"""Run V1 for a chosen number of generations and save final spatial plant counts."""

import argparse
from pathlib import Path

import numpy as np

if __package__:
    from .simulationV1 import PROJECT_ROOT, load_landscape, run_simulation
else:
    from simulationV1 import PROJECT_ROOT, load_landscape, run_simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--landscape", required=True, help="Input .npy landscape")
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--carrying-capacity", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7, help="Random seed (default: 7)")
    parser.add_argument("--output", type=Path, help="Output .npy file; existing files are not overwritten")
    parser.add_argument(
        "--snapshot", choices=("adults", "community"), default="adults",
        help="adults: after mortality; community: before mortality, including annuals",
    )
    parser.add_argument(
        "--dispersal-mode", choices=("adjacent", "intermediate", "universal"),
        default="adjacent",
    )
    parser.add_argument("--interspecific-strength", type=float, choices=(0.5, 1.0, 1.5), default=1.0)
    args = parser.parse_args()
    if args.generations < 0:
        parser.error("--generations must be nonnegative")
    if args.carrying_capacity < 0:
        parser.error("--carrying-capacity must be nonnegative")

    output = args.output or Path("data/simulations") / (
        f"{Path(args.landscape).stem}_g{args.generations}_seed{args.seed}_{args.snapshot}.npy"
    )
    output = output.expanduser()
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    if output.suffix != ".npy":
        parser.error("--output must end in .npy")
    if output.exists():
        parser.error(f"output already exists: {output}; choose another --output")

    try:
        landscape = load_landscape(args.landscape)
        result = run_simulation(
            landscape, steps=args.generations, seed=args.seed,
            dispersal_mode=args.dispersal_mode,
            interspecific_strength=args.interspecific_strength,
            carrying_capacity=args.carrying_capacity,
        )
        plants = getattr(result, args.snapshot)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as file:
            np.save(file, plants, allow_pickle=False)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print(f"Saved {args.snapshot} after {args.generations} generations to {output}")
    print(f"Array shape (species, row, column): {plants.shape}")
    print(f"Plant counts per species: {plants.sum(axis=(1, 2)).tolist()}")
    if args.snapshot == "adults" and args.generations > 0:
        print("Annual adults die at year end; use --snapshot community to include annual plants.")


if __name__ == "__main__":
    main()
