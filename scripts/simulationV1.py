"""Spatial plant community simulation V1.

Lifecycle: random initialization, initial competition to carrying capacity, then
repeated germination, competition, reproduction, and dispersal phases.
Competition protects established adults and uses fixed seedling removal weights.
Reproduction samples each individual's fractional seed independently.
For V1, use the same landscape as the abiotic input for G, C, and R.
"""

from pathlib import Path
from dataclasses import dataclass

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_landscape(path):
    """Load a square NumPy landscape; relative paths start at the project root."""
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    landscape = np.load(path, allow_pickle=False)
    _validate_landscape(landscape)
    return landscape


def _validate_landscape(landscape):
    if landscape.ndim != 2 or landscape.shape[0] != landscape.shape[1]:
        raise ValueError("landscape must be a square two-dimensional array")
    if landscape.size == 0:
        raise ValueError("landscape must not be empty")
    if not np.issubdtype(landscape.dtype, np.number) or np.iscomplexobj(landscape):
        raise ValueError("landscape must contain real numeric values")
    if not np.isfinite(landscape).all():
        raise ValueError("landscape must contain only finite values")


def initialize_densities(landscape, number_of_species=4, seed=None):
    """Return integer plant counts with shape (species, rows, columns).

    Place round(landscape.size / number_of_species) individuals per species
    uniformly at random, allowing multiple individuals at a microsite.
    No remainder is distributed (65 x 65 with four species gives 1,056 each).
    A seed or an existing NumPy Generator may be supplied for reproducibility.
    """
    landscape = np.asarray(landscape)
    _validate_landscape(landscape)
    if (
        not isinstance(number_of_species, (int, np.integer))
        or isinstance(number_of_species, (bool, np.bool_))
        or number_of_species <= 0
    ):
        raise ValueError("number_of_species must be a positive integer")

    rng = np.random.default_rng(seed)
    site_count = landscape.size
    totals = np.full(number_of_species, round(site_count / number_of_species), dtype=np.int64)

    densities = np.zeros((number_of_species, *landscape.shape), dtype=np.int64)
    for species, count in enumerate(totals):
        sites = rng.integers(site_count, size=int(count))
        densities[species] = np.bincount(sites, minlength=site_count).reshape(
            landscape.shape
        )
    return densities


def _species_parameter(value, number_of_species, name):
    """Broadcast a scalar or species vector over the spatial dimensions."""
    value = np.asarray(value, dtype=float)
    if value.ndim == 0:
        value = np.full(number_of_species, value.item())
    if value.shape != (number_of_species,) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite and scalar or one value per species")
    return value[:, None, None]


def _validate_performance(performance):
    performance = np.asarray(performance, dtype=float)
    if performance.ndim != 3 or any(size == 0 for size in performance.shape):
        raise ValueError("performance must have shape (species, rows, columns)")
    if not np.isfinite(performance).all() or np.any(
        (performance < 0) | (performance > 1)
    ):
        raise ValueError("performance must be finite and between zero and one")
    return performance


def _validate_densities(densities, shape):
    densities = np.asarray(densities, dtype=float)
    if densities.shape != shape:
        raise ValueError("densities and performance must have the same shape")
    if not np.isfinite(densities).all() or np.any(densities < 0):
        raise ValueError("densities must be finite and nonnegative")
    return densities


def gaussian_performance(landscape, optima, niche_widths=np.sqrt(2)):
    """Compute P_Xi = exp(-(X - mu_Xi)^2 / (2 sigma_Xi^2)).

    optima is a nonempty vector with one entry per species. niche_widths may
    be a positive scalar or a vector of standard deviations (not variances).
    The default sqrt(2) corresponds to the paper's variance sigma^2 = 2. The result
    has shape (species, rows, columns). Call with the same landscape for G,
    C, and R; phase-specific optima and widths can still differ.
    """
    landscape = np.asarray(landscape)
    _validate_landscape(landscape)
    optima = np.asarray(optima, dtype=float)
    if optima.ndim != 1 or optima.size == 0:
        raise ValueError("optima must contain one value per species")
    means = _species_parameter(optima, optima.size, "optima")
    widths = _species_parameter(niche_widths, optima.size, "niche_widths")
    if np.any(widths <= 0):
        raise ValueError("niche_widths must be positive")
    with np.errstate(over="ignore"):
        standardized = (landscape[None, :, :] - means) / widths
        return np.exp(-0.5 * standardized**2)


def germination_probability(performance, h):
    """Return per-seed germination probability g_i = h_i P_Gi.

    h is a maximum germination probability, scalar or one value per species.
    This function does not yet sample germination events or update a seed bank.
    """
    performance = _validate_performance(performance)
    maximum = _species_parameter(h, performance.shape[0], "h")
    if np.any((maximum < 0) | (maximum > 1)):
        raise ValueError("h must be between zero and one")
    return maximum * performance


def competition_pressure(densities, performance, alpha, eligible=None):
    """Return unnormalized removal weights e_j*(alpha_jj*n_j + sum_i!=j alpha_ji*P_Ci*n_i).

    The species axis represents j in the output and i in competitor inputs.
    alpha[j, i] is the effect of competitor i on focal species j. The diagonal
    term is NOT performance-weighted. n is pre-competition adults + seedlings;
    e is the eligible seedling count (defaults to n for initialization).
    The paper describes both total pre-competition density and germinating
    individuals without explicitly separating them in Eq. 2. We interpret the
    outer count as eligible individuals and the bracket as effects of all plants.
    Z is computed by competition_phase as the sum of eligible weights.
    """
    performance = _validate_performance(performance)
    densities = _validate_densities(densities, performance.shape)
    species_count = performance.shape[0]
    alpha = np.asarray(alpha, dtype=float)
    if alpha.shape != (species_count, species_count):
        raise ValueError("alpha must have shape (species, species)")
    if not np.isfinite(alpha).all() or np.any(alpha < 0):
        raise ValueError("competition strengths must be finite and nonnegative")
    eligible = densities if eligible is None else _validate_densities(eligible, densities.shape)
    if np.any(eligible > densities):
        raise ValueError("eligible counts cannot exceed total densities")

    interspecific = alpha.copy()
    np.fill_diagonal(interspecific, 0.0)
    competitors = np.einsum("ji,ixy->jxy", interspecific, performance * densities)
    self_competition = np.diag(alpha)[:, None, None] * densities
    return eligible * (self_competition + competitors)


def individual_fecundity(performance, fecundity):
    """Return expected seeds per individual f_j = F_j P_Rj.

    F may be scalar or species-specific, allowing distinct annual/perennial
    fecundities without prescribing either group's survival rules.
    """
    performance = _validate_performance(performance)
    maximum = _species_parameter(fecundity, performance.shape[0], "fecundity")
    if np.any(maximum < 0):
        raise ValueError("fecundity must be nonnegative")
    return maximum * performance


def expected_seed_production(densities, performance, fecundity):
    """Return n_j F_j P_Rj expected seeds per species and microsite."""
    per_individual = individual_fecundity(performance, fecundity)
    densities = _validate_densities(densities, per_individual.shape)
    return densities * per_individual


def _counts(values):
    values = np.asarray(values)
    if values.ndim != 3 or any(size == 0 for size in values.shape):
        raise ValueError("counts must have shape (species, rows, columns)")
    if not np.issubdtype(values.dtype, np.integer) or np.any(values < 0):
        raise ValueError("counts must be nonnegative integers")
    if np.any(values > np.iinfo(np.int64).max):
        raise ValueError("counts exceed int64 range")
    return values.astype(np.int64, copy=True)


def _annual_flags(annual, species_count):
    annual = np.asarray(annual)
    if annual.shape != (species_count,) or annual.dtype != np.dtype(bool):
        raise ValueError("annual must contain one boolean per species")
    return annual


def _nonnegative_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def competition_phase(densities, performance, alpha, carrying_capacity=3, seed=None, *, adults=None):
    """Return surviving eligible plants; the caller keeps protected adults separately.

    densities contains seedlings, or all plants during initialization. Original
    weights remain fixed; only species with remaining eligible plants enter
    subsequent draws. Z is their weight sum. Adults occupy capacity and exert
    pressure. Overcrowded sites with no positive eligible weight raise an error.
    """
    _nonnegative_integer(carrying_capacity, "carrying_capacity")
    counts = _counts(densities)
    adults = np.zeros_like(counts) if adults is None else _counts(adults)
    if adults.shape != counts.shape or np.any(adults.sum(axis=0) > carrying_capacity):
        raise ValueError("adults must match seedling shape and fit within capacity")
    rates = competition_pressure(adults + counts, performance, alpha, eligible=counts)
    rng = np.random.default_rng(seed)
    while True:
        rows, cols = np.nonzero((adults + counts).sum(axis=0) > carrying_capacity)
        if rows.size == 0:
            return counts
        site_rates = rates[:, rows, cols] * (counts[:, rows, cols] > 0)
        totals = site_rates.sum(axis=0)
        if not np.isfinite(totals).all() or np.any(totals <= 0):
            raise ValueError("overcrowded sites must have finite positive total mortality")
        cumulative = np.cumsum(site_rates, axis=0)
        cumulative /= cumulative[-1]
        deaths = (rng.random(rows.size)[None, :] >= cumulative).sum(axis=0)
        counts[deaths, rows, cols] -= 1


def germination_phase(seeds, performance, h, seed=None):
    """Return (germinated plants, ungerminated seeds) using binomial trials."""
    seeds = _counts(seeds)
    probabilities = germination_probability(performance, h)
    _validate_densities(seeds, probabilities.shape)
    germinated = np.random.default_rng(seed).binomial(seeds, probabilities)
    return germinated, seeds - germinated


def reproduction_phase(densities, performance, fecundity, seed=None):
    """Sample n*floor(f) + Binomial(n, f-floor(f)), where f = F*P_R."""
    counts = _counts(densities)
    fecundity = individual_fecundity(performance, fecundity)
    _validate_densities(counts, fecundity.shape)
    whole = np.floor(fecundity).astype(np.int64)
    return counts * whole + np.random.default_rng(seed).binomial(counts, fecundity - whole)


def dispersal_phase(seeds, annual, mode="adjacent", p=0.5, seed=None):
    """Return (deposited seeds, lost seeds per species).

    Perennial radii are 1, 10, and the landscape dimension for adjacent,
    intermediate, and universal dispersal. Annual radii double only for adjacent
    and intermediate dispersal. Each seed stays with probability p.
    As an implementation assumption, dispersers choose uniformly from
    integer offsets in [-radius, radius]^2 excluding (0, 0); thus distance is
    measured by the maximum absolute row/column offset (diagonals included).
    Out-of-bounds destinations are lost, without wrapping or resampling.
    Universal uses this same offset rule, so it can also lose seeds at edges.
    """
    seeds = _counts(seeds)
    annual = _annual_flags(annual, seeds.shape[0])
    if seeds.shape[1] != seeds.shape[2]:
        raise ValueError("dispersal requires a square landscape")
    radii = {"adjacent": 1, "intermediate": 10, "universal": seeds.shape[1]}
    if mode not in radii:
        raise ValueError("mode must be adjacent, intermediate, or universal")
    if not np.isscalar(p) or not np.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("p must be between zero and one")
    rng = np.random.default_rng(seed)
    moving = rng.binomial(seeds, 1 - p)
    deposited = seeds - moving
    lost = np.zeros(seeds.shape[0], dtype=np.int64)
    dimension = seeds.shape[1]
    for species in range(seeds.shape[0]):
        radius = radii[mode] * (2 if annual[species] and mode != "universal" else 1)
        width = 2 * radius + 1
        center = radius * width + radius
        for row, col in zip(*np.nonzero(moving[species])):
            # Bound temporary memory even for highly fecund populations.
            remaining = int(moving[species, row, col])
            while remaining:
                size = min(remaining, 100_000)
                offsets = rng.integers(width * width - 1, size=size)
                offsets += offsets >= center
                target_rows = row + offsets // width - radius
                target_cols = col + offsets % width - radius
                inside = (
                    (target_rows >= 0) & (target_rows < dimension)
                    & (target_cols >= 0) & (target_cols < dimension)
                )
                np.add.at(deposited[species], (target_rows[inside], target_cols[inside]), 1)
                lost[species] += size - int(inside.sum())
                remaining -= size
    return deposited, lost


def adult_survival_phase(densities, annual, m=4.0, seed=None):
    """Annuals die; each perennial independently survives with probability 1-1/m."""
    counts = _counts(densities)
    annual = _annual_flags(annual, counts.shape[0])
    lifespan = _species_parameter(m, counts.shape[0], "m")
    if np.any(lifespan < 1):
        raise ValueError("m must be at least one")
    survival = np.where(annual[:, None, None], 0.0, 1.0 - 1.0 / lifespan)
    return np.random.default_rng(seed).binomial(counts, survival)


@dataclass
class SimulationResult:
    """Final spatial arrays, each shaped (species, rows, columns).

    community is an independent post-competition, pre-mortality snapshot.
    adults are post-mortality survivors; seeds are dispersed offspring for the
    next year. With zero steps, adults and community both hold the initialized
    post-competition plants (independent arrays), and seeds are empty.
    No time history or persistent seed bank is retained.
    """

    adults: np.ndarray
    seeds: np.ndarray
    community: np.ndarray


def run_simulation(
    landscape, steps, *, optima=None, niche_widths=np.sqrt(2), annual=None,
    h=1.0, fecundity=None, alpha=None, interspecific_strength=1.0,
    number_of_species=4, carrying_capacity=3, dispersal_mode="adjacent",
    p=0.5, m=4.0, seed=None,
):
    """Run initialization, initial competition, and the annual phase loop.

    G=C=R=landscape. Default optima span min(A)+range(A)/S to max(A)-range(A)/S
    evenly (the midpoint for S=1). Explicit optima determine species count;
    otherwise annual flags determine it, or number_of_species is used.
    The four-species baseline is two annuals followed by two perennials; other
    species counts require explicit annual flags. Defaults are h=1, F=20/5 for
    annuals/perennials, niche standard deviation sqrt(2), and m=4.
    alpha defaults to diagonal 1 and symmetric interspecific_strength (0.5, 1,
    or 1.5); a custom matrix overrides this shorthand.

    First-year survivors of initial competition reproduce, disperse, then die
    according to life history. Later years germinate last year's seeds, protect
    established adults during competition, reproduce, disperse, and apply adult
    mortality. Ungerminated seeds are discarded. Reproduction remains locally
    abiotic-dependent. Only the final community, adults, and seeds are returned.

    h, fecundity, and niche_widths accept scalars or species vectors. alpha is
    a species-by-species matrix; annual is a boolean vector. The entire run
    shares one random generator for reproducibility.
    """
    _nonnegative_integer(steps, "steps")
    landscape = np.asarray(landscape)
    _validate_landscape(landscape)
    species_count = len(optima) if optima is not None else (
        len(annual) if annual is not None else number_of_species
    )
    _nonnegative_integer(species_count, "number_of_species")
    if species_count == 0:
        raise ValueError("number_of_species must be positive")
    if optima is None:
        low, high = float(landscape.min()), float(landscape.max())
        margin = (high - low) / species_count
        optima = (
            np.linspace(low + margin, high - margin, species_count)
            if species_count > 1 else [(low + high) / 2]
        )
    if annual is None:
        if species_count != 4:
            raise ValueError("provide annual flags for communities other than four species")
        annual = [True, True, False, False]
    performance = gaussian_performance(landscape, optima, niche_widths)
    annual = _annual_flags(annual, species_count)
    if fecundity is None:
        fecundity = np.where(annual, 20.0, 5.0)
    if alpha is None:
        if interspecific_strength not in (0.5, 1.0, 1.5):
            raise ValueError("interspecific_strength must be 0.5, 1.0, or 1.5; use alpha for custom strengths")
        alpha = np.full((species_count, species_count), interspecific_strength)
        np.fill_diagonal(alpha, 1.0)
    # Validate phase parameters even for zero-step runs.
    germination_probability(performance, h)
    individual_fecundity(performance, fecundity)
    empty = np.zeros(performance.shape, dtype=np.int64)
    adult_survival_phase(empty, annual, m, seed=0)
    dispersal_phase(empty, annual, dispersal_mode, p, seed=0)
    rng = np.random.default_rng(seed)
    adults = initialize_densities(landscape, species_count, seed=rng)
    adults = competition_phase(adults, performance, alpha, carrying_capacity, seed=rng)
    seeds = empty
    community = adults.copy()
    for year in range(steps):
        if year > 0:
            recruits, _ = germination_phase(seeds, performance, h, rng)
            recruits = competition_phase(recruits, performance, alpha, carrying_capacity, seed=rng, adults=adults)
            adults = adults + recruits
        community = adults.copy()
        produced = reproduction_phase(adults, performance, fecundity, rng)
        seeds, _ = dispersal_phase(produced, annual, dispersal_mode, p, rng)
        adults = adult_survival_phase(adults, annual, m, rng)
    return SimulationResult(adults, seeds, community)
