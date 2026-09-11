"""Spatial plant community simulation V1.

Lifecycle: random initialization, initial competition to carrying capacity, then
repeated germination, competition, reproduction, and dispersal phases.
Competition removes individuals until carrying capacity is met, with species
death-event rates given by r_j. Reproduction uses Poisson seed counts.
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

    Place one landscape's worth of individuals uniformly at random, allowing
    multiple individuals at a microsite. Each species receives approximately
    landscape.size / number_of_species individuals. Randomly selected species
    receive any remainder, keeping the total equal to the number of sites.
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
    per_species, remainder = divmod(site_count, number_of_species)
    totals = np.full(number_of_species, per_species, dtype=np.int64)
    if remainder:
        totals[rng.choice(number_of_species, size=remainder, replace=False)] += 1

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


def gaussian_performance(landscape, optima, niche_widths):
    """Compute P_Xi = exp(-(X - mu_Xi)^2 / (2 sigma_Xi^2)).

    optima is a nonempty vector with one entry per species. niche_widths may
    be a positive scalar or a vector with one entry per species. The result
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


def competition_pressure(densities, performance, alpha, z):
    """Evaluate r_j = n_j/z * (alpha_jj*n_j + sum_i!=j alpha_ji*P_Ci*n_i).

    The species axis represents j in the output and i in competitor inputs.
    alpha[j, i] is the effect of competitor i on focal species j. The diagonal
    term is NOT performance-weighted. Here n_j is supplied as species j's
    density at each site; no conversion of r_j to deaths is imposed.
    z must be a positive scalar. Pressures are not clipped to probabilities.
    """
    performance = _validate_performance(performance)
    densities = _validate_densities(densities, performance.shape)
    species_count = performance.shape[0]
    alpha = np.asarray(alpha, dtype=float)
    if alpha.shape != (species_count, species_count):
        raise ValueError("alpha must have shape (species, species)")
    if not np.isfinite(alpha).all() or np.any(alpha < 0):
        raise ValueError("competition strengths must be finite and nonnegative")
    z = np.asarray(z, dtype=float)
    if z.ndim != 0 or not np.isfinite(z) or z <= 0:
        raise ValueError("z must be a finite positive scalar")

    interspecific = alpha.copy()
    np.fill_diagonal(interspecific, 0.0)
    competitors = np.einsum("ji,ixy->jxy", interspecific, performance * densities)
    self_competition = np.diag(alpha)[:, None, None] * densities
    return densities / z * (self_competition + competitors)


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


def competition_phase(densities, performance, alpha, carrying_capacity=3, z=1.0, seed=None):
    """Remove one plant per overcrowded site per round, recomputing r_j.

    Treat r_j as the total death-event rate for species j at a site. Conditional
    on a death, its species is sampled with probability r_j / sum(r). Sites at
    or below capacity have no competition deaths. This runs to capacity rather
    than for a fixed duration, so a shared z changes no survival probabilities.
    Zero total mortality at an overcrowded site raises an error.
    """
    _nonnegative_integer(carrying_capacity, "carrying_capacity")
    counts = _counts(densities)
    rates = competition_pressure(counts, performance, alpha, z)
    rng = np.random.default_rng(seed)
    while True:
        rows, cols = np.nonzero(counts.sum(axis=0) > carrying_capacity)
        if rows.size == 0:
            return counts
        site_rates = rates[:, rows, cols]
        totals = site_rates.sum(axis=0)
        if not np.isfinite(totals).all() or np.any(totals <= 0):
            raise ValueError("overcrowded sites must have finite positive total mortality")
        cumulative = np.cumsum(site_rates / totals, axis=0)
        cumulative[-1] = 1.0
        deaths = (rng.random(rows.size)[None, :] >= cumulative).sum(axis=0)
        counts[deaths, rows, cols] -= 1
        rates = competition_pressure(counts, performance, alpha, z)


def germination_phase(seeds, performance, h, seed=None):
    """Return (germinated plants, ungerminated seeds) using binomial trials."""
    seeds = _counts(seeds)
    probabilities = germination_probability(performance, h)
    _validate_densities(seeds, probabilities.shape)
    germinated = np.random.default_rng(seed).binomial(seeds, probabilities)
    return germinated, seeds - germinated


def reproduction_phase(densities, performance, fecundity, seed=None):
    """Sample Poisson seeds with mean n_j F_j P_Rj at each microsite."""
    counts = _counts(densities)
    means = expected_seed_production(counts, performance, fecundity)
    return np.random.default_rng(seed).poisson(means)


def dispersal_phase(seeds, annual, mode="adjacent", p=0.5, seed=None):
    """Return (deposited seeds, lost seeds per species).

    Perennial radii are 1, 10, and the landscape dimension for adjacent,
    intermediate, and universal dispersal. Annual radii are twice these.
    Each seed disperses with probability p. Dispersers choose uniformly from
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
    moving = rng.binomial(seeds, p)
    deposited = seeds - moving
    lost = np.zeros(seeds.shape[0], dtype=np.int64)
    dimension = seeds.shape[1]
    for species in range(seeds.shape[0]):
        radius = radii[mode] * (2 if annual[species] else 1)
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


def adult_survival_phase(densities, annual, m=3.0, seed=None):
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
    """Final adults and seeds, each shaped (species, rows, columns).

    Adults are recorded after end-of-year mortality; seeds are available for
    the next step. No time history is retained.
    """

    adults: np.ndarray
    seeds: np.ndarray


def run_simulation(
    landscape, steps, *, optima, niche_widths, annual, h, fecundity, alpha,
    carrying_capacity=3, z=1.0, dispersal_mode="adjacent", p=0.5, m=3.0,
    retain_ungerminated=False, seed=None,
):
    """Run initialization, initial competition, and the annual phase loop.

    Species count is determined by optima. G=C=R=landscape, using shared optima
    and widths for all three phases. Initial plants are reproductive adults;
    the initial seed pool is empty. Each step germinates seeds, adds recruits
    to surviving adults, competes to capacity, reproduces, and disperses new
    seeds. Adult survival is applied after reproduction/dispersal. Ungerminated
    seeds are discarded by default or retained unchanged when requested.

    h, fecundity, and niche_widths accept scalars or species vectors. alpha is
    a species-by-species matrix; annual is a boolean vector. The entire run
    shares one random generator for reproducibility.
    """
    _nonnegative_integer(steps, "steps")
    performance = gaussian_performance(landscape, optima, niche_widths)
    species_count = performance.shape[0]
    annual = _annual_flags(annual, species_count)
    # Validate phase parameters even for zero-step runs.
    germination_probability(performance, h)
    individual_fecundity(performance, fecundity)
    empty = np.zeros(performance.shape, dtype=np.int64)
    adult_survival_phase(empty, annual, m, seed=0)
    dispersal_phase(empty, annual, dispersal_mode, p, seed=0)
    rng = np.random.default_rng(seed)
    adults = initialize_densities(landscape, species_count, seed=rng)
    adults = competition_phase(adults, performance, alpha, carrying_capacity, z, rng)
    seeds = empty
    for _ in range(steps):
        recruits, remaining = germination_phase(seeds, performance, h, rng)
        adults = competition_phase(adults + recruits, performance, alpha, carrying_capacity, z, rng)
        produced = reproduction_phase(adults, performance, fecundity, rng)
        seeds, _ = dispersal_phase(produced, annual, dispersal_mode, p, rng)
        if retain_ungerminated:
            seeds += remaining
        adults = adult_survival_phase(adults, annual, m, rng)
    return SimulationResult(adults, seeds)
