from pathlib import Path

import numpy as np

import settings
from createlandscape import generate_landscape
from simulationV2_helpers import (
    calculate_niche_optima,
    calculate_performance,
    place_initial_plants,
    compete,
    reproduce,
    disperse,
    apply_mortality,
    germinate,
    save_results,
)


def main():
    random_generator = np.random.default_rng(settings.simulation_seed)

    # Create and save the landscape.
    landscape = generate_landscape(
        dimension=settings.landscape_size,
        fractal_dimension=settings.fractal_dimension,
        standard_deviation=settings.landscape_standard_deviation,
        seed=settings.landscape_seed,
    )

    Path(settings.landscape_file).parent.mkdir(
        parents=True, exist_ok=True
    )
    np.savez_compressed(
        settings.landscape_file,
        landscape=landscape,
    )

    # All stages use the same landscape for this first version.
    germination_landscape = landscape
    competition_landscape = landscape
    reproduction_landscape = landscape

    germination_optima = calculate_niche_optima(
        germination_landscape, settings.number_of_species
    )
    competition_optima = calculate_niche_optima(
        competition_landscape, settings.number_of_species
    )
    reproduction_optima = calculate_niche_optima(
        reproduction_landscape, settings.number_of_species
    )

    # Calculate local performance or fixed spatial averages.
    germination_performance = calculate_performance(
        germination_landscape,
        germination_optima,
        settings.germination_niche_variance,
        settings.germination_uses_landscape,
    )
    competition_performance = calculate_performance(
        competition_landscape,
        competition_optima,
        settings.competition_niche_variance,
        settings.competition_uses_landscape,
    )
    reproduction_performance = calculate_performance(
        reproduction_landscape,
        reproduction_optima,
        settings.reproduction_niche_variance,
        settings.reproduction_uses_landscape,
    )

    # Initialize plants and apply initial competition.
    initial_plants = place_initial_plants(
        settings.landscape_size,
        settings.number_of_species,
        random_generator,
    )
    established_adults = np.zeros_like(initial_plants)

    plants = compete(
        initial_plants,
        established_adults,
        competition_performance,
        settings.carrying_capacity,
        settings.intraspecific_competition,
        settings.interspecific_competition,
        random_generator,
    )

    # Complete the first year before entering the main loop.
    new_seeds = reproduce(
        plants,
        reproduction_performance,
        settings.max_seeds_per_plant,
        random_generator,
    )
    seeds = disperse(
        new_seeds,
        settings.annual_species,
        settings.dispersal_type,
        settings.seed_stay_probability,
        random_generator,
    )
    established_adults = apply_mortality(
        plants,
        settings.annual_species,
        settings.perennial_lifespan,
        random_generator,
    )

    # Run years 2 through the final year.
    for year in range(2, settings.years + 1):
        seedlings = germinate(
            seeds,
            germination_performance,
            settings.max_germination_probability,
            random_generator,
        )

        surviving_seedlings = compete(
            seedlings,
            established_adults,
            competition_performance,
            settings.carrying_capacity,
            settings.intraspecific_competition,
            settings.interspecific_competition,
            random_generator,
        )
        plants = established_adults + surviving_seedlings

        new_seeds = reproduce(
            plants,
            reproduction_performance,
            settings.max_seeds_per_plant,
            random_generator,
        )
        seeds = disperse(
            new_seeds,
            settings.annual_species,
            settings.dispersal_type,
            settings.seed_stay_probability,
            random_generator,
        )
        established_adults = apply_mortality(
            plants,
            settings.annual_species,
            settings.perennial_lifespan,
            random_generator,
        )

    # Mortality returns a separate array, so plants still includes annuals.
    final_plants = plants.copy()

    abiotic_effects = np.array([
        settings.germination_uses_landscape,
        settings.competition_uses_landscape,
        settings.reproduction_uses_landscape,
    ])
    landscape_files = np.array([
        settings.landscape_file,
        settings.landscape_file,
        settings.landscape_file,
    ])

    Path(settings.simulation_file).parent.mkdir(
        parents=True, exist_ok=True
    )
    save_results(
        settings.simulation_file,
        final_plants,
        seeds,
        abiotic_effects,
        landscape_files,
        settings.simulation_seed,
        settings.years,
    )

    print(f"Saved simulation to {settings.simulation_file}")
    print("Final plants per species:", final_plants.sum(axis=(1, 2)))


if __name__ == "__main__":
    main()