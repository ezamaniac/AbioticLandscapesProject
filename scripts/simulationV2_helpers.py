"""Helper functions for the plant community simulation.

Plant and seed counts use the array layout:
    counts[species, row, column]

Performance arrays use the same layout and contain values from 0 to 1.
Niche breadth is expressed as variance.

Established adults contribute to competition but are protected from removal.
Removal weights use total pre-competition counts and remain fixed while
eligible plants are removed.

Functions receive their settings as arguments. Random events use the
NumPy generator created in simulationV2.py.
"""

import numpy as np


"""
Calculate the niche optima for a the given number of species
@param landscape: 2D array of environmental values
@param number_of_species: number of species in the simulation
@return: 1D array of niche optima for each species
"""

def calculate_niche_optima(landscape, number_of_species):
    minimum = landscape.min()
    maximum = landscape.max()
    landscape_range = maximum - minimum

    first_optimum = minimum + landscape_range / number_of_species
    last_optimum = maximum - landscape_range / number_of_species

    return np.linspace(
        first_optimum,
        last_optimum,
        number_of_species,
    )



def calculate_performance(
    landscape,
    niche_optima,
    niche_variance,
    uses_landscape,
):
    number_of_species = len(niche_optima)

    performance = np.zeros(
        (number_of_species, *landscape.shape),
        dtype=float,
    )

    for species in range(number_of_species):
        optimum = niche_optima[species]

        local_performance = np.exp(
            -(landscape - optimum) ** 2 / (2 * niche_variance)
        )

        if uses_landscape:
            performance[species] = local_performance
        else:
            baseline = local_performance.mean()
            performance[species] = baseline

    return performance

def place_initial_plants(
    landscape_size,
    number_of_species,
    random_generator,
):
    number_of_sites = landscape_size ** 2
    plants_per_species = round(number_of_sites / number_of_species)

    plant_counts = np.zeros(
        (number_of_species, landscape_size, landscape_size),
        dtype=int,
    )

    for species in range(number_of_species):
        rows = random_generator.integers(
            landscape_size, size=plants_per_species
        )
        columns = random_generator.integers(
            landscape_size, size=plants_per_species
        )

        np.add.at(plant_counts[species], (rows, columns), 1)

    return plant_counts


def compete(
    removable_plants,
    established_adults,
    competition_performance,
    carrying_capacity,
    intraspecific_competition,
    interspecific_competition,
    random_generator,
):
    survivors = removable_plants.copy()
    total_plants = established_adults + removable_plants

    overcrowded_rows, overcrowded_columns = np.nonzero(
        total_plants.sum(axis=0) > carrying_capacity
    )

    for row, column in zip(overcrowded_rows, overcrowded_columns):
        starting_counts = total_plants[:, row, column]
        removable_counts = survivors[:, row, column]

        number_of_species = len(starting_counts)
        removal_weights = np.zeros(number_of_species)

        # Calculate weights once, using pre-competition counts.
        for species in range(number_of_species):
            pressure = (
                intraspecific_competition * starting_counts[species]
            )

            for competitor in range(number_of_species):
                if competitor != species:
                    pressure += (
                        interspecific_competition
                        * competition_performance[competitor, row, column]
                        * starting_counts[competitor]
                    )

            removal_weights[species] = starting_counts[species] * pressure

        plants_to_remove = int(
            starting_counts.sum() - carrying_capacity
        )

        for _ in range(plants_to_remove):
            # Exclude species that have no removable plants left.
            available_weights = removal_weights.copy()
            available_weights[removable_counts == 0] = 0

            removal_probabilities = (
                available_weights / available_weights.sum()
            )

            species_to_remove = random_generator.choice(
                number_of_species,
                p=removal_probabilities,
            )

            removable_counts[species_to_remove] -= 1

    return survivors

def reproduce(
    plants,
    reproduction_performance,
    max_seeds_per_plant,
    random_generator,
):
    seeds = np.zeros_like(plants)

    for species in range(len(plants)):
        expected_seeds = (
            max_seeds_per_plant[species]
            * reproduction_performance[species]
        )

        whole_seeds = np.floor(expected_seeds).astype(int)
        fractional_seeds = expected_seeds - whole_seeds

        additional_seeds = random_generator.binomial(
            plants[species],
            fractional_seeds,
        )

        seeds[species] = (
            plants[species] * whole_seeds + additional_seeds
        )

    return seeds


def disperse(
    new_seeds,
    annual_species,
    dispersal_type,
    seed_stay_probability,
    random_generator,
):
    landscape_size = new_seeds.shape[1]
    perennial_distance = {
        "adjacent": 1,
        "intermediate": 10,
        "universal": 65,
    }[dispersal_type]

    staying_seeds = random_generator.binomial(
        new_seeds, seed_stay_probability
    )
    moving_seeds = new_seeds - staying_seeds
    deposited_seeds = staying_seeds.copy()

    for species in range(len(new_seeds)):
        distance = perennial_distance
        if annual_species[species]:
            distance *= 2

        offsets = np.array([
            (row_offset, column_offset)
            for row_offset in range(-distance, distance + 1)
            for column_offset in range(-distance, distance + 1)
            if 0 < abs(row_offset) + abs(column_offset) <= distance
        ])

        occupied_rows, occupied_columns = np.nonzero(
            moving_seeds[species]
        )

        for row, column in zip(occupied_rows, occupied_columns):
            number_moving = moving_seeds[species, row, column]

            chosen_offsets = random_generator.integers(
                len(offsets), size=number_moving
            )

            destination_rows = row + offsets[chosen_offsets, 0]
            destination_columns = column + offsets[chosen_offsets, 1]

            inside_landscape = (
                (destination_rows >= 0)
                & (destination_rows < landscape_size)
                & (destination_columns >= 0)
                & (destination_columns < landscape_size)
            )

            np.add.at(
                deposited_seeds[species],
                (
                    destination_rows[inside_landscape],
                    destination_columns[inside_landscape],
                ),
                1,
            )

    return deposited_seeds

def apply_mortality(
    plants,
    annual_species,
    perennial_lifespan,
    random_generator,
):
    surviving_adults = np.zeros_like(plants)
    survival_probability = 1 - 1 / perennial_lifespan

    for species in range(len(plants)):
        if not annual_species[species]:
            surviving_adults[species] = random_generator.binomial(
                plants[species],
                survival_probability,
            )

    return surviving_adults


def germinate(
    seeds,
    germination_performance,
    max_germination_probability,
    random_generator,
):
    seedlings = np.zeros_like(seeds)

    for species in range(len(seeds)):
        germination_probability = (
            max_germination_probability[species]
            * germination_performance[species]
        )

        seedlings[species] = random_generator.binomial(
            seeds[species],
            germination_probability,
        )

    return seedlings

def save_results(
    output_file,
    final_plants,
    remaining_seeds,
    abiotic_effects,
    landscape_files,
    simulation_seed,
    years,
):
    np.savez_compressed(
        output_file,
        final_plants=final_plants,
        remaining_seeds=remaining_seeds,
        abiotic_effects=abiotic_effects,
        landscape_files=landscape_files,
        simulation_seed=simulation_seed,
        years=years,
    )