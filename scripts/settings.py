"""Settings for landscape generation and the plant community simulation."""

import numpy as np

# Landscape generation
landscape_size = 65
fractal_dimension = 2.5
landscape_standard_deviation = 2.5
landscape_seed = 7

# Simulation duration and randomness
years = 200
simulation_seed = 9

# Output files
germination_landscape_file = "data/landscapes/landscape_001.npz"
competition_landscape_file = "data/landscapes/landscape_002.npz"
reproduction_landscape_file = "data/landscapes/landscape_003.npz"

simulation_file = "data/simulations/simulation_008.npz"

# Species order: annual, perennial, annual, perennial
number_of_species = 4
annual_species = np.array([True, False, True, False])
max_seeds_per_plant = np.array([20, 5, 20, 5])
max_germination_probability = np.array([1.0, 1.0, 1.0, 1.0])
perennial_lifespan = 4

# Competition
carrying_capacity = 3
intraspecific_competition = 1.0
interspecific_competition = 1.0  # Paper treatments: 0.5, 1.0, 1.5

# Dispersal
dispersal_type = "adjacent"  # "adjacent", "intermediate", "universal"
seed_stay_probability = 0.5

# Dependence on local abiotic conditions
germination_uses_landscape = True
competition_uses_landscape = True
reproduction_uses_landscape = True

# Niche variance for each life stage
germination_niche_variance = 2.0
competition_niche_variance = 2.0
reproduction_niche_variance = 2.0