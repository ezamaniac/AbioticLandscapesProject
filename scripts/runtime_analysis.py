import pstats

stats = pstats.Stats("simulation_profile0.prof")
stats.strip_dirs().sort_stats("cumulative").print_stats(20)