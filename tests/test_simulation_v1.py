"""Check V1 equations against small, independently calculated examples."""

import unittest
from unittest.mock import patch

import numpy as np

from scripts.simulationV1 import (
    competition_pressure,
    expected_seed_production,
    gaussian_performance,
    germination_probability,
    initialize_densities,
    competition_phase,
    germination_phase,
    reproduction_phase,
    dispersal_phase,
    adult_survival_phase,
    run_simulation,
)


class SimulationEquationTests(unittest.TestCase):
    def test_initialization_conserves_abundance(self):
        landscape = np.zeros((65, 65))
        counts = initialize_densities(landscape, seed=7)
        self.assertEqual(counts.shape, (4, 65, 65))
        self.assertEqual(counts.sum(), 4 * 1056)
        self.assertEqual(sorted(counts.sum(axis=(1, 2))), [1056] * 4)
        np.testing.assert_array_equal(
            counts, initialize_densities(landscape, seed=7)
        )

    def test_gaussian_optimum_and_one_niche_width(self):
        performance = gaussian_performance(np.array([[0., 2.], [2., 0.]]), [0., 2.], 2.)
        edge = np.exp(-0.5)
        np.testing.assert_allclose(
            performance, [[[1., edge], [edge, 1.]], [[edge, 1.], [1., edge]]]
        )

    def test_competitor_performance_and_unweighted_diagonal(self):
        densities = np.array([[[2.]], [[3.]]])
        performance = np.array([[[0.25]], [[0.5]]])
        # w_0 = 2 * (1*2 + 4*0.5*3) = 16
        # w_1 = 3 * (5*3 + 2*0.25*2) = 48
        np.testing.assert_allclose(
            competition_pressure(densities, performance, [[1., 4.], [2., 5.]]),
            [[[16.]], [[48.]]],
        )
        # Zero performance still permits self-competition.
        np.testing.assert_allclose(
            competition_pressure(densities[:1], np.zeros((1, 1, 1)), [[1.]]),
            [[[4.]]],
        )

    def test_germination_and_seed_expectations(self):
        performance = np.array([[[0., 1.]], [[0.5, 1.]]])
        np.testing.assert_allclose(
            germination_probability(performance, [0.8, 0.4]),
            [[[0., 0.8]], [[0.2, 0.4]]],
        )
        np.testing.assert_allclose(
            expected_seed_production(
                np.array([[[3., 2.]], [[4., 0.]]]), performance, [10., 20.]
            ),
            [[[0., 20.]], [[40., 0.]]],
        )

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            gaussian_performance(np.zeros((2, 2)), [0.], 0.)
        with self.assertRaises(ValueError):
            germination_probability(np.ones((1, 2, 2)), 1.1)
        with self.assertRaises(ValueError):
            competition_pressure(np.ones((1, 2, 2)), np.ones((1, 2, 2)), [[-1.]])
        with self.assertRaises(ValueError):
            expected_seed_production(np.ones((1, 2, 2)), np.ones((1, 2, 2)), -1.)

    def test_competition_stops_at_capacity(self):
        counts = np.array([[[5, 1], [0, 0]], [[4, 1], [0, 2]]])
        survivors = competition_phase(counts, np.ones_like(counts), np.ones((2, 2)), seed=5)
        np.testing.assert_array_equal(survivors.sum(axis=0), [[3, 2], [0, 2]])
        self.assertTrue(np.all((survivors >= 0) & (survivors <= counts)))
        np.testing.assert_array_equal(
            survivors, competition_phase(counts, np.ones_like(counts), np.ones((2, 2)), seed=5)
        )
        with self.assertRaises(ValueError):
            competition_phase(counts, np.ones_like(counts), np.zeros((2, 2)))

    def test_germination_and_reproduction_boundaries(self):
        seeds = np.full((2, 2, 2), 10, dtype=int)
        recruits, remaining = germination_phase(seeds, np.ones_like(seeds), [1, 0], seed=4)
        np.testing.assert_array_equal(recruits + remaining, seeds)
        self.assertEqual(recruits[0].sum(), 40)
        self.assertEqual(recruits[1].sum(), 0)
        self.assertEqual(reproduction_phase(seeds, np.ones_like(seeds), 0, seed=4).sum(), 0)

    def test_dispersal_distances_and_conservation(self):
        for mode, radius, dimension in (("adjacent", 1, 9), ("intermediate", 10, 45)):
            seeds = np.zeros((2, dimension, dimension), dtype=int)
            center = dimension // 2
            seeds[:, center, center] = 20000
            deposited, lost = dispersal_phase(seeds, [False, True], mode, p=0, seed=3)
            np.testing.assert_array_equal(deposited.sum(axis=(1, 2)) + lost, [20000, 20000])
            self.assertEqual(lost.sum(), 0)
            self.assertEqual(deposited[:, center, center].sum(), 0)
            for species, expected in enumerate((radius, 2 * radius)):
                rows, cols = np.nonzero(deposited[species])
                self.assertEqual(max(abs(rows - center).max(), abs(cols - center).max()), expected)
        seeds = np.full((2, 3, 3), 100, dtype=int)
        deposited, lost = dispersal_phase(seeds, [False, True], "universal", p=0, seed=3)
        np.testing.assert_array_equal(deposited.sum(axis=(1, 2)) + lost, seeds.sum(axis=(1, 2)))
        self.assertTrue(np.all(lost > 0))
        deposited, lost = dispersal_phase(seeds, [False, True], p=1, seed=3)
        np.testing.assert_array_equal(deposited, seeds)
        self.assertEqual(lost.sum(), 0)

    def test_adult_survival(self):
        adults = np.full((2, 2, 2), 10000, dtype=int)
        survivors = adult_survival_phase(adults, [True, False], seed=4)
        self.assertEqual(survivors[0].sum(), 0)
        self.assertAlmostEqual(survivors[1].sum() / adults[1].sum(), 3/4, delta=0.01)
        self.assertEqual(adult_survival_phase(adults, [False, False], m=1).sum(), 0)

    def test_complete_loop_and_reproducibility(self):
        settings = dict(
            optima=[0, 0], niche_widths=1, annual=[True, False], h=1,
            fecundity=[8, 4], alpha=np.ones((2, 2)), seed=7,
        )
        first = run_simulation(np.zeros((9, 9)), 4, **settings)
        second = run_simulation(np.zeros((9, 9)), 4, **settings)
        self.assertEqual(first.adults.shape, (2, 9, 9))
        self.assertEqual(first.seeds.shape, (2, 9, 9))
        self.assertTrue(np.all(first.adults.sum(axis=0) <= 3))
        self.assertEqual(first.adults[0].sum(), 0)
        self.assertGreater(first.seeds[0].sum(), 0)
        self.assertEqual(set(vars(first)), {"adults", "seeds", "community"})
        for name in ("adults", "seeds", "community"):
            np.testing.assert_array_equal(getattr(first, name), getattr(second, name))

    def test_ungerminated_seeds_discarded_and_first_year_reproduces(self):
        settings = dict(optima=[0], annual=[True], h=0, fecundity=10, p=1, seed=1)
        first = run_simulation(np.zeros((3, 3)), 1, **settings)
        self.assertGreater(first.community.sum(), 0)
        np.testing.assert_array_equal(first.seeds, first.community * 10)
        self.assertEqual(first.adults.sum(), 0)
        second = run_simulation(np.zeros((3, 3)), 2, **settings)
        self.assertEqual(second.seeds.sum(), 0)
        self.assertEqual(second.community.sum(), 0)

    def test_adults_protected_and_contribute_pressure(self):
        adults = np.array([[[3, 1]], [[0, 0]]])
        recruits = np.array([[[0, 0]], [[5, 4]]])
        saved_adults = adults.copy()
        survivors = competition_phase(recruits, np.ones_like(recruits), np.ones((2, 2)), adults=adults, seed=8)
        np.testing.assert_array_equal(adults, saved_adults)
        np.testing.assert_array_equal(survivors, [[[0, 0]], [[0, 2]]])
        # Eligible species 1 feels adults of species 0, weighted by P_C0.
        weights = competition_pressure(adults + recruits, np.full(recruits.shape, 0.5),
                                       [[1, 2], [2, 1]], eligible=recruits)
        np.testing.assert_array_equal(weights, [[[0, 0]], [[40, 20]]])

    def test_fixed_weights_and_exhausted_species(self):
        # Original weights 1 and 4: species 0 must be excluded once exhausted.
        counts = np.array([[[1]], [[2]]])
        class Draws:
            def __init__(self, values):
                self.values = iter(values)
            def random(self, size):
                return np.full(size, next(self.values))
        with patch("scripts.simulationV1.np.random.default_rng", return_value=Draws([0.1, 0.1])):
            survivors = competition_phase(counts, np.ones_like(counts), np.eye(2), carrying_capacity=1)
        np.testing.assert_array_equal(survivors, [[[0]], [[1]]])
        # Weights start 4:4 and must stay 4:4 after the first removal.
        counts = np.array([[[2]], [[2]]])
        with patch("scripts.simulationV1.np.random.default_rng", return_value=Draws([0.4, 0.4])):
            survivors = competition_phase(counts, np.ones_like(counts), np.eye(2), carrying_capacity=2)
        np.testing.assert_array_equal(survivors, [[[0]], [[2]]])

    def test_fractional_seed_sampling_is_per_individual(self):
        counts = np.full((1, 100, 100), 10, dtype=int)
        seeds = reproduction_phase(counts, np.ones_like(counts), 2.5, seed=2)
        self.assertTrue(np.all((seeds >= 20) & (seeds <= 30)))
        self.assertAlmostEqual(seeds.mean(), 25, delta=0.1)
        self.assertAlmostEqual(seeds.var(), 2.5, delta=0.15)
        np.testing.assert_array_equal(reproduction_phase(counts, np.ones_like(counts), 2, seed=2), counts * 2)

    def test_universal_distance_independent_of_life_history(self):
        seeds = np.full((1, 5, 5), 100, dtype=int)
        perennial = dispersal_phase(seeds, [False], "universal", p=0, seed=4)
        annual = dispersal_phase(seeds, [True], "universal", p=0, seed=4)
        for actual, expected in zip(annual, perennial):
            np.testing.assert_array_equal(actual, expected)

    def test_snapshot_timing_and_zero_steps(self):
        initial = run_simulation(np.zeros((5, 5)), 0, seed=3)
        first = run_simulation(np.zeros((5, 5)), 1, seed=3)
        np.testing.assert_array_equal(initial.adults, initial.community)
        self.assertEqual(initial.seeds.sum(), 0)
        np.testing.assert_array_equal(first.community, initial.community)
        self.assertGreater(first.community[:2].sum(), 0)
        self.assertEqual(first.adults[:2].sum(), 0)
        self.assertTrue(np.all(first.adults <= first.community))
        for result in (initial, first):
            self.assertFalse(np.shares_memory(result.adults, result.community))
            self.assertFalse(np.shares_memory(result.seeds, result.community))

    def test_paper_defaults_and_explicit_overrides(self):
        landscape = np.arange(25).reshape(5, 5) / 6 - 2
        # min=-2, max=2; inset=range/4=1 -> optima -1 to +1.
        expected = run_simulation(landscape, 2, optima=np.linspace(-1, 1, 4),
            annual=[True, True, False, False], niche_widths=np.sqrt(2), h=1,
            fecundity=[20, 20, 5, 5], alpha=np.ones((4, 4)), m=4, seed=2)
        actual = run_simulation(landscape, 2, seed=2)
        for name in vars(actual):
            np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
        for strength in (0.5, 1, 1.5):
            matrix = np.full((4, 4), strength, dtype=float)
            np.fill_diagonal(matrix, 1)
            a = run_simulation(landscape, 2, interspecific_strength=strength, seed=3)
            b = run_simulation(landscape, 2, alpha=matrix, seed=3)
            np.testing.assert_array_equal(a.community, b.community)
        performance = gaussian_performance(np.array([[np.sqrt(2)]]), [0])
        np.testing.assert_allclose(performance, np.exp(-0.5))


if __name__ == "__main__":
    unittest.main()
