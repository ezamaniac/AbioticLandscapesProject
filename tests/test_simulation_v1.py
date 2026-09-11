"""Check V1 equations against small, independently calculated examples."""

import unittest

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
        self.assertEqual(counts.sum(), 65**2)
        self.assertEqual(sorted(counts.sum(axis=(1, 2))), [1056, 1056, 1056, 1057])
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
        # r_0 = 2/10 * (1*2 + 4*0.5*3) = 1.6
        # r_1 = 3/10 * (5*3 + 2*0.25*2) = 4.8
        np.testing.assert_allclose(
            competition_pressure(densities, performance, [[1., 4.], [2., 5.]], 10.),
            [[[1.6]], [[4.8]]],
        )
        # Zero performance still permits self-competition.
        np.testing.assert_allclose(
            competition_pressure(densities[:1], np.zeros((1, 1, 1)), [[1.]], 2.),
            [[[2.]]],
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
            competition_pressure(np.ones((1, 2, 2)), np.ones((1, 2, 2)), [[1.]], 0.)
        with self.assertRaises(ValueError):
            expected_seed_production(np.ones((1, 2, 2)), np.ones((1, 2, 2)), -1.)

    def test_competition_stops_at_capacity(self):
        counts = np.array([[[5, 1], [0, 0]], [[4, 1], [0, 2]]])
        survivors = competition_phase(counts, np.ones_like(counts), np.ones((2, 2)), seed=5)
        np.testing.assert_array_equal(survivors.sum(axis=0), [[3, 2], [0, 2]])
        self.assertTrue(np.all((survivors >= 0) & (survivors <= counts)))
        np.testing.assert_array_equal(
            survivors, competition_phase(counts, np.ones_like(counts), np.ones((2, 2)), z=10, seed=5)
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
            deposited, lost = dispersal_phase(seeds, [False, True], mode, p=1, seed=3)
            np.testing.assert_array_equal(deposited.sum(axis=(1, 2)) + lost, [20000, 20000])
            self.assertEqual(lost.sum(), 0)
            self.assertEqual(deposited[:, center, center].sum(), 0)
            for species, expected in enumerate((radius, 2 * radius)):
                rows, cols = np.nonzero(deposited[species])
                self.assertEqual(max(abs(rows - center).max(), abs(cols - center).max()), expected)
        seeds = np.full((2, 3, 3), 100, dtype=int)
        deposited, lost = dispersal_phase(seeds, [False, True], "universal", p=1, seed=3)
        np.testing.assert_array_equal(deposited.sum(axis=(1, 2)) + lost, seeds.sum(axis=(1, 2)))
        self.assertTrue(np.all(lost > 0))
        deposited, lost = dispersal_phase(seeds, [False, True], p=0, seed=3)
        np.testing.assert_array_equal(deposited, seeds)
        self.assertEqual(lost.sum(), 0)

    def test_adult_survival(self):
        adults = np.full((2, 2, 2), 10000, dtype=int)
        survivors = adult_survival_phase(adults, [True, False], m=3, seed=4)
        self.assertEqual(survivors[0].sum(), 0)
        self.assertAlmostEqual(survivors[1].sum() / adults[1].sum(), 2/3, delta=0.01)
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
        self.assertEqual(set(vars(first)), {"adults", "seeds"})
        for name in ("adults", "seeds"):
            np.testing.assert_array_equal(getattr(first, name), getattr(second, name))

    def test_ungerminated_seed_retention(self):
        settings = dict(
            optima=[0], niche_widths=1, annual=[True], h=0,
            fecundity=10, alpha=[[1]], p=0, seed=1,
        )
        retained = run_simulation(np.zeros((3, 3)), 2, retain_ungerminated=True, **settings)
        discarded = run_simulation(np.zeros((3, 3)), 2, **settings)
        self.assertGreater(retained.seeds.sum(), 0)
        first_year = run_simulation(np.zeros((3, 3)), 1, retain_ungerminated=True, **settings)
        np.testing.assert_array_equal(first_year.seeds, retained.seeds)
        self.assertEqual(discarded.seeds.sum(), 0)


if __name__ == "__main__":
    unittest.main()
