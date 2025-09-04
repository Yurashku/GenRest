"""Genetic algorithm for data stratification."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class Stratification:
    """Mapping from column name to sorted list of cut points."""
    boundaries: Dict[str, List[float]]


class GeneticStratifier:
    """Genetic algorithm to minimize stratified variance of a target column.

    Parameters
    ----------
    strat_columns:
        List of feature columns used to build stratification boundaries.
    target_col:
        Name of the target column for variance evaluation.
    population_size:
        Number of individuals in each generation.
    generations:
        Number of generations to evolve.
    mutation_rate:
        Probability of mutating a boundary in an individual.
    random_state:
        Seed for the internal random number generator.
    """

    def __init__(
        self,
        strat_columns: List[str],
        target_col: str,
        population_size: int = 20,
        generations: int = 50,
        mutation_rate: float = 0.1,
        random_state: int | None = None,
    ) -> None:
        self.strat_columns = strat_columns
        self.target_col = target_col
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self._rng = np.random.default_rng(random_state)
        self._ranges: Dict[str, tuple[float, float]] = {}
        self.best_stratification_: Stratification | None = None
        self.best_score_: float | None = None

    # ------------------------------------------------------------------
    def _random_stratification(self) -> Stratification:
        boundaries: Dict[str, List[float]] = {}
        for col in self.strat_columns:
            lo, hi = self._ranges[col]
            pts = self._rng.uniform(lo, hi, size=2)
            boundaries[col] = sorted(pts.tolist())
        return Stratification(boundaries)

    def _mutate(self, strat: Stratification) -> Stratification:
        boundaries = {}
        for col, pts in strat.boundaries.items():
            lo, hi = self._ranges[col]
            new_pts = []
            for p in pts:
                if self._rng.random() < self.mutation_rate:
                    span = hi - lo
                    p = float(np.clip(p + self._rng.normal(scale=0.1 * span), lo, hi))
                new_pts.append(p)
            boundaries[col] = sorted(new_pts)
        return Stratification(boundaries)

    def _crossover(self, a: Stratification, b: Stratification) -> Stratification:
        cut = self._rng.integers(1, len(self.strat_columns))
        boundaries = {}
        for i, col in enumerate(self.strat_columns):
            parent = a if i < cut else b
            boundaries[col] = parent.boundaries[col][:]
        return Stratification(boundaries)

    def _assign_strata(self, data: pd.DataFrame, strat: Stratification) -> np.ndarray:
        indices = np.zeros(len(data), dtype=int)
        for col in self.strat_columns:
            bins = [-np.inf] + strat.boundaries[col] + [np.inf]
            idx = np.digitize(data[col], bins) - 1
            indices = indices * 3 + idx
        return indices

    def _stratified_variance(self, data: pd.DataFrame, strat: Stratification) -> float:
        idx = self._assign_strata(data, strat)
        grouped = data.groupby(idx)[self.target_col]
        total = len(data)
        var = 0.0
        for _, grp in grouped:
            if len(grp) > 1:
                w = len(grp) / total
                var += w * w * grp.var(ddof=1)
        return float(var)

    def _next_generation(
        self, data: pd.DataFrame, population: List[Stratification]
    ) -> List[Stratification]:
        scores = [self._stratified_variance(data, s) for s in population]
        order = np.argsort(scores)
        sorted_pop = [population[i] for i in order]
        new_pop = sorted_pop[:2]
        while len(new_pop) < self.population_size:
            parents = self._rng.choice(sorted_pop[:5], size=2, replace=False)
            child = self._crossover(parents[0], parents[1])
            child = self._mutate(child)
            new_pop.append(child)
        return new_pop

    # ------------------------------------------------------------------
    def fit(self, data: pd.DataFrame) -> Stratification:
        """Run the genetic algorithm and return the best stratification."""
        self._ranges = {
            col: (float(data[col].min()), float(data[col].max()))
            for col in self.strat_columns
        }
        population = [self._random_stratification() for _ in range(self.population_size)]
        for _ in range(self.generations):
            population = self._next_generation(data, population)
        scores = [self._stratified_variance(data, s) for s in population]
        best_idx = int(np.argmin(scores))
        self.best_stratification_ = population[best_idx]
        self.best_score_ = scores[best_idx]
        return population[best_idx]


__all__ = ["GeneticStratifier", "Stratification"]
