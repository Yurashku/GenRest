"""Генетический алгоритм для стратификации по категориальным признакам."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class Stratification:
    """Отображение: столбец -> категория -> номер группы (0..2)."""
    boundaries: Dict[str, Dict[str, int]]


class GeneticStratifier:
    """Минимизирует стратифицированную дисперсию целевой переменной.

    Алгоритм принимает категориальные признаки и случайно разбивает значения
    каждого признака на три группы. Группы по всем признакам комбинируются в
    номер страты, после чего вычисляется взвешенная дисперсия целевой
    переменной внутри страт. Генетический алгоритм эволюционирует разбиения,
    чтобы минимизировать эту дисперсию.

    Parameters
    ----------
    strat_columns:
        Список категориальных столбцов для стратификации.
    target_col:
        Название числового столбца с целевой переменной.
    population_size:
        Размер популяции в каждом поколении.
    generations:
        Количество поколений эволюции.
    mutation_rate:
        Вероятность изменения группы у отдельной категории.
    random_state:
        Значение для воспроизводимости.
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
        self._categories: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------
    def _random_stratification(self) -> Stratification:
        boundaries: Dict[str, Dict[str, int]] = {}
        for col in self.strat_columns:
            cats = self._categories[col]
            boundaries[col] = {c: int(self._rng.integers(0, 3)) for c in cats}
        return Stratification(boundaries)

    def _mutate(self, strat: Stratification) -> Stratification:
        boundaries: Dict[str, Dict[str, int]] = {}
        for col, mapping in strat.boundaries.items():
            new_map: Dict[str, int] = {}
            for cat, grp in mapping.items():
                if self._rng.random() < self.mutation_rate:
                    grp = int(self._rng.integers(0, 3))
                new_map[cat] = grp
            boundaries[col] = new_map
        return Stratification(boundaries)

    def _crossover(self, a: Stratification, b: Stratification) -> Stratification:
        cut = self._rng.integers(1, len(self.strat_columns))
        boundaries: Dict[str, Dict[str, int]] = {}
        for i, col in enumerate(self.strat_columns):
            parent = a if i < cut else b
            boundaries[col] = dict(parent.boundaries[col])
        return Stratification(boundaries)

    def _assign_strata(self, data: pd.DataFrame, strat: Stratification) -> np.ndarray:
        indices = np.zeros(len(data), dtype=int)
        for col in self.strat_columns:
            mapping = strat.boundaries[col]
            idx = data[col].map(mapping).to_numpy()
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
        """Запустить алгоритм и вернуть лучшую стратификацию."""
        # собираем уникальные значения категориальных признаков
        self._categories = {
            col: sorted(map(str, data[col].astype(str).unique()))
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
