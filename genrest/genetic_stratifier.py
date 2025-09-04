"""Генетический алгоритм для стратификации по категориальным признакам.

Алгоритм работает только с категориальными признаками. Если исходные
данные содержат числовые столбцы, преобразуйте их в категории с помощью
`genrest.binning.bin_numeric`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


@dataclass
class Stratification:
    """Отображение: столбец -> категория -> номер группы (0..n_groups-1)."""
    boundaries: Dict[str, Dict[str, int]]


class GeneticStratifier:
    """Минимизирует стратифицированную дисперсию целевой переменной.

    Алгоритм принимает категориальные признаки и случайно разбивает значения
    каждого признака на заданное число групп. Группы по всем признакам
    комбинируются в номер страты, после чего вычисляется взвешенная дисперсия
    целевой
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
    n_groups:
        Количество групп для каждой категории. Итоговое число страт
        равно ``n_groups ** len(strat_columns)``.
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
        n_groups: int = 3,
        random_state: Optional[int] = None,
    ) -> None:
        self.strat_columns = strat_columns
        self.target_col = target_col
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.n_groups = n_groups
        self._rng = np.random.default_rng(random_state)
        self._categories: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------
    def _random_stratification(self) -> Stratification:
        boundaries: Dict[str, Dict[str, int]] = {}
        for col in self.strat_columns:
            cats = self._categories[col]
            boundaries[col] = {
                c: int(self._rng.integers(0, self.n_groups)) for c in cats
            }
        return Stratification(boundaries)

    def _mutate(self, strat: Stratification) -> Stratification:
        boundaries: Dict[str, Dict[str, int]] = {}
        for col, mapping in strat.boundaries.items():
            new_map: Dict[str, int] = {}
            for cat, grp in mapping.items():
                if self._rng.random() < self.mutation_rate:
                    grp = int(self._rng.integers(0, self.n_groups))
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
            indices = indices * self.n_groups + idx
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
        """Запустить алгоритм и вернуть лучшую стратификацию.

        Перед запуском проверяется, что все `strat_columns` являются
        категориальными. Если встречается числовой столбец, возбуждается
        ``TypeError`` с рекомендацией воспользоваться функцией
        ``bin_numeric``.
        """
        for col in self.strat_columns:
            if is_numeric_dtype(data[col]):
                raise TypeError(
                    f"Колонка '{col}' числовая. Используйте bin_numeric() "
                    "для преобразования в категории."
                )

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


def stratify_with_inheritance(
    data: pd.DataFrame,
    strat_columns: List[str],
    target_col: str,
    mandatory_columns: List[str],
    *,
    population_size: int = 20,
    generations: int = 50,
    mutation_rate: float = 0.1,
    n_groups: int = 3,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Стратифицировать данные с сохранением обязательных колонок.

    Для каждой комбинации ``mandatory_columns`` генетический алгоритм
    запускается отдельно на остальных колонках.
    """
    optional_cols = [c for c in strat_columns if c not in mandatory_columns]
    if not optional_cols:
        raise ValueError(
            "Не осталось колонок для объединения после учета обязательных."
        )

    strata = np.zeros(len(data), dtype=int)
    offset = 0
    for _, grp in data.groupby(mandatory_columns):
        strat = GeneticStratifier(
            strat_columns=optional_cols,
            target_col=target_col,
            population_size=population_size,
            generations=generations,
            mutation_rate=mutation_rate,
            n_groups=n_groups,
            random_state=random_state,
        )
        best = strat.fit(grp)
        local = strat._assign_strata(grp, best)
        strata[grp.index] = offset + local
        offset += n_groups ** len(optional_cols)
    return strata


__all__ = ["GeneticStratifier", "Stratification", "stratify_with_inheritance"]
