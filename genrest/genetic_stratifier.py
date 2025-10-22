"""Генетический алгоритм для стратификации по категориальным признакам.

Алгоритм работает только с категориальными признаками. Если исходные
данные содержат числовые столбцы, преобразуйте их в категории с помощью
`genrest.binning.bin_numeric`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from math import prod
from pandas.api.types import is_numeric_dtype


@dataclass
class Stratification:
    """Отображение: столбец -> категория -> номер группы."""
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
        Количество групп для каждой категории. Используется, если не задан
        ``total_strata``.
    total_strata:
        Общее число страт. Количество групп по столбцам распределяется
        автоматически так, чтобы произведение равнялось ``total_strata``.
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
        total_strata: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self.strat_columns = strat_columns
        self.target_col = target_col
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self._rng = np.random.default_rng(random_state)
        self._categories: Dict[str, List[str]] = {}

        if total_strata is None:
            self._group_counts = [n_groups] * len(strat_columns)
            self.total_strata = n_groups ** len(strat_columns)
        else:
            self.total_strata = total_strata
            self._group_counts = self._compute_group_counts(total_strata)

    # ------------------------------------------------------------------
    def _compute_group_counts(self, total: int) -> List[int]:
        groups = [1] * len(self.strat_columns)
        n = total
        factor = 2
        factors: List[int] = []
        while factor * factor <= n:
            while n % factor == 0:
                factors.append(factor)
                n //= factor
            factor += 1
        if n > 1:
            factors.append(n)
        for f in sorted(factors, reverse=True):
            i = int(np.argmin(groups))
            groups[i] *= f
        return groups

    def _random_stratification(self) -> Stratification:
        boundaries: Dict[str, Dict[str, int]] = {}
        for col, g in zip(self.strat_columns, self._group_counts):
            cats = self._categories[col]
            boundaries[col] = {c: int(self._rng.integers(0, g)) for c in cats}
        return Stratification(boundaries)

    def _mutate(self, strat: Stratification) -> Stratification:
        boundaries: Dict[str, Dict[str, int]] = {}
        for i, col in enumerate(self.strat_columns):
            mapping = strat.boundaries[col]
            new_map: Dict[str, int] = {}
            for cat, grp in mapping.items():
                if self._rng.random() < self.mutation_rate:
                    grp = int(self._rng.integers(0, self._group_counts[i]))
                new_map[cat] = grp
            boundaries[col] = new_map
        return Stratification(boundaries)

    def _crossover(self, a: Stratification, b: Stratification) -> Stratification:
        if len(self.strat_columns) == 1:
            parent = a if self._rng.random() < 0.5 else b
            col = self.strat_columns[0]
            return Stratification({col: dict(parent.boundaries[col])})

        cut = self._rng.integers(1, len(self.strat_columns))
        boundaries: Dict[str, Dict[str, int]] = {}
        for i, col in enumerate(self.strat_columns):
            parent = a if i < cut else b
            boundaries[col] = dict(parent.boundaries[col])
        return Stratification(boundaries)

    def _assign_strata(self, data: pd.DataFrame, strat: Stratification) -> np.ndarray:
        indices = np.zeros(len(data), dtype=int)
        for col, base in zip(self.strat_columns, self._group_counts):
            mapping = strat.boundaries[col]
            series = data[col].astype(str)
            mapped = series.map(mapping)
            if mapped.isnull().any():
                missing = sorted(series[mapped.isnull()].unique())
                raise KeyError(
                    "Для колонки "
                    f"'{col}' найдены неизвестные категории: {missing}."
                )
            idx = mapped.to_numpy(dtype=int)
            indices = indices * base + idx
        return indices

    def _encode_with_labels(
        self,
        data: pd.DataFrame,
        strat: Stratification,
        *,
        offset: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Вернуть глобальные индексы страт и их строковые описания."""

        if len(self.strat_columns) == 0:
            raise ValueError("Не указаны стратифицируемые колонки.")

        inverted: Dict[str, Dict[int, List[str]]] = {}
        encoded_columns: List[np.ndarray] = []
        for col in self.strat_columns:
            mapping = strat.boundaries[col]
            inverse: Dict[int, List[str]] = {}
            for category, group_id in mapping.items():
                inverse.setdefault(group_id, []).append(str(category))
            inverted[col] = {k: sorted(v) for k, v in inverse.items()}

            series = data[col].astype(str)
            encoded = series.map(mapping)
            if encoded.isnull().any():
                missing = sorted(series[encoded.isnull()].unique())
                raise KeyError(
                    "Для колонки "
                    f"'{col}' найдены неизвестные категории: {missing}."
                )
            encoded_columns.append(encoded.to_numpy(dtype=int))

        group_matrix = np.column_stack(encoded_columns)
        local_indices = np.zeros(len(data), dtype=int)
        for encoded, base in zip(encoded_columns, self._group_counts):
            local_indices = local_indices * base + encoded

        labels = np.empty(len(data), dtype=object)
        cache: Dict[Tuple[int, ...], str] = {}
        for i, (local_idx, combo) in enumerate(zip(local_indices, group_matrix)):
            combo_tuple = tuple(int(v) for v in combo)
            if combo_tuple not in cache:
                parts = []
                for col, grp in zip(self.strat_columns, combo_tuple):
                    cats = ", ".join(inverted[col][grp])
                    parts.append(f"{col}={{{cats}}}")
                strata_id = offset + int(local_idx)
                cache[combo_tuple] = f"strata_{strata_id}: " + "; ".join(parts)
            labels[i] = cache[combo_tuple]
        return local_indices + offset, labels

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

    def transform(
        self,
        data: pd.DataFrame,
        *,
        column_name: str = "strata_label",
        drop_original: bool = True,
    ) -> pd.DataFrame:
        """Заменить категориальные колонки новой с описаниями страт.

        Parameters
        ----------
        data:
            Таблица с исходными признаками, совместимыми с обучением.
        column_name:
            Название новой колонки с текстовым описанием страты.
        drop_original:
            Удалять ли исходные ``strat_columns`` из результата.
        """

        if not hasattr(self, "best_stratification_"):
            raise AttributeError("Сначала вызовите fit().")

        _, labels = self._encode_with_labels(data, self.best_stratification_)
        result = data.copy()
        result[column_name] = labels
        if drop_original:
            result = result.drop(columns=[c for c in self.strat_columns if c in result])
        return result


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
    total_strata: Optional[int] = None,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Стратифицировать данные с сохранением обязательных колонок.

    Для каждой комбинации ``mandatory_columns`` генетический алгоритм
    запускается отдельно на остальных колонках.

    Параметр ``total_strata`` задаёт общее число страт для объединяемых
    колонок. Если он не указан, используется ``n_groups`` для каждого столбца.
    """
    optional_cols = [c for c in strat_columns if c not in mandatory_columns]
    if not optional_cols:
        raise ValueError(
            "Не осталось колонок для объединения после учета обязательных."
        )

    strata = np.zeros(len(data), dtype=int)
    offset = 0
    for _, grp in data.groupby(mandatory_columns, sort=False):
        strat = GeneticStratifier(
            strat_columns=optional_cols,
            target_col=target_col,
            population_size=population_size,
            generations=generations,
            mutation_rate=mutation_rate,
            n_groups=n_groups,
            total_strata=total_strata,
            random_state=random_state,
        )
        best = strat.fit(grp)
        local = strat._assign_strata(grp, best)
        strata[grp.index] = offset + local
        offset += prod(strat._group_counts)
    return strata


def _stratify_with_inheritance_transform(
    data: pd.DataFrame,
    strat_columns: List[str],
    target_col: str,
    mandatory_columns: List[str],
    *,
    column_name: str = "strata_label",
    drop_original: bool = True,
    population_size: int = 20,
    generations: int = 50,
    mutation_rate: float = 0.1,
    n_groups: int = 3,
    total_strata: Optional[int] = None,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Стратифицировать данные и заменить признаки описанием страты.

    Параметр ``drop_original`` управляет удалением исходных колонок из
    возвращаемого датафрейма. Название новой колонки задаётся через
    ``column_name``.
    """

    optional_cols = [c for c in strat_columns if c not in mandatory_columns]
    if not optional_cols:
        raise ValueError(
            "Не осталось колонок для объединения после учета обязательных."
        )

    result = data.copy()
    labels = pd.Series(index=data.index, dtype=object)
    offset = 0

    for key, grp in data.groupby(mandatory_columns, sort=False):
        strat = GeneticStratifier(
            strat_columns=optional_cols,
            target_col=target_col,
            population_size=population_size,
            generations=generations,
            mutation_rate=mutation_rate,
            n_groups=n_groups,
            total_strata=total_strata,
            random_state=random_state,
        )
        best = strat.fit(grp)
        _, local_labels = strat._encode_with_labels(grp, best, offset=offset)

        if mandatory_columns:
            if not isinstance(key, tuple):
                key_values = (key,)
            else:
                key_values = key
            prefix_parts = [
                f"{col}={str(value)}"
                for col, value in zip(mandatory_columns, key_values)
            ]
            local_labels = np.array(
                [f"{'; '.join(prefix_parts)}; {lbl}" for lbl in local_labels],
                dtype=object,
            )

        labels.loc[grp.index] = local_labels
        offset += prod(strat._group_counts)

    result[column_name] = labels
    if drop_original:
        drop_cols = [c for c in strat_columns if c in result.columns]
        result = result.drop(columns=drop_cols)
    return result


stratify_with_inheritance.transform = _stratify_with_inheritance_transform


__all__ = ["GeneticStratifier", "Stratification", "stratify_with_inheritance"]
