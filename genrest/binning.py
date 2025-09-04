"""Утилиты для преобразования числовых признаков в категориальные.

Функция `bin_numeric` принимает числовую колонку и заменяет её
категориями, соответствующими интервалам ("бинам"). Это позволяет
использовать числовые признаки в `GeneticStratifier`, который работает
только с категориями.

Пример
-------
>>> import pandas as pd
>>> from genrest.binning import bin_numeric
>>> df = pd.DataFrame({"x": [1.0, 2.5, 3.0, 4.1]})
>>> bin_numeric(df, "x", bins=2)
>>> df["x"].unique()
['(-inf, 2.75]', '(2.75, inf]']
"""
from __future__ import annotations

from pandas import DataFrame
import pandas as pd


def bin_numeric(data: DataFrame, column: str, bins: int = 5, strategy: str = "quantile") -> None:
    """Преобразовать числовую колонку в категориальную.

    Parameters
    ----------
    data:
        Таблица с данными, которую нужно изменить на месте.
    column:
        Название числовой колонки.
    bins:
        Количество бинов.
    strategy:
        "quantile" – равное число объектов в каждом бине (`pd.qcut`),
        "uniform" – равная ширина бина (`pd.cut`).
    """
    col = data[column]
    if strategy == "quantile":
        binned = pd.qcut(col, q=bins, duplicates="drop")
    elif strategy == "uniform":
        binned = pd.cut(col, bins=bins)
    else:
        raise ValueError("strategy must be 'quantile' or 'uniform'")
    data[column] = binned.astype(str)


__all__ = ["bin_numeric"]
