"""Пример работы алгоритмов стратификации на синтетических данных."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from genrest import (
    GeneticStratificationAlgorithm,
    InheritedGeneticStratificationAlgorithm,
    bin_numeric,
)


def generate_data(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    colors = rng.choice(["ruby", "amber", "teal"], size=n, p=[0.45, 0.35, 0.2])
    shapes = rng.choice(["circle", "square", "triangle"], size=n, p=[0.4, 0.4, 0.2])
    age = rng.normal(42, 11, size=n)

    color_effect = np.select(
        [colors == "ruby", colors == "amber"],
        [4.5, 2.0],
        default=-1.5,
    )
    shape_effect = np.select(
        [shapes == "circle", shapes == "square"],
        [3.0, -0.5],
        default=-2.0,
    )
    age_effect = np.select(
        [age < 35, age < 45, age < 55],
        [-1.0, 0.5, 2.0],
        default=3.2,
    )

    y = color_effect + shape_effect + age_effect + rng.normal(0, 0.4, size=n)
    return pd.DataFrame({"color": colors, "shape": shapes, "age": age, "y": y})


def main() -> None:
    data = generate_data()
    # преобразуем числовой признак age в категории
    bin_numeric(data, "age", bins=4)
    stratifier = GeneticStratificationAlgorithm(
        strat_columns=["color", "shape", "age"],
        target_col="y",
        population_size=25,
        generations=40,
        n_groups=2,
        random_state=0,
    )
    best = stratifier.fit(data)
    print("Best stratification:", best)
    print("Best score:", stratifier.best_score_)
    transformed = stratifier.transform(data, column_name="strata")
    print("Transformed head:\n", transformed.head())

    # пример с обязательной колонкой color
    inherited_algo = InheritedGeneticStratificationAlgorithm(
        strat_columns=["color", "shape", "age"],
        target_col="y",
        mandatory_columns=["color"],
        n_groups=2,
        generations=40,
        population_size=25,
        random_state=0,
    )
    inherited_algo.fit(data)
    strata = inherited_algo.transform_to_indices(data)
    print("With inheritance (first 10):", strata[:10])
    inherited = inherited_algo.transform(
        data,
        column_name="strata",
        drop_original=False,
    )
    print("Transformed with inheritance head:\n", inherited.head())


if __name__ == "__main__":
    main()
