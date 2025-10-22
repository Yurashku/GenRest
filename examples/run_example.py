"""Пример работы алгоритмов стратификации на синтетических данных."""
import numpy as np
import pandas as pd

from genrest import (
    GeneticStratificationAlgorithm,
    InheritedGeneticStratificationAlgorithm,
    bin_numeric,
)


def generate_data(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    colors = rng.choice(["red", "blue", "green"], size=n)
    shapes = rng.choice(["circle", "square", "triangle"], size=n)
    age = rng.normal(30, 10, size=n)
    y = (
        (colors == "red").astype(int)
        + (shapes == "circle").astype(int)
        + age / 10
        + rng.normal(0, 1, n)
    )
    return pd.DataFrame({"color": colors, "shape": shapes, "age": age, "y": y})


def main() -> None:
    data = generate_data()
    # преобразуем числовой признак age в категории
    bin_numeric(data, "age", bins=3)
    stratifier = GeneticStratificationAlgorithm(
        strat_columns=["color", "shape", "age"],
        target_col="y",
        population_size=10,
        generations=5,
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
        generations=5,
        population_size=10,
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
