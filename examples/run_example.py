"""Пример работы алгоритма на синтетических данных."""
import numpy as np
import pandas as pd

from genrest import GeneticStratifier, bin_numeric, stratify_with_inheritance


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
    stratifier = GeneticStratifier(
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

    # пример с обязательной колонкой color
    strata = stratify_with_inheritance(
        data,
        strat_columns=["color", "shape", "age"],
        target_col="y",
        mandatory_columns=["color"],
        n_groups=2,
        generations=5,
        population_size=10,
        random_state=0,
    )
    print("With inheritance (first 10):", strata[:10])


if __name__ == "__main__":
    main()
