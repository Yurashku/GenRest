"""Пример работы алгоритма с синтетическими категориальными данными."""
import numpy as np
import pandas as pd

from genrest.genetic_stratifier import GeneticStratifier


def generate_data(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    colors = rng.choice(["red", "blue", "green"], size=n)
    shapes = rng.choice(["circle", "square", "triangle"], size=n)
    y = (
        (colors == "red").astype(int)
        + (shapes == "circle").astype(int)
        + rng.normal(0, 1, n)
    )
    return pd.DataFrame({"color": colors, "shape": shapes, "y": y})


def main() -> None:
    data = generate_data()
    stratifier = GeneticStratifier(
        strat_columns=["color", "shape"],
        target_col="y",
        population_size=10,
        generations=5,
        random_state=0,
    )
    best = stratifier.fit(data)
    print("Best stratification:", best)
    print("Best score:", stratifier.best_score_)


if __name__ == "__main__":
    main()
