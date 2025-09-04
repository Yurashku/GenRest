"""Minimal example using synthetic data."""
import numpy as np
import pandas as pd

from genrest.genetic_stratifier import GeneticStratifier


def generate_data(n: int = 1000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(5, 2, n)
    y = x1 * 0.5 + x2 * 0.2 + rng.normal(0, 1, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def main() -> None:
    data = generate_data()
    stratifier = GeneticStratifier(
        strat_columns=["x1", "x2"],
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
