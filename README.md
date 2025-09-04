# GenRest

Пример генетического алгоритма, который разбивает **категориальные** признаки
на группы так, чтобы уменьшить разброс целевой числовой переменной.

## Как работает алгоритм
1. Вы задаёте столбцы с категориальными признаками и целевой столбец.
2. Формируется случайная популяция разбиений: каждой категории каждого
   признака назначается одна из трёх групп.
3. Комбинация групп по всем признакам образует номер страты.
4. Для каждой страты считается дисперсия целевой переменной, затем их
   взвешенная сумма.
5. Лучшие разбиения скрещиваются и немного мутируют, создавая новое поколение.
6. После нескольких поколений выбирается разбиение с минимальной
   стратифицированной дисперсией.

## Установка
```
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Пример
```python
import numpy as np
import pandas as pd
from genrest.genetic_stratifier import GeneticStratifier

# синтетические данные
rng = np.random.default_rng(0)
data = pd.DataFrame({
    "color": rng.choice(["red", "blue", "green"], 200),
    "shape": rng.choice(["circle", "square", "triangle"], 200),
})
# целевая переменная зависит от категорий
data["y"] = (
    (data["color"] == "red").astype(int)
    + (data["shape"] == "circle").astype(int)
    + rng.normal(0, 1, 200)
)

stratifier = GeneticStratifier(
    strat_columns=["color", "shape"],
    target_col="y",
    population_size=10,
    generations=5,
    random_state=0,
)

best = stratifier.fit(data)
print(best)
print("score:", stratifier.best_score_)
```

Более развёрнутый пример см. в [examples/tutorial.ipynb](examples/tutorial.ipynb).
