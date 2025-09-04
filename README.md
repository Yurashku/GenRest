# GenRest

Генетический алгоритм, который разбивает **категориальные** признаки на группы
и тем самым уменьшает разброс целевой числовой переменной. Алгоритм работает
только с категориями. Числовые признаки перед использованием нужно разбить на
интервалы — для этого в пакете есть функция `bin_numeric`.

## Как работает алгоритм
1. Указываются столбцы с категориальными признаками и целевой столбец.
2. Каждое значение признака случайно попадает в одну из `n_groups` групп.
3. Комбинация групп по всем признакам образует номер страты.
4. Для каждой страты считается дисперсия целевой переменной и их взвешенная
   сумма.
5. Генетический алгоритм скрещивает и мутирует разбиения, выбирая то, где
   дисперсия минимальна.

Параметр `n_groups` управляет финальным числом страт: при двух признаках и
`n_groups=3` получится `3^2=9` страт, при `n_groups=2` — `4` и т.д.

## Установка
```bash
python3.8 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Пример использования
```python
import numpy as np
import pandas as pd
from genrest import GeneticStratifier, bin_numeric

# синтетические данные
data = pd.DataFrame({
    "color": np.random.choice(["red", "blue", "green"], 200),
    "shape": np.random.choice(["circle", "square", "triangle"], 200),
    "age": np.random.normal(30, 10, 200),
})
# целевая переменная зависит от признаков
rng = np.random.default_rng(0)
data["y"] = ((data["color"] == "red").astype(int)
             + (data["shape"] == "circle").astype(int)
             + data["age"] / 10
             + rng.normal(0, 1, len(data)))

# превращаем age в категории
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
print(best)
print("score:", stratifier.best_score_)
```

Более развёрнутый пример см. в [examples/tutorial.ipynb](examples/tutorial.ipynb).
