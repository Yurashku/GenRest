# GenRest

GenRest — это генетический алгоритм, который объединяет **категориальные**
признаки в группы и тем самым уменьшает разброс целевой числовой переменной.
Алгоритм работает только с категориями; числовые признаки перед использованием
нужно разбить на интервалы — для этого в пакете есть функция `bin_numeric`.
Проект написан на Python, а основными зависимостями являются библиотеки NumPy
и Pandas.

## Как работает алгоритм
1. Указываются столбцы с категориальными признаками и целевой столбец.
2. Каждое значение признака случайно попадает в одну из групп. Количество
   групп по столбцам определяется параметром `n_groups` или общим числом
   страт `total_strata`.
3. Комбинация групп по всем признакам образует номер страты.
4. Для каждой страты считается дисперсия целевой переменной и их взвешенная
   сумма.
5. Генетический алгоритм скрещивает и мутирует разбиения, выбирая то, где
   дисперсия минимальна.

По умолчанию используется параметр `n_groups`, задающий количество групп для
каждого столбца. Итоговое число страт при этом равно
`n_groups ** len(strat_columns)`. Альтернативно можно указать `total_strata`, и
алгоритм автоматически распределит количество групп по столбцам так, чтобы
произведение равнялось этому числу.

### Наследуемые признаки

Если нужно сохранить отдельные значения признаков (например, не смешивать
мужчин и женщин), используйте функцию `stratify_with_inheritance`. Она запускает
генетический алгоритм отдельно внутри каждой комбинации обязательных колонок.

```python
from genrest import stratify_with_inheritance

strata = stratify_with_inheritance(
    data,
    strat_columns=["color", "shape", "age"],
    target_col="y",
    mandatory_columns=["color"],  # раздельно для каждого цвета
    total_strata=8,
    generations=5,
    population_size=10,
)
data["strata"] = strata
```

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
    total_strata=8,
    random_state=0,
)

best = stratifier.fit(data)
print(best)
print("score:", stratifier.best_score_)
```

Более развёрнутый пример см. в [examples/tutorial.ipynb](examples/tutorial.ipynb).
