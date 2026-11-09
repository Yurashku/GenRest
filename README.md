# GenRest

**Genetic restratification for categorical data.** A Python research implementation that searches for category groupings under a custom within-stratum variance objective, with optional mandatory groups. It includes a reproducible synthetic example and a notebook; it is not a production experimentation platform.

GenRest — это набор взаимозаменяемых генетических алгоритмов, которые объединяют **категориальные**
признаки в группы и ищут разбиение с меньшим значением целевой функции стратификации.
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
4. Для каждой непустой страты считается выборочная дисперсия целевой переменной.
   Целевая функция кода: `sum((n_h / n) ** 2 * s_h ** 2)`; страты из одной строки
   не вносят вклад.
5. Генетический алгоритм скрещивает и мутирует разбиения, выбирая то, где
   дисперсия минимальна.

По умолчанию используется параметр `n_groups`, задающий количество групп для
каждого столбца. Верхняя граница числа возможных страт при этом равна
`n_groups ** len(strat_columns)`; реально занятых страт может быть меньше. Альтернативно можно указать `total_strata`, и
алгоритм автоматически распределит количество групп по столбцам так, чтобы
произведение равнялось этому числу.

## Доступные алгоритмы

* `GeneticStratificationAlgorithm` — базовый генетический алгоритм, который
  объединяет категории во всех указанных столбцах.
* `InheritedGeneticStratificationAlgorithm` — обёртка над базовым алгоритмом,
  запускающая его отдельно внутри каждой комбинации обязательных колонок.

Для обратной совместимости имя `GeneticStratifier` остаётся доступным как
псевдоним для `GeneticStratificationAlgorithm`, но новые проекты стоит строить
на новых названиях — так проще компоновать взаимозаменяемые алгоритмы.
Аналогично функция `stratify_with_inheritance` остаётся в пакете, но помечена
как устаревшая и делегирует вызовы новому классу.

### Наследуемые признаки

Если нужно сохранить отдельные значения признаков (например, не смешивать
мужчин и женщин), используйте класс `InheritedGeneticStratificationAlgorithm`.
Он запускает генетический алгоритм отдельно внутри каждой комбинации
обязательных колонок и выдаёт результат в том же формате, что и основной
алгоритм.

```python
from genrest import InheritedGeneticStratificationAlgorithm

inherited_algo = InheritedGeneticStratificationAlgorithm(
    strat_columns=["color", "shape", "age"],
    target_col="y",
    mandatory_columns=["color"],  # раздельно для каждого цвета
    total_strata=8,
    generations=5,
    population_size=10,
)
inherited_algo.fit(data)
data["strata"] = inherited_algo.transform_to_indices(data)
```

## Установка и первый запуск

Запускайте команды из клонированного репозитория: проект импортируется из папки
`genrest`, отдельного устанавливаемого пакета в этой ревизии нет.

```bash
git clone https://github.com/Yurashku/GenRest.git
cd GenRest
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python examples/run_example.py
```

Скрипт создаёт 600 синтетических строк, переводит возраст в категории и запускает
обычный алгоритм и вариант с обязательным сохранением `color`. В терминале будут
отображение категорий в группы, значение целевой функции, первые строки со стратами
и сравнение метрик. Полный список зависимостей включает Jupyter для ноутбука;
для Python-скрипта достаточно NumPy и pandas.

### Проверенный результат скрипта

На Python 3.14.0, NumPy 2.5.4 и pandas 3.1.1 скрипт завершился успешно.
Для обычного алгоритма `Best score` равен `0.4990934492798214`; для исходного
детального разбиения печатается `0.013610`, а для варианта с обязательным `color`
— `0.110075`. Укрупнение категорий в этом примере **увеличивает** целевую функцию
относительно исходных мелких страт. Это демонстрация работы и компромисса числа
страт, а не пример гарантированного улучшения статистического теста.

## Пример использования
```python
import numpy as np
import pandas as pd
from genrest import GeneticStratificationAlgorithm, bin_numeric

rng = np.random.default_rng(0)
data = pd.DataFrame({
    "color": rng.choice(["ruby", "amber", "teal"], size=600, p=[0.45, 0.35, 0.2]),
    "shape": rng.choice(["circle", "square", "triangle"], size=600, p=[0.4, 0.4, 0.2]),
    "age": rng.normal(42, 11, size=600),
})

color_effect = np.select(
    [data["color"] == "ruby", data["color"] == "amber"],
    [4.5, 2.0],
    default=-1.5,
)
shape_effect = np.select(
    [data["shape"] == "circle", data["shape"] == "square"],
    [3.0, -0.5],
    default=-2.0,
)
age_effect = np.select(
    [data["age"] < 35, data["age"] < 45, data["age"] < 55],
    [-1.0, 0.5, 2.0],
    default=3.2,
)
data["y"] = color_effect + shape_effect + age_effect + rng.normal(0, 0.4, len(data))

bin_numeric(data, "age", bins=4)

stratifier = GeneticStratificationAlgorithm(
    strat_columns=["color", "shape", "age"],
    target_col="y",
    population_size=25,
    generations=40,
    total_strata=8,
    random_state=0,
)

best = stratifier.fit(data)
print(best)
print("score:", stratifier.best_score_)

stratified = stratifier.transform(data, column_name="strata")
print(stratified.head())
```

При вызове `transform` библиотека автоматически выводит четыре показателя:

* цельную (обычную) дисперсию целевой переменной;
* стратифицированную дисперсию по исходным комбинациям категориальных признаков;
* стратифицированную дисперсию после объединения категорий генетическим алгоритмом;
* процент понижения стратифицированной дисперсии относительно исходной (если
  понижения нет, библиотека сообщает об отсутствии эффекта и выводит размер
  роста). Показатель сравнивает исходное детальное разбиение и укрупнённые
  страты по целевой функции кода. При сжатии числа страт значение может вырасти;
  обычная дисперсия `overall_var` и эта целевая функция имеют разный смысл.
  Их отношение не следует интерпретировать как выигрыш A/B-теста.

Это помогает сразу понять эффект от объединения страт без дополнительных расчётов.

Более развёрнутый пример см. в [examples/tutorial.ipynb](examples/tutorial.ipynb).

Чтобы при этом учитывать обязательные признаки, которые нельзя объединять
между собой, используйте ``InheritedGeneticStratificationAlgorithm`` и его метод
``transform``:

```python
from genrest import InheritedGeneticStratificationAlgorithm

algo = InheritedGeneticStratificationAlgorithm(
    strat_columns=["color", "shape", "age"],
    target_col="y",
    mandatory_columns=["color"],
    n_groups=2,
    generations=40,
    population_size=25,
)
algo.fit(data)
collapsed = algo.transform(data)
print(collapsed.head())
```

Метод `transform` у `InheritedGeneticStratificationAlgorithm` также печатает
аналогичные метрики, позволяя сравнить исходное и итоговое состояние для
случая с обязательными колонками.

## Как интерпретировать результат

- Алгоритм оптимизирует указанную функцию на данных обучения. Сравнивайте разбиения
  на отдельной выборке и контролируйте размер страт; малые группы могут давать
  оптимистичную оценку.
- В функции нет множителя `1 / n_h`, входящего в стандартную формулу дисперсии
  стратифицированного среднего при независимом отборе внутри страт. Поэтому
  `best_score_` и печатаемый процент — диагностические величины этого алгоритма,
  а не доказанный прирост мощности A/B-теста или универсальное снижение дисперсии
  оценщика. Нельзя напрямую трактовать отношение `best_score_` к обычной дисперсии
  целевой переменной как выигрыш эксперимента.
- При преобразовании новых данных неизвестные категории вызывают `KeyError`.
  Числовое разбиение также нужно применять согласованно: `bin_numeric` меняет
  входную колонку и заново вычисляет границы на переданной таблице.

## Код и происхождение

[Реализация алгоритмов](https://github.com/Yurashku/GenRest/blob/main/genrest/genetic_stratifier.py)
и [история изменений](https://github.com/Yurashku/GenRest/commits/main/) доступны для
проверки вклада. Репозиторий поддерживается Yurashku; алгоритмы, примеры и пояснения
сохранены вместе. Лицензия — [MIT](https://github.com/Yurashku/GenRest/blob/main/LICENSE).
