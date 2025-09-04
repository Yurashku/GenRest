# GenRest

GenRest предоставляет простой пайплайн генетического алгоритма для поиска стратификаций
в табличных данных. Алгоритм минимизирует стратифицированную дисперсию целевого признака.

## Установка

```bash
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Использование

Ниже приведён краткий пример работы с синтетическими данными:

```python
import pandas as pd
import numpy as np
from genrest.genetic_stratifier import GeneticStratifier

rng = np.random.default_rng(0)
x1 = rng.normal(0,1,1000)
x2 = rng.normal(5,2,1000)
y = x1*0.5 + x2*0.2 + rng.normal(0,1,1000)

data = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})

stratifier = GeneticStratifier(
    strat_columns=['x1', 'x2'],
    target_col='y',
    population_size=10,
    generations=5,
    random_state=0,
)

best = stratifier.fit(data)
print(best)
print(stratifier.best_score_)
```

Более подробный пример находится в [examples/tutorial.ipynb](examples/tutorial.ipynb).
