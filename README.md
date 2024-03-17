# Как запускать
подготовить 2 датасета (тренировочный, тестировочный). тренировочный нужен, чтобы снять метрики для признаков.

```python
from sk_pred import SKPredModel

model = SKPredModel()
model.prepare_df(some_df4test, train_df)
model.predict()
```

в папке saved_model разархивировать объект.
