# AutoGluon TimeSeries Quickstart

`autogluon.timeseries` provides automated forecasting for univariate and multivariate time series collections.

## Basic Usage
```python
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# Data must have item_id and timestamp columns
ts_train = TimeSeriesDataFrame.from_data_frame(
    df, id_column="item_id", timestamp_column="timestamp"
)

predictor = TimeSeriesPredictor(target="target", prediction_length=14, path="out/models")
predictor.fit(ts_train, time_limit=60)
predictions = predictor.predict(ts_train)
```
