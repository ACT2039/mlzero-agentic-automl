Optional user instruction:
"Run a time-series forecasting task on value using timestamp and series_id. Detect irregular ordering and missing values. Forecast the next timestamp for each series and save forecasts.csv."

Faults:
- timestamps are out of order for series A
- one missing value
- one missing timestamp
- missing hourly observation at 11:00 for series B

Expected:
- Task = time-series forecasting
- Time column = timestamp
- Group/series column = series_id
- Pipeline should reject, repair, or explicitly report the malformed timestamp row; then sort by time and handle missing value/gaps.
- Expected artifact: forecasts.csv with a forecast field and series/timestamp identifiers.
