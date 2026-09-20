# AutoGluon Tabular: Data Quality, Non-Numeric Values, and Preprocessing

AutoGluon handles many types of missing values automatically, but malformed targets or dirty numeric columns with unexpected string values require preprocessing.

## Common Data Quality Issues
1. **Malformed Target Values**: If target values contain strings like `'not_available'`, `'unknown'`, or `None` in a numeric regression task, conversion will fail.
   - Fix: Coerce target column to numeric and drop rows with invalid target values:
     ```python
     train_df["price"] = pd.to_numeric(train_df["price"], errors="coerce")
     train_df = train_df.dropna(subset=["price"])
     ```
2. **Invalid Numeric Feature Values**: If numeric columns contain mixed text like `'1200 sqft'` or `'high'`:
   - Fix: Coerce to numeric, and impute missing values with median or mean:
     ```python
     for col in feature_cols:
         s_num = pd.to_numeric(train_df[col], errors="coerce")
         if s_num.notna().sum() / len(train_df) >= 0.5:
             train_df[col] = s_num.fillna(s_num.median())
     ```
3. **Missing Feature Imputation**: Ensure missing values in test set are filled using train set statistics to avoid shape or type mismatches during inference.
