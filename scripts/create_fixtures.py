import os

import pandas as pd

base_dir = "tests/data"

def create_timeseries():
    ts_dir = os.path.join(base_dir, "tiny_timeseries")
    os.makedirs(ts_dir, exist_ok=True)
    df_train = pd.DataFrame({
        "item_id": ["A"]*10 + ["B"]*10,
        "timestamp": pd.date_range("2023-01-01", periods=10).tolist() * 2,
        "target": list(range(10)) + list(range(10, 20))
    })
    df_train.to_csv(os.path.join(ts_dir, "train.csv"), index=False)
    
    df_test = pd.DataFrame({
        "item_id": ["A"]*2 + ["B"]*2,
        "timestamp": pd.date_range("2023-01-11", periods=2).tolist() * 2,
    })
    df_test.to_csv(os.path.join(ts_dir, "test.csv"), index=False)

def create_multimodal():
    mm_dir = os.path.join(base_dir, "tiny_multimodal")
    os.makedirs(mm_dir, exist_ok=True)
    df_train = pd.DataFrame({
        "text": ["this is good", "this is bad", "excellent", "terrible"],
        "target": [1, 0, 1, 0]
    })
    df_train.to_csv(os.path.join(mm_dir, "train.csv"), index=False)
    
    df_test = pd.DataFrame({
        "text": ["very good", "very bad"]
    })
    df_test.to_csv(os.path.join(mm_dir, "test.csv"), index=False)

def create_retrieval():
    ret_dir = os.path.join(base_dir, "tiny_retrieval")
    os.makedirs(ret_dir, exist_ok=True)
    # the exact structure for retrieval adapter doesn't matter too much initially, just make a dataset
    df_train = pd.DataFrame({"document": ["doc A", "doc B"]})
    df_train.to_csv(os.path.join(ret_dir, "train.csv"), index=False)

def create_regression():
    reg_dir = os.path.join(base_dir, "tiny_regression")
    os.makedirs(reg_dir, exist_ok=True)
    df_train = pd.DataFrame({
        "feature1": [1.1, 2.2, 3.3, 4.4, 5.5],
        "feature2": [1, 2, 3, 4, 5],
        "target": [2.1, 4.2, 6.3, 8.4, 10.5]
    })
    df_train.to_csv(os.path.join(reg_dir, "train.csv"), index=False)
    
    df_test = pd.DataFrame({
        "feature1": [6.6, 7.7],
        "feature2": [6, 7]
    })
    df_test.to_csv(os.path.join(reg_dir, "test.csv"), index=False)


if __name__ == "__main__":
    create_timeseries()
    create_multimodal()
    create_retrieval()
    create_regression()
