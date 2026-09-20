# AutoGluon MultiModal Quickstart

`autogluon.multimodal` enables training deep learning models on multimodal data combining text, tabular features, and images.

## Basic Usage
```python
from autogluon.multimodal import MultiModalPredictor

predictor = MultiModalPredictor(label="target", path="out/models")
predictor.fit(train_data, time_limit=120)
predictions = predictor.predict(test_data)
```

## Constraints and Requirements
- Optimal performance requires CUDA GPU acceleration.
- Image columns must contain valid image paths.
- Text columns should contain clean UTF-8 strings.
