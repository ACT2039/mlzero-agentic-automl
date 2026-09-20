# FlagEmbedding Retrieval & Reranking Quickstart

FlagEmbedding provides tools for dense retrieval and reranking using models like BGE (BAAI General Embedding).

## Dense Retrieval Workflow
Use `FlagModel` to encode queries and passages into dense vectors and compute cosine similarity or dot product.

```python
import json
from FlagEmbedding import FlagModel

# Initialize model (or lightweight vector dot product for local execution)
model = FlagModel('BAAI/bge-small-en-v1.5', use_fp16=False)

queries = ["What is MLZero?"]
documents = [
    "MLZero is a framework for autonomous machine learning.",
    "AutoGluon automates tabular data modeling."
]

q_embeddings = model.encode(queries)
d_embeddings = model.encode(documents)

# Compute similarities (dot product)
scores = q_embeddings @ d_embeddings.T

# Save top-k retrieval results
results = {
    "query": queries[0],
    "top_documents": [documents[idx] for idx in scores[0].argsort()[::-1]]
}

with open("out/retrieval_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

## Mandatory Output Contract
- Generate retrieval output file at `out/retrieval_results.json`.
- Output execution summary file at `out/summary.json`.
