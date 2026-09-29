# Semantic Text Compression & Sentiment Analysis

A lightweight semantic text compression framework that removes redundant
sentences while preserving meaning and emotional tone. Implementation of the
pipeline described in *"Meaning Matters: A Lightweight Semantic Text
Compression Framework Using NLP and Sentiment Analysis."*

## How it works

1. **Pre-processing** — the input is split into sentences with NLTK's Punkt
   tokenizer and cleaned (whitespace/noise removal).
2. **Sentence embeddings** — each sentence is encoded into a 384-dimensional
   vector with a MiniLM sentence-transformer model.
3. **Semantic similarity** — pairwise cosine similarity is computed between
   sentence embeddings.
4. **Clustering** — Agglomerative Hierarchical Clustering (average linkage,
   cosine distance, threshold = 0.6) groups semantically similar sentences
   without requiring a predefined number of clusters.
5. **Representative selection** — for each cluster, the sentence closest to
   the cluster centroid is kept; the rest are dropped as redundant.
6. **Sentiment analysis** — a DistilBERT sentiment classifier scores both the
   original and compressed text so emotional polarity can be checked before
   and after compression.
7. **Metrics** — Compression Ratio (CR), Redundancy Reduction (RR), Semantic
   Similarity (SS), and Sentiment Consistency (SC) are computed and returned
   alongside the compressed text.

## Project structure

```
semantic-text-compression/
├── app.py                       # Flask app (single POST /compress endpoint)
├── evaluate.py                  # Evaluation harness (CR/RR/SS/SC metrics)
├── requirements.txt
├── src/
│   ├── __init__.py
│   └── semantic_compression.py  # Core pipeline (preprocessing, embeddings,
│                                 # clustering, sentiment, metrics)
├── templates/
│   └── index.html               # Web UI
└── static/
    ├── style.css
    └── script.js
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The first run downloads the MiniLM (`all-MiniLM-L6-v2`), DistilBERT
sentiment (`distilbert-base-uncased-finetuned-sst-2-english`), and NLTK
Punkt models automatically — this needs an internet connection and may take
a minute or two.

## Run the web demo

```bash
python app.py
```

Open `http://127.0.0.1:5000/` in a browser, paste in a paragraph, and click
**Run Compression** to see the compressed text alongside the CR, RR, SS
metrics and the sentiment labels for the original and compressed text.

## Use the pipeline directly

```python
from src.semantic_compression import SemanticCompressor

compressor = SemanticCompressor()
result = compressor.compress("Your paragraph of text goes here...")

print(result.compressed_text)
print(result.to_dict())
```

## Notes / tuning

- The clustering distance threshold (`0.6`) and linkage method (`average`)
  are set on `SemanticCompressor(distance_threshold=..., linkage=...)` and
  match the values reported as best-performing in the paper's clustering
  evaluation (Table IV).
- `Compression Ratio` is `len(compressed_text) / len(original_text)` — lower
  means stronger compression.
- `Redundancy Reduction` is `(1 - num_clusters / num_original_sentences) *
  100`.
- `Semantic Similarity` is the cosine similarity between the mean sentence
  embedding of the original text and of the compressed text.
- `Sentiment Consistency` is `1` if the DistilBERT sentiment label is the
  same before and after compression, else `0`.
