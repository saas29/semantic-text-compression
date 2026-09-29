# Semantic Text Compression & Sentiment Analysis

A lightweight semantic text compression framework that removes redundant
sentences from text while preserving its meaning and emotional tone. Instead
of compressing at the byte level like ZIP or Huffman coding, it compresses at
the **meaning level** — it detects sentences that say the same thing and keeps
only one representative sentence from each group.

This is an implementation of the framework described in the paper
*"Meaning Matters: A Lightweight Semantic Text Compression Framework Using NLP
and Sentiment Analysis."*

## Key Features

- Meaning-based compression that removes semantically redundant sentences.
- MiniLM sentence embeddings for fast, lightweight semantic representation.
- Agglomerative (hierarchical) clustering that groups similar sentences
  without needing a fixed number of clusters.
- DistilBERT sentiment check to confirm the emotional tone is unchanged after
  compression.
- Four evaluation metrics (CR, RR, SS, SC) computed automatically.
- A Flask web app for interactive use, plus a command-line evaluation harness.

## How It Works (Pipeline)

```
input text
   -> sentence split (NLTK)
   -> sentence embeddings (MiniLM, 384-dim)
   -> pairwise cosine similarity
   -> agglomerative clustering (average linkage, threshold 0.6)
   -> keep the sentence closest to each cluster's centre (drop the rest)
   -> sentiment check before vs. after (DistilBERT)
   -> compressed text + metrics (CR, RR, SS, SC)
```

Each step in plain terms:

1. **Sentence split** — the input paragraph is broken into individual
   sentences and cleaned (extra spaces and noise removed).
2. **Embeddings** — every sentence is turned into a 384-number vector using
   the MiniLM model. Sentences with similar meaning get similar vectors.
3. **Similarity** — cosine similarity measures how close each pair of
   sentence vectors is.
4. **Clustering** — sentences that are close in meaning are grouped into the
   same cluster. Two sentences saying the same thing land together.
5. **Representative selection** — from each cluster, only the single sentence
   nearest the cluster's centre is kept; the rest are dropped as redundant.
6. **Sentiment check** — DistilBERT reads the original and the compressed text
   and reports the sentiment (POSITIVE / NEGATIVE) of each, so we can verify
   the tone survived compression.
7. **Metrics** — the four evaluation numbers are computed (see below).

## Project Structure

```
semantic-text-compression/
├── app.py                       # Flask web app (single POST /compress endpoint)
├── evaluate.py                  # Evaluation harness (CR/RR/SS/SC over many samples)
├── requirements.txt             # Python dependencies
├── README.md
├── .gitignore
├── src/
│   ├── __init__.py
│   └── semantic_compression.py  # Core pipeline: preprocessing, embeddings,
│                                 # clustering, sentiment, metrics
├── templates/
│   └── index.html               # Web UI page
└── static/
    ├── style.css                # Web UI styling
    └── script.js                # Web UI logic (calls the backend)
```

## Requirements

- Python 3.9 or newer
- The packages listed in `requirements.txt`:
  `flask`, `nltk`, `numpy`, `scikit-learn`, `sentence-transformers`,
  `transformers`, `torch`

## Installation

```bash
# 1. (optional) create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 2. install dependencies
pip install -r requirements.txt
```

The first run automatically downloads three models (needs an internet
connection, one-time, a few minutes):

- MiniLM: `all-MiniLM-L6-v2` (sentence embeddings)
- DistilBERT: `distilbert-base-uncased-finetuned-sst-2-english` (sentiment)
- NLTK Punkt sentence tokenizer

## Usage

Run all commands from the **project root** (the folder that contains `app.py`),
because the scripts import the core module as `src.semantic_compression`.

### 1. Web app

```bash
python app.py
```

Then open `http://127.0.0.1:5000/` in a browser. Paste a paragraph, click
**Run Compression**, and the page shows the compressed text, the three metric
cards (CR, RR, SS), and the sentiment label before and after.

### 2. Evaluation harness

Run the built-in set of sample paragraphs:

```bash
python evaluate.py
```

Run it on your own dataset (for example IMDb or Amazon reviews in a CSV):

```bash
python evaluate.py --csv reviews.csv --column review --limit 50
```

- `--csv`    path to a CSV file of texts
- `--column` name of the text column in that CSV (default: `review`)
- `--limit`  how many rows to evaluate (default: 20)
- `--out`    output CSV file for the results (default: `evaluation_results.csv`)

It prints a per-sample table, the average metrics, and writes the full results
to a CSV file.

### 3. Use the pipeline directly in Python

```python
from src.semantic_compression import SemanticCompressor

compressor = SemanticCompressor()
result = compressor.compress("Your paragraph of text goes here...")

print(result.compressed_text)   # the shortened text
print(result.to_dict())         # all metrics as a dictionary
```

## Evaluation Metrics

The system reports four metrics, all defined in the paper:

| Metric | Meaning | Formula |
|--------|---------|---------|
| **CR** — Compression Ratio | How much shorter the text got. Lower = stronger compression. | `len(compressed_text) / len(original_text)` |
| **RR** — Redundancy Reduction (%) | Percentage of sentences removed as redundant. Higher = more removed. | `(1 - kept_sentences / original_sentences) * 100` |
| **SS** — Semantic Similarity | How well the meaning was preserved. Higher = better. | cosine similarity between the mean sentence embedding of the original text and of the compressed text |
| **SC** — Sentiment Consistency | Whether the emotional tone stayed the same. | `1` if the DistilBERT sentiment label is the same before and after compression, else `0` |

## Configuration

The main settings are passed to `SemanticCompressor` and can be changed when
creating it:

```python
SemanticCompressor(
    embedding_model_name="all-MiniLM-L6-v2",
    sentiment_model_name="distilbert-base-uncased-finetuned-sst-2-english",
    distance_threshold=0.6,   # higher = fewer, looser clusters (less compression)
    linkage="average",        # how clusters are merged
)
```

The defaults `distance_threshold=0.6` and `linkage="average"` are the
best-performing values reported in the paper's clustering evaluation
(Table IV).

## Notes

- **This is one pipeline, not an off-the-shelf tool.** The compression logic
  (clustering similar sentences and keeping one per cluster) is the project's
  own code. MiniLM and DistilBERT are used only as components — one turns a
  sentence into a vector, the other reads sentiment. No pre-built summarizer
  such as BART or GPT produces the output.
- Short paragraphs (5–7 sentences) show modest redundancy reduction. Longer,
  more repetitive text (like full product or movie reviews) compresses more,
  closer to the numbers reported in the paper.
