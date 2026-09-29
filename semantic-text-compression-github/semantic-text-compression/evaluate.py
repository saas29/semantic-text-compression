"""
evaluate.py

Evaluation harness for the semantic text compression model. Runs the pipeline
over a set of texts and reports the four evaluation metrics from the paper:

    CR  - Compression Ratio        (lower = stronger compression)
    RR  - Redundancy Reduction (%)  (higher = more redundant sentences removed)
    SS  - Semantic Similarity       (higher = meaning better preserved)
    SC  - Sentiment Consistency     (1 = sentiment label unchanged after compression)

It prints a per-sample table plus aggregate averages, and writes the results
to a CSV file.

Usage:
    # Built-in sample set
    python evaluate.py

    # Your own dataset (e.g. IMDb / Amazon reviews)
    python evaluate.py --csv reviews.csv --column review --limit 50

    # Change the output file
    python evaluate.py --out my_results.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
from typing import List

from src.semantic_compression import SemanticCompressor


# ---------------------------------------------------------------------------
# Built-in sample set (review-style paragraphs with redundant sentences)
# ---------------------------------------------------------------------------

BUILTIN_SAMPLES: List[str] = [
    "I recently bought this laptop and I am really happy with it. This laptop has "
    "made me very satisfied since I purchased it. The battery life is excellent and "
    "easily lasts a full day of work. Performance is fast and it handles heavy "
    "applications without lag. It runs demanding software smoothly without any "
    "slowdown. Overall it is one of the best purchases I have made this year.",

    "The movie was a complete waste of time. Honestly, watching this film felt like "
    "throwing my evening away. The plot made no sense and the pacing was painfully "
    "slow. The storyline was confusing and dragged on far too long. The acting was "
    "wooden and unconvincing throughout. I would not recommend this to anyone.",

    "This restaurant exceeded all my expectations. The food here was far better than "
    "I had hoped for. The pasta was rich, creamy, and full of flavour. The dessert "
    "was equally delicious and beautifully presented. Service was quick and the staff "
    "were friendly and attentive. I will definitely be coming back again soon.",

    "The hotel room was clean and comfortable. Everything in the room was tidy and "
    "well maintained. The bed was soft and I slept very well. The view from the "
    "balcony was stunning at sunrise. The staff were polite and helpful during "
    "check-in. It was a relaxing stay overall.",

    "This phone is a huge disappointment. I really regret buying this device. The "
    "battery drains extremely fast even with light use. The camera takes blurry "
    "photos in low light. Photos come out grainy and unclear when it is dark. The "
    "screen also has an annoying flicker. I expected much better for the price.",

    "The book kept me hooked from the very first page. I could not put this novel "
    "down once I started reading it. The characters were deep and well developed. "
    "The plot twists were genuinely surprising and clever. The writing style was "
    "elegant and easy to follow. It is easily one of my favourite reads this year.",

    "Customer service was outstanding from start to finish. The support team helped "
    "me resolve my issue in minutes. The representative was patient and explained "
    "everything clearly. My problem was fixed quickly and without any hassle. I felt "
    "truly valued as a customer. I would happily recommend this company to others.",

    "The concert was an unforgettable experience. The performance that night was "
    "absolutely magical. The band sounded incredible live. The lighting and stage "
    "design were spectacular. The crowd was full of energy and enthusiasm. I will "
    "cherish the memory of this show for a long time.",

    "This software is buggy and frustrating to use. The application crashes constantly "
    "during normal work. It freezes randomly and loses my unsaved changes. The "
    "interface is cluttered and confusing to navigate. Loading times are painfully "
    "slow as well. I am seriously considering switching to something else.",

    "The gym has fantastic facilities and equipment. All the machines here are modern "
    "and well maintained. There is plenty of space even during peak hours. The "
    "trainers are knowledgeable and supportive. The locker rooms are always clean and "
    "tidy. Joining this gym was a great decision.",

    "The flight experience was terrible from the beginning. The whole journey was "
    "stressful and exhausting. The flight was delayed by over three hours. The seats "
    "were cramped and uncomfortable. The cabin crew seemed rude and inattentive. I "
    "will avoid this airline in the future.",

    "This coffee maker works perfectly every single morning. It brews a great cup of "
    "coffee consistently. The design is sleek and fits nicely on my counter. It is "
    "very easy to clean after use. The coffee stays hot for a long time. I use it "
    "every day without any complaints.",

    "The online course was informative and well structured. The lessons were clear "
    "and easy to follow. The instructor explained complex topics in a simple way. "
    "The assignments reinforced what I learned effectively. The pace was comfortable "
    "for a beginner. I feel much more confident in the subject now.",

    "The headphones have amazing sound quality. The audio is crisp, clear, and "
    "balanced. The bass is deep without being overpowering. They are comfortable to "
    "wear for hours at a time. The noise cancellation works remarkably well. These "
    "are the best headphones I have owned.",

    "The apartment was smaller than the photos suggested. It looked much bigger in "
    "the listing pictures. The neighbourhood was noisy late at night. The plumbing "
    "had issues that were never fixed. The landlord was slow to respond to requests. "
    "I would not renew the lease again.",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_from_csv(path: str, column: str, limit: int) -> List[str]:
    """Read a text column from a CSV file (e.g. IMDb / Amazon reviews)."""
    texts: List[str] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError("CSV file appears to be empty.")
        if column not in reader.fieldnames:
            raise ValueError(
                f"Column '{column}' not found. Available columns: {reader.fieldnames}"
            )
        for row in reader:
            value = (row.get(column) or "").strip()
            if value:
                texts.append(value)
            if 0 < limit <= len(texts):
                break
    return texts


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(compressor: SemanticCompressor, texts: List[str]) -> List[dict]:
    rows = []
    for i, text in enumerate(texts, start=1):
        result = compressor.compress(text)
        rows.append(
            {
                "sample": i,
                "orig_sentences": result.num_original_sentences,
                "kept_sentences": result.num_clusters,
                "CR": round(result.compression_ratio, 3),
                "RR(%)": round(result.redundancy_reduction, 2),
                "SS": round(result.semantic_similarity, 3),
                "SC": result.sentiment_consistency,
                "orig_sentiment": result.original_sentiment.label,
                "comp_sentiment": result.compressed_sentiment.label,
            }
        )
    return rows


def print_table(rows: List[dict]) -> None:
    header = (
        f"{'#':>3}  {'Sent':>4}  {'Kept':>4}  {'CR':>5}  {'RR(%)':>6}  "
        f"{'SS':>5}  {'SC':>2}  {'Orig':>8}  {'Comp':>8}"
    )
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['sample']:>3}  {r['orig_sentences']:>4}  {r['kept_sentences']:>4}  "
            f"{r['CR']:>5.3f}  {r['RR(%)']:>6.2f}  {r['SS']:>5.3f}  {r['SC']:>2}  "
            f"{r['orig_sentiment']:>8}  {r['comp_sentiment']:>8}"
        )


def print_summary(rows: List[dict]) -> dict:
    n = len(rows)
    avg_cr = statistics.mean(r["CR"] for r in rows)
    avg_rr = statistics.mean(r["RR(%)"] for r in rows)
    avg_ss = statistics.mean(r["SS"] for r in rows)
    sc_rate = 100.0 * sum(r["SC"] for r in rows) / n

    print("\n" + "=" * 40)
    print("AGGREGATE METRICS (average over samples)")
    print("=" * 40)
    print(f"Samples evaluated          : {n}")
    print(f"Compression Ratio (CR)     : {avg_cr:.3f}")
    print(f"Redundancy Reduction (RR)  : {avg_rr:.2f}%")
    print(f"Semantic Similarity (SS)   : {avg_ss:.3f}")
    print(f"Sentiment Consistency (SC) : {sc_rate:.1f}%")
    print("=" * 40)

    return {
        "samples": n,
        "avg_CR": round(avg_cr, 3),
        "avg_RR(%)": round(avg_rr, 2),
        "avg_SS": round(avg_ss, 3),
        "SC_rate(%)": round(sc_rate, 1),
    }


def write_csv(rows: List[dict], summary: dict, out_path: str) -> None:
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        # Blank line + aggregate summary appended at the bottom.
        fh.write("\n")
        fh.write("AGGREGATE\n")
        for key, value in summary.items():
            fh.write(f"{key},{value}\n")
    print(f"\nResults written to: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the semantic text compression model.")
    parser.add_argument("--csv", help="Path to a CSV dataset (e.g. IMDb/Amazon reviews).")
    parser.add_argument("--column", default="review", help="Text column name in the CSV (default: review).")
    parser.add_argument("--limit", type=int, default=20, help="Max samples to evaluate from the CSV (default: 20).")
    parser.add_argument("--out", default="evaluation_results.csv", help="Output CSV path (default: evaluation_results.csv).")
    args = parser.parse_args()

    if args.csv:
        print(f"Loading up to {args.limit} samples from '{args.csv}' (column '{args.column}')...")
        texts = load_from_csv(args.csv, args.column, args.limit)
    else:
        print("Using built-in sample set (pass --csv to evaluate your own dataset).")
        texts = BUILTIN_SAMPLES

    if not texts:
        print("No texts to evaluate.")
        return

    print(f"Loading models and evaluating {len(texts)} samples (first run downloads models)...")
    compressor = SemanticCompressor()
    compressor.warm_up()

    rows = evaluate(compressor, texts)
    print_table(rows)
    summary = print_summary(rows)
    write_csv(rows, summary, args.out)


if __name__ == "__main__":
    main()
