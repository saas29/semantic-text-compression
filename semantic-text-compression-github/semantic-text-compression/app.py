"""
app.py

Flask backend for the semantic text compression web demo described in
Section III-D of "Meaning Matters" (Fig. 4). Exposes a single POST
endpoint, /compress, that runs the semantic compression pipeline on the
submitted text and returns the compressed text plus evaluation metrics
as JSON.

Run locally:
    python app.py

Then open http://127.0.0.1:5000/ in a browser.
"""

from flask import Flask, jsonify, render_template, request

from src.semantic_compression import SemanticCompressor

app = Flask(__name__)

# The embedding + sentiment models are loaded once at startup so individual
# requests stay fast (loading DistilBERT/MiniLM per-request would be slow).
compressor = SemanticCompressor()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/compress", methods=["POST"])
def compress():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Please provide some text to compress."}), 400

    try:
        result = compressor.compress(text)
    except Exception as exc:  # pragma: no cover - defensive guard for the demo UI
        return jsonify({"error": f"Compression failed: {exc}"}), 500

    return jsonify(result.to_dict())


if __name__ == "__main__":
    # Warm up the models once, before serving requests, so the first
    # request from the UI isn't stuck waiting on a cold model load.
    compressor.warm_up()
    app.run(host="0.0.0.0", port=5000, debug=True)
