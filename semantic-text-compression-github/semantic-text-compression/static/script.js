const runBtn = document.getElementById("run-btn");
const inputText = document.getElementById("input-text");
const errorMessage = document.getElementById("error-message");
const loading = document.getElementById("loading");
const results = document.getElementById("results");

const metricCR = document.getElementById("metric-cr");
const metricRR = document.getElementById("metric-rr");
const metricSS = document.getElementById("metric-ss");
const originalTextEl = document.getElementById("original-text");
const compressedTextEl = document.getElementById("compressed-text");
const originalSentimentEl = document.getElementById("original-sentiment");
const compressedSentimentEl = document.getElementById("compressed-sentiment");

function setLoading(isLoading) {
  runBtn.disabled = isLoading;
  loading.classList.toggle("hidden", !isLoading);
}

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.classList.remove("hidden");
}

function clearError() {
  errorMessage.textContent = "";
  errorMessage.classList.add("hidden");
}

function formatSentiment(sentiment) {
  return `${sentiment.label} (${sentiment.score.toFixed(4)})`;
}

async function runCompression() {
  const text = inputText.value.trim();
  clearError();

  if (!text) {
    showError("Please enter some text before running compression.");
    return;
  }

  setLoading(true);
  results.classList.add("hidden");

  try {
    const response = await fetch("/compress", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Something went wrong.");
    }

    metricCR.textContent = data.compression_ratio;
    metricRR.textContent = `${data.redundancy_reduction}%`;
    metricSS.textContent = data.semantic_similarity;

    originalTextEl.textContent = data.original_text;
    compressedTextEl.textContent = data.compressed_text;
    originalSentimentEl.textContent = formatSentiment(data.original_sentiment);
    compressedSentimentEl.textContent = formatSentiment(data.compressed_sentiment);

    results.classList.remove("hidden");
  } catch (err) {
    showError(err.message || "Compression failed. Please try again.");
  } finally {
    setLoading(false);
  }
}

runBtn.addEventListener("click", runCompression);
