"""
app.py — REST API for real-time network traffic anomaly detection
Run: python app.py
"""

from flask import Flask, request, jsonify, render_template_string
from src.pipeline import NetworkTrafficAnalyzer
import os, json

app = Flask(__name__)
analyzer = NetworkTrafficAnalyzer(model_dir="models")

# Load trained models at startup
try:
    analyzer.load()
    MODELS_READY = True
except Exception:
    MODELS_READY = False
    print("[!] No trained models found. Run 'python train.py' first.")


DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Network Traffic Anomaly Detector</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', sans-serif; background: #0f1117; color: #e2e8f0; min-height: 100vh; }
  .header { background: #1a1d27; border-bottom: 1px solid #2d3748; padding: 1.5rem 2rem; display: flex; align-items: center; gap: 1rem; }
  .header h1 { font-size: 1.4rem; font-weight: 600; color: #fff; }
  .badge { background: #22543d; color: #68d391; font-size: 11px; padding: 3px 10px; border-radius: 100px; font-weight: 500; }
  .badge.off { background: #742a2a; color: #fc8181; }
  .container { max-width: 900px; margin: 2rem auto; padding: 0 1.5rem; }
  .card { background: #1a1d27; border: 1px solid #2d3748; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; }
  .card h2 { font-size: 1rem; font-weight: 600; margin-bottom: 1rem; color: #a0aec0; text-transform: uppercase; letter-spacing: 0.05em; }
  .form-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.75rem; }
  label { font-size: 12px; color: #718096; display: block; margin-bottom: 4px; }
  input, select { width: 100%; background: #2d3748; border: 1px solid #4a5568; border-radius: 6px; padding: 8px 10px; color: #e2e8f0; font-size: 13px; }
  button { background: #4299e1; color: white; border: none; padding: 10px 24px; border-radius: 8px; font-size: 14px; font-weight: 600; cursor: pointer; margin-top: 1rem; }
  button:hover { background: #3182ce; }
  .result { display: none; margin-top: 1.5rem; padding: 1.5rem; border-radius: 10px; }
  .result.attack { background: #2d1515; border: 1px solid #742a2a; }
  .result.normal { background: #0f2d1a; border: 1px solid #22543d; }
  .result-title { font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem; }
  .result.attack .result-title { color: #fc8181; }
  .result.normal .result-title { color: #68d391; }
  .metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.75rem; margin-top: 1rem; }
  .metric { background: rgba(255,255,255,0.05); border-radius: 8px; padding: 0.75rem; text-align: center; }
  .metric-val { font-size: 1.4rem; font-weight: 700; color: #fff; }
  .metric-label { font-size: 11px; color: #718096; margin-top: 2px; }
  .api-block { background: #2d3748; border-radius: 8px; padding: 1rem; font-family: monospace; font-size: 12px; color: #a0aec0; overflow-x: auto; }
  .endpoint { display: flex; align-items: center; gap: 8px; margin-bottom: 0.5rem; }
  .method { background: #2b4c7e; color: #90cdf4; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }
  .method.get { background: #1a3a2a; color: #68d391; }
</style>
</head>
<body>
<div class="header">
  <h1>🛡️ Network Traffic Anomaly Detector</h1>
  <span class="badge {{ 'on' if models_ready else 'off' }}">
    {{ '● Models Loaded' if models_ready else '● Models Not Found' }}
  </span>
</div>
<div class="container">
  <div class="card">
    <h2>Live Traffic Classifier</h2>
    <form id="classifyForm">
      <div class="form-grid">
        <div><label>Protocol</label>
          <select name="protocol_type"><option>tcp</option><option>udp</option><option>icmp</option></select></div>
        <div><label>Service</label>
          <select name="service"><option>http</option><option>ftp</option><option>smtp</option><option>ssh</option><option>dns</option><option>other</option></select></div>
        <div><label>Flag</label>
          <select name="flag"><option>SF</option><option>S0</option><option>REJ</option><option>RSTO</option><option>SH</option></select></div>
        <div><label>Duration (s)</label><input type="number" name="duration" value="0" min="0"></div>
        <div><label>Src Bytes</label><input type="number" name="src_bytes" value="181" min="0"></div>
        <div><label>Dst Bytes</label><input type="number" name="dst_bytes" value="5450" min="0"></div>
        <div><label>Count</label><input type="number" name="count" value="8" min="0"></div>
        <div><label>Srv Count</label><input type="number" name="srv_count" value="8" min="0"></div>
        <div><label>Serror Rate</label><input type="number" name="serror_rate" value="0.0" step="0.01" min="0" max="1"></div>
        <div><label>Same Srv Rate</label><input type="number" name="same_srv_rate" value="1.0" step="0.01" min="0" max="1"></div>
        <div><label>Logged In</label>
          <select name="logged_in"><option value="1">Yes</option><option value="0">No</option></select></div>
        <div><label>Dst Host Count</label><input type="number" name="dst_host_count" value="9" min="0" max="255"></div>
      </div>
      <button type="submit">Classify Traffic →</button>
    </form>
    <div class="result" id="result">
      <div class="result-title" id="result-title"></div>
      <div id="result-sub" style="font-size:13px; color:#718096; margin-bottom:0.5rem;"></div>
      <div class="metrics">
        <div class="metric"><div class="metric-val" id="conf">-</div><div class="metric-label">Confidence</div></div>
        <div class="metric"><div class="metric-val" id="risk">-</div><div class="metric-label">Risk Level</div></div>
        <div class="metric"><div class="metric-val" id="model">2</div><div class="metric-label">Models Used</div></div>
      </div>
    </div>
  </div>
  <div class="card">
    <h2>API Endpoints</h2>
    <div class="api-block">
      <div class="endpoint"><span class="method get">GET</span> /health — System status</div>
      <div class="endpoint"><span class="method">POST</span> /predict — Classify single record</div>
      <div class="endpoint"><span class="method">POST</span> /predict/batch — Classify multiple records</div>
      <div class="endpoint"><span class="method get">GET</span> /models/info — Model performance metrics</div>
    </div>
  </div>
</div>
<script>
document.getElementById('classifyForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const body = {};
  fd.forEach((v, k) => body[k] = isNaN(v) ? v : Number(v));

  const res = await fetch('/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body)
  });
  const data = await res.json();
  const el = document.getElementById('result');
  el.className = 'result ' + (data.prediction === 'ATTACK' ? 'attack' : 'normal');
  el.style.display = 'block';
  document.getElementById('result-title').textContent =
    (data.prediction === 'ATTACK' ? '⚠️ ATTACK DETECTED' : '✅ NORMAL TRAFFIC');
  document.getElementById('result-sub').textContent =
    `Ensemble of Random Forest + XGBoost`;
  document.getElementById('conf').textContent = data.confidence + '%';
  document.getElementById('risk').textContent = data.risk_level;
});
</script>
</body>
</html>
"""


@app.route("/")
def dashboard():
    return render_template_string(DASHBOARD_HTML, models_ready=MODELS_READY)


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": MODELS_READY,
        "available_models": list(analyzer.models.keys())
    })


@app.route("/predict", methods=["POST"])
def predict():
    if not MODELS_READY:
        return jsonify({"error": "Models not loaded. Run train.py first."}), 503
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body provided."}), 400
    try:
        result = analyzer.predict(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    if not MODELS_READY:
        return jsonify({"error": "Models not loaded."}), 503
    records = request.get_json()
    if not isinstance(records, list):
        return jsonify({"error": "Expected a list of records."}), 400
    results = [analyzer.predict(r) for r in records]
    return jsonify(results)


@app.route("/models/info")
def models_info():
    if not MODELS_READY:
        return jsonify({"error": "Models not loaded."}), 503
    try:
        with open("models/results.json") as f:
            return jsonify(json.load(f))
    except Exception:
        return jsonify({"available_models": list(analyzer.models.keys())})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
