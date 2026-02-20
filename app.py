import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from cerebras.cloud.sdk import Cerebras
from datetime import datetime

app = Flask(__name__)
CORS(app)

# ---- Config ----
CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY")
DAILY_LIMIT = 15

# ---- Cerebras client ----
client = Cerebras(api_key=CEREBRAS_API_KEY)

# ---- Usage tracking ----
usage = {}  # { user_id: {"count": 0, "date": "YYYY-MM-DD"} }

def check_limit(user_id):
    today = datetime.utcnow().strftime("%Y-%m-%d")
    if user_id not in usage or usage[user_id]["date"] != today:
        usage[user_id] = {"count": 0, "date": today}
    if usage[user_id]["count"] >= DAILY_LIMIT:
        return False
    usage[user_id]["count"] += 1
    return True

# ---- Model selection ----
def get_best_model():
    """
    Fetch available models from Cerebras API and pick the one with the highest quota.
    Note: public API may not include quota, so this can use fallback criteria.
    """
    MODELS_URL = "https://api.cerebras.ai/v1/models"  # correct endpoint
    headers = {
        "Authorization": f"Bearer {CEREBRAS_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.get(MODELS_URL, headers=headers)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print("Error fetching models:", e)
        return "llama-3.3-70b"  # fallback default

    models = []
    for model in data.get("data", []):
        # Some endpoints do not return quotas; you can use context length or other fields
        models.append({
            "id": model.get("id"),
            "max_context": model.get("max_context_length", 0),
        })

    if not models:
        return "llama-3.3-70b"

    # Pick model with largest context length
    best_model = max(models, key=lambda m: m["max_context"])
    return best_model["id"]

# ---- Routes ----
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Cerebras Flask API with daily limit running"})

@app.route("/generate", methods=["POST", "OPTIONS"])
def generate():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    user_id = request.remote_addr

    if not check_limit(user_id):
        return jsonify({
            "response": f"You’ve reached your daily limit of {DAILY_LIMIT} prompts. Please try again tomorrow."
        }), 429

    data = request.json or {}
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    model_id = get_best_model()

    try:
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_id,
            max_completion_tokens=1024,
            temperature=0.2,
            top_p=1,
            stream=False
        )
        generated_text = completion.choices[0].message.content
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"response": generated_text, "model_used": model_id})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
