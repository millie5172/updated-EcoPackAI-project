from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
import random
from datetime import datetime
import os
from urllib.parse import urlparse

# For production (Render, Heroku, Railway, etc.)
DATABASE_URL = os.environ.get('DATABASE_URL')

if DATABASE_URL:
    # Parse the URL for psycopg2
    result = urlparse(DATABASE_URL)
    conn = psycopg2.connect(
        host=result.hostname,
        database=result.path[1:],
        user=result.username,
        password=result.password,
        port=result.port
    )
else:
    # Local development fallback
    conn = psycopg2.connect(
        host="localhost",
        database="infosys",
        user="postgres",
        password=os.environ.get('DB_PASSWORD', 'meeta')
    )

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# Database setup - SQLite for easy deployment (switch to PostgreSQL later if needed)
import sqlite3
from contextlib import closing

DB_PATH = 'ecopackai.db'


def init_db():
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                weight REAL,
                durability INTEGER,
                recyclable INTEGER,
                material TEXT,
                predicted_cost REAL,
                predicted_co2 REAL,
                environment_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()


# Load models with fallback to dummy models if files don't exist
def load_models():
    try:
        cost_model = joblib.load("models/cost_model.pkl")
        co2_model = joblib.load("models/co2_model.pkl")
        print("✅ Models loaded successfully")
        return cost_model, co2_model
    except Exception as e:
        print(f"⚠️  Could not load models ({e}), using dummy models")
        # Create dummy models for testing
        from sklearn.ensemble import RandomForestRegressor
        dummy_model = RandomForestRegressor(n_estimators=10)
        # Fit with dummy data so it can predict
        X_dummy = np.random.rand(10, 11)
        y_dummy = np.random.rand(10)
        dummy_model.fit(X_dummy, y_dummy)
        return dummy_model, dummy_model


cost_model, co2_model = load_models()

# Material database from your CSV
MATERIALS_DB = [
    {"name": "Bagasse Fiber", "co2_per_kg": 0.15, "cost_per_unit": 0.42, "biodegradability": 98, "durability": 85,
     "recyclability": 100},
    {"name": "Mushroom Mycelium", "co2_per_kg": 0.08, "cost_per_unit": 0.85, "biodegradability": 100, "durability": 90,
     "recyclability": 100},
    {"name": "PLA Bioplastic", "co2_per_kg": 0.68, "cost_per_unit": 0.65, "biodegradability": 85, "durability": 75,
     "recyclability": 60},
    {"name": "Recycled Cardboard", "co2_per_kg": 0.32, "cost_per_unit": 0.35, "biodegradability": 95, "durability": 70,
     "recyclability": 95},
    {"name": "Recycled Paperboard", "co2_per_kg": 0.30, "cost_per_unit": 0.40, "biodegradability": 90, "durability": 65,
     "recyclability": 90},
    {"name": "Corn Starch Polymer", "co2_per_kg": 0.45, "cost_per_unit": 0.58, "biodegradability": 90, "durability": 72,
     "recyclability": 70},
    {"name": "Biodegradable Plastic", "co2_per_kg": 0.55, "cost_per_unit": 0.70, "biodegradability": 80,
     "durability": 78, "recyclability": 50},
    {"name": "Glass", "co2_per_kg": 0.85, "cost_per_unit": 0.90, "biodegradability": 0, "durability": 95,
     "recyclability": 100},
    {"name": "Aluminum", "co2_per_kg": 1.20, "cost_per_unit": 1.10, "biodegradability": 0, "durability": 98,
     "recyclability": 100},
    {"name": "Molded Pulp", "co2_per_kg": 0.25, "cost_per_unit": 0.45, "biodegradability": 95, "durability": 68,
     "recyclability": 85}
]

FEATURES = [
    'weight', 'durability', 'recyclable',
    'material_Bagasse Fiber', 'material_Biodegradable Plastic',
    'material_Corn Starch Polymer', 'material_Glass',
    'material_Molded Pulp', 'material_PLA Bioplastic',
    'material_Recycled Cardboard', 'material_Recycled Paperboard'
]


def calculate_environment_score(co2, recyclable, durability):
    score = (1 / (1 + co2)) * 0.5 + (recyclable / 100) * 0.3 + (durability / 100) * 0.2
    return round(score, 3)


# Serve Frontend
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/api/materials', methods=['GET'])
def get_materials():
    return jsonify(MATERIALS_DB)


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        material_name = data.get("material", "PLA Bioplastic")
        weight = data.get("weight", 1.0)
        durability = data.get("durability", 5)
        recyclable = data.get("recyclable", 1)

        # Prepare input features
        row = {feature: 0 for feature in FEATURES}
        row["weight"] = weight
        row["durability"] = durability
        row["recyclable"] = 1 if recyclable else 0

        material_col = f"material_{material_name}"
        if material_col in row:
            row[material_col] = 1
        else:
            return jsonify({"error": "Invalid material"}), 400

        input_df = pd.DataFrame([row], columns=FEATURES)

        # Predict
        cost = float(cost_model.predict(input_df)[0])
        co2 = float(co2_model.predict(input_df)[0])
        env_score = calculate_environment_score(co2, recyclable, durability)

        # Save to database
        with closing(sqlite3.connect(DB_PATH)) as conn:
            conn.execute("""
                INSERT INTO predictions 
                (weight, durability, recyclable, material, predicted_cost, predicted_co2, environment_score)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (weight, durability, recyclable, material_name, cost, co2, env_score))
            conn.commit()

        return jsonify({
            "predicted_cost": round(cost, 2),
            "predicted_co2": round(co2, 3),
            "environment_score": env_score,
            "material": material_name
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/recommend", methods=["POST"])
def recommend_material():
    data = request.get_json() or {}

    results = []
    for m in MATERIALS_DB:
        score = calculate_environment_score(
            m["co2_per_kg"],
            m["recyclability"],
            m["durability"]
        )
        results.append({
            "material": m["name"],
            "environment_score": score,
            "co2": m["co2_per_kg"],
            "cost": m["cost_per_unit"]
        })

    best = max(results, key=lambda x: x["environment_score"])

    return jsonify({
        "recommended_material": best["material"],
        "ranking": sorted(results, key=lambda x: x["environment_score"], reverse=True)
    })


@app.route('/api/history', methods=['GET'])
def get_history():
    with closing(sqlite3.connect(DB_PATH)) as conn:
        cursor = conn.execute("SELECT * FROM predictions ORDER BY created_at DESC LIMIT 50")
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        return jsonify([dict(zip(columns, row)) for row in rows])


@app.route("/dashboard")
def dashboard():
    try:
        with closing(sqlite3.connect(DB_PATH)) as conn:
            df = pd.read_sql_query("SELECT * FROM predictions", conn)

            if df.empty:
                return "<h1>No prediction data available yet.</h1><a href='/'>Go back to app</a>"

            avg_co2 = df["predicted_co2"].mean()
            best_co2 = df["predicted_co2"].min()
            co2_reduction = round(((avg_co2 - best_co2) / avg_co2) * 100, 2) if avg_co2 > 0 else 0

            avg_cost = df["predicted_cost"].mean()
            min_cost = df["predicted_cost"].min()
            cost_savings = round(((avg_cost - min_cost) / avg_cost) * 100, 2) if avg_cost > 0 else 0

            html = f"""
            <html>
            <head><title>EcoPackAI Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; padding: 20px; background: #f0f2f5; }}
                .metric {{ background: white; padding: 20px; border-radius: 10px; margin: 10px; display: inline-block; }}
                .value {{ font-size: 2em; color: #2d6a4f; font-weight: bold; }}
                table {{ border-collapse: collapse; width: 100%; background: white; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background: #2d6a4f; color: white; }}
                tr:nth-child(even) {{ background: #f9f9f9; }}
            </style>
            </head>
            <body>
                <h1>🌍 EcoPackAI Sustainability Dashboard</h1>
                <a href="/">← Back to Application</a>

                <div style="margin: 20px 0;">
                    <div class="metric">
                        <div>CO₂ Reduction Potential</div>
                        <div class="value">{co2_reduction}%</div>
                    </div>
                    <div class="metric">
                        <div>Cost Savings Potential</div>
                        <div class="value">{cost_savings}%</div>
                    </div>
                    <div class="metric">
                        <div>Total Predictions</div>
                        <div class="value">{len(df)}</div>
                    </div>
                </div>

                <h2>Recent Predictions</h2>
                {df.to_html(classes='data', index=False, max_rows=20)}
            </body>
            </html>
            """
            return html
    except Exception as e:
        return f"<h1>Error loading dashboard</h1><p>{str(e)}</p>"


@app.route('/health')
def health():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})


if __name__ == "__main__":
    init_db()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)


