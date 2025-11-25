from flask_cors import CORS
from flask import Flask, request, jsonify, render_template
from NHS_SPC_Engine import (
    SPCConfig, fit_baseline_model, run_spc_with_baseline, build_wales_overall, build_deviation_table
)
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

cfg = SPCConfig()

# ============================================================
# LOAD HISTORICAL BASELINE ONCE
# ============================================================
aut = pd.read_csv("uptake_trend_Autumn_2024.csv")
aut["cohort"] = "Autumn 2024"
aut["week_end"] = pd.to_datetime(aut["week_end"], format="%d%b%Y")

spr = pd.read_csv("uptake_trend_Spring_2025.csv")
spr["cohort"] = "Spring 2025"
spr["week_end"] = pd.to_datetime(spr["week_end"], format="%d/%m/%Y")
spr = spr.rename(columns={"vacc": "vacc_alive", "Group": "group"})

history = pd.concat([aut, spr], ignore_index=True)
baseline = fit_baseline_model(history, cfg, by_cohort=False)



# ============================================================
# RUN SPC ENDPOINT (ONLY ONE VERSION NOW)
# ============================================================
@app.post("/run_spc")
def run_spc():
    try:
        file = request.files.get("file")
        if file is None:
            return jsonify({"error": "No file uploaded"}), 400

        df = pd.read_csv(file)
        df["week_end"] = pd.to_datetime(df["week_end"])

        # Run SPC with baseline
        full_new, dash_new, trend_new = run_spc_with_baseline(
            df, baseline, cfg, by_cohort=False
        )
        deviations = build_deviation_table(full_new, cfg)
        # Compute Wales-wide curve
        wales_df = build_wales_overall(full_new, cfg)

        # Prepare JSON-safe data
        def clean(t):
            return t.replace([np.nan, np.inf, -np.inf], None)

        return jsonify({
            "status": "ok",
            "full": clean(full_new).to_dict(orient="records"),
            "dashboard": clean(dash_new).to_dict(orient="records"),
            "trend": clean(trend_new).to_dict(orient="records"),
            "wales": clean(wales_df).to_dict(orient="records"),
            "deviations": clean(deviations).to_dict(orient="records")
        })

    except Exception as e:
        print("ERROR in /run_spc:", e)
        return jsonify({"error": str(e)}), 500



# ============================================================
# BASIC ROUTES
# ============================================================
@app.get("/")
def home():
    return render_template("index.html")

@app.get("/api/status")
def status():
    return {"status": "SPC backend running"}



# ============================================================
# RUN SERVER
# ============================================================
if __name__ == "__main__":
    app.run(debug=True)
