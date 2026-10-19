# app.py
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import json
import os

app = Flask(__name__)

if not os.path.exists("models/model.pkl"):
    print("❌ فایل مدل پیدا نشد. لطفاً train_model.py را اجرا کنید.")
    exit()

model = joblib.load("models/model.pkl")
with open("models/options.json", "r", encoding="utf-8") as f:
    options = json.load(f)

@app.route("/")
def home():
    return render_template("index.html", options=options)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        est_h = float(data["estimated_h"])
        
        # دریافت مجری از ورودی
        input_data = pd.DataFrame([{
            "module": data["module"],
            "workload": data["workload"],
            "task_type": data["task_type"],
            "priority": data["priority"],
            "assignee": data["assignee"],
            "estimated_h": est_h
        }])
        
        X_transformed = model.named_steps['preprocessor'].transform(input_data)
        rf_model = model.named_steps['regressor']
        
        all_tree_preds = np.array([tree.predict(X_transformed)[0] for tree in rf_model.estimators_])
        p10, p50, p90 = np.percentile(all_tree_preds, [10, 50, 90])
        
        # شبیه‌سازی مونت‌کارلو
        simulated_times = np.random.choice(all_tree_preds, size=1000)
        prob_finish_on_time = (simulated_times <= est_h).mean() * 100
        
        risk_level = "پایین"
        if prob_finish_on_time < 30: risk_level = "بحرانی"
        elif prob_finish_on_time < 60: risk_level = "بالا"
        elif prob_finish_on_time < 80: risk_level = "متوسط"

        return jsonify({
            "success": True,
            "p10": round(max(0, p10), 1),
            "p50": round(max(0, p50), 1),
            "p90": round(max(0, p90), 1),
            "prob_on_time": round(prob_finish_on_time, 1),
            "risk_level": risk_level,
            "assumptions": f"تحلیل بر اساس {options['total_tasks_analyzed']} تسک و سابقه عملکرد مجری انتخاب‌شده."
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == "__main__":
    app.run(debug=True, port=5000)