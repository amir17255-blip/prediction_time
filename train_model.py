# train_model.py
import pandas as pd
import numpy as np
import re
import os
import joblib
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def parse_duration(s):
    if pd.isna(s): return np.nan
    if isinstance(s, (int, float)): return float(s)
    s = str(s).strip()
    if s in ("", "0:00:00", "0", "nan"): return 0.0
    days = 0.0
    m = re.match(r"(\d+)\s*day[s]?,?\s*(.*)", s)
    if m:
        days = float(m.group(1))
        s = m.group(2)
    parts = s.split(":")
    try:
        h = float(parts[0]) if len(parts) > 0 else 0
        mnt = float(parts[1]) if len(parts) > 1 else 0
        sec = float(parts[2]) if len(parts) > 2 else 0
    except ValueError:
        return np.nan
    return days * 24 + h + mnt / 60 + sec / 3600

def train():
    print("⏳ در حال بارگذاری داده‌ها...")
    try:
        df = pd.read_excel("data/tasks.xlsx", sheet_name="Sheet1", header=1)
        if str(df.columns[0]).strip() != "شناسه":
            df = pd.read_excel("data/tasks.xlsx", sheet_name="Sheet1", header=None)
            header_row = 0
            for i, row in df.iterrows():
                if 'شناسه' in [str(x).strip() for x in row.values]:
                    header_row = i
                    break
            df.columns = df.iloc[header_row]
            df = df.iloc[header_row + 1:].reset_index(drop=True)
    except Exception as e:
        print(f"❌ خطا در خواندن فایل: {e}")
        return

    df.columns = [str(c).strip() for c in df.columns]
    rename_dict = {
        "شناسه": "task_id", "حجم کار": "workload", "ماژول": "module",
        "نوع مسئله": "task_type", "اولویت": "priority", "مجری": "assignee",
        "نفر ساعت پیش بینی شده": "estimated_h", "مجموع زمان گزارش شده": "actual_h",
        "وضعیت": "status", "تاریخ شروع": "start_date", "تاریخ اتمام": "end_date"
    }
    
    final_rename = {}
    for col in df.columns:
        col_str = str(col).strip()
        for key, val in rename_dict.items():
            if key in col_str:
                final_rename[col] = val
                break
    df = df.rename(columns=final_rename)
    
    df["estimated_h"] = df["estimated_h"].apply(parse_duration)
    df["actual_h"] = df["actual_h"].apply(parse_duration)
    
    # محاسبه تاخیر (تفاوت تاریخ اتمام با تاریخ اتمام پیش بینی شده)
    # برای سادگی در این نسخه، صرفاً از نسبت زمان واقعی به برآورد استفاده می‌کنیم
    df = df.dropna(subset=["estimated_h", "actual_h", "workload", "module", "task_type", "priority", "assignee"])
    df = df[(df["status"] == "done") & (df["actual_h"] > 0) & (df["estimated_h"] > 0)]
    
    print(f"✅ تعداد رکوردهای معتبر: {len(df)}")

    # محاسبه ضریب کالیبراسیون
    calibration_factor = float((df["actual_h"] / df["estimated_h"]).median())
    print(f"🔧 ضریب کالیبراسیون: {calibration_factor:.2f}")

    # افزودن مجری به عنوان متغیر ورودی
    X = df[["module", "workload", "task_type", "priority", "assignee", "estimated_h"]]
    y = df["actual_h"]

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['module', 'workload', 'task_type', 'priority', 'assignee'])
        ],
        remainder='passthrough'
    )

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=150, random_state=42))
    ])

    print("🧠 در حال آموزش مدل...")
    model.fit(X, y)
    mae = mean_absolute_error(y, model.predict(X))
    print(f"📊 خطای میانگین مدل (MAE): {mae:.2f} ساعت")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")
    
    # استخراج لیست مجریان برای نمایش در وب‌سایت
    options = {
        "modules": [str(x) for x in df["module"].dropna().unique().tolist()],
        "workloads": [str(x) for x in df["workload"].dropna().unique().tolist()],
        "task_types": [str(x) for x in df["task_type"].dropna().unique().tolist()],
        "priorities": [str(x) for x in df["priority"].dropna().unique().tolist()],
        "assignees": [str(x) for x in df["assignee"].dropna().unique().tolist()], # اضافه شدن مجریان
        "calibration_factor": calibration_factor,
        "total_tasks_analyzed": len(df)
    }
    with open("models/options.json", "w", encoding="utf-8") as f:
        json.dump(options, f, ensure_ascii=False, indent=4)

    print("💾 مدل و تنظیمات جدید ذخیره شدند!")

if __name__ == "__main__":
    train()