# export_model.py
import joblib
import json
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType, FloatTensorType

# 1. بارگذاری مدل آموزش‌دیده
model = joblib.load("models/model.pkl")

# 2. تعریف ساختار ورودی‌ها برای تبدیل (دقیقاً مثل ترتیبی که در train_model استفاده کردیم)
# 4 ورودی متنی و 1 ورودی عددی
initial_types = [
    ('module', StringTensorType([None, 1])),
    ('workload', StringTensorType([None, 1])),
    ('task_type', StringTensorType([None, 1])),
    ('priority', StringTensorType([None, 1])),
    ('assignee', StringTensorType([None, 1])),
    ('estimated_h', FloatTensorType([None, 1]))
]

# 3. تبدیل به ONNX
print("در حال تبدیل مدل به فرمت وب...")
onx = convert_sklearn(model, initial_types=initial_types, target_opset=12)

# 4. ذخیره فایل مدل وب
with open("model.onnx", "wb") as f:
    f.write(onx.SerializeToString())
print("✅ فایل model.onnx ساخته شد.")

# 5. کپی کردن فایل options.json برای خواندن لیست‌ها در وب
with open("models/options.json", "r", encoding="utf-8") as f:
    options = json.load(f)
with open("options.json", "w", encoding="utf-8") as f:
    json.dump(options, f, ensure_ascii=False)
print("✅ فایل options.json کپی شد.")