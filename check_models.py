from ultralytics import YOLO
import sys

# User suggests 'yolo12' (no v) logic
models_to_check = ["yolo12n.pt"]

print(f"Checking models...")
for m in models_to_check:
    print(f"--- Checking {m} ---")
    try:
        model = YOLO(m)
        print(f"✅ SUCCESS: {m} loaded/downloaded.")
    except Exception as e:
        print(f"❌ FAILED: {m} - {e}")
