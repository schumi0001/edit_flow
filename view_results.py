import os
import glob
import pandas as pd
import joblib

FEATURES_DIR = "data/lake/features"
MODEL_PATH = "models/anomaly_detector.joblib"

def main():
    # 1. Load your processed test data
    file_pattern = os.path.join(FEATURES_DIR, "**", "*.parquet")
    parquet_files = glob.glob(file_pattern, recursive=True)
    
    if not parquet_files:
        print("❌ Error: No parquet data found. Did you delete the features folder?")
        return
        
    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    df = df.drop_duplicates(subset=["page_title"]).dropna()

    # 2. Load your trained model brain
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file not found at {MODEL_PATH}")
        return
        
    pipeline = joblib.load(MODEL_PATH)

    # 3. Re-run predictions to view the log table
    feature_cols = ["edit_count", "unique_editors", "total_byte_changes", "bot_ratio", "minor_edit_ratio", "relative_growth", "human_bot_friction", "editor_concentration"]
    X = df[feature_cols]
    
    df["anomaly_score"] = pipeline.decision_function(X)
    df["is_anomaly"] = (pipeline.predict(X) == -1).astype(int)

    # 4. Print clean results to terminal
    print("\n" + "="*50)
    print("       🔍 ISOLATION FOREST RESULTS SUMMARY")
    print("="*50)
    print(df[["page_title", "edit_count", "total_byte_changes", "is_anomaly"]].to_string(index=False))
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
