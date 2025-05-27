import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import mlflow

def preprocessing_obesity(filepath, output_dir):
    df = pd.read_csv(filepath)

    # Hapus outlier dengan metode IQR
    numeric_cols = ["Age", "Height", "Weight", "BMI", "PhysicalActivityLevel"]
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]

    # Encode fitur kategorikal
    label_encoder = LabelEncoder()
    df["Gender"] = label_encoder.fit_transform(df["Gender"])
    df["ObesityCategory"] = label_encoder.fit_transform(df["ObesityCategory"])

    # Standarisasi fitur numerik
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Split fitur dan target
    X = df.drop("ObesityCategory", axis=1)
    y = df["ObesityCategory"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Simpan hasil preprocessing
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

    return {
        "rows_clean": df.shape[0],
        "files": [
            os.path.join(output_dir, "X_train.csv"),
            os.path.join(output_dir, "X_test.csv"),
            os.path.join(output_dir, "y_train.csv"),
            os.path.join(output_dir, "y_test.csv"),
        ]
    }

if __name__ == "__main__":
    input_file = os.path.join("..", "obes_raw", "obesity_data.csv")
    output_dir = os.path.join(os.environ.get("GITHUB_WORKSPACE", "."), "preprocessing/output")

    print(f"Input file: {input_file}")
    print(f"Output dir: {output_dir}")

    mlruns_path = os.path.join(output_dir, "mlruns")
    os.makedirs(mlruns_path, exist_ok=True)
    mlflow.set_tracking_uri(f"file:{mlruns_path}")
    mlflow.set_experiment("Preprocessing_Obesity")

    with mlflow.start_run(run_name="Obesity_Preprocessing"):
        result = preprocessing_obesity(input_file, output_dir)

        mlflow.log_param("input_file", input_file)
        mlflow.log_param("output_dir", output_dir)
        mlflow.log_metric("rows_clean", result["rows_clean"])

        for file in result["files"]:
            mlflow.log_artifact(file)
