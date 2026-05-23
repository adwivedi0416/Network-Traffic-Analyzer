"""
Network Traffic Anomaly Detection Pipeline
==========================================
Detects malicious network traffic patterns using ML classifiers.
Supports: Random Forest, XGBoost, Gradient Boosting
Dataset: NSL-KDD / KDD Cup 99 format
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, accuracy_score
)
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib
import json
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Feature columns (NSL-KDD format)
# ─────────────────────────────────────────────
COLUMNS = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment",
    "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty"
]

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]

ATTACK_CATEGORIES = {
    "normal": "normal",
    "back": "DoS", "land": "DoS", "neptune": "DoS",
    "pod": "DoS", "smurf": "DoS", "teardrop": "DoS",
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe", "satan": "Probe",
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L",
    "multihop": "R2L", "phf": "R2L", "spy": "R2L", "warezclient": "R2L",
    "warezmaster": "R2L", "buffer_overflow": "U2R", "loadmodule": "U2R",
    "perl": "U2R", "rootkit": "U2R"
}


class NetworkTrafficAnalyzer:
    """
    End-to-end ML pipeline for network intrusion detection.
    Trains and evaluates Random Forest and XGBoost classifiers.
    """

    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.encoders = {}
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}

    # ─────────────────────────────────────────────
    # Data loading & preprocessing
    # ─────────────────────────────────────────────

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load NSL-KDD format dataset."""
        df = pd.read_csv(filepath, header=None, names=COLUMNS)
        print(f"[✓] Loaded {len(df):,} records from {filepath}")
        return df

    def generate_synthetic_data(self, n_samples: int = 50000) -> pd.DataFrame:
        """
        Generate realistic synthetic network traffic data for demo purposes.
        Mirrors the statistical distribution of NSL-KDD dataset.
        """
        np.random.seed(42)
        n = n_samples

        labels = np.random.choice(
            ["normal", "neptune", "smurf", "ipsweep", "satan", "portsweep",
             "back", "guess_passwd", "buffer_overflow", "rootkit"],
            size=n,
            p=[0.53, 0.15, 0.10, 0.07, 0.05, 0.04, 0.03, 0.01, 0.01, 0.01]
        )

        df = pd.DataFrame({
            "duration":             np.where(labels == "normal",
                                             np.random.exponential(10, n),
                                             np.random.exponential(2, n)).astype(int),
            "protocol_type":        np.random.choice(["tcp", "udp", "icmp"], n, p=[0.6, 0.3, 0.1]),
            "service":              np.random.choice(
                                        ["http", "ftp", "smtp", "ssh", "dns", "other"], n,
                                        p=[0.4, 0.15, 0.1, 0.1, 0.1, 0.15]),
            "flag":                 np.random.choice(
                                        ["SF", "S0", "REJ", "RSTO", "SH"], n,
                                        p=[0.7, 0.1, 0.1, 0.05, 0.05]),
            "src_bytes":            np.where(labels == "normal",
                                             np.random.randint(100, 50000, n),
                                             np.random.randint(0, 1000, n)),
            "dst_bytes":            np.where(labels == "normal",
                                             np.random.randint(100, 50000, n),
                                             np.zeros(n, dtype=int)),
            "land":                 np.random.choice([0, 1], n, p=[0.99, 0.01]),
            "wrong_fragment":       np.random.choice([0, 1, 2, 3], n, p=[0.95, 0.02, 0.02, 0.01]),
            "urgent":               np.zeros(n, dtype=int),
            "hot":                  np.random.randint(0, 30, n),
            "num_failed_logins":    np.random.choice([0, 1, 2, 5], n, p=[0.93, 0.04, 0.02, 0.01]),
            "logged_in":            np.where(labels == "normal",
                                             np.random.choice([0, 1], n, p=[0.3, 0.7]),
                                             np.random.choice([0, 1], n, p=[0.7, 0.3])),
            "num_compromised":      np.random.randint(0, 10, n),
            "root_shell":           np.where(np.isin(labels, ["rootkit", "buffer_overflow"]),
                                             np.random.choice([0, 1], n, p=[0.3, 0.7]),
                                             np.zeros(n, dtype=int)),
            "su_attempted":         np.zeros(n, dtype=int),
            "num_root":             np.random.randint(0, 5, n),
            "num_file_creations":   np.random.randint(0, 10, n),
            "num_shells":           np.zeros(n, dtype=int),
            "num_access_files":     np.random.randint(0, 10, n),
            "num_outbound_cmds":    np.zeros(n, dtype=int),
            "is_host_login":        np.zeros(n, dtype=int),
            "is_guest_login":       np.random.choice([0, 1], n, p=[0.97, 0.03]),
            "count":                np.where(np.isin(labels, ["neptune", "smurf"]),
                                             np.random.randint(200, 512, n),
                                             np.random.randint(1, 100, n)),
            "srv_count":            np.random.randint(1, 512, n),
            "serror_rate":          np.where(np.isin(labels, ["neptune", "back"]),
                                             np.random.uniform(0.7, 1.0, n),
                                             np.random.uniform(0.0, 0.1, n)),
            "srv_serror_rate":      np.random.uniform(0.0, 1.0, n),
            "rerror_rate":          np.random.uniform(0.0, 0.5, n),
            "srv_rerror_rate":      np.random.uniform(0.0, 0.5, n),
            "same_srv_rate":        np.where(labels == "normal",
                                             np.random.uniform(0.6, 1.0, n),
                                             np.random.uniform(0.0, 0.5, n)),
            "diff_srv_rate":        np.random.uniform(0.0, 1.0, n),
            "srv_diff_host_rate":   np.random.uniform(0.0, 1.0, n),
            "dst_host_count":       np.random.randint(1, 256, n),
            "dst_host_srv_count":   np.random.randint(1, 256, n),
            "dst_host_same_srv_rate":       np.random.uniform(0.0, 1.0, n),
            "dst_host_diff_srv_rate":       np.random.uniform(0.0, 1.0, n),
            "dst_host_same_src_port_rate":  np.random.uniform(0.0, 1.0, n),
            "dst_host_srv_diff_host_rate":  np.random.uniform(0.0, 1.0, n),
            "dst_host_serror_rate":         np.random.uniform(0.0, 1.0, n),
            "dst_host_srv_serror_rate":     np.random.uniform(0.0, 1.0, n),
            "dst_host_rerror_rate":         np.random.uniform(0.0, 1.0, n),
            "dst_host_srv_rerror_rate":     np.random.uniform(0.0, 1.0, n),
            "label":    labels,
            "difficulty": np.random.randint(1, 21, n)
        })

        print(f"[✓] Generated {n:,} synthetic traffic records")
        return df

    def preprocess(self, df: pd.DataFrame, fit: bool = True) -> tuple:
        """Encode categoricals, scale numerics, map binary labels."""
        df = df.copy()
        df.drop(columns=["difficulty"], inplace=True, errors="ignore")

        # Map labels to binary (normal=0, attack=1)
        y_binary = (df["label"] != "normal").astype(int)
        y_category = df["label"].map(
            lambda x: ATTACK_CATEGORIES.get(x, "Other")
        )

        df.drop(columns=["label"], inplace=True)

        # Encode categoricals
        for col in CATEGORICAL_COLS:
            if col in df.columns:
                if fit:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.encoders[col] = le
                else:
                    le = self.encoders[col]
                    df[col] = df[col].astype(str).map(
                        lambda x: le.transform([x])[0]
                        if x in le.classes_ else -1
                    )

        X = df.values.astype(float)

        if fit:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)

        return X, y_binary.values, y_category.values

    # ─────────────────────────────────────────────
    # Model training
    # ─────────────────────────────────────────────

    def train(self, X_train, y_train):
        """Train Random Forest and XGBoost classifiers."""

        print("\n[→] Training Random Forest ...")
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        )
        rf.fit(X_train, y_train)
        self.models["random_forest"] = rf
        print("[✓] Random Forest trained")

        print("[→] Training XGBoost ...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train)],
            verbose=False
        )
        self.models["xgboost"] = xgb_model
        print("[✓] XGBoost trained")

    # ─────────────────────────────────────────────
    # Evaluation
    # ─────────────────────────────────────────────

    def evaluate(self, X_test, y_test) -> dict:
        """Evaluate all trained models and return results."""
        results = {}

        for name, model in self.models.items():
            y_pred  = model.predict(X_test)
            y_prob  = model.predict_proba(X_test)[:, 1]

            acc     = accuracy_score(y_test, y_pred)
            f1      = f1_score(y_test, y_pred, average="weighted")
            roc_auc = roc_auc_score(y_test, y_prob)
            report  = classification_report(
                y_test, y_pred,
                target_names=["Normal", "Attack"],
                output_dict=True
            )

            results[name] = {
                "accuracy":  round(acc * 100, 2),
                "f1_score":  round(f1 * 100, 2),
                "roc_auc":   round(roc_auc, 4),
                "report":    report
            }

            print(f"\n{'='*50}")
            print(f"  {name.upper()} RESULTS")
            print(f"{'='*50}")
            print(f"  Accuracy : {acc*100:.2f}%")
            print(f"  F1 Score : {f1*100:.2f}%")
            print(f"  ROC-AUC  : {roc_auc:.4f}")
            print(f"\n{classification_report(y_test, y_pred, target_names=['Normal','Attack'])}")

        self.results = results
        return results

    # ─────────────────────────────────────────────
    # Feature importance
    # ─────────────────────────────────────────────

    def feature_importance(self, feature_names: list, top_n: int = 15) -> pd.DataFrame:
        """Extract top features from Random Forest model."""
        rf = self.models.get("random_forest")
        if rf is None:
            raise ValueError("Train the model first.")

        importances = rf.feature_importances_
        df_imp = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False).head(top_n)

        print(f"\n[→] Top {top_n} Features:")
        for _, row in df_imp.iterrows():
            bar = "█" * int(row["importance"] * 300)
            print(f"  {row['feature']:<35} {bar}  {row['importance']:.4f}")

        return df_imp

    # ─────────────────────────────────────────────
    # Save / load
    # ─────────────────────────────────────────────

    def save(self):
        """Save trained models and preprocessors."""
        for name, model in self.models.items():
            joblib.dump(model, f"{self.model_dir}/{name}.pkl")
        joblib.dump(self.scaler,   f"{self.model_dir}/scaler.pkl")
        joblib.dump(self.encoders, f"{self.model_dir}/encoders.pkl")
        with open(f"{self.model_dir}/results.json", "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\n[✓] Models saved to ./{self.model_dir}/")

    def load(self):
        """Load saved models and preprocessors."""
        self.models["random_forest"] = joblib.load(f"{self.model_dir}/random_forest.pkl")
        self.models["xgboost"]       = joblib.load(f"{self.model_dir}/xgboost.pkl")
        self.scaler   = joblib.load(f"{self.model_dir}/scaler.pkl")
        self.encoders = joblib.load(f"{self.model_dir}/encoders.pkl")
        print("[✓] Models loaded")

    # ─────────────────────────────────────────────
    # Real-time prediction
    # ─────────────────────────────────────────────

    def predict(self, raw_record: dict) -> dict:
        """
        Predict whether a single network record is an attack.
        raw_record: dict with same keys as COLUMNS (excluding label/difficulty)
        """
        df = pd.DataFrame([raw_record])
        X, _, _ = self.preprocess(df, fit=False)

        rf_pred  = self.models["random_forest"].predict(X)[0]
        xgb_pred = self.models["xgboost"].predict(X)[0]
        rf_prob  = self.models["random_forest"].predict_proba(X)[0][1]
        xgb_prob = self.models["xgboost"].predict_proba(X)[0][1]

        ensemble_prob = (rf_prob + xgb_prob) / 2
        ensemble_pred = int(ensemble_prob >= 0.5)

        return {
            "prediction":    "ATTACK" if ensemble_pred else "NORMAL",
            "confidence":    round(ensemble_prob * 100, 2),
            "rf_confidence":  round(rf_prob * 100, 2),
            "xgb_confidence": round(xgb_prob * 100, 2),
            "risk_level":    "HIGH" if ensemble_prob > 0.8
                             else "MEDIUM" if ensemble_prob > 0.5
                             else "LOW"
        }
