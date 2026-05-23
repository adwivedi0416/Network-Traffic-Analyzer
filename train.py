"""
train.py — Train and evaluate the Network Traffic Anomaly Detector
Run: python train.py
Run with real data: python train.py --data path/to/kddcup.data
"""

import argparse
from src.pipeline import NetworkTrafficAnalyzer
from sklearn.model_selection import train_test_split


def main(data_path: str = None):
    analyzer = NetworkTrafficAnalyzer(model_dir="models")

    # Load data
    if data_path:
        df = analyzer.load_data(data_path)
    else:
        print("[!] No dataset path provided — using synthetic data for demo.")
        print("    Download NSL-KDD: https://www.unb.ca/cic/datasets/nsl.html\n")
        df = analyzer.generate_synthetic_data(n_samples=50000)

    # Preprocess
    X, y_binary, y_category = analyzer.preprocess(df, fit=True)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )
    print(f"\n[✓] Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"[✓] Attack rate in train: {y_train.mean()*100:.1f}%\n")

    # Train
    analyzer.train(X_train, y_train)

    # Evaluate
    results = analyzer.evaluate(X_test, y_test)

    # Feature importance
    feature_cols = [c for c in df.columns if c not in ["label", "difficulty"]]
    analyzer.feature_importance(feature_cols, top_n=15)

    # Save
    analyzer.save()

    print("\n" + "="*50)
    print("  TRAINING COMPLETE")
    print("="*50)
    for name, res in results.items():
        print(f"  {name:<20} Acc: {res['accuracy']}%  F1: {res['f1_score']}%  AUC: {res['roc_auc']}")
    print("\nRun 'python app.py' to start the REST API server.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Network Traffic Anomaly Detector")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to NSL-KDD dataset CSV (optional)")
    args = parser.parse_args()
    main(data_path=args.data)
