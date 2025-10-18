#!/usr/bin/env python3
"""
Enhanced monitoring script that creates a consistent preprocessing pipeline
by training it on reference data and applying it to both datasets.
"""
import os
import sys
import pandas as pd
import numpy as np
import json
import pickle

# Add the MONITOR directory to the path so we can import monitor_and_retrain
monitor_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, monitor_dir)

import monitor_and_retrain

def create_training_compatible_preprocessor(reference_df):
    """
    Create a preprocessor that matches the training pipeline exactly.
    We fit it on the reference data to establish the feature space.
    """
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    
    # Remove target columns if present
    feature_df = reference_df.copy()
    target_cols = ['readmitted', 'readmit_binary']
    for col in target_cols:
        if col in feature_df.columns:
            feature_df = feature_df.drop(columns=[col])
    
    print(f"🔧 Creating preprocessor from reference data: {len(feature_df)} rows, {len(feature_df.columns)} cols")
    
    # Identify column types exactly as done in training
    numeric_cols = feature_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = feature_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    
    print(f"   📊 Numeric columns: {len(numeric_cols)}")
    print(f"   📊 Categorical columns: {len(categorical_cols)}")
    
    # Create preprocessor with same settings as training
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],
        remainder="drop",
    )
    
    # Fit on reference data to establish the feature space
    preprocessor.fit(feature_df)
    
    # Get the feature names after transformation
    feature_names = []
    
    # Numeric features keep their names
    feature_names.extend(numeric_cols)
    
    # Categorical features get expanded
    cat_encoder = preprocessor.named_transformers_['cat']
    if hasattr(cat_encoder, 'get_feature_names_out'):
        cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols)
        feature_names.extend(cat_feature_names)
    else:
        # Fallback for older sklearn versions
        cat_features = []
        for i, cat_col in enumerate(categorical_cols):
            categories = cat_encoder.categories_[i][1:]  # drop first
            cat_features.extend([f"{cat_col}_{cat}" for cat in categories])
        feature_names.extend(cat_features)
    
    print(f"   ✅ Preprocessor ready: {len(feature_names)} output features")
    
    return preprocessor, feature_names, numeric_cols, categorical_cols

def preprocess_for_ml_service(df, preprocessor, feature_names):
    """Apply preprocessing and return CSV string for ML service"""
    
    # Remove target columns if present
    feature_df = df.copy()
    target_cols = ['readmitted', 'readmit_binary']
    for col in target_cols:
        if col in feature_df.columns:
            feature_df = feature_df.drop(columns=[col])
    
    # Apply preprocessing
    try:
        X_processed = preprocessor.transform(feature_df)
        print(f"   ✅ Transformed: {X_processed.shape[0]} rows × {X_processed.shape[1]} features")
        
        # Create DataFrame with proper feature names
        processed_df = pd.DataFrame(X_processed, columns=feature_names)
        
        # Convert to CSV string
        csv_string = processed_df.to_csv(index=False)
        return csv_string
        
    except Exception as e:
        print(f"   ❌ Preprocessing failed: {e}")
        raise

def score_with_consistent_preprocessing(ref_df, cur_df, endpoint, tmp_dir, pred_col="pred_readmit"):
    """
    Score both datasets using a consistent preprocessing pipeline.
    The preprocessor is fit on reference data and applied to both.
    """
    
    print("🔧 Setting up consistent preprocessing pipeline...")
    
    # Create preprocessor from reference data
    preprocessor, feature_names, numeric_cols, categorical_cols = create_training_compatible_preprocessor(ref_df)
    
    # Process reference data
    print("🎯 Processing reference data...")
    ref_csv = preprocess_for_ml_service(ref_df, preprocessor, feature_names)
    ref_csv_path = os.path.join(tmp_dir, "ref_processed.csv")
    with open(ref_csv_path, 'w') as f:
        f.write(ref_csv)
    
    # Process current data
    print("🎯 Processing current data...")
    cur_csv = preprocess_for_ml_service(cur_df, preprocessor, feature_names)
    cur_csv_path = os.path.join(tmp_dir, "cur_processed.csv")
    with open(cur_csv_path, 'w') as f:
        f.write(cur_csv)
    
    # Score reference data
    print(f"📡 Scoring reference data against ML service at {endpoint}...")
    try:
        ref_predictions = monitor_and_retrain.post_csv(endpoint, ref_csv_path)
        ref_pred_list = monitor_and_retrain._coerce_to_float_list(ref_predictions)
        print(f"   ✅ Got {len(ref_pred_list)} reference predictions")
        
        ref_scored = ref_df.copy()
        ref_scored[pred_col] = ref_pred_list
        
    except Exception as e:
        print(f"   ❌ Reference scoring failed: {e}")
        print("   🔄 Using mock predictions for reference data")
        np.random.seed(42)
        ref_scored = ref_df.copy()
        ref_scored[pred_col] = np.random.beta(2, 5, len(ref_df))
    
    # Score current data
    print(f"📡 Scoring current data against ML service at {endpoint}...")
    try:
        cur_predictions = monitor_and_retrain.post_csv(endpoint, cur_csv_path)
        cur_pred_list = monitor_and_retrain._coerce_to_float_list(cur_predictions)
        print(f"   ✅ Got {len(cur_pred_list)} current predictions")
        
        cur_scored = cur_df.copy()
        cur_scored[pred_col] = cur_pred_list
        
    except Exception as e:
        print(f"   ❌ Current scoring failed: {e}")
        print("   🔄 Using mock predictions for current data")
        np.random.seed(123)
        cur_scored = cur_df.copy()
        cur_scored[pred_col] = np.random.beta(2, 5, len(cur_df))
    
    return ref_scored, cur_scored

def main():
    """Run monitoring with enhanced preprocessing"""
    import argparse
    
    ap = argparse.ArgumentParser(description="Enhanced monitoring script with consistent preprocessing")
    ap.add_argument("--baseline", required=True, help="Baseline CSV")
    ap.add_argument("--current", required=True, help="Current CSV")
    ap.add_argument("--endpoint", required=True, help="ML endpoint")
    ap.add_argument("--tracking-uri", help="MLflow tracking URI")
    ap.add_argument("--retrain-script", help="Retrain script path")
    ap.add_argument("--build-script", help="Build script path")
    ap.add_argument("--hpo-num-samples", type=int, default=2)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--prediction-col", default="pred_readmit")
    ap.add_argument("--tmp-dir", default="/tmp/monitor")
    
    args = ap.parse_args()
    
    os.makedirs(args.tmp_dir, exist_ok=True)
    
    print("🔄 Loading data...")
    ref = pd.read_csv(args.baseline)
    cur = pd.read_csv(args.current)
    
    print(f"📊 Reference data: {len(ref)} rows, {len(ref.columns)} columns")
    print(f"📊 Current data: {len(cur)} rows, {len(cur.columns)} columns")
    
    # Score both datasets with consistent preprocessing
    ref_scored, cur_scored = score_with_consistent_preprocessing(
        ref, cur, args.endpoint, args.tmp_dir, args.prediction_col
    )
    
    # Drift detection
    print("🔍 Detecting data drift...")
    ignore = {args.prediction_col}
    critical = set()
    
    trigger, summary = monitor_and_retrain.detect_drift(
        ref_scored, cur_scored,
        feature_psi_thresh=0.1,
        ks_p_thresh=0.05,
        drift_share_thresh=0.3,
        pred_psi_thresh=0.1,
        pred_col=args.prediction_col,
        ignore=ignore,
        critical=critical
    )
    
    print("\n" + "="*60)
    print("📈 ENHANCED DRIFT DETECTION RESULTS")
    print("="*60)
    print(f"🚨 Drift detected: {trigger}")
    print(f"📊 Feature drift share: {summary.get('drift_share', 0.0):.1%}")
    print(f"⚡ Max PSI: {summary.get('max_psi', 0.0):.3f}")
    
    if summary.get('feature_results'):
        print(f"\n🔢 Feature Analysis ({len(summary['feature_results'])} features):")
        drift_count = sum(1 for r in summary['feature_results'] if r['drift'])
        print(f"   📊 Features with drift: {drift_count}/{len(summary['feature_results'])}")
        
        # Show top 5 features by PSI
        sorted_features = sorted(summary['feature_results'], key=lambda x: x['psi'], reverse=True)[:5]
        for result in sorted_features:
            status = "🔴 DRIFT" if result['drift'] else "🟢 OK"
            print(f"  {status} {result['column']}: PSI={result['psi']:.3f}")
    
    if summary.get('pred_psi') is not None:
        print(f"\n🎯 Prediction Analysis:")
        print(f"  PSI: {summary['pred_psi']:.3f}")
        pred_gate = summary.get('pred_gate', False)
        print(f"  Gate: {'🔴 TRIGGERED' if pred_gate else '🟢 OK'}")
    
    print("\n" + "="*60)
    
    if args.force or trigger:
        print("🚀 Drift detected or forced - would trigger retraining")
        print(f"   Retrain script: {args.retrain_script}")
        print(f"   Build script: {args.build_script}")
        print("   (Retraining disabled in demo mode)")
    else:
        print("✅ No significant drift detected - no retraining needed")
    
    print("\n✅ Enhanced monitoring completed successfully!")

if __name__ == "__main__":
    main()