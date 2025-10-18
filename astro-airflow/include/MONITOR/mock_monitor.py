#!/usr/bin/env python3
"""
Mock monitoring script that bypasses ML service prediction calls
and demonstrates drift detection with synthetic predictions.
"""
import os
import sys
import pandas as pd
import numpy as np

# Add the MONITOR directory to the path so we can import monitor_and_retrain
monitor_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, monitor_dir)

import monitor_and_retrain

def add_mock_predictions(df, pred_col="pred_readmit", seed=42):
    """Add mock predictions to demonstrate monitoring"""
    np.random.seed(seed)
    # Generate realistic-looking predictions (0-1 probabilities)
    n = len(df)
    predictions = np.random.beta(2, 5, n)  # Beta distribution biased toward lower values
    df = df.copy()
    df[pred_col] = predictions
    return df

def main():
    """Run monitoring with mock predictions"""
    import argparse
    
    ap = argparse.ArgumentParser(description="Mock monitoring script")
    ap.add_argument("--baseline", required=True, help="Baseline CSV")
    ap.add_argument("--current", required=True, help="Current CSV")
    ap.add_argument("--endpoint", help="ML endpoint (ignored in mock)")
    ap.add_argument("--tracking-uri", help="MLflow tracking URI")
    ap.add_argument("--retrain-script", help="Retrain script path")
    ap.add_argument("--build-script", help="Build script path")
    ap.add_argument("--hpo-num-samples", type=int, default=2)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--prediction-col", default="pred_readmit")
    ap.add_argument("--tmp-dir", default="/tmp/monitor")
    
    args = ap.parse_args()
    
    print("🔄 Loading data...")
    ref = pd.read_csv(args.baseline)
    cur = pd.read_csv(args.current)
    
    print(f"📊 Reference data: {len(ref)} rows, {len(ref.columns)} columns")
    print(f"📊 Current data: {len(cur)} rows, {len(cur.columns)} columns")
    
    # Add mock predictions
    print("🎯 Adding mock predictions...")
    ref_scored = add_mock_predictions(ref, args.prediction_col, seed=42)
    cur_scored = add_mock_predictions(cur, args.prediction_col, seed=123)
    
    # Use the original drift detection logic
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
    print("📈 DRIFT DETECTION RESULTS")
    print("="*60)
    print(f"🚨 Drift detected: {trigger}")
    print(f"� Feature drift share: {summary.get('drift_share', 0.0):.1%}")
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
        print("   (Retraining disabled in mock mode)")
    else:
        print("✅ No significant drift detected - no retraining needed")
    
    print("\n✅ Monitoring completed successfully!")

if __name__ == "__main__":
    main()