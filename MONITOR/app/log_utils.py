import os
import datetime as dt
import pandas as pd

LOG_DIR = os.getenv("PRED_LOG_DIR", "/logs")
os.makedirs(LOG_DIR, exist_ok=True)

def append_prediction_log(features_df: pd.DataFrame, y_pred, req_meta: dict | None = None):
    df = features_df.copy()
    df["__prediction__"] = y_pred
    df["__timestamp__"] = dt.datetime.utcnow().isoformat()
    if req_meta:
        for k, v in req_meta.items():
            df[f"__{k}__"] = v

    out = os.path.join(LOG_DIR, f"preds_{dt.date.today().isoformat()}.parquet")
    if os.path.exists(out):
        old = pd.read_parquet(out)
        df = pd.concat([old, df], ignore_index=True)
    df.to_parquet(out, index=False)
