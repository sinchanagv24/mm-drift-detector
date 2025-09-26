import io, json, numpy as np, pandas as pd
from typing import Tuple, List, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score
from imblearn.over_sampling import RandomOverSampler, SMOTE

def split(df: pd.DataFrame, target: str, test_size: float, seed: int):
    X = df.drop(columns=[target])
    y = df[target].astype(int).values
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

def make_pipeline(cat_cols: list, num_cols: list, model="logreg", C=1.0, seed=42):
    preproc = ColumnTransformer(transformers=[
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
    ])
    if model == "logreg":
        clf = LogisticRegression(max_iter=200, C=C, random_state=seed)
    else:
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1, class_weight=None)
    pipe = Pipeline(steps=[("prep", preproc), ("clf", clf)])
    return pipe

def resample(X, y, method: str, seed: int):
    if method == "smote":
        sampler = SMOTE(random_state=seed)
    elif method == "ros":
        sampler = RandomOverSampler(random_state=seed)
    else:
        return X, y
    X_res, y_res = sampler.fit_resample(X, y)
    return X_res, y_res

def metrics_at_threshold(y_true, y_proba, thresh=0.5):
    y_pred = (y_proba >= thresh).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    auc = roc_auc_score(y_true, y_proba) if len(np.unique(y_true))==2 else np.nan
    return {"precision": pr, "recall": rc, "f1": f1, "auc": auc, "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}

def expected_cost(cm: dict, c_fp=1.0, c_fn=5.0):
    # Cost only for FP and FN by default; you can add rewards if needed.
    return c_fp*cm["fp"] + c_fn*cm["fn"]

def df_to_csv_bytes(df: pd.DataFrame):
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf
