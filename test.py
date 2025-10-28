# --- PyCharm scientific/datalore hotfix: מנטרל הזרקה שגורמת ל-reload של numpy ---
import sys, os
sys.path[:] = [p for p in sys.path if ("pycharm_display" not in p and "datalore" not in p)]
os.environ["PYCHARM_DISPLAY_PORT"] = "0"
# ------------------------------------------------------------------------------

import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import SimpleImputer, KNNImputer
import faiss


FEATURE_COLUMNS = [
    'src2dst_avg_pkt_size','src2dst_avg_pkt_size_mean','src2dst_avg_pkt_size_median',
    'src2dst_avg_pkt_size_std','src2dst_avg_pkt_size_min','src2dst_avg_pkt_size_max',
    'src2dst_pps','src2dst_pps_mean','src2dst_pps_median',
    'src2dst_pps_std','src2dst_pps_min','src2dst_pps_max',
    'dst2src_avg_pkt_size','dst2src_avg_pkt_size_mean','dst2src_avg_pkt_size_median',
    'dst2src_avg_pkt_size_std','dst2src_avg_pkt_size_min','dst2src_avg_pkt_size_max',
    'dst2src_pps','dst2src_pps_mean','dst2src_pps_median',
    'dst2src_pps_std','dst2src_pps_min','dst2src_pps_max',
    'pkt_dir_ratio','pkt_dir_ratio_mean','pkt_dir_ratio_median',
    'pkt_dir_ratio_std','pkt_dir_ratio_min','pkt_dir_ratio_max'
]

# ===== פונקציות עזר =====
def sanitize_numeric(df: pd.DataFrame, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[cols] = df[cols].replace([np.inf, -np.inf], np.nan)
    return df

def build_labels_from_ping(df: pd.DataFrame):
    from rank import rank_qoe_3_classes, rank_str_to_int_mapping_3_classes
    if 'ping' not in df.columns:
        raise ValueError("חסרה עמודת 'ping'.")
    y = df['ping'].apply(lambda p: rank_str_to_int_mapping_3_classes[rank_qoe_3_classes(p)]).astype('int64')
    return y

def evaluate_faiss_knn(X, y, k=5, seed=42):
    # ספליט עקבי לכל השיטות
    strat = y if len(np.unique(y)) > 1 else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=strat)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr).astype('float32')
    Xte = scaler.transform(Xte).astype('float32')

    dim = Xtr.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(Xtr)
    D, I = index.search(Xte, k)

    # Majority vote
    ypred = []
    for nn in I:
        votes = ytr[nn]
        vals, cnts = np.unique(votes, return_counts=True)
        ypred.append(vals[np.argmax(cnts)])
    ypred = np.array(ypred)

    return accuracy_score(yte, ypred), f1_score(yte, ypred, average='macro')

# ===== אסטרטגיות ניקוי =====
def strategy_dropna(df):
    """מוחק כל שורה עם NaN בפיצ'רים או ב-ping"""
    # t = df.copy()
    # # t = sanitize_numeric(t, FEATURE_COLUMNS)
    # # t = t.dropna(subset=FEATURE_COLUMNS + ['ping'])
    # return t

def strategy_fillna_zero(df):
    """מחליף חסרים ב-0 בפיצ'רים (אחרי ניקוי inf)"""
    t = df.copy()
    t = sanitize_numeric(t, FEATURE_COLUMNS)
    for c in FEATURE_COLUMNS:
        t[c] = t[c].fillna(0)
    return t.dropna(subset=['ping'])

def strategy_impute_median(df):
    """אימפיוטציה למדיאן רק על FEATURE_COLUMNS"""
    t = df.copy()
    t = sanitize_numeric(t, FEATURE_COLUMNS)
    imp = SimpleImputer(strategy="median")
    t[FEATURE_COLUMNS] = imp.fit_transform(t[FEATURE_COLUMNS])
    return t.dropna(subset=['ping'])

def strategy_impute_knn(df, n_neighbors=5):
    """KNNImputer לפיצ'רים"""
    t = df.copy()
    t = sanitize_numeric(t, FEATURE_COLUMNS)
    imp = KNNImputer(n_neighbors=n_neighbors, weights="uniform")
    t[FEATURE_COLUMNS] = imp.fit_transform(t[FEATURE_COLUMNS])
    return t.dropna(subset=['ping'])

def strategy_drop_cols_then_median(df, col_miss_thresh=0.95):
    """זורק עמודות עם >95% חסר, ואז אימפיוט למדיאן"""
    t = df.copy()
    t = sanitize_numeric(t, FEATURE_COLUMNS)
    miss = t[FEATURE_COLUMNS].isna().mean()
    keep_cols = [c for c in FEATURE_COLUMNS if miss.get(c, 0.0) <= col_miss_thresh]
    imp = SimpleImputer(strategy="median")
    t[keep_cols] = imp.fit_transform(t[keep_cols])
    # אם נשארו פחות פיצ'רים, נמיר רק את הקיימים
    return t.dropna(subset=['ping']), keep_cols

def strategy_row_thresh_then_median(df, row_non_na_ratio=0.7):
    """דורש בשורה >=70% פיצ'רים מלאים (אחרי sanitize), ואז אימפיוט למדיאן"""
    t = df.copy()
    t = sanitize_numeric(t, FEATURE_COLUMNS)
    min_non_na = int(np.ceil(row_non_na_ratio * len(FEATURE_COLUMNS)))
    # שומר רק שורות שעוברות את הסף
    t = t.dropna(subset=['ping'])  # ping חייב קיים לתווית
    mask = t[FEATURE_COLUMNS].notna().sum(axis=1) >= min_non_na
    t = t.loc[mask].copy()
    imp = SimpleImputer(strategy="median")
    t[FEATURE_COLUMNS] = imp.fit_transform(t[FEATURE_COLUMNS])
    return t

# ===== הרצה השוואתית =====
def run_ablation(df, results_path="output/cleaning_results.csv"):
    experiments = []

    def run_one(name, make_df, keep_cols=None):
        start = time.time()
        try:
            out = make_df(df)
            cols = FEATURE_COLUMNS
            if isinstance(out, tuple):
                tdf, cols = out
            else:
                tdf = out
            if len(tdf) < 100 or 'ping' not in tdf.columns:
                raise ValueError(f"Too few rows ({len(tdf)}) או ping חסר.")
            y = build_labels_from_ping(tdf).values
            X = tdf[cols].values.astype('float32')
            if len(np.unique(y)) < 2:
                raise ValueError("מחלקה אחת בלבד אחרי ניקוי.")
            acc, f1 = evaluate_faiss_knn(X, y)
            dur = time.time() - start
            experiments.append([name, len(tdf), len(cols), acc, f1, dur])
            print(f"{name:28s} → rows={len(tdf):6d} feats={len(cols):2d} | Acc={acc:.3f} F1={f1:.3f} | {dur:.2f}s")
        except Exception as e:
            dur = time.time() - start
            experiments.append([name, 0, 0, np.nan, np.nan, dur, str(e)])
            print(f"{name:28s} → ❌ {e}")

    print("\n=== ניסוי השפעת ניקוי על המודל (FAISS-kNN) ===")
    # run_one("dropna(strict)", strategy_dropna)
    run_one("fillna(0)+sanitize", strategy_fillna_zero)
    run_one("SimpleImputer(median)", strategy_impute_median)
    run_one("KNNImputer(k=5)", strategy_impute_knn)
    run_one("drop cols>95% + median", strategy_drop_cols_then_median)
    run_one("row>=70% + median", strategy_row_thresh_then_median)

    cols = ["strategy","rows","features","accuracy","macro_f1","seconds"]
    # בדיקה אם יש עמודת error
    if any(len(r) == 7 for r in experiments):
        cols.append("error")
    res = pd.DataFrame(experiments, columns=cols)
    Path(Path(results_path).parent).mkdir(parents=True, exist_ok=True)
    res.to_csv(results_path, index=False)
    print(f"\nנשמר: {results_path}")
    return res

# === שימוש לדוגמה ===
df = pd.read_csv("output/merged_selected.csv")
print(f"len(df) = {len(df)}")
results = run_ablation(df)
print(results)
