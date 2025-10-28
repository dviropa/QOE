# agent.py

import argparse, json, time, os, numpy as np, pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer, KNNImputer
import faiss

# נסיון לייבא מודלים חזקים—מדלגים אם לא מותקן
HAS_XGB, HAS_CAT = False, False
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    pass
try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except Exception:
    pass

FEATURES = [
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

# ---------------------- ניקוי / עזר ----------------------
def sanitize_numeric(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df[cols] = df[cols].replace([np.inf, -np.inf], np.nan)
    return df

def drop_dead_cols(df, cols, thr=0.95):
    miss = df[cols].isna().mean()
    keep = [c for c in cols if miss.get(c, 0) <= thr]
    return df, keep

def _split_valid_empty_cols(t: pd.DataFrame, cols):
    valid = [c for c in cols if t[c].notna().any()]
    empty = [c for c in cols if not t[c].notna().any()]
    return valid, empty

def impute_median(df, cols):
    t = df.copy()
    valid, empty = _split_valid_empty_cols(t, cols)
    if valid:
        imp = SimpleImputer(strategy="median")
        t[valid] = imp.fit_transform(t[valid])
    for c in empty:
        t[c] = 0.0
    return t, valid + empty

def impute_knn(df, cols, n_neighbors=5):
    t = df.copy()
    valid, empty = _split_valid_empty_cols(t, cols)
    if valid:
        imp = KNNImputer(n_neighbors=n_neighbors, weights="uniform")
        t[valid] = imp.fit_transform(t[valid])
    for c in empty:
        t[c] = 0.0
    return t, valid + empty

def build_labels_from_ping(df):
    # שימוש במיפוי שלך
    from rank import rank_qoe_3_classes, rank_str_to_int_mapping_3_classes
    return df['ping'].apply(lambda p: rank_str_to_int_mapping_3_classes[rank_qoe_3_classes(p)]).astype('int64').values

# ---------------------- מודלים ----------------------
def faiss_knn_predict(Xtr, ytr, Xte, k=5):
    index = faiss.IndexFlatL2(Xtr.shape[1])
    index.add(Xtr.astype('float32'))
    D, I = index.search(Xte.astype('float32'), k)
    ypred = []
    for nn in I:
        votes = ytr[nn]
        vals, cnts = np.unique(votes, return_counts=True)
        ypred.append(vals[np.argmax(cnts)])
    return np.array(ypred)

def evaluate_single_split(X, y, seed=42):
    # פיצול עקבי
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    # סקיילינג נדרש ל-FAISS/LR
    scaler = StandardScaler()
    Xtr_sc = scaler.fit_transform(Xtr).astype('float32')
    Xte_sc = scaler.transform(Xte).astype('float32')

    results = []

    # 1) FAISS-kNN
    ypred = faiss_knn_predict(Xtr_sc, ytr, Xte_sc, k=5)
    results.append(("faiss_knn", ypred))

    # 2) Logistic Regression
    lr = LogisticRegression(max_iter=1000)  # ללא n_jobs—לא תמיד רלוונטי
    lr.fit(Xtr_sc, ytr)
    results.append(("logreg", lr.predict(Xte_sc)))

    # 3) RandomForest
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed)
    rf.fit(Xtr, ytr)  # עצים לא צריכים סקיילינג
    results.append(("rf200", rf.predict(Xte)))

    # 4) XGBoost (אם קיים)
    if HAS_XGB:
        xgb = XGBClassifier(
            n_estimators=600, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            objective='multi:softmax', num_class=len(np.unique(y)),
            tree_method="hist", random_state=seed
        )
        xgb.fit(Xtr, ytr)
        results.append(("xgb", xgb.predict(Xte)))

    # 5) CatBoost (אם קיים)
    if HAS_CAT:
        cat = CatBoostClassifier(
            iterations=800, depth=6, learning_rate=0.06,
            l2_leaf_reg=3.0, loss_function='MultiClass',
            eval_metric='TotalF1', verbose=False, random_seed=seed
        )
        cat.fit(Xtr, ytr)
        results.append(("cat", cat.predict(Xte).astype(int)))

    # מדדים לכל מודל
    scores = []
    for name, ypred in results:
        acc = float(accuracy_score(yte, ypred))
        f1  = float(f1_score(yte, ypred, average='macro'))
        scores.append((name, acc, f1))

    # בוחר מנצח לפי Macro-F1
    best = max(scores, key=lambda x: x[2])
    # Confusion matrix של המנצח
    best_name = best[0]
    ypred_best = dict(results)[best_name]
    cm = confusion_matrix(yte, ypred_best)
    return scores, best_name, cm

# ---------------------- אסטרטגיות ניקוי ----------------------
def strategy_fillna0(df):
    t = sanitize_numeric(df.copy(), FEATURES)
    for c in FEATURES:
        t[c] = t[c].fillna(0)
    t = t.dropna(subset=['ping'])
    return t, FEATURES

def strategy_dropcols95_median(df):
    t = sanitize_numeric(df.copy(), FEATURES)
    t, keep = drop_dead_cols(t, FEATURES, thr=0.95)
    t, keep = impute_median(t, keep)
    t = t.dropna(subset=['ping'])
    return t, keep

def strategy_row70_median(df):
    t = sanitize_numeric(df.copy(), FEATURES)
    t = t.dropna(subset=['ping'])
    min_non_na = int(np.ceil(0.7 * len(FEATURES)))
    t = t.loc[t[FEATURES].notna().sum(axis=1) >= min_non_na].copy()
    t, keep = impute_median(t, FEATURES)
    return t, keep

def strategy_knn_imputer(df):
    t = sanitize_numeric(df.copy(), FEATURES)
    t, keep = impute_knn(t, FEATURES, n_neighbors=5)
    t = t.dropna(subset=['ping'])
    return t, keep

# ---------------------- הערכה לכל אסטרטגיה ----------------------
def evaluate_strategy(df, cols, run_dir, name):
    y = build_labels_from_ping(df)
    X = df[cols].values.astype('float32')

    if np.unique(y).size < 2 or len(y) < 50:
        return {"strategy": name, "rows": int(len(df)), "features": int(len(cols)),
                "best_model": None, "accuracy": None, "macro_f1": None, "note": "too_few_classes_or_rows"}

    scores, best_model, cm = evaluate_single_split(X, y, seed=42)
    # שמירת CM למודל המנצח
    pd.DataFrame(cm).to_csv(Path(run_dir)/f"cm_{name}_{best_model}.csv", index=False)

    # שמירה מפורטת לכל המודלים
    detail_rows = []
    for model_name, acc, f1 in scores:
        detail_rows.append({
            "strategy": name, "model": model_name,
            "rows": int(len(df)), "features": int(len(cols)),
            "accuracy": acc, "macro_f1": f1
        })
    return detail_rows, {"strategy": name, "best_model": best_model,
                         "rows": int(len(df)), "features": int(len(cols)),
                         "accuracy": max(scores, key=lambda x: x[1])[1],
                         "macro_f1": max(scores, key=lambda x: x[2])[2]}

def run_agent(csv_path):
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path("output")/f"agent_run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    print("rows in:", len(df))

    strategies = [
        ("fillna0", strategy_fillna0),
        ("dropcols95_med", strategy_dropcols95_median),
        ("row70_med", strategy_row70_median),
        ("knn_imputer", strategy_knn_imputer),
    ]

    all_details = []
    best_summaries = []

    for name, func in strategies:
        try:
            tdf, cols = func(df)
            res = evaluate_strategy(tdf, cols, run_dir, name)
            if isinstance(res, dict):  # מעט מדי דגימות/מחלקות
                best_summaries.append(res)
                print(f"{name:16s} → {res.get('note')}")
                continue

            detail_rows, best_summary = res
            all_details.extend(detail_rows)
            best_summaries.append(best_summary)
            # הדפסה קצרה
            print(f"{name:16s} → best={best_summary['best_model']}"
                  f" | rows={best_summary['rows']} feats={best_summary['features']}"
                  f" | Acc={best_summary['accuracy']:.3f} F1={best_summary['macro_f1']:.3f}")
        except Exception as e:
            print(f"{name:16s} → ❌ {e}")

    # שמירה
    if all_details:
        pd.DataFrame(all_details).to_csv(Path(run_dir)/"models_by_strategy.csv", index=False)
    pd.DataFrame(best_summaries).to_csv(Path(run_dir)/"summary_best_by_strategy.csv", index=False)
    print(f"✅ results → {Path(run_dir)/'models_by_strategy.csv'}")
    print(f"✅ summary → {Path(run_dir)/'summary_best_by_strategy.csv'}")

if __name__ == "__main__":
    # רץ ישירות על הקובץ שלך כפי שביקשת
    run_agent("output/merged_selected.csv")
