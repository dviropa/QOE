import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import faiss  # pip install faiss-cpu
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
FEATURE_COLUMNS = [
    'src2dst_avg_pkt_size', 'src2dst_avg_pkt_size_mean', 'src2dst_avg_pkt_size_median',
    'src2dst_avg_pkt_size_std', 'src2dst_avg_pkt_size_min', 'src2dst_avg_pkt_size_max',
    'src2dst_pps', 'src2dst_pps_mean', 'src2dst_pps_median',
    'src2dst_pps_std', 'src2dst_pps_min', 'src2dst_pps_max',
    'dst2src_avg_pkt_size', 'dst2src_avg_pkt_size_mean', 'dst2src_avg_pkt_size_median',
    'dst2src_avg_pkt_size_std', 'dst2src_avg_pkt_size_min', 'dst2src_avg_pkt_size_max',
    'dst2src_pps', 'dst2src_pps_mean', 'dst2src_pps_median',
    'dst2src_pps_std', 'dst2src_pps_min', 'dst2src_pps_max',
    'pkt_dir_ratio', 'pkt_dir_ratio_mean', 'pkt_dir_ratio_median',
    'pkt_dir_ratio_std', 'pkt_dir_ratio_min', 'pkt_dir_ratio_max'
]
# def vectordb5filde(df):
#         from rank import rank_qoe_3_classes, rank_str_to_int_mapping_3_classes
#         df = df.dropna(subset=FEATURE_COLUMNS + ['ping']).copy()
#         df['y_qoe_by_ping'] = df['ping'].apply(lambda p: rank_str_to_int_mapping_3_classes[rank_qoe_3_classes(p)])



#         # rain/Test
#         a=0
#         f=0
#         for i in range(0,5):
#             df = df.sample(frac=1, random_state=42).reset_index(drop=True)
#             X = df[FEATURE_COLUMNS].values.astype('float32')
#             y = df['y_qoe_by_ping'].values.astype('int64')

#             X_train, X_test, y_train, y_test = train_test_split(
#                 X, y, test_size=0.2, random_state=42, stratify=y
#             )
#             acc,f1=vectordb(X_train, X_test, y_train, y_test)
#             a+=acc
#             f+=f1
#             print(f"  FAISS-kNN → Accuracy: {acc:.3f} | Macro-F1: {f1:.3f}")
#         print(f" avg FAISS-kNN → Accuracy: {a/5:.3f} |avg Macro-F1: {f/5:.3f}")

from sklearn.model_selection import StratifiedKFold  # אפשר להחליף ל-KFold אם מתעקשים

def vectordb5filde(df):
    from rank import rank_qoe_3_classes, rank_str_to_int_mapping_3_classes
    df = df.dropna(subset=FEATURE_COLUMNS + ['ping']).copy()
    df['y_qoe_by_ping'] = df['ping'].apply(
        lambda p: rank_str_to_int_mapping_3_classes[rank_qoe_3_classes(p)]
    )

    X = df[FEATURE_COLUMNS].values.astype('float32')
    y = df['y_qoe_by_ping'].values.astype('int64')

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    accs, f1s = [], []
    for fold, (train_index, test_index) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]


        acc, f1 = vectordb(X_train, X_test, y_train, y_test)
        accs.append(acc); f1s.append(f1)
        print(f"Fold {fold}/5 → Accuracy: {acc:.3f} | Macro-F1: {f1:.3f}")

    print(f"AVG over 5 folds → Accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f} | "
          f"Macro-F1: {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")

def vectordb(X_train, X_test, y_train, y_test):
    try:



        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train).astype('float32')
        X_test_sc = scaler.transform(X_test).astype('float32')

        d = X_train_sc.shape[1]
        index = faiss.IndexFlatL2(d)  
        index.add(X_train_sc)  # מוסיפים את הווקטורים

        # 5) חיפוש k שכנים ו־Majority Vote
        k  = min(5, len(X_train_sc))
        D, I = index.search(X_test_sc, k)  # D=מרחקים, I=אינדקסים של השכנים (ב-X_train_sc)

        def majority_vote(neighbor_indices):
            labels = y_train[neighbor_indices]
            vals, counts = np.unique(labels, return_counts=True)
            return vals[np.argmax(counts)]

        y_pred = np.array([majority_vote(I[i]) for i in range(len(X_test_sc))])

        # 6) הערכה
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        # print(f"FAISS-kNN → Accuracy: {acc:.3f} | Macro-F1: {f1:.3f}")
        return acc,f1
    except Exception as e:
        print(f"   שגיאה: {e}")





# df = pd.read_csv("output/window_10_1754904064.csv")

# t_df = df.dropna(subset=FEATURE_COLUMNS + ['ping']).copy()
# # print("dropna")
# vectordb5filde(df)


# # DATA_FILE = "output/window_15_1756035248.csv"
# # vectordb("output/subset_window_1754898203.csv")
# # vectordb("output/window_1_1756028067.csv")
# # vectordb("output/window_3_1756030271.csv")
# # vectordb("output/window_5_1756032959.csv")
# # vectordb("output/window_10_1754904064.csv")
# # vectordb("output/window_15_1756035248.csv")

# # files = [
# #     "output/subset_window_1754898203.csv","output/window_1_1756028067.csv",
# #     "output/window_3_1756030271.csv",
# #     "output/window_5_1756032959.csv","output/window_10_1754904064.csv","output/window_15_1756035248.csv"
# # ]
# # merged_df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
# # merged_df.to_csv("output/merged_selected.csv", index=False)


# # t_df = df.fillna(0).copy()
# # print("fillna")
# # vectordb(t_df)

# # t_df = df.copy()
# # imp = SimpleImputer(strategy="median")   # או "mean", או "most_frequent"
# # X = imp.fit_transform(t_df)
# # print("median")
# # vectordb(t_df)

























import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import faiss

# מניח ש-FEATURE_COLUMNS כבר מוגדר
from rank import rank_qoe_3_classes, rank_str_to_int_mapping_3_classes

def mini_bootstrap_ensemble(df, n_models=100, frac_per_model=1/100, k=5, seed=42):
    # --- הכנת X,y ---
    df = df.dropna(subset=FEATURE_COLUMNS + ['ping']).copy()
    df['y'] = df['ping'].apply(lambda p: rank_str_to_int_mapping_3_classes[rank_qoe_3_classes(p)])
    X = df[FEATURE_COLUMNS].values.astype('float32')
    y = df['y'].values.astype('int64')

    # --- חלוקה 80/20 ---
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)

    rng = np.random.default_rng(seed)
    shard_size = max(1, int(len(Xtr) * frac_per_model))  # ~1/100 מה-train

    # --- פונקציות עזר קצרות ---
    def fit_faiss(X_train_s, y_train_s):
        sc = StandardScaler()
        Xs = sc.fit_transform(X_train_s).astype('float32')
        idx = faiss.IndexFlatL2(Xs.shape[1])
        idx.add(Xs)
        return sc, idx, y_train_s

    def predict_faiss(model, X_test):
        sc, idx, y_train_s = model
        Xt = sc.transform(X_test).astype('float32')
        kk = min(k, len(y_train_s))
        D, I = idx.search(Xt, kk)
        # majority vote לשורת אינדקסים
        preds = np.empty(Xt.shape[0], dtype=y_train_s.dtype)
        for i, row in enumerate(I):
            labs = y_train_s[row]
            vals, cnts = np.unique(labs, return_counts=True)
            preds[i] = vals[np.argmax(cnts)]
        return preds

    # --- אימון 100 מודלים רנדומליים (Bootstrap קטן) + חיזוי על ה-20% ---
    per_model_preds = np.empty((n_models, len(Xte)), dtype=yte.dtype)
    for m in range(n_models):
        idx = rng.choice(len(Xtr), size=shard_size, replace=True)  # עם החזרה
        model = fit_faiss(Xtr[idx], ytr[idx])
        per_model_preds[m] = predict_faiss(model, Xte)

    # --- רוב על פני 100 המודלים לכל דגימת טסט ---
    y_pred = np.empty(len(Xte), dtype=yte.dtype)
    for j in range(per_model_preds.shape[1]):
        col = per_model_preds[:, j]
        vals, cnts = np.unique(col, return_counts=True)
        y_pred[j] = vals[np.argmax(cnts)]

    # --- מדדים ---
    acc = accuracy_score(yte, y_pred)
    f1  = f1_score(yte, y_pred, average='macro')
    print(f"Mini-Bagging (n={n_models}, shard≈{shard_size}) → Acc: {acc:.3f} | Macro-F1: {f1:.3f}")

    # מחזיר גם את מטריצת התחזיות אם תרצה לנתח
    return acc, f1, yte, y_pred, per_model_preds

# דוגמת הרצה:
df = pd.read_csv("output/window_10_1754904064.csv")
acc, f1, y_true, y_hat, preds_mat = mini_bootstrap_ensemble(df, n_models=100, frac_per_model=0.3, k=5)
