import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import faiss  # pip install faiss-cpu
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
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

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
RESET = "\033[0m"
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

    print(f"AVG over 5 folds →{GREEN} Accuracy:{RESET} {np.mean(accs):.3f} ± {np.std(accs):.3f} | "
          f"{RED}Macro-F1:{RESET} {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")

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





df = pd.read_csv("output/merged_selected.csv")

t_df = df.dropna(subset=FEATURE_COLUMNS + ['ping']).copy()
# print("dropna")
vectordb5filde(t_df)


# DATA_FILE = "output/window_15_1756035248.csv"
# vectordb("output/subset_window_1754898203.csv")
# vectordb("output/window_1_1756028067.csv")
# vectordb("output/window_3_1756030271.csv")
# vectordb("output/window_5_1756032959.csv")
# vectordb("output/window_10_1754904064.csv")
# vectordb("output/window_15_1756035248.csv")

# files = [
#     "output/subset_window_1754898203.csv","output/window_1_1756028067.csv",
#     "output/window_3_1756030271.csv",
#     "output/window_5_1756032959.csv","output/window_10_1754904064.csv","output/window_15_1756035248.csv"
# ]
# merged_df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
# merged_df.to_csv("output/merged_selected.csv", index=False)


# t_df = df.fillna(0).copy()
# print("fillna")
# vectordb(t_df)

# t_df = df.copy()
# imp = SimpleImputer(strategy="median")   # או "mean", או "most_frequent"
# X = imp.fit_transform(t_df)
# print("median")
# vectordb(t_df)









