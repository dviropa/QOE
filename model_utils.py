import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# === 1) הכנת הדאטה (התאם לשדות שלך) ===
from rank import rank_qoe_3_classes, rank_str_to_int_mapping_3_classes

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

# אם כבר יש לך merged_df – השתמש בו; אחרת טען מקובץ
df = pd.read_csv("output/subset_window_1754898203.csv")  # שנה אם צריך
df = df.dropna(subset=FEATURE_COLUMNS + ['ping']).copy()
df['y'] = df['ping'].apply(lambda p: rank_str_to_int_mapping_3_classes[rank_qoe_3_classes(p)])

# X = df[FEATURE_COLUMNS].values.astype('float32')
# y = df['y'].values.astype('int64')

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, stratify=y, random_state=42
# )
from sklearn.preprocessing import LabelEncoder

# ... אחרי שיצרת df['y'] ...
X = df[FEATURE_COLUMNS].values.astype('float32')

le = LabelEncoder()
y = le.fit_transform(df['y'].values)   # יהפוך {1,2,3} ל-{0,1,2}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
# === 2) המודלים (כפי שהגדרת) ===
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

def make_models(n_classes: int):
    models = [
        ('RF', RandomForestClassifier(random_state=42)),
        ('DT', DecisionTreeClassifier(random_state=42)),
        # LR ו-SVM נהנים מ־StandardScaler → נעטוף אותם ב־Pipeline
        ('LR', make_pipeline(StandardScaler(), LogisticRegression(random_state=42, max_iter=1000, multi_class='auto'))),
        ('SVM', make_pipeline(StandardScaler(), SVC(random_state=42)))  # אם צריך הסתברויות: SVC(probability=True)
    ]
    if n_classes == 2:
        xgb = XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42)
    else:
        xgb = XGBClassifier(objective='multi:softprob', num_class=n_classes, eval_metric='mlogloss', random_state=42)
    models.append(('XGBoost', xgb))
    return models

models = make_models(n_classes=len(np.unique(y_train)))

# === 3) ריצה, מדדים, ומטריצת בלבול ===
from sklearn.metrics import confusion_matrix
import seaborn as sns, matplotlib.pyplot as plt



for name, clf in models:
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='macro')
    print(f"{name}: Accuracy={acc:.3f} | Macro-F1={f1:.3f}")


