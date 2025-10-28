
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd


def make_models(n_classes: int):
    # Fresh instances per call
    models = [
        ('RF', RandomForestClassifier(random_state=42)),
        ('DT', DecisionTreeClassifier(random_state=42)),
        ('LR', LogisticRegression(random_state=42, max_iter=1000, multi_class='auto')),
        ('SVM', SVC(random_state=42))
    ]
    # Configure XGB based on number of classes
    if n_classes == 2:
        xgb = XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42
        )
    else:
        xgb = XGBClassifier(
            objective='multi:softprob',
            num_class=n_classes,
            eval_metric='mlogloss',
            random_state=42
        )
    models.append(('XGBoost', xgb))
    return models

def plot_cm(y_true, y_pred, labels=None, normalize=False, title="", show_plot=True, save_path=None):
    if normalize:
        cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
        fmt = ".2f"
    else:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fmt = "d"

    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=labels if labels is not None else sorted(pd.Series(y_true).unique()),
                yticklabels=labels if labels is not None else sorted(pd.Series(y_true).unique()))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    if show_plot:
        plt.show()
    else:
        plt.close()