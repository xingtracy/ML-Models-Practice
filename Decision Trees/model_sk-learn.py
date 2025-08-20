import numpy as np
import pandas as pd
import dataset as ds
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

df = ds.prepare_dataset()  

# ---------- Target handling ----------
TARGET_COL = df.columns[-1]

feature_cols = [c for c in df.columns if c != TARGET_COL]
X_full = pd.get_dummies(df[feature_cols], drop_first=False)  # numeric only
y_full = df[TARGET_COL]

X_train, X_temp, y_train, y_temp = train_test_split(
    X_full, y_full, test_size=0.30, random_state=42, stratify=y_full
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# ---------- Label encoding for sklearn / PyTorch ----------

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_val_enc   = le.transform(y_val)
y_test_enc  = le.transform(y_test)

print("\n=== scikit-learn: DecisionTreeClassifier ===")
sk_tree = DecisionTreeClassifier(
    criterion="gini",      # or "entropy", "log_loss"
    max_depth=None,        # tune as needed
    min_samples_split=2,
    random_state=42
)

sk_tree.fit(X_train, y_train_enc)

val_pred_sk  = sk_tree.predict(X_val)
test_pred_sk = sk_tree.predict(X_test)

print("Validation Accuracy:", accuracy_score(y_val_enc, val_pred_sk))
print("Test Accuracy:", accuracy_score(y_test_enc, test_pred_sk))
print("Classification report (test):\n", classification_report(y_test_enc, test_pred_sk, target_names=le.classes_))

# Validation Accuracy: 0.873015873015873
# Test Accuracy: 0.889763779527559
# Classification report (test):
#                precision    recall  f1-score   support

#     Distress       0.80      0.80      0.80         5
#     Eustress       0.94      0.94      0.94       115
#    No Stress       0.14      0.14      0.14         7

#     accuracy                           0.89       127
#    macro avg       0.63      0.63      0.63       127
# weighted avg       0.89      0.89      0.89       127
