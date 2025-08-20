import pandas as pd
import dataset as ds
from hummingbird.ml import convert
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_val_enc   = le.transform(y_val)
y_test_enc  = le.transform(y_test)

print("\n=== PyTorch (via hummingbird-ml conversion) ===")


torch_model = convert(sk_tree, backend="pytorch", extra_config={"tree_implementation": "gemm"})
# The converter expects numpy; outputs probabilities or logits depending on model.
# For classifiers, Hummingbird's predict_proba -> we argmax for class predictions.
val_proba_torch  = torch_model.predict_proba(X_val.to_numpy())
test_proba_torch = torch_model.predict_proba(X_test.to_numpy())

val_pred_torch  = np.argmax(val_proba_torch, axis=1)
test_pred_torch = np.argmax(test_proba_torch, axis=1)

print("Validation Accuracy:", accuracy_score(y_val_enc, val_pred_torch))
print("Test Accuracy:", accuracy_score(y_test_enc, test_pred_torch))