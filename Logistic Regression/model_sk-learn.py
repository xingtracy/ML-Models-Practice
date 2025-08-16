import dataset as ds
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = ds.prepare_dataset()

# Features & target
X = df.drop(columns=["Dialysis_Needed"])  # Features
y = df["Dialysis_Needed"]                 # Binary target (0/1)

# Train/Val/Test split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
val_pred = model.predict(X_val)
test_pred = model.predict(X_test)

print("\n=== Scikit-learn Logistic Regression ===")
print("Validation Accuracy:", accuracy_score(y_val, val_pred))
print("Test Accuracy:", accuracy_score(y_test, test_pred))
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Validation Accuracy: 1.0
# Test Accuracy: 0.9971098265895953
# Coefficients: [-4.98113398e-02  3.42161376e-01 -2.32652060e-02 -1.45489914e-01, 1.45500670e-01 -1.35441468e+00  6.40732527e-04 -1.15269564e-02]
# Intercept: [22.7765105]

