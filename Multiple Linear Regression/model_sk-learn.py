import dataset as ds
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load dataset
df = ds.prepare_dataset()

# Prepare features and target
X = df.drop(columns=["Performance Index"])  # Drop target column
y = df["Performance Index"]

# Train-test split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
# Test-Validation split
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print("\n=== Scikit-learn Linear Regression ===")

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
val_pred = model.predict(X_val)
test_pred = model.predict(X_test)

# Evaluation
print(f"Validation MSE: {mean_squared_error(y_val, val_pred):.2f}")
print(f"Test MSE: {mean_squared_error(y_test, test_pred):.2f}")
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Validation MSE: 4.07
# Test MSE: 4.07
# Intercept: -33.83654380714247
# Coefficients: [2.85895223 1.01551979 0.58171285 0.47967676 0.19039416]




