import torch
import torch.nn as nn
import dataset as ds
from sklearn.model_selection import train_test_split

df = ds.prepare_dataset()

print("\n=== Using Sklearn for Data Splitting ===")
# Prepare features and target
X = df.drop(columns=["Performance Index"])  # Drop target column
y = df["Performance Index"]

X = X.to_numpy()
y = y.to_numpy()

# Train-test split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
# Test-Validation split
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print("\n=== PyTorch Linear Regression ===")

# Convert to tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Model definition
torch_model = nn.Linear(X_train.shape[1], 1)

# CRITICAL: Use the exact same closed-form solution as scikit-learn
# Normal equation: β = (X^T X)^(-1) X^T y
print("Computing closed-form solution (same as scikit-learn)...")

# Add bias term (intercept)
X_train_with_bias = torch.cat([torch.ones(X_train_t.shape[0], 1), X_train_t], dim=1)

# Compute (X^T X)^(-1) X^T y
X_transpose = X_train_with_bias.T
X_transpose_X = X_transpose @ X_train_with_bias
X_transpose_y = X_transpose @ y_train_t

# Solve the system using torch.linalg.solve (more stable than inverse)
beta = torch.linalg.solve(X_transpose_X, X_transpose_y)

print(f"Closed-form solution computed!")

# Set the model parameters
with torch.no_grad():
    torch_model.bias.data = beta[0]
    torch_model.weight.data = beta[1:].T

# Final evaluation
torch_model.eval()
with torch.no_grad():
    val_mse = torch.mean((torch_model(X_val_t) - y_val_t) ** 2).item()
    test_mse = torch.mean((torch_model(X_test_t) - y_test_t) ** 2).item()

# Print model parameters for comparison
print(f"\nModel Parameters:")
print(f"Validation MSE: {val_mse:.2f}")
print(f"Test MSE: {test_mse:.2f}")
print("Intercept:", torch_model.bias.item())
print("Coefficients:", torch_model.weight.data.numpy().flatten())



