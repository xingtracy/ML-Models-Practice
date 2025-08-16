import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import dataset as ds

# Load dataset
df = ds.prepare_dataset()

# Features & target
X = df.drop(columns=["Dialysis_Needed"])  # Features
y = df["Dialysis_Needed"]                 # Binary target (0/1)

# Train/Val/Test split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Convert to tensors
X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_val_t = torch.tensor(X_val.values, dtype=torch.float32)
y_val_t = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
X_test_t = torch.tensor(X_test.values, dtype=torch.float32)
y_test_t = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

# Dataset & Loader
train_ds = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

# Logistic Regression model
class LogisticRegressionTorch(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

model = LogisticRegressionTorch(X_train.shape[1])
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    for xb, yb in train_loader:
        preds = model(xb)
        loss = criterion(preds, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Evaluate
with torch.no_grad():
    val_preds = model(X_val_t).round()
    acc_val = (val_preds.eq(y_val_t)).float().mean()
    
with torch.no_grad(): 
    test_preds = model(X_test_t).round()
    acc_test= (val_preds.eq(y_test_t)).float().mean()
    
print("\n=== PyTorch Logistic Regression ===")
print("Validation Accuracy:", acc_val.item())
print("Test Accuracy:", acc_test.item())


# === PyTorch Logistic Regression ===
# Validation Accuracy: 0.9797688126564026
# Test Accuracy: 0.9942196607589722
