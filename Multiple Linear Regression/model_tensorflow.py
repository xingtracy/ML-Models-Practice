import tensorflow as tf
import dataset as ds
import pandas as pd
from sklearn.model_selection import train_test_split

df=ds.prepare_dataset()

print("\n=== Using Sklearn for Data Splitting ===")
# Prepare features and target
X = df.drop(columns=["Performance Index"])  # Drop target column
y = df["Performance Index"]

# Train-test split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
# Test-Validation split
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


print("\n=== TensorFlow / Keras Linear Regression ===")
tf_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(1)
])

tf_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),loss='mse')

history = tf_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=500,
    verbose=0
)

# Evaluate
val_mse = tf_model.evaluate(X_val, y_val, verbose=0)
test_mse = tf_model.evaluate(X_test, y_test, verbose=0)

print(f"Validation MSE: {val_mse:.2f}")
print(f"Test MSE: {test_mse:.2f}")

# Validation MSE: 4.16
# Test MSE: 4.08