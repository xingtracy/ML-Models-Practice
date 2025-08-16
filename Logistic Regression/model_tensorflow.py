from tensorflow import keras
import dataset as ds
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Load dataset
df = ds.prepare_dataset()

# Features & target
X = df.drop(columns=["Dialysis_Needed"])  # Features
y = df["Dialysis_Needed"]                 # Binary target (0/1)

# Train/Val/Test split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Convert to TensorFlow tensors
X_train_tf = tf.convert_to_tensor(X_train.values, dtype=tf.float32)
y_train_tf = tf.convert_to_tensor(y_train.values, dtype=tf.float32)
X_val_tf = tf.convert_to_tensor(X_val.values, dtype=tf.float32)
y_val_tf = tf.convert_to_tensor(y_val.values, dtype=tf.float32)
X_test_tf = tf.convert_to_tensor(X_test.values, dtype=tf.float32)
y_test_tf = tf.convert_to_tensor(y_test.values, dtype=tf.float32)

# Model
model = keras.Sequential([
    keras.layers.Input(shape=(X_train.shape[1],)),
    keras.layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train_tf, y_train_tf, validation_data=(X_val_tf, y_val_tf), epochs=50, batch_size=16, verbose=0)

# Evaluate
val_loss, val_acc = model.evaluate(X_val_tf, y_val_tf, verbose=0)
test_loss, test_acc = model.evaluate(X_test_tf, y_test_tf, verbose=0)


print("\n=== TensorFlow Logistic Regression ===")
print("Validation Accuracy:", val_acc)
print("Test Accuracy:", test_acc)

# === TensorFlow Logistic Regression ===
# Validation Accuracy: 0.9942196607589722
# Test Accuracy: 0.9942196607589722