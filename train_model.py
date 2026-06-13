"""
=============================================================
  Handwritten Digit Recognizer — CNN Model Training Script
  Dataset : MNIST (70,000 images, 28×28 grayscale)
  Framework: TensorFlow / Keras
  Run this once to generate model/digit_model.h5
=============================================================
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend (no display needed)
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix

# ── Directories ──────────────────────────────────────────────────────────────
MODEL_DIR  = "model"
STATIC_IMG = "static/images"
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(STATIC_IMG, exist_ok=True)

# ── Hyper-parameters ──────────────────────────────────────────────────────────
EPOCHS     = 15
BATCH_SIZE = 128
NUM_CLASSES = 10

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load & Pre-process MNIST
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Loading MNIST …")
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
X_test  = X_test .reshape(-1, 28, 28, 1).astype("float32") / 255.0

y_train_cat = to_categorical(y_train, NUM_CLASSES)
y_test_cat  = to_categorical(y_test,  NUM_CLASSES)

print(f"[INFO] Train: {X_train.shape}  Test: {X_test.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Build CNN Architecture
# ─────────────────────────────────────────────────────────────────────────────
model = Sequential([
    # ── Block 1 ──────────────────────────────────────────
    Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(28, 28, 1)),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    # ── Block 2 ──────────────────────────────────────────
    Conv2D(64, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    # ── Block 3 ──────────────────────────────────────────
    Conv2D(128, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    # ── Classifier ───────────────────────────────────────
    Flatten(),
    Dense(256, activation="relu"),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(NUM_CLASSES, activation="softmax"),
], name="DigitCNN")

model.summary()

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Callbacks
# ─────────────────────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
]

# ─────────────────────────────────────────────────────────────────────────────
# 4. Train
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Training …")
history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1,
)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Evaluate
# ─────────────────────────────────────────────────────────────────────────────
train_loss, train_acc = model.evaluate(X_train, y_train_cat, verbose=0)
test_loss,  test_acc  = model.evaluate(X_test,  y_test_cat,  verbose=0)
print(f"[INFO] Train Acc: {train_acc*100:.2f}%  Test Acc: {test_acc*100:.2f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Save Model
# ─────────────────────────────────────────────────────────────────────────────
model.save(os.path.join(MODEL_DIR, "digit_model.h5"))
print("[INFO] Model saved → model/digit_model.h5")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Save Training Stats as JSON  (consumed by Flask API)
# ─────────────────────────────────────────────────────────────────────────────
stats = {
    "train_accuracy": round(train_acc * 100, 2),
    "test_accuracy" : round(test_acc  * 100, 2),
    "train_loss"    : round(train_loss, 4),
    "test_loss"     : round(test_loss,  4),
    "epochs_run"    : len(history.history["loss"]),
    "total_params"  : model.count_params(),
    "acc_history"   : [round(v * 100, 2) for v in history.history["accuracy"]],
    "val_acc_history": [round(v * 100, 2) for v in history.history["val_accuracy"]],
    "loss_history"  : [round(v, 4) for v in history.history["loss"]],
    "val_loss_history": [round(v, 4) for v in history.history["val_loss"]],
}
with open(os.path.join(MODEL_DIR, "stats.json"), "w") as f:
    json.dump(stats, f, indent=2)
print("[INFO] Stats saved → model/stats.json")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Accuracy / Loss Graphs
# ─────────────────────────────────────────────────────────────────────────────
plt.style.use("dark_background")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#0f0f1a")

epochs_range = range(1, len(history.history["accuracy"]) + 1)

# Accuracy
axes[0].plot(epochs_range, history.history["accuracy"],     color="#00d4ff", lw=2.5, label="Train Acc")
axes[0].plot(epochs_range, history.history["val_accuracy"], color="#ff6b6b", lw=2.5, label="Val Acc", linestyle="--")
axes[0].set_title("Model Accuracy", fontsize=14, color="#fff", pad=12)
axes[0].set_xlabel("Epoch", color="#aaa"); axes[0].set_ylabel("Accuracy", color="#aaa")
axes[0].legend(facecolor="#1a1a2e", edgecolor="#444"); axes[0].set_facecolor("#1a1a2e")
axes[0].tick_params(colors="#aaa"); axes[0].grid(alpha=0.2)

# Loss
axes[1].plot(epochs_range, history.history["loss"],     color="#00d4ff", lw=2.5, label="Train Loss")
axes[1].plot(epochs_range, history.history["val_loss"], color="#ff6b6b", lw=2.5, label="Val Loss", linestyle="--")
axes[1].set_title("Model Loss", fontsize=14, color="#fff", pad=12)
axes[1].set_xlabel("Epoch", color="#aaa"); axes[1].set_ylabel("Loss", color="#aaa")
axes[1].legend(facecolor="#1a1a2e", edgecolor="#444"); axes[1].set_facecolor("#1a1a2e")
axes[1].tick_params(colors="#aaa"); axes[1].grid(alpha=0.2)

plt.tight_layout(pad=2)
plt.savefig(os.path.join(STATIC_IMG, "training_curves.png"), dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
plt.close()
print("[INFO] Training curves saved → static/images/training_curves.png")

# ─────────────────────────────────────────────────────────────────────────────
# 9. Confusion Matrix
# ─────────────────────────────────────────────────────────────────────────────
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
cm     = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor("#0f0f1a")
ax.set_facecolor("#0f0f1a")
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=range(10), yticklabels=range(10),
    linewidths=0.5, ax=ax,
    cbar_kws={"shrink": 0.8}
)
ax.set_title("Confusion Matrix — Test Set", fontsize=14, color="#fff", pad=12)
ax.set_xlabel("Predicted Label", color="#aaa"); ax.set_ylabel("True Label", color="#aaa")
ax.tick_params(colors="#aaa")
plt.tight_layout()
plt.savefig(os.path.join(STATIC_IMG, "confusion_matrix.png"), dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
plt.close()
print("[INFO] Confusion matrix saved → static/images/confusion_matrix.png")

# ─────────────────────────────────────────────────────────────────────────────
# 10. Sample Predictions Grid
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 10, figsize=(20, 5))
fig.patch.set_facecolor("#0f0f1a")
fig.suptitle("Sample Predictions (Test Set)", fontsize=14, color="#fff", y=1.02)

for digit in range(10):
    idx = np.where(y_test == digit)[0][0]
    for row, (img_arr, label_arr, title_color) in enumerate(
        [(X_test[idx].reshape(28, 28), y_pred[idx], "#00d4ff"),
         (X_test[idx].reshape(28, 28), y_test[idx], "#aaa")]
    ):
        axes[row][digit].imshow(img_arr, cmap="gray")
        axes[row][digit].set_title(str(label_arr), color=title_color, fontsize=11)
        axes[row][digit].axis("off")
        axes[row][digit].set_facecolor("#0f0f1a")

axes[0][0].set_ylabel("Predicted", color="#00d4ff", fontsize=9)
axes[1][0].set_ylabel("True",      color="#aaa",    fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(STATIC_IMG, "sample_predictions.png"), dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
plt.close()
print("[INFO] Sample predictions saved → static/images/sample_predictions.png")

print("\n[DONE] All assets generated. Run 'python app.py' to start the server.")
