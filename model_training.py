# ==========================================
# TRAFFIC SIGN CNN MODEL
# ==========================================

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ==========================================
# 1️⃣ LOAD DATA
# ==========================================

X_train = np.load("Indian Traffic Sign/X_train.npy")
X_val   = np.load("Indian Traffic Sign/X_val.npy")
X_test  = np.load("Indian Traffic Sign/X_test.npy")

y_train = np.load("Indian Traffic Sign/y_train.npy")
y_val   = np.load("Indian Traffic Sign/y_val.npy")
y_test  = np.load("Indian Traffic Sign/y_test.npy")

NUM_CLASSES = int(np.max(np.concatenate([y_train, y_val, y_test])) + 1)

print("Training shape:", X_train.shape)
print("Number of classes:", NUM_CLASSES)

# ==========================================
# 2️⃣ BUILD OPTIMIZED MODEL
# ==========================================

model = models.Sequential([

    # Proper Input Layer
    layers.Input(shape=(224,224,3)),

    layers.Conv2D(32, (3,3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(256, (3,3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),

    layers.GlobalAveragePooling2D(),

    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.summary()

# ==========================================
# 3️⃣ COMPILE MODEL
# ==========================================

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ==========================================
# 4️⃣ CALLBACKS
# ==========================================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "best_traffic_sign_model.h5",
    save_best_only=True
)

# ==========================================
# 5️⃣ TRAIN MODEL
# ==========================================

history = model.fit(
    X_train, y_train,
    epochs=25,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stop, checkpoint]
)

# ==========================================
# 6️⃣ EVALUATE MODEL
# ==========================================

test_loss, test_acc = model.evaluate(X_test, y_test)
print("\nTest Accuracy:", test_acc)

# ==========================================
# 7️⃣ SAVE FINAL MODEL
# ==========================================

model.save("traffic_sign_final_model.h5")
print("✅ Model saved successfully!")