import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Load preprocessed data
print("Loading preprocessed data...")
X = np.load("X.npy")
y = np.load("y.npy")

print(f"Data shape: X={X.shape}, y={y.shape}")

# Take a smaller subset for faster training (demonstration purposes)
subset_size = 5000
indices = np.random.choice(len(X), subset_size, replace=False)
X_subset = X[indices]
y_subset = y[indices]

print(f"Using subset: X={X_subset.shape}, y={y_subset.shape}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_subset, y_subset, test_size=0.2, random_state=42)

# Define a simpler CNN model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 emotions
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Model architecture:")
model.summary()

# Train the model with fewer epochs for demonstration
print("Training the model...")
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Save the trained model
model.save("emotion_detection_model.h5")
print("Model saved as emotion_detection_model.h5")

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}")

