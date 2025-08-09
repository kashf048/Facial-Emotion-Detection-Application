
import pandas as pd
import numpy as np

def preprocess_fer2013(csv_file_path):
    data = pd.read_csv(csv_file_path)

    pixels = data["pixels"].tolist()
    emotions = data["emotion"].tolist()

    X = []
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(" ")]
        face = np.asarray(face).reshape(48, 48)
        X.append(face.astype("float32"))

    X = np.asarray(X)
    X = np.expand_dims(X, -1)  # Add channel dimension
    X = X / 255.0  # Normalize pixel values to [0, 1]

    y = np.asarray(emotions)

    return X, y

if __name__ == "__main__":
    csv_file = "fer2013.csv"
    print(f"Preprocessing {csv_file}...")
    X, y = preprocess_fer2013(csv_file)
    print(f"Shape of preprocessed images (X): {X.shape}")
    print(f"Shape of preprocessed labels (y): {y.shape}")

    # Save preprocessed data
    np.save("X.npy", X)
    np.save("y.npy", y)
    print("Preprocessed data saved as X.npy and y.npy")


