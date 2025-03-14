import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.model_selection import train_test_split

# Define dataset path
dataset_path = "part1"

# Load image file paths and labels
image_paths = []
labels = []

for file in os.listdir(dataset_path):
    if file.endswith(".jpg"):
        try:
            parts = file.split("_")
            if len(parts) >= 3:  # Check file format
                age = int(parts[0])
                image_paths.append(os.path.join(dataset_path, file))
                labels.append(age)
            else:
                print(f"Skipping file {file} due to incorrect format.")
        except ValueError as e:
            print(f"Skipping file {file} due to error: {e}")

# Ensure data consistency
if len(image_paths) != len(labels):
    print(f"Mismatch in data: {len(image_paths)} images and {len(labels)} labels.")
    exit()

labels = np.array(labels, dtype=np.float32).reshape(-1, 1)  # ✅ reshape เป็น (num_samples, 1)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(image_paths, labels, test_size=0.2, random_state=42)

# Image preprocessing function
image_size = (128, 128)

def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, image_size) / 255.0  # Normalize to [0,1]
    return image,label 

# Create dataset
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.map(load_and_preprocess_image).batch(32).shuffle(1000)

test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_dataset = test_dataset.map(load_and_preprocess_image).batch(32)

# Define CNN model
inputs = keras.Input(shape=(128, 128, 3))

x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation="relu")(x)

#  Output layers
age_output = layers.Dense(1, activation="linear", name="age")(x)


# Create model
model = keras.Model(inputs=inputs, outputs=age_output)



model.compile(optimizer="adam",
              loss="mean_absolute_error",
              metrics=["mae"])

#  Train model
model.fit(train_dataset, epochs=10, validation_data=test_dataset)

#  Save model
model.save("cnn_model.keras")

print("Model training complete! Model saved as cnn_model.keras")
