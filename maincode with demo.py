# -----------------------------------------------------------
# Handwritten Digit Recognition with CNN + Custom Image Upload
# -----------------------------------------------------------

# Step 1: Import libraries
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from google.colab import files
from PIL import Image

# Step 2: Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize
x_train, x_test = x_train / 255.0, x_test / 255.0

# Reshape for CNN
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Step 3: Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 4: Train model
history = model.fit(x_train, y_train, epochs=5, validation_split=0.1, verbose=2)

# Step 5: Evaluate model
loss, acc = model.evaluate(x_test, y_test, verbose=2)
print("\n🎯 Test Accuracy:", round(acc*100, 2), "%")

# Step 6: Upload your own handwritten digit
print("\n📤 Upload your handwritten digit image (JPG/PNG)...")
uploaded = files.upload()

for fn in uploaded.keys():
    # Load image
    img = Image.open(fn).convert("L")   # Convert to grayscale
    img = img.resize((28,28))           # Resize to 28x28
    img_array = np.array(img)

    # Invert colors if background is white
    if img_array.mean() > 127:
        img_array = 255 - img_array

    # Normalize and reshape
    img_array = img_array / 255.0
    img_array = img_array.reshape(1,28,28,1)

    # Prediction
    pred = model.predict(img_array)
    digit = np.argmax(pred)

    # Show image and prediction
    plt.imshow(img_array.reshape(28,28), cmap="gray")
    plt.title(f"Predicted Digit: {digit}")
    plt.axis("off")
    plt.show()
