# -----------------------------------------------------------
# Handwritten Digit Recognition using CNN (TensorFlow/Keras)
# -----------------------------------------------------------

# Step 1: Import libraries
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Step 2: Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values (0–255 → 0–1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Reshape (add channel dimension for CNN input)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print("✅ Dataset loaded")
print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)

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

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Model summary
model.summary()

# Step 4: Train the model
history = model.fit(x_train, y_train, epochs=5, validation_split=0.1, verbose=2)

# Step 5: Evaluate model
loss, acc = model.evaluate(x_test, y_test, verbose=2)
print("\n🎯 Test Accuracy:", round(acc*100, 2), "%")

# Step 6: Predict and visualize results
predictions = model.predict(x_test[:5])

for i in range(5):
    plt.imshow(x_test[i].reshape(28,28), cmap="gray")
    plt.title(f"True: {y_test[i]}, Pred: {predictions[i].argmax()}")
    plt.axis("off")
    plt.show()
