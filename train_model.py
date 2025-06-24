import tensorflow as tf
from tensorflow.keras import layers, models

# Load the MNIST dataset
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.astype('float32') / 255.0

# Build a simple neural network
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5)

# Save the model to the same directory as XYCordinate.py
model.save('mnist_model.h5')
print("Model saved as 'mnist_model.h5'")