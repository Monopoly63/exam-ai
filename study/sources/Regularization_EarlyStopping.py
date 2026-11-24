"""
Step 1: Import Required Libraries

First, import the required libraries for building the neural network, loading the dataset and visualizing the training process.
"""
#===========================================================================
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
#===========================================================================
"""
Step 2: Load and Preprocess the Dataset

Load the MNIST dataset and normalize the pixel values to the range [0, 1] for faster and more stable training.
"""
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0
#===========================================================================
"""
Step 3: Build the Neural Network

Create a simple fully connected neural network for handwritten digit classification.
"""
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation="relu"),
    Dense(10, activation="softmax")
])
#A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor.
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
#===========================================================================
"""
Step 4: Create the Early Stopping Callback

Configure the Early Stopping callback to monitor the validation loss. Training will stop if the validation loss does not improve for 3 consecutive epochs and the best model weights will be restored.

    monitor="val_loss" monitors the validation loss after each epoch.
    patience=3 allows training to continue for three additional epochs without improvement.
    restore_best_weights=True restores the model weights from the epoch with the lowest validation loss. 
"""
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)
#===========================================================================
"""
Step 5: Train the Model with Early Stopping

Train the model while reserving 20% of the training data for validation.
"""
history = model.fit(
    x_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)
#===========================================================================
"""
Step 6: Evaluate the Model

Evaluate the trained model on the test dataset.
"""
test_loss, test_accuracy = model.evaluate(x_test, y_test)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
#===========================================================================
"""
Step 7: Visualize the Training Process

Plot the training loss and validation loss to observe where Early Stopping terminated the training.
"""
plt.figure(figsize=(8, 5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()

plt.show()
#===========================================================================






