"""
Implementation using TensorFlow
Step 1: Import the Required Library

    Import the TensorFlow library to build and train the neural network.
    The Keras API in TensorFlow provides the BatchNormalization() layer used in this example. 
"""
import tensorflow as tf
#===========================================================================
"""
Step 2: Load and Preprocess the Dataset

    Load the handwritten digit dataset using TensorFlow.
    Normalize the pixel values to the range 0–1 for faster and more stable training.
    Flatten each image into a one-dimensional vector before passing it to the Dense layer.
    Select only 5,000 training samples so the example runs quickly. 
"""
# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Flatten each 28×28 image into a 784-dimensional vector
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# Use a small subset for faster execution
x_train = x_train[:5000]
y_train = y_train[:5000]
#===========================================================================
"""
Step 3: Create the Neural Network with Batch Normalization

    Create a Sequential neural network.
    Add an input layer with 784 features.
    Add a Dense layer containing 64 neurons. 
"""
model = tf.keras.Sequential([
    tf.keras.Input(shape=(784,)),
    tf.keras.layers.Dense(64),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])
#===========================================================================
"""
Step 4: Compile the Model

    Use the Adam optimizer for efficient weight updates.
    Use Sparse Categorical Crossentropy because the labels are integer values.
    Track the model's performance using accuracy. 
"""
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
#===========================================================================
"""
Step 5: Train the Model

    Train the model using the training dataset.
    Run the training process for 2 epochs to keep execution time short. 
"""
model.fit(
    x_train,
    y_train,
    epochs=2,
    batch_size=32
)
#===========================================================================