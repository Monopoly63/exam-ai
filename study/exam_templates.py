# -*- coding: utf-8 -*-
"""Exam-ready Python snippets — copy ONE section, do not run this whole file.

The blocks below are a reference collection and intentionally reuse names such as
`optimizer`, `model`, and `X_train`. Open study/index.html for the Arabic guide.
"""

# =============================================================================
# 1) The 5-step whiteboard pipeline
# =============================================================================

# 1) Import Library — خزّنها بالذاكرة
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import mean_squared_error, r2_score

# Optimization / nnfs
import nnfs
from nnfs.datasets import spiral_data

# Regularization / TensorFlow + Keras
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

# Regularization / PyTorch
import torch
import torch.nn as nn
import torch.optim as optim


# =============================================================================
# 2) Optimization: common nnfs frame; swap the optimizer line
# =============================================================================

# إطار مشترك لكل أسئلة Optimization: غيّر بس سطر optimizer
import numpy as np
from nnfs.datasets import spiral_data

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Softmax_CrossEntropy:
    def forward(self, inputs, y_true):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        clipped = np.clip(self.output, 1e-7, 1 - 1e-7)
        return np.mean(-np.log(clipped[range(len(y_true)), y_true]))
    def backward(self, y_true):
        samples = len(y_true)
        self.dinputs = self.output.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs /= samples

X, y = spiral_data(samples=100, classes=3)
dense1, activation1 = Layer_Dense(2, 64), Activation_ReLU()
dense2 = Layer_Dense(64, 3)
loss_activation = Softmax_CrossEntropy()

# ====== بدّل هالسطر حسب المطلوب ======
optimizer = Optimizer_Adam(learning_rate=0.05, decay=5e-7)
# optimizer = Optimizer_SGD()
# optimizer = Optimizer_SGD(decay=1e-3, momentum=0.9)
# optimizer = Optimizer_Adagrad(decay=1e-4)
# optimizer = Optimizer_RMSprop(learning_rate=0.02, decay=1e-5, rho=0.999)

for epoch in range(10_001):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y)
    acc = np.mean(np.argmax(loss_activation.output, axis=1) == y)

    loss_activation.backward(y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    if hasattr(optimizer, "pre_update_params"):
        optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    if hasattr(optimizer, "post_update_params"):
        optimizer.post_update_params()

    if epoch % 100 == 0:
        print(f"epoch={epoch}, loss={loss:.3f}, acc={acc:.3f}")


# =============================================================================
# 3) Optimizer classes — copy the one the question asks for
# =============================================================================

# SGD — من SGD_Stochastic_Gradient_Descent.py
class Optimizer_SGD:
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate

    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases

optimizer = Optimizer_SGD()          # lr=1.0 في ملف المحاضرة
optimizer.update_params(dense1)
optimizer.update_params(dense2)


# SGD + LR decay + Momentum — من SGD_with_Momentum.py / Adaptive_Gradient.py
class Optimizer_SGD:
    def __init__(self, learning_rate=1.0, decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (
                1. / (1. + self.decay * self.iterations)
            )

    def update_params(self, layer):
        if self.momentum:
            if not hasattr(layer, "weight_momentums"):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            weight_updates = (self.momentum * layer.weight_momentums) \
                - (self.current_learning_rate * layer.dweights)
            bias_updates = (self.momentum * layer.bias_momentums) \
                - (self.current_learning_rate * layer.dbiases)
            layer.weight_momentums = weight_updates
            layer.bias_momentums = bias_updates
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        self.iterations += 1

optimizer = Optimizer_SGD(decay=1e-3, momentum=0.9)


# AdaGrad — cache يجمع مربّعات التدرّجات، فـ lr يصغر مع الوقت
class Optimizer_Adagrad:
    def __init__(self, learning_rate=1.0, decay=0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (
                1. / (1. + self.decay * self.iterations)
            )

    def update_params(self, layer):
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        layer.weights += (-self.current_learning_rate * layer.dweights) / (
            np.sqrt(layer.weight_cache) + self.epsilon
        )
        layer.biases += (-self.current_learning_rate * layer.dbiases) / (
            np.sqrt(layer.bias_cache) + self.epsilon
        )

    def post_update_params(self):
        self.iterations += 1

optimizer = Optimizer_Adagrad(decay=1e-4)


# RMSprop — متوسّط أُسّي لمربّعات التدرّجات بدل الجمع التراكمي
class Optimizer_RMSprop:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (
                1. / (1. + self.decay * self.iterations)
            )

    def update_params(self, layer):
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache = self.rho * layer.weight_cache \
            + (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache \
            + (1 - self.rho) * layer.dbiases**2

        layer.weights += (-self.current_learning_rate * layer.dweights) / (
            np.sqrt(layer.weight_cache) + self.epsilon
        )
        layer.biases += (-self.current_learning_rate * layer.dbiases) / (
            np.sqrt(layer.bias_cache) + self.epsilon
        )

    def post_update_params(self):
        self.iterations += 1

optimizer = Optimizer_RMSprop(learning_rate=0.02, decay=1e-5, rho=0.999)


# Adam = Momentum + RMSprop + bias correction
class Optimizer_Adam:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
                 beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (
                1. / (1. + self.decay * self.iterations)
            )

    def update_params(self, layer):
        if not hasattr(layer, "weight_cache"):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_momentums = self.beta_1 * layer.weight_momentums \
            + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums \
            + (1 - self.beta_1) * layer.dbiases
        layer.weight_cache = self.beta_2 * layer.weight_cache \
            + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache \
            + (1 - self.beta_2) * layer.dbiases**2

        step = self.iterations + 1
        m_hat_w = layer.weight_momentums / (1 - self.beta_1 ** step)
        m_hat_b = layer.bias_momentums / (1 - self.beta_1 ** step)
        v_hat_w = layer.weight_cache / (1 - self.beta_2 ** step)
        v_hat_b = layer.bias_cache / (1 - self.beta_2 ** step)

        layer.weights += (-self.current_learning_rate * m_hat_w) / (
            np.sqrt(v_hat_w) + self.epsilon
        )
        layer.biases += (-self.current_learning_rate * m_hat_b) / (
            np.sqrt(v_hat_b) + self.epsilon
        )

    def post_update_params(self):
        self.iterations += 1

optimizer = Optimizer_Adam(learning_rate=0.05, decay=5e-7)


# =============================================================================
# 4) Regularization
# =============================================================================

# Early Stopping — TensorFlow / Keras — من المحاضرة الخامسة
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    x_train, y_train,
    epochs=20, batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")


# Dropout — PyTorch — أهم سطرين: nn.Dropout(p) + model.train()/eval()
import torch
import torch.nn as nn
import torch.optim as optim

class CNN_WithDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=p)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(torch.relu(self.fc1(x)))
        return self.fc2(x)

criterion = nn.CrossEntropyLoss()
model = CNN_WithDropout(p=0.5)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# داخل التدريب
model.train()
optimizer.zero_grad()
loss = criterion(model(images), labels)
loss.backward()
optimizer.step()

# داخل الـ validation / test
model.eval()
with torch.no_grad():
    outputs = model(images)
    _, preds = torch.max(outputs, 1)


# Batch Normalization — TensorFlow / Keras
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train.reshape(-1, 784)[:5000]
x_test = x_test.reshape(-1, 784)
y_train = y_train[:5000]

model = tf.keras.Sequential([
    tf.keras.Input(shape=(784,)),
    tf.keras.layers.Dense(64),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.fit(x_train, y_train, epochs=2, batch_size=32)


# =============================================================================
# 5) SVM / SVC / SVR
# =============================================================================

# SVC كامل — من SVM2.py (تصنيف سرطان الثدي)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

data = load_breast_cancer()
X = data.data[:, [0, 1]]     # أول ميزتين للرسم فقط؛ بالامتحان غالباً كل الأعمدة
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm_classifier = SVC(kernel="rbf", C=100.0, random_state=42)
svm_classifier.fit(X_train_scaled, y_train)

y_pred = svm_classifier.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred,
                            target_names=data.target_names))


# SVR — y رقم مستمر، مو Class — من SVR.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - np.random.rand(8))   # noise

svr_rbf = SVR(kernel="rbf", C=1, gamma=10, epsilon=0.1)
svr_lin = SVR(kernel="linear", C=100, gamma="auto")
svr_poly = SVR(kernel="poly", C=100, gamma="auto",
               degree=3, epsilon=0.1, coef0=1)

for model in [svr_rbf, svr_lin, svr_poly]:
    model.fit(X, y)
    pred = model.predict(X)
    print("MSE:", mean_squared_error(y, pred),
          "R2:", r2_score(y, pred),
          "support vectors:", len(model.support_))


# نسخة آمنة للامتحان: الـ scaling داخل CV بواسطة Pipeline (بدون leakage)
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe = make_pipeline(StandardScaler(), SVC())
param_grid = {
    "svc__C": np.logspace(-2, 3, 6),
    "svc__gamma": np.logspace(-4, 1, 6),
    "svc__kernel": ["rbf", "linear"],
}
grid = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train)

print(grid.best_params_, grid.best_score_)
y_pred = grid.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

