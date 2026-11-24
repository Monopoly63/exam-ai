#!/usr/bin/env python3
"""Generate the self-contained Arabic exam-review page and the Python templates."""
from __future__ import annotations

from html import escape
from pathlib import Path
import json
import textwrap

ROOT = Path(__file__).resolve().parent

# Deployed on Vercel from a small static bundle; original lecture files stay on GitHub.
GITHUB_SOURCE_BASE = (
    "https://github.com/Monopoly63/exam-ai/raw/refs/heads/"
    "arena/872c0405-exam-ai/study/sources/"
)
GITHUB_TEMPLATE_URL = (
    "https://github.com/Monopoly63/exam-ai/raw/refs/heads/"
    "arena/872c0405-exam-ai/study/exam_templates.py"
)

# ---------------------------------------------------------------------------
# Code snippets. They are intentionally short: the exam-relevant lines only.
# ---------------------------------------------------------------------------
CODE = {
    "imports": textwrap.dedent(r'''
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
    '''),

    "pipeline": textwrap.dedent(r'''
        # خارطة البورد: 1 Library → 2 Data → 3 Pre-processing → 4 Train → 5 Test / Figure out
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC                     # SVR لو y رقم مستمر
        from sklearn.metrics import accuracy_score     # classification_report كمان

        # 2) Import Data
        df = pd.read_csv("data.csv")
        X = df.drop("target_class", axis=1)
        y = df["target_class"]

        # 3) Pre-processing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)        # fit + transform على الترين فقط
        X_test = scaler.transform(X_test)              # transform فقط على التيست

        # 4) Train Algorithm
        model = SVC(kernel="rbf", C=1.0, gamma="scale")
        model.fit(X_train, y_train)

        # 5) Test / Validation / Prediction / Figure out
        y_pred = model.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        print("support vectors:", model.support_vectors_.shape)
    '''),

    "nnfs_frame": textwrap.dedent(r'''
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
    '''),

    "sgd_vanilla": textwrap.dedent(r'''
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
    '''),

    "sgd_momentum": textwrap.dedent(r'''
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
    '''),

    "adagrad": textwrap.dedent(r'''
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
    '''),

    "rmsprop": textwrap.dedent(r'''
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
    '''),

    "adam": textwrap.dedent(r'''
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
    '''),

    "train_loop": textwrap.dedent(r'''
        # ترتيب التدريب اللي لازم تكتبه غيباً
        for epoch in range(10_001):
            # Forward pass
            dense1.forward(X)
            activation1.forward(dense1.output)
            dense2.forward(activation1.output)
            loss = loss_activation.forward(dense2.output, y)

            # Accuracy
            predictions = np.argmax(loss_activation.output, axis=1)
            accuracy = np.mean(predictions == y)

            # Backward pass
            loss_activation.backward(loss_activation.output, y)
            dense2.backward(loss_activation.dinputs)
            activation1.backward(dense2.dinputs)
            dense1.backward(activation1.dinputs)

            # Optimizer: pre مرة → update لكل طبقة → post مرة
            optimizer.pre_update_params()
            optimizer.update_params(dense1)
            optimizer.update_params(dense2)
            optimizer.post_update_params()
    '''),

    "early_stopping": textwrap.dedent(r'''
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
    '''),

    "dropout": textwrap.dedent(r'''
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
    '''),

    "batchnorm": textwrap.dedent(r'''
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
    '''),

    "l2_extra": textwrap.dedent(r'''
        # إضافي للفهم — مو موجود كملف مستقل بكود المحاضرة
        # L2 / weight decay تقلّل كبر الأوزان، فتقلّل الـ overfitting

        # Keras
        dense = tf.keras.layers.Dense(
            64, activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        )

        # PyTorch
        optimizer = optim.SGD(model.parameters(), lr=0.01,
                              momentum=0.9, weight_decay=1e-4)

        # SVM: C هو معامل الـ regularization بطريقة معكوسة
        clf = SVC(C=0.01)   # C صغير = تنظيم أقوى / margin أوسع
        clf = SVC(C=100.0)  # C كبير = تنظيم أضعف / محاولة تصحيح كل العيّنات
    '''),

    "svc_basic": textwrap.dedent(r'''
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
    '''),

    "svc_kernels": textwrap.dedent(r'''
        # مقارنة Kernels و C — مختصر من svm.py
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score

        # SVC() الافتراضي: C=1.0 و kernel='rbf'
        # ملاحظة: gamma الافتراضي تغيّر بين إصدارات sklearn (auto / scale)
        experiments = [
            ("default", SVC()),
            ("rbf C=100", SVC(kernel="rbf", C=100.0)),
            ("rbf C=1000", SVC(kernel="rbf", C=1000.0)),
            ("linear", SVC(kernel="linear", C=1.0)),
            ("poly", SVC(kernel="poly", C=1.0, degree=3)),
            ("sigmoid", SVC(kernel="sigmoid", C=1.0)),
        ]

        for name, clf in experiments:
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            print(name, accuracy_score(y_test, pred))
    '''),

    "svr": textwrap.dedent(r'''
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
    '''),

    "gridsearch": textwrap.dedent(r'''
        # GridSearchCV على C و gamma — من RBF_SVM Param.py
        import numpy as np
        from sklearn.datasets import load_iris
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
        from sklearn.svm import SVC

        iris = load_iris()
        X, y = iris.data, iris.target
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        C_range = np.logspace(-2, 10, 13)
        gamma_range = np.logspace(-9, 3, 13)
        param_grid = dict(gamma=gamma_range, C=C_range)

        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
        grid.fit(X, y)

        print("best params:", grid.best_params_)
        print(f"best CV score: {grid.best_score_:.2f}")
        print("best estimator:", grid.best_estimator_)
    '''),

    "gridsearch_safe": textwrap.dedent(r'''
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
    '''),

    "exam_switch": textwrap.dedent(r'''
        # سطر واحد يقرّر المسار الصح
        # y = أصناف / labels  → تصنيف → SVC + accuracy_score
        # y = رقم مستمر       → انحدار → SVR + mean_squared_error / r2_score

        def build_model(y):
            if len(np.unique(y)) <= 10 and np.all(y == y.astype(int)):
                return SVC(kernel="rbf", C=1.0, gamma="scale")
            return SVR(kernel="rbf", C=1.0, gamma="scale", epsilon=0.1)

        model = build_model(y)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
    '''),
}

TEMPLATE_FILE = """# -*- coding: utf-8 -*-
\"\"\"Exam-ready Python snippets — copy ONE section, do not run this whole file.

The blocks below are a reference collection and intentionally reuse names such as
`optimizer`, `model`, and `X_train`. Open study/index.html for the Arabic guide.
\"\"\"

# =============================================================================
# 1) The 5-step whiteboard pipeline
# =============================================================================
{imports}

# =============================================================================
# 2) Optimization: common nnfs frame; swap the optimizer line
# =============================================================================
{nnfs_frame}

# =============================================================================
# 3) Optimizer classes — copy the one the question asks for
# =============================================================================
{sgd_vanilla}
{sgd_momentum}
{adagrad}
{rmsprop}
{adam}

# =============================================================================
# 4) Regularization
# =============================================================================
{early_stopping}
{dropout}
{batchnorm}

# =============================================================================
# 5) SVM / SVC / SVR
# =============================================================================
{svc_basic}
{svr}
{gridsearch_safe}
""".format(**CODE)

(ROOT / "exam_templates.py").write_text(TEMPLATE_FILE, encoding="utf-8")

# ---------------------------------------------------------------------------
# Quiz + checklist + source map
# ---------------------------------------------------------------------------
QUIZ = [
    {
        "q": "عندك y فيه قيم 0/1 وعليك تتوقع الصنف. أي موديل تختار؟",
        "options": ["SVR", "SVC", "Optimizer_Adam", "EarlyStopping"],
        "answer": 1,
        "why": "SVC = Support Vector Classification. SVR للانحدار لمّا y رقم مستمر."
    },
    {
        "q": "لماذا نستعمل StandardScaler قبل SVC/SVR؟",
        "options": [
            "ليزيد عدد الأعمدة",
            "لأن الـ kernel والمسافات تتأثر بمقياس الميزات",
            "لأنه يحوّل y إلى one-hot",
            "لأنه يلغي الحاجة لـ train_test_split"
        ],
        "answer": 1,
        "why": "RBF وغيره يعتمد على المسافات؛ لو ميزة واحدة بمقياس كبير بتسيطر على الحساب."
    },
    {
        "q": "السطر الصح للتيست بعد ما درّبت الـ scaler على الترين؟",
        "options": [
            "X_test = scaler.fit_transform(X_test)",
            "X_test = scaler.transform(X_test)",
            "X_test = scaler.fit(X_test)",
            "ما في داعي للـ transform"
        ],
        "answer": 1,
        "why": "fit_transform على الترين فقط. على التيست نطبّق نفس المتوسط والانحراف بـ transform."
    },
    {
        "q": "زيادة gamma في SVC مع RBF ماذا تعمل غالباً؟",
        "options": [
            "حدّ قرار أنعم وأبسط",
            "حدّ قرار أكثر التواءً وخطر overfitting أعلى",
            "تغيّر عدد الطبقات",
            "توقف التدريب مبكراً"
        ],
        "answer": 1,
        "why": "gamma كبير = تأثير كل نقطة أقرب وأضيق = boundary أدق وأكثر تعقيداً."
    },
    {
        "q": "C كبير جداً في SVC يعني…",
        "options": [
            "تنظيم أقوى و margin أوسع",
            "تنظيم أضعف، margin أضيق، والموديل يحاول يصحّح كل عيّنات الترين",
            "learning rate أعلى",
            "عدد support vectors صفر"
        ],
        "answer": 1,
        "why": "C يعاقب أخطاء الـ margin؛ كل ما كبر قلّت tolerance وقلّ التنظيم فعلياً."
    },
    {
        "q": "ما معنى epsilon في SVR؟",
        "options": [
            "معدّل التعلّم",
            "عرض أنبوب عدم الحساسية حول المنحنى",
            "عدد الـ epochs",
            "درجة كثير الحدود"
        ],
        "answer": 1,
        "why": "ε-insensitive tube: أخطاء داخل الأنبوب ما بتنحسب بالـ loss."
    },
    {
        "q": "في PyTorch مع Dropout، شو لازم قبل الـ validation؟",
        "options": ["model.train()", "model.eval()", "optimizer.step()", "loss.backward()"],
        "answer": 1,
        "why": "model.eval() يطفئ Dropout ويجمّد سلوك BatchNorm؛ ومعها غالباً torch.no_grad()."
    },
    {
        "q": "EarlyStopping(monitor='val_loss') يحتاج…",
        "options": [
            "validation_split أو validation_data داخل fit",
            "SVR بدل SVC",
            "np.logspace",
            "model.support_vectors_"
        ],
        "answer": 0,
        "why": "بدون بيانات validation ما رح يكون عندك val_loss أصلاً."
    },
    {
        "q": "أي سطر يميّز AdaGrad عن RMSprop؟",
        "options": [
            "cache += dW**2",
            "cache = rho*cache + (1-rho)*dW**2",
            "m_hat = m / (1-beta_1**step)",
            "weight_updates = momentum*v - lr*dW"
        ],
        "answer": 0,
        "why": "AdaGrad يجمع مربّعات التدرّجات تراكمياً؛ RMSprop يستعمل متوسطاً أُسّياً. الخيار الثالث Adam."
    },
    {
        "q": "ترتيب استدعاء الـ optimizer في حلقة التدريب؟",
        "options": [
            "post → update → pre",
            "update → pre → post",
            "pre → update_params لكل طبقة → post",
            "pre → post → update"
        ],
        "answer": 2,
        "why": "pre مرة واحدة قبل تحديث الطبقات، ثم update لكل طبقة، ثم post مرة واحدة لزيادة iterations."
    },
    {
        "q": "GridSearchCV بعد التدريب، وين أفضل نتيجة؟",
        "options": [
            "grid.best_score_ و grid.best_params_",
            "grid.support_",
            "grid.epsilon",
            "grid.current_learning_rate"
        ],
        "answer": 0,
        "why": "best_params_ يعطي أفضل C/gamma، و best_score_ درجة الـ CV، و best_estimator_ الموديل نفسه."
    },
    {
        "q": "BatchNormalization في كود المحاضرة جاي…",
        "options": [
            "بعد softmax",
            "بين Dense(64) و Activation('relu')",
            "داخل optimizer",
            "بدل train_test_split"
        ],
        "answer": 1,
        "why": "الكود: Dense(64) → BatchNormalization() → Activation('relu') → Dense(10, softmax)."
    },
]

CHECKS = [
    "قرأت السؤال وميّزت: تصنيف SVC ولا انحدار SVR؟",
    "كتبت كل الـ imports قبل ما أبدأ (sklearn / tf / torch / nnfs).",
    "قسمت البيانات بـ train_test_split قبل الـ scaling.",
    "استعملت fit_transform على X_train و transform فقط على X_test.",
    " SVC/SVR: اخترت kernel و C و gamma بشكل مقصود.",
    " SVR: انتبهت لـ epsilon ولـ degree/coef0 لو kernel='poly'.",
    "قِست التصنيف بـ accuracy_score / classification_report.",
    "قِست الانحدار بـ mean_squared_error / r2_score.",
    " GridSearchCV: طبّعت best_params_ و best_score_.",
    " Optimization: حدّثت weights و biases مع بعض.",
    " Optimization: حطّيت epsilon بالقاسم و bias correction لـ Adam.",
    " Regularization: كتبت callbacks=[early_stopping] و validation_split.",
    " Dropout: حطّيت model.eval() وقت الـ validation/test.",
    " Figure out: فسّرت النتيجة مو بس طبعتها (overfit؟ margin؟ val_loss؟).",
]

SOURCES = [
    ("3rd_Lecture_Codes_AI_Practical.zip", "SGD_Stochastic_Gradient_Descent.py", "Optimization_SGD.py", "SGD عادي"),
    ("3rd_Lecture_Codes_AI_Practical.zip", "SGD_with_Momentum.py", "Optimization_SGD_Momentum.py", "Momentum + decay"),
    ("3rd_Lecture_Codes_AI_Practical.zip", "Adaptive_Gradient.py", "Optimization_AdaGrad.py", "AdaGrad"),
    ("3rd_Lecture_Codes_AI_Practical.zip", "Opt_RMS.py", "Optimization_RMSprop.py", "RMSprop"),
    ("3rd_Lecture_Codes_AI_Practical.zip", "Opt_Adam.py", "Optimization_Adam.py", "Adam"),
    ("5th_Lecture_Codes.zip", "Batch_Normalization.py", "Regularization_BatchNorm.py", "Batch Normalization"),
    ("5th_Lecture_Codes.zip", "Dropout.py", "Regularization_Dropout.py", "Dropout"),
    ("5th_Lecture_Codes.zip", "Early_Stopping_Regularization.py", "Regularization_EarlyStopping.py", "Early Stopping"),
    ("AI_Lectture_4.zip", "RBF_SVM Param.py", "SVM_RBF_Params.py", "C / gamma + GridSearchCV"),
    ("AI_Lectture_4.zip", "SVM2.py", "SVM_SVC_BreastCancer.py", "SVC + تقييم + حدّ القرار"),
    ("AI_Lectture_4.zip", "SVR.py", "SVM_SVR.py", "SVR بثلاث kernels"),
    ("AI_Lectture_4.zip", "svm.py", "SVM_SVC_Pulsar.py", "CSV + SVC + مقارنة kernels"),
]

# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------
CARD_TEMPLATE = """
<article class="card searchable" id="__ID__" data-search>
  <div class="card-head">
    <div class="card-title">
      <span class="eyebrow">__TAG__</span>
      <h3>__TITLE__</h3>
      <p>__SUB__</p>
    </div>
    <div class="card-actions noprint">
      __SOURCE__
      <button type="button" class="btn tiny copy-btn" data-copy="__ID__">نسخ الكود</button>
    </div>
  </div>
  <div class="code-wrap">
    <pre dir="ltr"><code class="python" id="code-__ID__">__CODE__</code></pre>
  </div>
  __NOTES__
</article>
"""


def code_card(cid, tag, title, sub, code_key, note, source=None):
    src = ""
    if source:
        src = ('<a class="source-link" href="' + GITHUB_SOURCE_BASE + escape(source) +
               '" target="_blank" rel="noopener">الأصل .py</a>')
    notes = ""
    if note:
        notes = '<div class="notes"><span class="notes-label">احفظ</span><div>' + note + '</div></div>'
    return (CARD_TEMPLATE
            .replace("__ID__", cid)
            .replace("__TAG__", tag)
            .replace("__TITLE__", title)
            .replace("__SUB__", sub)
            .replace("__SOURCE__", src)
            .replace("__CODE__", escape(CODE[code_key].strip("\n")))
            .replace("__NOTES__", notes))


def section(sid, number, title, kicker, body):
    return """
<section class="topic" id="__SID__">
  <div class="section-head">
    <div class="section-num">__NUM__</div>
    <div>
      <p class="kicker">__KICKER__</p>
      <h2>__TITLE__</h2>
    </div>
  </div>
  __BODY__
</section>
""".replace("__SID__", sid).replace("__NUM__", number).replace("__KICKER__", kicker).replace("__TITLE__", title).replace("__BODY__", body)


# ---------------------------------------------------------------------------
# Section bodies
# ---------------------------------------------------------------------------
hero_body = """
<div class="hero-grid">
  <div class="hero-copy">
    <div class="pill-row">
      <span class="pill">Optimization</span>
      <span class="pill">Regularization</span>
      <span class="pill">SVM · SVR · SVC</span>
    </div>
    <h1>مذاكرة عملية للامتحان<br><span>على خطوات البورد، سطر بسطر</span></h1>
    <p class="lead">هاي الصفحة مبنية على أكواد المحاضرات الموجودة بالريبو، مختصرة ومترجمة: شو بنستورد، شو بنعالج، شو بندرّب، وكيف بنقرأ النتيجة. ابدأ من «القالب الذهبي» وبعدين افتح القسم اللي الدكتور ركّز عليه.</p>
    <div class="hero-actions noprint">
      <a class="btn primary" href="#pipeline-section">ابدأ بالقالب</a>
      <a class="btn" href="#quiz">اختبر حالك</a>
      <button type="button" class="btn" id="exam-mode">وضع الحفظ</button>
    </div>
    <div class="mini-stats">
      <div><strong>5</strong><span>خطوات البورد</span></div>
      <div><strong>5</strong><span>Optimizers</span></div>
      <div><strong>3</strong><span>Regularizers</span></div>
      <div><strong>12</strong><span>ملف كود أصلي</span></div>
    </div>
  </div>

  <div class="board" aria-label="خطوات البورد من الصورة">
    <div class="board-header"><span>السبورة → الخريطة</span><span class="board-dots">● ● ●</span></div>
    <ol class="steps">
      <li><b>1</b><div><strong>Import Library</strong><code>import numpy, sklearn, torch, tf</code></div></li>
      <li><b>2</b><div><strong>Import Data</strong><code>read_csv / load_iris / load_breast_cancer</code></div></li>
      <li><b>3</b><div><strong>Pre-processing</strong><code>split → StandardScaler → fit_transform/transform</code></div></li>
      <li><b>4</b><div><strong>Train Algorithm</strong><code>.fit() / forward-backward-update</code></div></li>
      <li><b>5</b><div><strong>Test · Validation · Prediction · Figure out</strong><code>predict → metric → best_params_ / val_loss</code></div></li>
    </ol>
  </div>
</div>

<div class="summary-strip searchable" data-search>
  <h3>الخلاصة بـ ١٢ سطر قبل ما تنام</h3>
  <div class="summary-grid">
    <div><span>01</span><p><b>SVC</b> للتصنيف، <b>SVR</b> للأرقام المستمرة.</p></div>
    <div><span>02</span><p>قبل SVM: <code>split</code> ثم <code>fit_transform</code> على الترين و<code>transform</code> على التيست.</p></div>
    <div><span>03</span><p><code>C ↑</code> = تنظيم أضعف / حدّ أضيق. <code>gamma ↑</code> = حدّ أعقد / overfitting.</p></div>
    <div><span>04</span><p><code>epsilon</code> في SVR = عرض أنبوب التسامح.</p></div>
    <div><span>05</span><p>SGD: <code>W += -lr*dW</code>. Momentum يخزّن آخر تحديث.</p></div>
    <div><span>06</span><p>AdaGrad يجمع <code>dW²</code>؛ RMSprop متوسط أُسّي؛ Adam الاثنين + bias correction.</p></div>
    <div><span>07</span><p>كل optimizer يحدّث <b>weights و biases</b> ويحط <code>epsilon</code> بالقاسم.</p></div>
    <div><span>08</span><p>الترتيب: <code>pre</code> → <code>update_params</code> لكل طبقة → <code>post</code>.</p></div>
    <div><span>09</span><p>EarlyStopping يحتاج <code>validation_split</code> و <code>callbacks=[...]</code>.</p></div>
    <div><span>10</span><p>Dropout وقت التقييم: <code>model.eval()</code> + <code>torch.no_grad()</code>.</p></div>
    <div><span>11</span><p>BatchNorm بالكود: <code>Dense → BatchNormalization → Activation('relu')</code>.</p></div>
    <div><span>12</span><p>Figure out = اشرح النتيجة: accuracy / val_loss / best_params_ / support vectors.</p></div>
  </div>
</div>
"""

pipeline_body = """
<div class="callout">
  <b>الفكرة:</b> أغلب أسئلة العملي هي نفس القالب، بس يتغيّر <code>model</code> و <code>metric</code>. احفظ الترتيب وما تخلط بين <code>fit_transform</code> و <code>transform</code>.
</div>
""" + code_card("imports-card", "الخطوة ١", " شنو بنستورد؟", "قائمة imports جاهزة لكل المواضيع الثلاثة.", "imports",
    "ما لازم تحفظ كل سطر حرفياً، بس لازم تعرف من وين بيجي <code>SVC</code> و <code>EarlyStopping</code> و <code>nn.Dropout</code>.") + code_card("pipeline-card", "الخطوات ٢–٥", "القالب الذهبي", "من <code>read_csv</code> لحد طباعة الـ accuracy.", "pipeline",
    "<code>support_vectors_.shape</code> خطوة «Figure out»: بتحكي قدّيش عيّنة صارت مسؤولة عن الحدّ.")

optimization_body = """
<div class="callout accent">
  <b>قاعدة الامتحان:</b> Optimization = كيف نحدّث الأوزان بعد ما حسبنا <code>dweights</code> و <code>dbiases</code>. كل الكلاسات بنفس الشكل: <code>__init__</code> → <code>pre_update_params</code> → <code>update_params(layer)</code> → <code>post_update_params</code>.
</div>

<div class="table-wrap searchable" data-search>
  <table class="compare">
    <thead><tr><th>Algorithm</th><th>السطر اللي بيميّزه</th><th>إعدادات المحاضرة</th><th>انتبه</th></tr></thead>
    <tbody>
      <tr><td><b>SGD</b></td><td dir="ltr"><code>W += -lr * dW</code></td><td><code>Optimizer_SGD()</code></td><td>ما عنده pre/post</td></tr>
      <tr><td><b>+ Momentum</b></td><td dir="ltr"><code>v = m*v - lr*dW</code></td><td><code>decay=1e-3, momentum=0.9</code></td><td><code>np.zeros_like</code> أول مرة</td></tr>
      <tr><td><b>AdaGrad</b></td><td dir="ltr"><code>cache += dW**2</code></td><td><code>decay=1e-4</code></td><td>lr بيصغر تراكمياً</td></tr>
      <tr><td><b>RMSprop</b></td><td dir="ltr"><code>cache = rho*cache + (1-rho)*dW**2</code></td><td><code>lr=0.02, decay=1e-5, rho=0.999</code></td><td>ما بيجمع للأبد</td></tr>
      <tr><td><b>Adam</b></td><td dir="ltr"><code>m_hat / (sqrt(v_hat)+eps)</code></td><td><code>lr=0.05, decay=5e-7</code></td><td><code>iterations + 1</code> للتصحيح</td></tr>
    </tbody>
  </table>
</div>

<div class="formula-strip">
  <div><span>LR decay</span><code dir="ltr">lr_t = lr0 / (1 + decay * iterations)</code></div>
  <div><span>Adagrad / RMS / Adam</span><code dir="ltr">W += -lr * dW / (sqrt(cache) + eps)</code></div>
  <div><span>Adam correction</span><code dir="ltr">m̂ = m / (1 - β1^t), v̂ = v / (1 - β2^t)</code></div>
</div>
""" + code_card("nnfs-frame-card", "إطار مشترك", "حلقة التدريب + تبديل الـ optimizer بسطر", "نسخة مختصرة وصالحة للتشغيل؛ الأصل أطول بسبب الرسوم.", "nnfs_frame",
    "<code>hasattr</code> بيحميك لو استعملت SGD العادي اللي ما عنده <code>pre/post</code>. المصدر: ملفات المحاضرة الثالثة.", "Optimization_Adam.py") \
 + code_card("sgd-card", "Optimizer 1", "SGD العادي", "أبسط تحديث؛ هو أساس كل اللي بعده.", "sgd_vanilla",
    "الإشارة سالبة لأننا بننزل عكس التدرّج. ولا تنسَ الـ <code>biases</code>.", "Optimization_SGD.py") \
 + code_card("momentum-card", "Optimizer 2", "SGD + Decay + Momentum", "نفس الكلاس بيخدم الحالتين: مع أو بدون momentum.", "sgd_momentum",
    "<code>pre</code> بحسب الـ lr الحالي، <code>post</code> بيزيد <code>iterations</code>. لو <code>momentum=0</code> بيرجع SGD عادي مع decay.", "Optimization_SGD_Momentum.py") \
 + code_card("adagrad-card", "Optimizer 3", "AdaGrad", "Cache تراكمي لمربّعات التدرّج.", "adagrad",
    "الفرق الجوهري: <code>+= dW**2</code> مو <code>= rho*cache + ...</code>.", "Optimization_AdaGrad.py") \
 + code_card("rmsprop-card", "Optimizer 4", "RMSprop", "متوسط أُسّي بدل التراكم.", "rmsprop",
    "<code>rho</code> بالمحاضرة Final = <code>0.999</code> مع أن الافتراض بالكلاس <code>0.9</code>.", "Optimization_RMSprop.py") \
 + code_card("adam-card", "Optimizer 5", "Adam", "Momentum + RMSprop + bias correction.", "adam",
    "أكثر سطر ينِساه الطلاب: <code>step = self.iterations + 1</code> قبل <code>1 - beta ** step</code>.", "Optimization_Adam.py") \
 + code_card("train-loop-card", "ترتيب ثابت", "حلقة التدريب كما وردت بالمحاضرة", "Forward → Loss/Accuracy → Backward → Optimizer.", "train_loop",
    "كود الرسم <code>contourf / pause</code> مش مهم للامتحان؛ ركّز على الـ backward وتحديث الطبقتين.", "Optimization_SGD_Momentum.py")

regularization_body = """
<div class="callout warning">
  <b>تنبيه من ملفاتك:</b> المحاضرة الخامسة فيها ثلاث تقنيات: <b>Early Stopping</b>، <b>Dropout</b>، <b>Batch Normalization</b>. ما في ملف مستقل لـ L1/L2؛ أضفتلك卡片 صغيرة للفهم بس، و<code>C</code> الخاص بـ SVM بنشرحه بقسمه.
</div>

<div class="reg-grid">
  <div class="reg-card searchable" data-search><span class="reg-icon">⏹</span><h3>Early Stopping</h3><p>يوقف التدريب لمّا <code>val_loss</code> ما يتحسّن.</p><code dir="ltr">patience=3 · restore_best_weights=True</code></div>
  <div class="reg-card searchable" data-search><span class="reg-icon">◌</span><h3>Dropout</h3><p>يطفئ نسبة من العصبونات وقت التدريب فقط.</p><code dir="ltr">nn.Dropout(p=0.5) · model.eval()</code></div>
  <div class="reg-card searchable" data-search><span class="reg-icon">μ</span><h3>Batch Normalization</h3><p>يطبّع خرج الطبقة داخل كل batch؛ أثر تنظيمي خفيف.</p><code dir="ltr">Dense → BN → Activation('relu')</code></div>
</div>
""" + code_card("early-card", "Regularization 1", "Early Stopping / Keras", "أهم شيء: في <code>validation</code> وفي <code>callbacks</code>.", "early_stopping",
    "<code>monitor='val_loss'</code> بدون <code>validation_split=0.2</code> = خطأ. <code>restore_best_weights=True</code> بيرجّع أفضل أوزان.", "Regularization_EarlyStopping.py") \
 + code_card("dropout-card", "Regularization 2", "Dropout / PyTorch", "مقارنة موديل بدون ومع Dropout على MNIST.", "dropout",
    "وقت التدريب <code>model.train()</code>، وقت التقييم <code>model.eval()</code>. ومع التقييم <code>with torch.no_grad()</code>.", "Regularization_Dropout.py") \
 + code_card("batchnorm-card", "Regularization 3", "Batch Normalization / Keras", "MNIST → 784 → Dense(64) → BN → ReLU → Softmax.", "batchnorm",
    "وظيفته الأساسية تثبيت التدريب وتسريعه، بس المادة حطّته مع ملفات الـ Regularization.", "Regularization_BatchNorm.py") \
 + code_card("l2-card", "إضافي", "L2 / weight decay و C", "للفهم فقط: مش كود محاضرة مستقل.", "l2_extra",
    "في SVM: <b>C صغير = تنظيم أقوى</b>. في Keras: <code>kernel_regularizer</code>. في PyTorch: <code>weight_decay</code>.")

svm_body = """
<div class="decision searchable" data-search>
  <h3>Decision tree قبل ما تكتب أول سطر</h3>
  <div class="decision-tree">
    <div class="root">شو نوع <code>y</code>؟</div>
    <div class="branch"><span>أصناف / 0 و 1 / labels</span><b>→ SVC</b><small>accuracy_score · classification_report</small></div>
    <div class="branch"><span>رقم مستمر / sin(X) / سعر</span><b>→ SVR</b><small>mean_squared_error · r2_score · epsilon</small></div>
  </div>
</div>

<div class="table-wrap searchable" data-search>
  <table class="compare params">
    <thead><tr><th>Parameter</th><th>وين</th><th>ماذا يعمل</th><th>قيم من كود المحاضرة</th></tr></thead>
    <tbody>
      <tr><td><code>kernel</code></td><td>SVC + SVR</td><td>شكل حدّ القرار / المنحنى</td><td><code>'rbf', 'linear', 'poly', 'sigmoid'</code></td></tr>
      <tr><td><code>C</code></td><td>SVC + SVR</td><td>عقوبة أخطاء الـ margin؛ كل ما كبر قلّ التنظيم</td><td><code>0.01, 1, 100, 1000</code> · logspace(-2,10,13)</td></tr>
      <tr><td><code>gamma</code></td><td>غالباً RBF / poly / sigmoid</td><td>مدى تأثير النقطة الواحدة؛ كبير = حدّ أعقد</td><td><code>10</code> · <code>0.1, 1, 10</code> · logspace(-9,3,13)</td></tr>
      <tr><td><code>epsilon</code></td><td>SVR فقط</td><td>عرض أنبوب عدم الحساسية</td><td><code>0.1</code></td></tr>
      <tr><td><code>degree</code> / <code>coef0</code></td><td>SVR poly</td><td>درجة كثير الحدود / الثابت</td><td><code>degree=3, coef0=1</code></td></tr>
    </tbody>
  </table>
</div>

<div class="cg-matrix searchable" data-search>
  <h3>مصفوفة C / gamma — «Figure out»</h3>
  <div class="matrix">
    <div><span>C صغير · gamma صغير</span><p>حدّ ناعم جداً → غالباً <b>underfitting</b>.</p></div>
    <div><span>C كبير · gamma صغير</span><p>margin أضيق بس الحدّ لسه سلس.</p></div>
    <div><span>C صغير · gamma كبير</span><p>التعقيد موجود بس التنظيم بجمّعه.</p></div>
    <div class="hot"><span>C كبير · gamma كبير</span><p>حدّ ملتوي يلاحق كل نقطة → <b>overfitting</b>.</p></div>
  </div>
</div>
""" + code_card("svc-basic-card", "SVC", "تصنيف كامل: split → scale → fit → predict", "من <code>SVM2.py</code> مع <code>load_breast_cancer</code>.", "svc_basic",
    "<code>data.data[:, [0, 1]]</code> مخصوص للرسم؛ بالامتحان استعمل كل الميزات إلا إذا طلب غير هيك.", "SVM_SVC_BreastCancer.py") \
 + code_card("svc-kernels-card", "SVC", "مقارنة Kernels و C", "مختصر من <code>svm.py</code> على بيانات Pulsar.", "svc_kernels",
    "<code>SVC()</code> = <code>C=1.0</code> و <code>kernel='rbf'</code>. قيمة <code>gamma</code> الافتراضية اختلفت بين إصدارات sklearn.", "SVM_SVC_Pulsar.py") \
 + code_card("svr-card", "SVR", "انحدار بثلاث Kernels", "<code>y = sin(X)</code> + noise، مع <code>support_</code>.", "svr",
    "<code>epsilon</code> ما بيجي بـ SVC. و <code>degree/coef0</code> مهمين مع <code>kernel='poly'</code>.", "SVM_SVR.py") \
 + code_card("grid-card", "GridSearchCV", "بحث C و gamma على Iris", "من <code>RBF_SVM Param.py</code>: <code>np.logspace</code> + <code>StratifiedShuffleSplit</code>.", "gridsearch",
    "ملف المحاضرة بيعمل <code>fit_transform</code> على كل البيانات قبل الـ CV (الكود نفسه كاتب «cheating a bit»). استعمل <code>Pipeline</code> بالامتحان.", "SVM_RBF_Params.py") \
 + code_card("grid-safe-card", "نسخة آمنة", "GridSearchCV مع Pipeline", "يمنع leakage لأن الـ scaler بيتدرّب داخل كل fold.", "gridsearch_safe",
    "أسماء البارامترات بتبدل لـ <code>svc__C</code> و <code>svc__gamma</code> بسبب الـ <code>make_pipeline</code>.") \
 + code_card("exam-switch-card", "سطر الحسم", "SVC ولا SVR؟", "مجرّد تذكير منطقي قبل كتابة الموديل.", "exam_switch",
    "ما تعتمد على عدد القيم Unique لوحده؛ اقرأ وصف السؤال. لو قال «price / sin / continuous» → SVR.")

ready_body = """
<div class="callout">
  <b>لو جاك السؤال فجأة:</b> انسخ القالب، بدّل اسم ملف البيانات واسم عمود الهدف، وبعدين اختار <code>SVC</code> أو <code>SVR</code>. للـ Optimization بدّل سطر <code>optimizer = ...</code> فقط. كل القوالب كمان موجودة بملف <a href="__TEMPLATE_URL__" target="_blank" rel="noopener"><code>exam_templates.py</code></a>.
</div>
<div class="ready-grid">
  <div class="ready-card"><span class="step-tag">قالب A</span><h3>مسألة SVM / SVC / SVR</h3><ul><li><code>train_test_split</code></li><li><code>StandardScaler</code></li><li><code>SVC(...)</code> أو <code>SVR(...)</code></li><li><code>fit → predict → metric</code></li></ul><a class="btn tiny" href="#pipeline-card">افتح القالب</a></div>
  <div class="ready-card"><span class="step-tag">قالب B</span><h3>مسألة Optimizer</h3><ul><li>الإطار المشترك</li><li>كلاس الـ optimizer المطلوب</li><li><code>pre → update → post</code></li><li>Loss / Accuracy</li></ul><a class="btn tiny" href="#nnfs-frame-card">افتح الإطار</a></div>
  <div class="ready-card"><span class="step-tag">قالب C</span><h3>مسألة Regularization</h3><ul><li>Keras + <code>EarlyStopping</code></li><li>أو PyTorch + <code>Dropout</code></li><li>أو <code>BatchNormalization</code></li><li>validation + eval + تفسير المنحنى</li></ul><a class="btn tiny" href="#early-card">افتح الأمثلة</a></div>
</div>
"""
ready_body = ready_body.replace("__TEMPLATE_URL__", GITHUB_TEMPLATE_URL)

quiz_body = """
<div class="quiz-top noprint">
  <div><h3>١٢ سؤال سريع</h3><p>جاوب وبعدين اضغط «صحّح». الشرح بيطلع تحت كل سؤال.</p></div>
  <div class="quiz-actions"><button type="button" class="btn primary" id="quiz-submit">صحّح</button><button type="button" class="btn" id="quiz-reset">إعادة</button></div>
</div>
<div id="quiz-result" class="quiz-result" aria-live="polite"></div>
<div id="quiz-list" class="quiz-list"></div>
"""

checklist_body = """
<div class="check-head">
  <div><h3> Checklist قبل ما تسلّم</h3><p>العلامات بتنحفظ بالمتصفح، فتقدر ترجعلها قبل الامتحان.</p></div>
  <div class="progress-box"><div class="progress-track"><div id="progress-bar"></div></div><strong id="progress-text">0 / 0</strong></div>
</div>
<div class="checklist" id="checklist"></div>
"""

sources_body = """
<div class="callout">
  <b>المصدر:</b> أكواد ZIP الأصلية اللي بالريبو. نسخها الكاملة موجودة تحت <code>study/sources/</code> بالفرع الحالي على GitHub، وكل رابط بالجدول بيفتح الملف الأصلي مباشرة.
</div>
<div class="table-wrap">
  <table class="compare sources">
    <thead><tr><th>الـ ZIP الأصلي</th><th>الملف داخل الـ ZIP</th><th>النسخة هنا</th><th>الموضوع</th></tr></thead>
    <tbody>
      __ROWS__
    </tbody>
  </table>
</div>
<p class="footnote">الـ PDF-ات <code>practical.pdf</code> و <code>practical2-pdf.pdf</code> شرحوا أساسيات الـ Perceptron / Layer_Dense / Loss / أول فكرة optimization عشوائي؛ هالمراجعة ركّزت على أكواد المحاضرات ٣ و ٤ و ٥ حسب طلبك.</p>
"""

rows = "\n".join(
    "<tr><td><code>" + escape(z) + "</code></td><td dir='ltr'><code>" + escape(orig) +
    "</code></td><td><a href='" + GITHUB_SOURCE_BASE + escape(copy) +
    "' target='_blank' rel='noopener'>" + escape(copy) +
    "</a></td><td>" + escape(topic) + "</td></tr>"
    for z, orig, copy, topic in SOURCES
)
sources_body = sources_body.replace("__ROWS__", rows)

# ---------------------------------------------------------------------------
# Page skeleton
# ---------------------------------------------------------------------------
NAV = [
    ("home", "الخريطة"),
    ("pipeline-section", "القالب"),
    ("optimization", "Optimization"),
    ("regularization", "Regularization"),
    ("svm", "SVM · SVR · SVC"),
    ("ready", "كود جاهز"),
    ("quiz", "تدريب"),
    ("checklist", "Checklist"),
    ("sources", "المصادر"),
]

nav_html = "\n".join(
    '<a href="#' + sid + '" data-nav="' + sid + '">' + label + '</a>'
    for sid, label in NAV
)

checks_html = "\n".join(
    '<label class="check-item"><input type="checkbox" data-check="' + str(i) + '"><span>' + escape(text) + '</span></label>'
    for i, text in enumerate(CHECKS)
)
checklist_body = checklist_body.replace('<div class="checklist" id="checklist"></div>',
                                        '<div class="checklist" id="checklist">' + checks_html + '</div>')

CSS = r"""
:root{
  --bg:#0b1020; --bg2:#111832; --card:#151d38; --card2:#1b2547;
  --ink:#eef3ff; --muted:#a8b4d4; --line:rgba(170,190,255,.16);
  --teal:#4ee0c1; --blue:#7aa8ff; --yellow:#ffd166; --pink:#ff8fab; --red:#ff6b6b;
  --shadow:0 18px 45px rgba(0,0,0,.28); --radius:20px;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth;scroll-padding-top:92px}
body{margin:0;background:radial-gradient(1200px 700px at 85% -10%,#1d2b5c 0%,transparent 60%),radial-gradient(900px 600px at 5% 20%,#153c47 0%,transparent 55%),var(--bg);color:var(--ink);font-family:"Segoe UI",Tahoma,Arial,sans-serif;line-height:1.75;direction:rtl}
body[data-theme="light"]{--bg:#f5f7ff;--bg2:#ffffff;--card:#ffffff;--card2:#f0f4ff;--ink:#17213d;--muted:#5b6786;--line:rgba(30,50,100,.14);--shadow:0 12px 30px rgba(30,50,100,.10);background:radial-gradient(900px 500px at 90% -10%,#dce8ff 0%,transparent 60%),var(--bg)}
a{color:inherit;text-decoration:none}
code{font-family:"Cascadia Code",Consolas,Monaco,"Courier New",monospace;font-size:.92em}
.topbar{position:sticky;top:0;z-index:50;backdrop-filter:blur(18px);background:rgba(11,16,32,.86);border-bottom:1px solid var(--line)}
body[data-theme="light"] .topbar{background:rgba(245,247,255,.88)}
.topbar-inner{max-width:1240px;margin:auto;display:flex;align-items:center;gap:14px;padding:10px 20px}
.brand{display:flex;align-items:center;gap:10px;font-weight:800;white-space:nowrap}
.brand-mark{width:34px;height:34px;border-radius:11px;background:linear-gradient(135deg,var(--teal),var(--blue));display:grid;place-items:center;color:#08101f;font-weight:900}
.brand small{display:block;color:var(--muted);font-weight:500;font-size:11px;line-height:1.2}
nav.tabs{display:flex;gap:4px;overflow-x:auto;flex:1;scrollbar-width:none}
nav.tabs::-webkit-scrollbar{display:none}
nav.tabs a{white-space:nowrap;padding:7px 11px;border-radius:999px;color:var(--muted);font-size:13px;font-weight:650;border:1px solid transparent;transition:.2s}
nav.tabs a:hover,nav.tabs a.active{color:var(--ink);background:var(--card2);border-color:var(--line)}
.tools{display:flex;gap:7px;align-items:center}
.search{width:150px;border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:999px;padding:8px 13px;font:inherit;font-size:13px;outline:none}
.search:focus{border-color:var(--teal);width:190px}
.icon-btn{border:1px solid var(--line);background:var(--card);color:var(--ink);width:36px;height:36px;border-radius:11px;cursor:pointer;font-size:16px;display:grid;place-items:center}
.icon-btn:hover{border-color:var(--teal)}
.wrap{max-width:1240px;margin:auto;padding:30px 20px 80px}
section.topic{scroll-margin-top:90px;margin-bottom:62px}
.section-head{display:flex;align-items:center;gap:16px;margin:0 0 22px}
.section-num{width:48px;height:48px;flex:none;border-radius:16px;display:grid;place-items:center;background:linear-gradient(135deg,rgba(78,224,193,.18),rgba(122,168,255,.18));border:1px solid var(--line);color:var(--teal);font-weight:900;font-size:20px}
.kicker{margin:0;color:var(--teal);font-size:12px;font-weight:800;letter-spacing:.4px}
h1,h2,h3{line-height:1.35;margin:0}
h2{font-size:clamp(24px,3vw,34px)}
h3{font-size:18px}
.hero-grid{display:grid;grid-template-columns:1.08fr .92fr;gap:28px;align-items:stretch}
.hero-copy{padding:22px 6px}
.pill-row{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:18px}
.pill{padding:5px 12px;border:1px solid var(--line);border-radius:999px;color:var(--muted);font-size:12px;font-weight:750;background:var(--card)}
h1{font-size:clamp(32px,5vw,56px);letter-spacing:-.5px}
h1 span{background:linear-gradient(90deg,var(--teal),var(--blue),var(--pink));-webkit-background-clip:text;background-clip:text;color:transparent}
.lead{color:var(--muted);font-size:16px;max-width:680px;margin:18px 0 24px}
.hero-actions{display:flex;gap:10px;flex-wrap:wrap}
.btn{display:inline-flex;align-items:center;justify-content:center;gap:7px;border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:12px;padding:10px 16px;font:inherit;font-size:14px;font-weight:700;cursor:pointer;transition:.2s}
.btn:hover{transform:translateY(-1px);border-color:var(--teal)}
.btn.primary{background:linear-gradient(135deg,var(--teal),#59b8ff);color:#08101f;border:0}
.btn.tiny{padding:6px 11px;font-size:12px;border-radius:9px}
.mini-stats{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:30px}
.mini-stats div{background:var(--card);border:1px solid var(--line);border-radius:15px;padding:12px;text-align:center}
.mini-stats strong{display:block;font-size:22px;color:var(--yellow)}
.mini-stats span{font-size:11px;color:var(--muted)}
.board{background:linear-gradient(160deg,#f4f7fb,#dfe7f0);border-radius:var(--radius);padding:18px;color:#14203b;box-shadow:var(--shadow);border:6px solid #cfd8e6;position:relative;overflow:hidden;transform:rotate(-.4deg)}
.board:before{content:"";position:absolute;inset:0;background:repeating-linear-gradient(0deg,transparent 0 38px,rgba(30,60,100,.035) 39px 40px);pointer-events:none}
.board-header{display:flex;justify-content:space-between;align-items:center;font-size:12px;font-weight:800;color:#51617f;border-bottom:2px solid #cbd5e3;padding-bottom:10px;margin-bottom:8px;position:relative}
.board-dots{letter-spacing:5px;color:#ff7a7a}
.steps{list-style:none;margin:0;padding:0;position:relative}
.steps li{display:flex;gap:12px;align-items:flex-start;padding:9px 4px;border-bottom:1px dashed rgba(40,60,90,.16)}
.steps li:last-child{border-bottom:0}
.steps b{width:28px;height:28px;flex:none;border:2px solid #e06a6a;color:#d55d5d;border-radius:50%;display:grid;place-items:center;font-family:Georgia,serif}
.steps strong{display:block;font-family:"Segoe Print","Comic Sans MS",Tahoma,sans-serif;font-size:15px;letter-spacing:.2px}
.steps code{display:inline-block;direction:ltr;text-align:left;font-size:10.5px;color:#43577a;background:rgba(255,255,255,.65);padding:2px 6px;border-radius:6px;margin-top:2px}
.summary-strip{margin-top:28px;background:var(--card);border:1px solid var(--line);border-radius:var(--radius);padding:20px;box-shadow:var(--shadow)}
.summary-strip h3{margin-bottom:14px;color:var(--yellow)}
.summary-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
.summary-grid div{display:flex;gap:10px;align-items:flex-start;background:var(--card2);border:1px solid var(--line);border-radius:13px;padding:10px 12px}
.summary-grid span{color:var(--teal);font-weight:900;font-size:12px;flex:none}
.summary-grid p{margin:0;font-size:13px;color:var(--muted)}
.summary-grid b,.summary-grid code{color:var(--ink)}
.callout{background:linear-gradient(90deg,rgba(78,224,193,.10),rgba(122,168,255,.07));border:1px solid var(--line);border-right:4px solid var(--teal);border-radius:14px;padding:14px 17px;color:var(--muted);margin-bottom:18px}
.callout b,.callout code{color:var(--ink)}
.callout.accent{border-right-color:var(--blue)}
.callout.warning{border-right-color:var(--yellow)}
.card{background:var(--card);border:1px solid var(--line);border-radius:var(--radius);margin:16px 0;box-shadow:var(--shadow);overflow:hidden}
.card-head{display:flex;justify-content:space-between;gap:14px;align-items:flex-start;padding:16px 18px 12px;border-bottom:1px solid var(--line)}
.card-title p{margin:5px 0 0;color:var(--muted);font-size:13px}
.eyebrow{font-size:11px;font-weight:850;color:var(--teal);letter-spacing:.3px}
.card-actions{display:flex;align-items:center;gap:8px;flex-wrap:wrap;justify-content:flex-end}
.source-link{font-size:12px;color:var(--blue);border-bottom:1px dashed var(--blue);padding-bottom:1px;white-space:nowrap}
.code-wrap{position:relative;background:#0a0f20;border-top:1px solid rgba(255,255,255,.04);border-bottom:1px solid rgba(255,255,255,.04);overflow:auto}
body[data-theme="light"] .code-wrap{background:#101731}
pre{margin:0;padding:18px 20px;direction:ltr;text-align:left;overflow:auto;max-height:560px;tab-size:4}
pre code{color:#dbe6ff;font-size:13px;line-height:1.65;white-space:pre;display:block}
.tok-k{color:#ff9ecd;font-weight:700}.tok-s{color:#a8e6a1}.tok-c{color:#7d8bab;font-style:italic}.tok-n{color:#ffd166}.tok-f{color:#7fd7ff}
.notes{display:flex;gap:12px;padding:13px 18px 16px;color:var(--muted);font-size:13px;background:linear-gradient(90deg,rgba(255,209,102,.06),transparent)}
.notes-label{flex:none;background:rgba(255,209,102,.14);color:var(--yellow);border-radius:8px;padding:2px 9px;font-weight:850;height:fit-content}
.notes code{color:var(--teal)}
.table-wrap{overflow-x:auto;border:1px solid var(--line);border-radius:16px;background:var(--card);margin:16px 0}
table{border-collapse:collapse;width:100%;min-width:700px;font-size:13px}
th,td{padding:12px 14px;text-align:right;border-bottom:1px solid var(--line);vertical-align:top}
th{background:var(--card2);color:var(--teal);font-size:12px;letter-spacing:.2px}
tr:last-child td{border-bottom:0}
td code{color:var(--yellow)}
.formula-strip{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:16px 0}
.formula-strip div{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:12px}
.formula-strip span{display:block;color:var(--muted);font-size:11px;font-weight:800;margin-bottom:6px}
.formula-strip code{color:var(--teal);font-size:12px;display:block;text-align:left;direction:ltr}
.reg-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:16px 0}
.reg-card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:16px}
.reg-icon{display:grid;place-items:center;width:38px;height:38px;border-radius:12px;background:rgba(78,224,193,.12);color:var(--teal);font-size:20px;margin-bottom:10px}
.reg-card p{color:var(--muted);font-size:13px;margin:6px 0}
.reg-card code{display:block;direction:ltr;text-align:left;color:var(--yellow);font-size:11px;margin-top:10px}
.decision{background:var(--card);border:1px solid var(--line);border-radius:var(--radius);padding:18px;margin-bottom:16px}
.decision h3{margin-bottom:14px}
.decision-tree{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;align-items:stretch}
.root{background:linear-gradient(135deg,var(--teal),var(--blue));color:#08101f;border-radius:14px;padding:14px;display:grid;place-items:center;font-weight:900;text-align:center}
.branch{border:1px solid var(--line);border-radius:14px;padding:14px;background:var(--card2)}
.branch span{display:block;color:var(--muted);font-size:12px}
.branch b{display:block;color:var(--yellow);font-size:20px;margin:4px 0}
.branch small{color:var(--teal);font-size:11px;direction:ltr;display:block;text-align:right}
.cg-matrix{background:var(--card);border:1px solid var(--line);border-radius:var(--radius);padding:18px;margin:16px 0}
.cg-matrix h3{margin-bottom:12px}
.matrix{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
.matrix div{border:1px solid var(--line);border-radius:13px;padding:12px;background:var(--card2)}
.matrix span{display:block;color:var(--teal);font-size:11px;font-weight:800;margin-bottom:6px}
.matrix p{margin:0;color:var(--muted);font-size:12px}
.matrix .hot{border-color:rgba(255,107,107,.55);background:rgba(255,107,107,.08)}
.matrix .hot span{color:var(--red)}
.ready-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:16px}
.ready-card{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:18px}
.step-tag{font-size:11px;color:var(--pink);font-weight:850}
.ready-card h3{margin:8px 0}
.ready-card ul{padding-right:18px;color:var(--muted);font-size:13px;margin:10px 0 16px}
.ready-card li{margin:4px 0}
.ready-card code{color:var(--teal)}
.quiz-top{display:flex;justify-content:space-between;gap:16px;align-items:center;background:var(--card);border:1px solid var(--line);border-radius:18px;padding:16px 18px;margin-bottom:14px}
.quiz-top p{margin:4px 0 0;color:var(--muted);font-size:13px}
.quiz-actions{display:flex;gap:8px}
.quiz-result{margin-bottom:14px;font-weight:850;color:var(--yellow);min-height:20px}
.quiz-list{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}
.quiz-card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:16px}
.q-num{color:var(--teal);font-size:11px;font-weight:900}
.quiz-card h4{margin:6px 0 10px;font-size:15px;line-height:1.5}
.opt{display:flex;gap:9px;align-items:flex-start;padding:7px 9px;border:1px solid transparent;border-radius:10px;cursor:pointer;font-size:13px;color:var(--muted)}
.opt:hover{background:var(--card2)}
.opt input{margin-top:5px;accent-color:var(--teal)}
.opt.correct{background:rgba(78,224,193,.12);border-color:rgba(78,224,193,.45);color:var(--ink)}
.opt.wrong{background:rgba(255,107,107,.10);border-color:rgba(255,107,107,.45);color:var(--ink)}
.explain{display:none;margin-top:10px;padding:10px;border-radius:10px;background:var(--card2);color:var(--muted);font-size:12px;border-right:3px solid var(--yellow)}
.explain.show{display:block}
.check-head{display:flex;justify-content:space-between;gap:18px;align-items:center;background:var(--card);border:1px solid var(--line);border-radius:18px;padding:16px 18px;margin-bottom:14px}
.check-head p{margin:4px 0 0;color:var(--muted);font-size:13px}
.progress-box{display:flex;align-items:center;gap:10px;min-width:190px}
.progress-track{flex:1;height:9px;border-radius:99px;background:var(--card2);overflow:hidden;border:1px solid var(--line)}
#progress-bar{height:100%;width:0;background:linear-gradient(90deg,var(--teal),var(--blue));transition:.3s}
#progress-text{font-size:12px;color:var(--yellow);white-space:nowrap}
.checklist{display:grid;grid-template-columns:repeat(2,1fr);gap:9px}
.check-item{display:flex;gap:10px;align-items:flex-start;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:11px 13px;cursor:pointer;font-size:13px;color:var(--muted)}
.check-item input{margin-top:5px;accent-color:var(--teal);width:16px;height:16px}
.check-item:has(input:checked){border-color:rgba(78,224,193,.45);color:var(--ink);text-decoration:line-through;text-decoration-color:rgba(78,224,193,.5)}
.footnote{color:var(--muted);font-size:12px;margin-top:14px}
.footer{text-align:center;color:var(--muted);font-size:12px;border-top:1px solid var(--line);padding-top:20px;margin-top:50px}
.search-hidden{display:none!important}
body.exam-mode .notes,body.exam-mode .callout,body.exam-mode .summary-strip,body.exam-mode .reg-grid,body.exam-mode .cg-matrix,body.exam-mode .decision,body.exam-mode .formula-strip{display:none!important}
body.exam-mode .card{border-color:rgba(78,224,193,.35)}
body.exam-mode pre{max-height:360px}
@media(max-width:900px){
  .hero-grid,.ready-grid,.reg-grid,.decision-tree,.summary-grid,.formula-strip,.matrix,.quiz-list,.checklist{grid-template-columns:1fr}
  .board{transform:none}
  .topbar-inner{flex-wrap:wrap}
  nav.tabs{order:3;width:100%;flex-basis:100%}
  .search{width:120px}
  .mini-stats{grid-template-columns:repeat(2,1fr)}
  .check-head{flex-direction:column;align-items:stretch}
  .quiz-top{flex-direction:column;align-items:stretch}
}
@media(max-width:560px){.wrap{padding:20px 12px 60px}.topbar-inner{padding:8px 12px}.brand small{display:none}.tools{gap:5px}.search{width:95px}.search:focus{width:130px}.card-head{flex-direction:column}.card-actions{justify-content:flex-start}}
@media print{
  :root{--bg:#fff;--card:#fff;--card2:#f6f8ff;--ink:#111;--muted:#444;--line:#ccc;--shadow:none}
  body{background:#fff;color:#111}
  .topbar,.noprint,.hero-actions,.tools,.card-actions,.quiz-actions{display:none!important}
  .wrap{max-width:none;padding:0}
  .card,.summary-strip,.board{box-shadow:none;break-inside:avoid;page-break-inside:avoid}
  .board{background:#fff;color:#111;border:2px solid #bbb;transform:none}
  .code-wrap{background:#f4f6fb}
  pre{white-space:pre-wrap;max-height:none}
  pre code{color:#111}
  .tok-k,.tok-s,.tok-c,.tok-n,.tok-f{color:#111!important;font-weight:normal;font-style:normal}
  section.topic{margin-bottom:26px}
  details{display:block}
}
"""

JS = r"""
const QUIZ = __QUIZ__;
const CHECK_COUNT = __CHECK_COUNT__;

function esc(s){return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
const KEYWORDS = new Set(['import','from','as','class','def','return','for','in','if','else','elif','not','and','or','with','lambda','True','False','None','self','super','range','print','len','np','torch','tf','plt','pd','is','hasattr','mean','argmax','max','sum','sqrt','exp','clip','log','zeros_like','copy','reshape','astype','ravel','sort','rand','randn','sin','unique','all','int','float','str','dict','enumerate','zip','range','step','fit','predict','score','compile','evaluate','forward','backward','update_params','pre_update_params','post_update_params','train','eval','zero_grad','item','no_grad','load_data','train_test_split','accuracy_score','classification_report','mean_squared_error','r2_score','make_pipeline','fit_transform','transform','setdiff1d','arange','meshgrid','contourf','scatter','show','logspace','support_','best_params_','best_score_']);
const FUNCS = new Set(['SVC','SVR','StandardScaler','GridSearchCV','StratifiedShuffleSplit','EarlyStopping','Dropout','BatchNormalization','Conv2d','MaxPool2d','Linear','CrossEntropyLoss','SGD','Adam','Adagrad','RMSprop','Sequential','Dense','Flatten','Input','Activation','ToTensor','Normalize','Compose','DataLoader','random_split','load_breast_cancer','load_iris','spiral_data','np','torch','tf','optim','nn','model','optimizer','scaler','clf','grid','history']);
function highlightPython(el){
  if(el.dataset.raw) return;
  const src = el.textContent;
  el.dataset.raw = src;
  const re = /("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|#.*$|\b\d+(?:\.\d+)?(?:e[-+]?\d+)?\b|\b[A-Za-z_][A-Za-z0-9_]*\b)/gm;
  let out = '', last = 0, m;
  while((m = re.exec(src)) !== null){
    out += esc(src.slice(last, m.index));
    const t = m[0];
    let cls = '';
    if(t.startsWith('#')) cls = 'c';
    else if(t[0] === '"' || t[0] === "'") cls = 's';
    else if(/^\d/.test(t)) cls = 'n';
    else if(KEYWORDS.has(t)) cls = 'k';
    else if(FUNCS.has(t)) cls = 'f';
    out += cls ? '<span class="tok-'+cls+'">'+esc(t)+'</span>' : esc(t);
    last = re.lastIndex;
  }
  out += esc(src.slice(last));
  el.innerHTML = out;
}
document.querySelectorAll('code.python').forEach(highlightPython);

document.addEventListener('click', async (e) => {
  const btn = e.target.closest('.copy-btn');
  if(!btn) return;
  const el = document.getElementById('code-' + btn.dataset.copy);
  const text = el ? (el.dataset.raw || el.textContent) : '';
  try { await navigator.clipboard.writeText(text); }
  catch(err){
    const ta = document.createElement('textarea');
    ta.value = text; document.body.appendChild(ta); ta.select();
    try{document.execCommand('copy');}catch(_){}
    ta.remove();
  }
  const old = btn.textContent; btn.textContent = 'تم ✓'; btn.classList.add('copied');
  setTimeout(()=>{btn.textContent = old; btn.classList.remove('copied');}, 1200);
});

const search = document.getElementById('search');
if(search){
  search.addEventListener('input', () => {
    const q = search.value.trim().toLocaleLowerCase('ar');
    document.querySelectorAll('[data-search]').forEach(el => {
      const hit = !q || el.textContent.toLocaleLowerCase('ar').includes(q);
      el.classList.toggle('search-hidden', !hit);
    });
    document.querySelectorAll('section.topic').forEach(sec => {
      const cards = sec.querySelectorAll('[data-search]');
      if(cards.length){
        const visible = [...cards].some(c => !c.classList.contains('search-hidden'));
        sec.classList.toggle('search-hidden', q && !visible);
      }
    });
  });
}

document.getElementById('theme-btn').addEventListener('click', () => {
  const light = document.body.dataset.theme === 'light';
  document.body.dataset.theme = light ? 'dark' : 'light';
  document.getElementById('theme-btn').textContent = light ? '☾' : '☀';
});
document.getElementById('print-btn').addEventListener('click', () => window.print());
document.getElementById('exam-mode').addEventListener('click', (e) => {
  document.body.classList.toggle('exam-mode');
  e.target.textContent = document.body.classList.contains('exam-mode') ? 'وضع الشرح' : 'وضع الحفظ';
});

const navLinks = [...document.querySelectorAll('nav.tabs a')];
const obs = new IntersectionObserver(entries => {
  entries.forEach(en => {
    if(en.isIntersecting){
      navLinks.forEach(a => a.classList.toggle('active', a.dataset.nav === en.target.id));
    }
  });
}, {rootMargin:'-100px 0px -65% 0px', threshold:0});
document.querySelectorAll('section.topic').forEach(s => obs.observe(s));

const list = document.getElementById('quiz-list');
QUIZ.forEach((item, i) => {
  const opts = item.options.map((o, j) =>
    '<label class="opt" data-opt="'+j+'"><input type="radio" name="q'+i+'" value="'+j+'"><span>'+esc(o)+'</span></label>'
  ).join('');
  const card = document.createElement('div');
  card.className = 'quiz-card';
  card.innerHTML = '<span class="q-num">سؤال '+(i+1)+'</span><h4>'+esc(item.q)+'</h4><div class="opts">'+opts+'</div><div class="explain"><b>الصح: '+esc(item.options[item.answer])+'</b><br>'+esc(item.why)+'</div>';
  list.appendChild(card);
});
document.getElementById('quiz-submit').addEventListener('click', () => {
  let score = 0;
  [...list.children].forEach((card, i) => {
    const picked = card.querySelector('input:checked');
    const pick = picked ? Number(picked.value) : -1;
    card.querySelectorAll('.opt').forEach((o, j) => {
      o.classList.remove('correct','wrong');
      if(j === QUIZ[i].answer) o.classList.add('correct');
      if(j === pick && pick !== QUIZ[i].answer) o.classList.add('wrong');
    });
    card.querySelector('.explain').classList.add('show');
    if(pick === QUIZ[i].answer) score++;
  });
  const msg = score === QUIZ.length ? 'ممتاز! جاهز للامتحان 🎯' : score >= 9 ? 'منيح، بس راجع اللي غلط.' : 'رجّع الأقسام وركز على الشروحات الصغيرة.';
  document.getElementById('quiz-result').textContent = 'النتيجة: '+score+' / '+QUIZ.length+' — '+msg;
});
document.getElementById('quiz-reset').addEventListener('click', () => {
  list.querySelectorAll('input').forEach(i => i.checked = false);
  list.querySelectorAll('.opt').forEach(o => o.classList.remove('correct','wrong'));
  list.querySelectorAll('.explain').forEach(e => e.classList.remove('show'));
  document.getElementById('quiz-result').textContent = '';
});

const checks = [...document.querySelectorAll('[data-check]')];
let saved = [];
try { saved = JSON.parse(localStorage.getItem('examChecklist') || '[]'); } catch(_){}
function saveChecks(){
  const values = checks.map(c => c.checked);
  try { localStorage.setItem('examChecklist', JSON.stringify(values)); } catch(_){}
  const done = values.filter(Boolean).length;
  document.getElementById('progress-bar').style.width = (100 * done / CHECK_COUNT) + '%';
  document.getElementById('progress-text').textContent = done + ' / ' + CHECK_COUNT;
}
checks.forEach((c, i) => { c.checked = !!saved[i]; c.addEventListener('change', saveChecks); });
saveChecks();
"""

PAGE = r"""<!doctype html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark light">
<title>مذاكرة عملي AI — Optimization · Regularization · SVM</title>
<style>__CSS__</style>
</head>
<body data-theme="dark">
<header class="topbar noprint">
  <div class="topbar-inner">
    <a class="brand" href="#home"><span class="brand-mark">AI</span><span>Exam Review<small>بكرا عملي · خليني أركز</small></span></a>
    <nav class="tabs">__NAV__</nav>
    <div class="tools">
      <input id="search" class="search" type="search" placeholder="ابحث بالكود…" aria-label="بحث">
      <button id="theme-btn" class="icon-btn" type="button" title="تبديل الثيم">☀</button>
      <button id="print-btn" class="icon-btn" type="button" title="طباعة">⎙</button>
    </div>
  </div>
</header>

<main class="wrap">
  <section class="topic" id="home">__HERO__</section>
  __SECTIONS__
  <footer class="footer">مبنية من ملفات المحاضرة الموجودة بالريبو · كل الكود بدون مكتبات خارجية للصفحة · بالتوفيق بالامتحان ✦</footer>
</main>
<script>__JS__</script>
</body>
</html>
"""

sections = "\n".join([
    section("pipeline-section", "01", "القالب الذهبي", "Whiteboard → Code", pipeline_body),
    section("optimization", "02", "Optimization", "SGD · Momentum · AdaGrad · RMSprop · Adam", optimization_body),
    section("regularization", "03", "Regularization", "Early Stopping · Dropout · BatchNorm", regularization_body),
    section("svm", "04", "SVM · SVR · SVC", "Kernels · C / gamma / epsilon · GridSearchCV", svm_body),
    section("ready", "05", "كود جاهز للامتحان", "انسخ → بدّل → نفّذ", ready_body),
    section("quiz", "06", "تدريب سريع", "أسئلة اختيار مع شرح", quiz_body),
    section("checklist", "07", "قبل ما تسلّم", "Checklist محفوظة بالمتصفح", checklist_body),
    section("sources", "08", "المصادر", "Original .py files", sources_body),
])

html = (PAGE
        .replace("__CSS__", CSS)
        .replace("__NAV__", nav_html)
        .replace("__HERO__", hero_body)
        .replace("__SECTIONS__", sections)
        .replace("__JS__", JS)
        .replace("__QUIZ__", json.dumps(QUIZ, ensure_ascii=False))
        .replace("__CHECK_COUNT__", str(len(CHECKS))))

(ROOT / "index.html").write_text(html, encoding="utf-8")
print("wrote", ROOT / "index.html", len(html), "chars")
print("wrote", ROOT / "exam_templates.py", len(TEMPLATE_FILE), "chars")
