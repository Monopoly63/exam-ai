#RBF SVM parameters
#Utility class to move the midpoint of a colormap to be around the values of interest.
import numpy as np
from matplotlib.colors import Normalize


class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
#Load and prepare data set
#dataset for grid search
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target
#Dataset for decision function visualization: we only keep the first two features in X
#and sub-sample the dataset to keep only 2 classes and make it a binary classification problem.
X_2d = X[:, :2]
X_2d = X_2d[y > 0]
y_2d = y[y > 0]
y_2d -= 1
#It is usually a good idea to scale the data for SVM training.
#We are cheating a bit in this example in scaling all of the data,
#instead of fitting the transformation on the training set and just applying it on the test set.
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler() #Standardize features by removing the mean and scaling to unit variance.
X = scaler.fit_transform(X)
X_2d = scaler.fit_transform(X_2d)
#Train classifiers
#For an initial search, a logarithmic grid with basis 10 is often helpful.
#Using a basis of 2, a finer tuning can be achieved but at a much higher cost.
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.svm import SVC

C_range = np.logspace(-2, 10, 13)#Return numbers spaced evenly on a log scale
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)#Create a dictionary for example: x = dict(name = "John", age = 36, country = "Norway")
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
grid.fit(X, y)

print(
    f"The best parameters are {grid.best_params_} "
    f"with a score of {grid.best_score_:0.2f}"
)
#Now we need to fit a classifier for all parameters in the 2d version
#(we use a smaller set of parameters here because it takes a while to train)
C_2d_range = [0.01, 1, 100]
gamma_2d_range = [0.1, 1, 10]
classifiers = []
for C in C_2d_range:
    for gamma in gamma_2d_range:
        clf = SVC(C=C, gamma=gamma)# C:Regularization Parameter , Gamma:fitting level
        clf.fit(X_2d, y_2d)
        classifiers.append((C, gamma, clf))
#Visualization
#We plot the decision boundaries for different values of gamma and C.
import matplotlib.pyplot as plt

from sklearn.inspection import DecisionBoundaryDisplay

plt.figure(figsize=(8, 6))
for k, (C, gamma, clf) in enumerate(classifiers):
    plt.subplot(len(C_2d_range), len(gamma_2d_range), k + 1)
    plt.title(f"gamma={gamma}, C={C}", size="medium")
    # visualize decision boundary for these parameters
    DecisionBoundaryDisplay.from_estimator(
        clf,
        X_2d,
        eps=0.5,
        cmap="RdBu_r",
        ax=plt.gca(),
        response_method="decision_function",
        plot_method="pcolormesh",
    )
    # also plot the data
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap=plt.cm.RdBu_r, edgecolors="k")
plt.xticks(())
plt.yticks(())
plt.show()