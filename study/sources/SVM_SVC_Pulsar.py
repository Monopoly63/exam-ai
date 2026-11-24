import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#Seaborn is a Python data visualization library based on matplotlib.
#It provides a high-level interface for drawing attractive and informative statistical graphics.
import warnings #situations that may need attention but do not normally stop program execution

warnings.filterwarnings('ignore')
#data = 'C:/datasets/pulsar_stars.csv'
data = 'pulsar_stars.csv'
df = pd.read_csv(data)#read data file
#Now, I will explore the data to gain insights about the data.
# view dimensions of dataset

df.shape
# let's preview the dataset

df.head()
# view the column names of the dataframe

col_names = df.columns

col_names
#We can see that there are leading spaces (spaces at the start of the string name)
#in the dataframe. So, I will remove these leading spaces
# remove leading spaces from column names
df.columns = df.columns.str.strip()
#We have removed the leading spaces from the column names. Let's again view the column names to confirm the same.
# view column names again
df.columns
#We can see that the leading spaces are removed from the column name. But the column names are very long
#So, we will make them short by renaming them.
# rename column names

df.columns = ['IP Mean', 'IP Sd', 'IP Kurtosis', 'IP Skewness', 
              'DM-SNR Mean', 'DM-SNR Sd', 'DM-SNR Kurtosis', 'DM-SNR Skewness', 'target_class']
# view the renamed column names

df.columns
#We can see that the column names are shortened. IP stands for integrated profile and DM-SNR stands for delta modulation and signal to noise ratio.
#Now, it is much more easy to work with the columns.
#Our target variable is the target_class column. So, I will check its distribution.

# check distribution of target_class column
df['target_class'].value_counts()
# view the percentage distribution of target_class column

#df['target_class'].value_counts()/np.float(len(df))
# view summary of dataset

df.info()
# check for missing values in variables

df.isnull().sum()
# view summary statistics in numerical variables

round(df.describe(),2)
# draw boxplots to visualize outliers

plt.figure(figsize=(24,20))


plt.subplot(4, 2, 1)
fig = df.boxplot(column='IP Mean') # 'IP Sd' 'IP Kurtosis' 'IP Skewness' 'DM-SNR Mean' 'DM-SNR Sd' 'DM-SNR Kurtosis' 'DM-SNR Skewness'
fig.set_title('')
fig.set_ylabel('IP Mean') # 'IP Sd' 'IP Kurtosis' 'IP Skewness' 'DM-SNR Mean' 'DM-SNR Sd' 'DM-SNR Kurtosis' 'DM-SNR Skewness'

# plot histogram to check distribution
plt.figure(figsize=(24,20))

plt.subplot(4, 2, 1)
fig = df['IP Mean'].hist(bins=20)
#'IP Sd' 'IP Kurtosis' 'IP Skewness' 'DM-SNR Mean' 'DM-SNR Sd' 'DM-SNR Kurtosis' 'DM-SNR Skewness'
fig.set_xlabel('IP Mean')
#'IP Sd' 'IP Kurtosis' 'IP Skewness' 'DM-SNR Mean' 'DM-SNR Sd' 'DM-SNR Kurtosis' 'DM-SNR Skewness'
fig.set_ylabel('Number of pulsar stars')

#9. Declare feature vector and target variable
X = df.drop(['target_class'], axis=1)

y = df['target_class']
#10. Split data into separate training and test set
# split X and y into training and testing sets

from sklearn.model_selection import train_test_split
#used to split arrays or matrices into random train and test subsets.
#It is a fundamental step in machine learning workflows to evaluate how well your model generalizes to unseen data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#splits your features (X) and target labels (y) into a training set and a testing set with specific configurations.
# check the shape of X_train and X_test

print(X_train.shape, X_test.shape)
#11. Feature Scaling
cols = X_train.columns #This line extracts the column names (feature names) from your training feature DataFrame
#(X_train) and stores them in a variable named cols.
from sklearn.preprocessing import StandardScaler
#is a preprocessing class in scikit-learn (sklearn.preprocessing) used to standardize features by
#removing the mean and scaling them to unit variance.
scaler = StandardScaler() #(creates) an object of the StandardScaler class and assigns it to the variable scaler.

X_train = scaler.fit_transform(X_train) #This line computes the scaling parameters from your training features,
#applies the transformation, and overwrites X_train with the newly scaled values

X_test = scaler.transform(X_test)#This line applies the scaling parameters learned from your training set to your
# testing set, overwriting X_test with the newly scaled values.
X_train = pd.DataFrame(X_train, columns=[cols])#This line attempts to convert your scaled NumPy array back into a
#pandas DataFrame using your saved column names (cols).
X_test = pd.DataFrame(X_test, columns=[cols])
X_train.describe()#generates a statistical summary of your training features.
#If you call this after applying the StandardScaler, it is a great sanity check to verify that your scaling worked correctly.

#12. Run SVM with default hyperparameters
#Default hyperparameter means C=1.0, kernel=rbf and gamma=auto among other parameters.

# import SVC classifier
from sklearn.svm import SVC
#used primarily for classification tasks (predicting categories or labels

# import metrics to compute accuracy
from sklearn.metrics import accuracy_score
#is a metric function in scikit-learn (sklearn.metrics) used to evaluate classification models.
#It calculates the fraction of correctly predicted labels out of all the predictions made.

# instantiate classifier with default hyperparameters
svc=SVC() 


# fit classifier to training set
svc.fit(X_train,y_train)


# make predictions on test set
y_pred=svc.predict(X_test)


# compute and print accuracy score
print('Model accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

#Run SVM with rbf kernel and C=100.0
#We have seen that there are outliers in our dataset. So, we should increase the value of C as higher C means fewer outliers.
#So, I will run SVM with kernel=rbf and C=100.0.
# instantiate classifier with rbf kernel and C=100
svc=SVC(C=100.0) 


# fit classifier to training set
svc.fit(X_train,y_train)


# make predictions on test set
y_pred=svc.predict(X_test)


# compute and print accuracy score
print('Model accuracy score with rbf kernel and C=100.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
#Now, I will further increase the value of C=1000.0 and check accuracy.
#Run SVM with rbf kernel and C=1000.0
# instantiate classifier with rbf kernel and C=1000
svc=SVC(C=1000.0) 


# fit classifier to training set
svc.fit(X_train,y_train)


# make predictions on test set
y_pred=svc.predict(X_test)


# compute and print accuracy score
print('Model accuracy score with rbf kernel and C=1000.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
#13. Run SVM with linear kernel
#Run SVM with linear kernel and C=1.0
# instantiate classifier with linear kernel and C=1.0
linear_svc=SVC(kernel='linear', C=1.0) 


# fit classifier to training set
linear_svc.fit(X_train,y_train)


# make predictions on test set
y_pred_test=linear_svc.predict(X_test)


# compute and print accuracy score
print('Model accuracy score with linear kernel and C=1.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))
#Model accuracy score with linear kernel and C=1.0 : 0.9830
#Run SVM with linear kernel and C=100.0
# instantiate classifier with linear kernel and C=100.0
linear_svc100=SVC(kernel='linear', C=100.0) 


# fit classifier to training set
linear_svc100.fit(X_train, y_train)


# make predictions on test set
y_pred=linear_svc100.predict(X_test)


# compute and print accuracy score
print('Model accuracy score with linear kernel and C=100.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
#Run SVM with linear kernel and C=1000.0
# instantiate classifier with linear kernel and C=1000.0
linear_svc1000=SVC(kernel='linear', C=1000.0) 


# fit classifier to training set
linear_svc1000.fit(X_train, y_train)


# make predictions on test set
y_pred=linear_svc1000.predict(X_test)


# compute and print accuracy score
print('Model accuracy score with linear kernel and C=1000.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
#14. Run SVM with polynomial kernel
#Run SVM with polynomial kernel and C=1.0
# instantiate classifier with polynomial kernel and C=1.0
poly_svc=SVC(kernel='poly', C=1.0) 


# fit classifier to training set
poly_svc.fit(X_train,y_train)


# make predictions on test set
y_pred=poly_svc.predict(X_test)


# compute and print accuracy score
print('Model accuracy score with polynomial kernel and C=1.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
#Run SVM with polynomial kernel and C=100.0
# instantiate classifier with polynomial kernel and C=100.0
poly_svc100=SVC(kernel='poly', C=100.0) 


# fit classifier to training set
poly_svc100.fit(X_train, y_train)


# make predictions on test set
y_pred=poly_svc100.predict(X_test)


# compute and print accuracy score
print('Model accuracy score with polynomial kernel and C=1.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
#15. Run SVM with sigmoid kernel
#Run SVM with sigmoid kernel and C=1.0
# instantiate classifier with sigmoid kernel and C=1.0
sigmoid_svc=SVC(kernel='sigmoid', C=1.0) 


# fit classifier to training set
sigmoid_svc.fit(X_train,y_train)


# make predictions on test set
y_pred=sigmoid_svc.predict(X_test)


# compute and print accuracy score
print('Model accuracy score with sigmoid kernel and C=1.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
#Run SVM with sigmoid kernel and C=100.0
# instantiate classifier with sigmoid kernel and C=100.0
sigmoid_svc100=SVC(kernel='sigmoid', C=100.0) 


# fit classifier to training set
sigmoid_svc100.fit(X_train,y_train)


# make predictions on test set
y_pred=sigmoid_svc100.predict(X_test)


# compute and print accuracy score
print('Model accuracy score with sigmoid kernel and C=100.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

