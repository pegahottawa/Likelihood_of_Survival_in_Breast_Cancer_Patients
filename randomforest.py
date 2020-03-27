#!/usr/bin/env python
from sklearn.tree import export_graphviz
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import graphviz
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import math

path_to_file = '/home/pegah/project_discovery/clinical_cleaned_dead_alive.xlsx'
df1 = pd.read_excel(path_to_file)

path_to_file = '/home/pegah/project_discovery/gene_ex.xlsx'
df2 = pd.read_excel(path_to_file)

df3 = df2.transpose()


path_to_file = '/home/pegah/project_discovery/column.xlsx'  # column.xlsx file contain only column names for about 48804 gene names
df100 = pd.read_excel(path_to_file)

df3.columns = df3.iloc[0]  # after trasposing a df2 column names become the first row ,thats why I renamed the column names
                           # with the firts row
df4 = df3.drop(df3.index[[0]])     # after that because by renaming I would end up with same
                                   # column names and and same row values,I droped the first row

df6=df4.reset_index()              # I wanted to have dataframe format with all index for each row
df6.columns = df100.iloc[:,0]      # After all, now its time to rename columns
result = pd.merge(df1,df6,on='METABRIC_ID')


NON_value = result.isnull().sum()   # how many NON value we have in each column
del_NON = result.dropna(how = 'any')  # drop row if you find any NON value in it

Tar = del_NON[del_NON.columns[6]]
Descriptor = del_NON.drop(del_NON.columns[[0,2,6]], axis=1)
arr_descriptors=Descriptor.values
arr_Target=Tar.values


X = arr_descriptors
T = arr_Target
Target = np.reshape(T, (533,))
Y = Target

print 'X.shape =', X.shape
print 'Y.shape =', Y.shape


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3,random_state=0)

n_estimators = [int(x) for x in np.linspace(start = 1, stop = 100, num = 10)]

# Number of features to consider at every split
max_features = [ 'log2','sqrt', 'auto']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(1, 70, num = 10)]

# Minimum number of samples required to split a node
min_samples_split = [2, 3, 5 ]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4 ]

# Method of selecting samples for training each tree
bootstrap = [ False, True]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,cv = 3,
                               n_iter = 100,  random_state=42, n_jobs = -1)

# Fit the random search model
#rf_random.fit(X_train, y_train)
#y_pred = rf_random.predict(X_test)
#print "Accuracy test ", accuracy_score(y_pred, y_test) * 100
scores =cross_val_score( rf_random, X_train, y_train, cv=3)
print scores
print("Average 3-Fold CV Score: {}".format(np.mean(scores)))
print (rf_random.best_params_)


