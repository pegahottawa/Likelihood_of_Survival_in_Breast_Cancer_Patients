#!/usr/bin/env python
#SBATCH --mem=50G
#SBATCH --time=17-24:00:00

from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFE
import pandas as pd
import numpy as np
import math

path_to_file = '/home/peg/clinical_cleaned_dead_alive.xlsx'
df1 = pd.read_excel(path_to_file)

path_to_file = '/home/peg/gene_ex.xlsx'
df2 = pd.read_excel(path_to_file)
df3 = df2.transpose()

path_to_file = '/home/peg/column.xlsx'  # column.xlsx file contain only column names for about 48804 gene names
column_names = pd.read_excel(path_to_file)

df3.columns = df3.iloc[0]  
df4 = df3.drop(df3.index[[0]])     

df6 = df4.reset_index()              
df6.columns = column_names.iloc[:,0]      
df_merge = pd.merge(df1,df6,on='METABRIC_ID')

NON_value = df_merge.isnull().sum()   
#print NON_value
del_NON = df_merge.dropna(how = 'any') 
#print del_NON

Target = del_NON[del_NON.columns[6]]
#print Tar
Descriptor = del_NON.drop(del_NON.columns[[0,2,6]], axis=1)
arr_descriptors = Descriptor.values
arr_Target = Target.values

feature_lst = list(Descriptor.columns.values)

X = arr_descriptors
T = arr_Target
Targe_reshape = np.reshape(T, (554,)) 
#Target = np.reshape(T, (554,1))
Y = Target_reshape

print 'X.shape =', X.shape
print 'Y.shape =', Y.shape

#######################################
# RFE with SVM and Cross_validation
#######################################

X_train,X_test,y_train,y_test=train_test_split(X, Y, random_state=0)

print 'X_train.shape =', X_train.shape
print 'y_train.shape =', y_train.shape

reg = svm.SVC(kernel='linear', C=0.001,gamma=0.001)
#rfe = RFE(estimator = logreg , step=1)
#rfe.fit(X, Y)

rfecv = RFECV(estimator= reg, step=1, cv=StratifiedKFold(4), scoring='accuracy')
rfecv.fit(X, Y)

print " time for Ranking "
print("Num Features: %s" % (rfecv.n_features_))

bestfeature = []
for feature in zip(feature_lst, rfecv.ranking_):
   bestfeature.append(feature)
print "done appending"

Output = [item for item in bestfeature if item[1] == 1]
print(Output)

X_train_rfecv = rfecv.transform(X_train)
X_test_rfecv = rfecv.transform(X_test)

rfecv_model = reg.fit(X_train_rfecv ,y_train)
y_pred=reg.predict(X_test_rfecv)

print(reg.score(X_test_rfecv, y_test))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print 'rmse =', rmse


