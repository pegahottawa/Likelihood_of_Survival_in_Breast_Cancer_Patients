#!/usr/bin/env python
#SBATCH --mem=20G
#SBATCH --time=15-24:00:00

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

df6=df4.reset_index()              
df6.columns = column_names.iloc[:,0]      # rename columns
df_merge = pd.merge(df1,df6,on='METABRIC_ID')

NON_value = df_merge.isnull().sum()   # shows NON value in each column
#print NON_value
del_NON = df_merge.dropna(how = 'any')  # drop rows with NON value in it
#print del_NON

Target = del_NON[del_NON.columns[6]]
#print Target
Descriptor = del_NON.drop(del_NON.columns[[0,2,6]], axis=1)
arr_descriptors=Descriptor.values
arr_Target=Target.values

feature_lst = list(Descriptor.columns.values)

X = arr_descriptors
T = arr_Target
Target = np.reshape(T, (554,))
Y = Target

print 'X.shape =', X.shape
print 'Y.shape =', Y.shape

##############################################
#validation set
##############################################

path_to_file = '/home/peg/validation_clinical_cleaned.xlsx'
df_v = pd.read_excel(path_to_file)

path_to_file = '/home/peg/validation.xlsx'
df_v1 = pd.read_excel(path_to_file)
df_v2 = df_v1.transpose()

df_v2.columns = df_v2.iloc[0]
df_v3 = df_v2.drop(df_v2.index[[0]])
df_v4=df_v3.reset_index()
df_v4.columns = column_names.iloc[:,0]
Merging_2dataframe = pd.merge(df_v,df_v4,on='METABRIC_ID')
#print 'result.shape =', result.shape

del_NON_val = Merging_2dataframe.dropna(how = 'any')
#print 'del_NON_val.shape =', del_NON_val.shape

Target_val = del_NON_val[del_NON_val.columns[6]]
Test_val = del_NON_val.drop(del_NON_val.columns[[0,2,6]], axis=1)
#print 'Test_val.shape =', Test_val.shape
arr_test_val= Test_val.values
arr_Target_val = Target_val.values

X_val = arr_test_val
Target_val = arr_Target_val
Target_reshape = np.reshape(Target_val, (583,))
Y_val = Target_val

print 'validation:'

print 'Y_val.shape =', Y_val.shape
print 'X_val.shape =', X_val.shape

###################################################################
#Recursive feature elimination with cross-validation(Stratified)
###################################################################

X_train,X_test,y_train,y_test=train_test_split(X, Y, test_size=0.2, random_state=0)

X_test_validation = X_val
y_test_validation = Y_val

reg = svm.SVC(kernel='linear', C=0.001,gamma=0.001)
rfecv = RFECV(estimator= reg, step=1, cv=StratifiedKFold(4), scoring ='accuracy')
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

##########################################
#Validation test
##########################################

X_test_validation_rfecv = rfecv.transform(X_test_validation)

rfecv_model2 = reg.fit(X_train_rfecv ,y_train)
y_pred2=reg.predict(X_test_validation_rfecv)

print(reg.score(X_test_validation_rfecv, y_test_validation))
rmse2 = np.sqrt(mean_squared_error(y_test_validation, y_pred2))
print 'rmse =', rmse2


