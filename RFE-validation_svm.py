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
df100 = pd.read_excel(path_to_file)

df3.columns = df3.iloc[0]  # after trasposing a df2 column names become the first row ,thats why I renamed the column names
                           # with the firts row
df4 = df3.drop(df3.index[[0]])     # after that because by renaming I would end up with same
                                   # column names and and same row values,I droped the first row

df6=df4.reset_index()              # I wanted to have dataframe format with all index for each row
df6.columns = df100.iloc[:,0]      # After all, now its time to rename columns
result = pd.merge(df1,df6,on='METABRIC_ID')

NON_value = result.isnull().sum()   # how many NON value we have in each column
#print NON_value
del_NON = result.dropna(how = 'any')  # drop row if you find any NON value in it
#print del_NON

Tar = del_NON[del_NON.columns[6]]
#print Tar
Descriptor = del_NON.drop(del_NON.columns[[0,2,6]], axis=1)
arr_descriptors=Descriptor.values
arr_Target=Tar.values

feature_lst = list(Descriptor.columns.values)

X = arr_descriptors
T = arr_Target
Target = np.reshape(T, (554,))
Y = Target

print 'X.shape =', X.shape
print 'Y.shape =', Y.shape

##############################################
#validation
##############################################

path_to_file = '/home/peg/validation_clinical_cleaned.xlsx'
dfv1 = pd.read_excel(path_to_file)

path_to_file = '/home/peg/validation.xlsx'
dfv = pd.read_excel(path_to_file)
dfv2 = dfv.transpose()

dfv2.columns = dfv2.iloc[0]
dfv3 = dfv2.drop(dfv2.index[[0]])
dfv4=dfv3.reset_index()
dfv4.columns = df100.iloc[:,0]
Merging_2dataframe = pd.merge(dfv1,dfv4,on='METABRIC_ID')
#print 'result.shape =', result.shape

del_NON_val = Merging_2dataframe.dropna(how = 'any')
#print 'del_NON_val.shape =', del_NON_val.shape

Tar_val = del_NON_val[del_NON_val.columns[6]]
Test_val = del_NON_val.drop(del_NON_val.columns[[0,2,6]], axis=1)
#print 'Test_val.shape =', Test_val.shape
arr_test_val= Test_val.values
arr_Tar_val = Tar_val.values

X_val = arr_test_val
T_val = arr_Tar_val
Target_val = np.reshape(T_val, (583,)) # when using Kfold cv and using Holdout cv method
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

print "Final result recursive svm discovery cleaned "

print(reg.score(X_test_rfecv, y_test))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print 'rmse =', rmse

##########################################
#Validation test
##########################################

X_test_validation_rfecv = rfecv.transform(X_test_validation)

rfecv_model2 = reg.fit(X_train_rfecv ,y_train)
y_pred2=reg.predict(X_test_validation_rfecv)

print "Final result recursive svm validation cleaned "

print(reg.score(X_test_validation_rfecv, y_test_validation))
rmse2 = np.sqrt(mean_squared_error(y_test_validation, y_pred2))
print 'rmse =', rmse2


