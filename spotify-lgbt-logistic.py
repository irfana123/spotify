#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 19:02:24 2021

@author: irfana
"""
import pandas as pd
from datetime import datetime 
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score



#importing dataset 
dataset = pd.read_csv("spotifyfile.csv")

fc = dataset.pop('skipped')
dataset.insert(0, 'skipped', fc)

x=dataset.drop('skipped',axis=1)
y=dataset['skipped'].values

#feature scaling

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)
y=y.reshape(-1,1)

# apply PCA on features
'''
from sklearn.decomposition import PCA
pca=PCA(n_components=35)
X=pca.fit_transform(x)

explained_variance=pca.explained_variance_ratio_

sum(explained_variance)

'''
# splitting the dataset

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

x_train.shape,y_train.shape

#model fitting
# 1.RandomForestClassifier


RFC = RandomForestClassifier(random_state = 1).fit(x_train, y_train)
pred_RFC = RFC.predict(x_test)
score_RFC = f1_score(y_test, pred_RFC)
print('RandomForestClassifier: ',score_RFC)


#2. XGBClassifier
XGBC = XGBClassifier(random_state = 1).fit(x_train, y_train)
pred_XGBC = XGBC.predict(x_test)
score_XGBC = f1_score(y_test, pred_XGBC)
print('XGBClassifier: ',score_XGBC)

#3.Descision tree classifier
depth = []
score_DTC = [] 
for i in range(1,21):
    DTC = DecisionTreeClassifier(criterion="entropy", max_depth = i, random_state = 1).fit(x_train, y_train)
    pred_DTC = DTC.predict(x_test) 
    score = f1_score(y_test, pred_DTC)
    depth.append(i)
    score_DTC.append(score)
    
print('We get maximum f1_score {} for DecisionTreeClassifier when max_depth = {}.'.format
      (max(score_DTC), depth[score_DTC.index(max(score_DTC))])  )

DTC = DecisionTreeClassifier(criterion="entropy", max_depth = 10, random_state = 1).fit(x_train, y_train)
pred_DTC = DTC.predict(x_test)
score_DTC = f1_score(y_test, pred_DTC)
print('DecisionTreeClassifier: ',score_DTC)



#4.Logistic Regression

ml = LogisticRegression()
ml.fit(x_train, y_train)

#Predicted result
y_pred =ml.predict(x_test)

#confusion matrix

from sklearn.metrics import confusion_matrix
cm =confusion_matrix(y_test,y_pred)



# accuracy score,recall,precision
from sklearn.metrics import accuracy_score,classification_report

accuracy_score(y_test,y_pred)
cr = classification_report(y_test,y_pred)


# training the dataset with light gbm mdel


d_train = lgb.Dataset(x_train, label=y_train)


lgbm_params = {'learning_rate':0.02, 'boosting_type':'dart',    #Try dart for better accuracy
              'objective':'binary',
              'metric':['auc', 'binary_logloss'],
              'num_leaves':100,
              'max_depth':50}

start=datetime.now()
clf = lgb.train(lgbm_params, d_train, 120) #100 iterations. Increase iterations for small learning rates
stop=datetime.now()
execution_time_lgbm = stop-start
print("LGBM execution time is: ", execution_time_lgbm)



#Prediction on test data
y_pred_lgbm=clf.predict(x_test)

for i in range(0, x_test.shape[0]):
    if y_pred_lgbm[i]>=.5:       # setting threshold to .5
       y_pred_lgbm[i]=1
    else:  
       y_pred_lgbm[i]=0
 

# confusion metrix,accuracy score and classification report of the model 
#confusion matrix

from sklearn.metrics import confusion_matrix


# accuracy score,recall,precision
from sklearn.metrics import accuracy_score,classification_report


cm1 =confusion_matrix(y_test,y_pred_lgbm)
accuracy_score(y_test,y_pred_lgbm)
cr = classification_report(y_test,y_pred_lgbm)

#pickling the best model for deployment


model_pkl_filename = 'spotify_model.pkl'
model_pkl = open(model_pkl_filename, 'wb')
pickle.dump(clf, model_pkl)
model_pkl.close()

