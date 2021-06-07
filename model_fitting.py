import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2 # input must be non-negative
from sklearn.feature_selection import f_classif #ANOVA F-value between label/feature for classification tasks.

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

# data scaling

scaler = MinMaxScaler()
scaled1 = scaler.fit_transform(df1)
df1_scaled = pd.DataFrame(scaled1)
scaled2 = scaler.fit_transform(df2)
df2_scaled = pd.DataFrame(scaled2)
df1_scaled.columns = df1.columns
df2_scaled.columns = df2.columns

# seperate input and output

X1 = df1_scaled.drop('CurrentStatus', axis=1)
Y1 = df1['CurrentStatus']
X2 = df2_scaled.drop('CurrentStatus', axis=1)
Y2 = df2['CurrentStatus']

# undersample

from imblearn.under_sampling import RandomUnderSampler
undersample = RandomUnderSampler(sampling_strategy=0.3)
X1_us, Y1_us =  undersample.fit_resample(X1, Y1)
X2_us, Y2_us =  undersample.fit_resample(X2, Y2)

#SMOTE

sm2 = SMOTE(random_state=42, sampling_strategy=0.666)
X1_sm2, Y1_sm2 = sm2.fit_resample(X1_us, Y1_us)
X2_sm2, Y2_sm2 = sm2.fit_resample(X2_us, Y2_us)

#MODEL

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X2_sm2,Y2_sm2, test_size=0.2)

# parameter tuning pipeline::::

# Random Forest
scoring = {'Recall': 'recall', 'Accuracy': make_scorer(accuracy_score)}

pipe = Pipeline([
    # the reduce_dim stage is populated by the param_grid
    ('reduce_dim', SelectKBest(score_func=chi2)),
    ('classify', RandomForestClassifier(random_state=123))
])

N_FEATURES_OPTIONS = [300,800]
N_OPTIONS = [5, 10, 15, 20, 25]
DEPTH_OPTIONS = [3, 5, 7, 9, 11, 13]

param_grid = {
        'reduce_dim__k': N_FEATURES_OPTIONS,
        'classify__n_estimators': N_OPTIONS,
        'classify__max_depth': DEPTH_OPTIONS
    }


grid = GridSearchCV(pipe, n_jobs=1, param_grid=param_grid, cv=10, scoring=scoring, refit='Recall') #default is 5-fold
grid.fit(X_train, y_train)

print(grid.best_params_)    
print(grid.best_score_)
print(grid.score(X_test, y_test))

#SVM
from sklearn.svm import SVC  
scoring = {'Recall': 'recall', 'Accuracy': make_scorer(accuracy_score)}

pipe = Pipeline([
    # the reduce_dim stage is populated by the param_grid
    ('reduce_dim', SelectKBest(score_func=chi2)),
    ('classify', SVC(random_state=123, kernel='linear'))
])

N_FEATURES_OPTIONS = [200,600,800]
C_OPTIONS = [0.1, 10, 100]
GAMMA_OPTIONS = [1, 0.1, 0.01]

param_grid = {
        'reduce_dim__k': N_FEATURES_OPTIONS,
        'classify__C': C_OPTIONS,
        'classify__gamma': GAMMA_OPTIONS,
    }


grid = GridSearchCV(pipe, n_jobs=1, param_grid=param_grid, cv=10, scoring=scoring, refit='Recall') #default is 5-fold
grid.fit(X_train, y_train)

print(grid.best_params_)    
print(grid.best_score_)
print(grid.score(X_test, y_test))

# report production Random Forest
# including: confusion matrix, feature importance

def rf_report_chi(array1,array2,depth,n,k):
    model = SelectKBest(score_func=chi2,k=k).fit(array1,array2)
    ff = model.get_support(indices=True)
    df_new = array1.iloc[:,ff]
    X_train, X_test, y_train, y_test = train_test_split(df_new,array2, test_size=0.2)
    rf = RandomForestClassifier(n_estimators=n, max_depth=depth, random_state=123)
    fit = rf.fit(X_train,y_train)
    labels = rf.predict(X_test)
    print(classification_report(y_test, labels))
    print(sorted(zip(fit.feature_importances_, df_new.columns), reverse=True)[0:20])
    sns.set(font_scale=1.75)
    cf_matrix_under_1 = confusion_matrix(y_test, labels)
    f, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(cf_matrix_under_1, annot=True, ax=ax, cmap="YlGnBu", fmt=".0f", linewidths=.5)
    plt.title("Confusion matrix ")
    plt.xlabel('Predicted')
    plt.ylabel('Real');
    
    
# report production SVM
def sm_report(array1,array2,c,gamma,kernel,k):
    model = SelectKBest(score_func=chi2,k=k).fit(array1,array2)
    ff = model.get_support(indices=True)
    df_new = array1.iloc[:,ff]
    X_train, X_test, y_train, y_test = train_test_split(df_new,array2, test_size=0.2)
    svc = SVC(C=c, gamma=gamma,kernel=kernel)
    fit = svc.fit(X_train,y_train)
    labels = svc.predict(X_test)
    print(classification_report(y_test, labels))
    print(sorted(zip(abs(svc.coef_.ravel()), df_new.columns), reverse=True)[0:20])
    sns.set(font_scale=1.75)
    cf_matrix_under_1 = confusion_matrix(y_test, labels)
    f, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(cf_matrix_under_1, annot=True, ax=ax, cmap="YlGnBu", fmt=".0f", linewidths=.5)
    plt.title("Confusion matrix ")
    plt.xlabel('Predicted')
    plt.ylabel('Real');

    
# Predict enrolled patients
model = SelectKBest(score_func=chi2,k=1000).fit(X4_sm2,Y4_sm2)
ff = model.get_support(indices=True)
df_new = X4_sm2.iloc[:,ff]
df_enroll = jxzb_enroll_scaled.iloc[:,ff]
X_train, X_test, y_train, y_test = train_test_split(df_new,Y4_sm2, test_size=0.2)
svc = SVC(C=10, gamma=1,kernel='linear')
fit = svc.fit(X_train,y_train)
labels = svc.predict(df_enroll)
labels
