# feature selection

logreg = LogisticRegression()
rfe = RFE(logreg, n_features_to_select=15)
rfe = rfe.fit(X_sm, Y_sm)
print(rfe.support_)
print(rfe.ranking_)

# building

logit_model=sm.Logit(Y_sm,X_selected)
result=logit_model.fit()
print(result.summary2())

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

# result of cols
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# predict enrolled patients

pred = df_enroll[cols]
logreg.predict(pred)

# give probability of each class

logreg.predict_proba(pred)
