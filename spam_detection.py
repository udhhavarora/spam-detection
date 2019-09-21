import pandas as pd 
from sklearn.feature_extraction .text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import svm
from sklearn.model_selection import GridSearchCV

#Load Data
df=pd.read_csv("spam.csv")
print(df.head())

#splitting data into test and train

x=df['EmailText']
y=df['Label']

x_train, y_train=x[0:4457],y[0:4457]
x_test, y_test = x[4457:],y[4457:]

#extracting data
cv=CountVectorizer()
features=cv.fit_transform(x_train)

"""model=svm.SVC()

model.fit(features,y_train)

features_test=cv.fit_transform(x_test)
model.fit(features_test,y_test)

print('Accuracy is: ',model.score(features_test,y_test))
"""
#another model

tuned_params={'kernel':['linear','rbf'],'gamma':[1e-3,1e-4],'C':[1,10,100,1000]}

model2=GridSearchCV(svm.SVC(),tuned_params)
model2.fit(features,y_train)

print(model2.best_params_)

features_test=cv.transform(x_test)
print('New Accuracy is: ',model2.score(features_test,y_test))