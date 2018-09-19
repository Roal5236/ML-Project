# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 23:53:22 2018

@author: Rohaanoa Zoro
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 00:48:33 2018

@author: Rohaanoa Zoro
"""



import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm


#Reading the Datasets
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
train=train.copy()
test=test.copy()


y= train.iloc[:,[142]]

#PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X_opt)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, y], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 Component PCA', fontsize = 20)


targets = [1,2,3,4]
colors = ['y', 'g', 'b', 'r']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['Target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()


#Building the Optimal model using Backward Elimination
X_opt = train.iloc[:, [1, 4, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 31, 33, 36,
                       41, 42, 43, 47, 50, 53, 55, 56, 57, 58, 59, 63, 64, 65, 66, 68, 69,
                       72, 73, 74, 75, 76, 77, 78, 79, 81, 86, 92, 96, 97, 98, 99, 103, 104, 105, 106,
                       114, 115, 116, 117, 119, 121, 122, 123, 124,
                       125, 127, 128, 129, 130, 131]]
X_opt.fillna(0, inplace=True)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
result2 = sc.fit_transform(X_opt)


# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_opt, y)

# Predicting the Test set results
y_pred = classifier.predict(X_opt)

from sklearn.metrics import accuracy_score
accuracy_score(y_pred,y)
