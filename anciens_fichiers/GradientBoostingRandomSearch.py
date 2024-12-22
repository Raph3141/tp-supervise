import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV

df = pd.read_csv('alt_acsincome_ca_features_85.csv')
df2 = pd.read_csv('alt_acsincome_ca_labels_85.csv')

df = df.sample(frac=0.1,random_state=42)
df2 = df2.sample(frac=0.1,random_state=42)

#print(df.describe())
#df['AGEP'].plot(kind='box')
#plt.show()

X = df
y = df2['PINCP'].astype(int)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,train_size=0.8)

print(X_train)
print(y_train)

my_scaler = StandardScaler()
X_scaled = my_scaler.fit_transform(X_train)

param_grid = {
    'n_estimators':range(160,180),
    'learning_rate': np.linspace(0.4,0.8,100),
    'max_depth':[2],
    'min_samples_split':range(60,100),
    'subsample': np.linspace(0.7,0.9,100)
}

grid_search = RandomizedSearchCV(estimator=GradientBoostingClassifier(), cv=5, scoring='accuracy',n_iter=20,param_distributions=param_grid)
grid_search.fit(X_scaled,y_train)
print(f"Meilleurs hyperparametres: {grid_search.best_params_}")

meilleur_model = grid_search.best_estimator_