import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import time
import seaborn as sns

df = pd.read_csv('alt_acsincome_ca_features_85.csv')
df2 = pd.read_csv('alt_acsincome_ca_labels_85.csv')
X = df
y = df2['PINCP'].astype(int)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,train_size=0.8)

print(X_train)
print(y_train)

my_scaler = StandardScaler()
X_scaled = my_scaler.fit_transform(X_train)
dump(my_scaler, 'scaler.joblib')

gb = GradientBoostingClassifier(n_estimators=100, subsample=0.8, min_samples_split=95, max_depth=2, learning_rate=0.5)

model = gb.fit(X_scaled,y_train)
X_scaled_test = my_scaler.transform(X_test)
y_pred = cross_val_predict(model, X_scaled_test, y_test, cv=5)

importances = model.feature_importances_


feature_importance_df = pd.DataFrame({
   'Feature': X_train.columns,
   'Importance': importances
})


feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)


print("Importance des features : ", feature_importance_df)