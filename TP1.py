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

df.hist()
plt.suptitle('Distributions des variables numériques')
plt.show()

df['AGEP'].plot(kind='box')
plt.suptitle("Boxplot de l'age")
plt.show()

X = df
y = df2['PINCP'].astype(int)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,train_size=0.8)

print(X_train)
print(y_train)

my_scaler = StandardScaler()
X_scaled = my_scaler.fit_transform(X_train)
dump(my_scaler, 'scaler.joblib')

rf = RandomForestClassifier(n_estimators=100)

start_time = time.time()
model = rf.fit(X_scaled,y_train)
X_scaled_test = my_scaler.transform(X_test)
y_pred = cross_val_predict(model, X_scaled_test, y_test, cv=5)
print("classification report : ", classification_report(y_test,y_pred))
print("accuracy score : ", accuracy_score(y_test,y_pred))
print("matrice de confusion : ", confusion_matrix(y_test,y_pred))
end_time = time.time()

exec_time = end_time - start_time
print("temps d'execution = ", exec_time)

cm = confusion_matrix(y_test,y_pred)
classes = ['Classe 0', 'Classe 1']
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Prédictions')
plt.ylabel('Vérités')
plt.title('Matrice de Confusion')
plt.show()


