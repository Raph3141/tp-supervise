import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import time
import seaborn as sns

features = pd.read_csv('alt_acsincome_ca_features_85.csv')
labels = pd.read_csv('alt_acsincome_ca_labels_85.csv')

X = features
y = labels['PINCP'].astype(int)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,train_size=0.8)

my_scaler = StandardScaler()
X_scaled = my_scaler.fit_transform(X_train)
dump(my_scaler, 'scaler.joblib')

#random_forest_classifier = RandomForestClassifier(n_estimators=100)
random_forest_classifier = RandomForestClassifier(n_estimators=500, min_samples_split=19, max_depth=37)

model_classification = random_forest_classifier.fit(X_scaled,y_train)
X_scaled_test = my_scaler.transform(X_test)
y_pred = cross_val_predict(model_classification, X_scaled_test, y_test, cv=5)

accuracy_classification = accuracy_score(y_test,y_pred)
report_classification = classification_report(y_test, y_pred)
conf_matrix_classifier = confusion_matrix(y_test,y_pred)

print(f"Accuracy of RandomForestClassifier : {accuracy_classification}")
print("\nClassification Report of RandomForestClassifier")
print(report_classification)
print("\nConfusion Matrix of RandomForestClassifier")
print(conf_matrix_classifier)

classes = ['Classe 0', 'Classe 1']
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_classifier, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Prédictions')
plt.ylabel('Vérités')
plt.title('Matrice de Confusion pour RandomForestClassifier')
plt.show()