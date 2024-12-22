import pandas as pd
import matplotlib.pyplot as plt

features = pd.read_csv('alt_acsincome_ca_features_85.csv')
labels = pd.read_csv('alt_acsincome_ca_labels_85.csv')

features.hist()
plt.suptitle('Distributions des variables numériques')
plt.show()

features['AGEP'].plot(kind='box')
plt.suptitle("Boxplot de l'age")
plt.show()

print("Résumé de la distribution des données : ", features.describe(include='all'))
print(features.shape)
print(labels.shape)

