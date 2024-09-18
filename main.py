import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import networkx as nx
import matplotlib.pyplot as plt

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/new-thyroid.data'
columns = ['Label', 'T3', 'T4', 'TSH', 'Goiter']
data = pd.read_csv(url, names=columns, sep=',', na_values='?')

print("Dataset shape before preprocessing:", data.shape)
print("First few rows of data:\n", data.head())

data.dropna(inplace=True)

print("Dataset shape after preprocessing (after dropping NaN):", data.shape)

if data.shape[0] == 0:
    print("Warning: No data remaining after dropping missing values.")

X = data.drop('Label', axis=1)
y = data['Label']

if X.empty or y.empty:
    raise ValueError("The dataset is empty after preprocessing.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_data = pd.concat([X_train, y_train], axis=1)

model = BayesianNetwork([('T3', 'Label'), ('T4', 'Label'), ('TSH', 'Label'), ('Goiter', 'Label')])

model.fit(train_data, estimator=MaximumLikelihoodEstimator)

plt.figure(figsize=(8, 6))
G = model.to_networkx()
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=10, font_weight='bold')
plt.title("Bayesian Network Structure")
plt.show()

inference = VariableElimination(model)

query_result = inference.query(variables=['Label'], evidence={'T3': 1, 'T4': 2, 'TSH': 3, 'Goiter': 0})
print(query_result)

predictions = []
for index, row in X_test.iterrows():
    evidence = {'T3': row['T3'], 'T4': row['T4'], 'TSH': row['TSH'], 'Goiter': row['Goiter']}
    result = inference.map_query(variables=['Label'], evidence=evidence)
    predictions.append(result['Label'])

accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")