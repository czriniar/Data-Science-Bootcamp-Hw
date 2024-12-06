# Logistic Regression
# 1 Try different thresholds for computing predictions
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Example probabilities from a logistic regression model
# Replace with `model.predict_proba(X_test)[:, 1]` in a real scenario
probabilities = np.array([0.1, 0.4, 0.6, 0.8, 0.9])
y_test = np.array([0, 0, 1, 1, 1])  # True labels

# Evaluate for different thresholds
thresholds = [0.3, 0.5, 0.7]
for threshold in thresholds:
    predictions = (probabilities >= threshold).astype(int)
    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions)
    rec = recall_score(y_test, predictions)
    print(f"Threshold: {threshold}")
    print(f"Accuracy: {acc:.2f}, Precision: {prec:.2f}, Recall: {rec:.2f}")
    print("-" * 30)

# 3 Fit a Logistic Regression Model on all features
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd

# Example dataset
df = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5],
    'feature2': ['A', 'B', 'A', 'B', 'A'],
    'target': [0, 1, 0, 1, 1]
})

# Define features and target
X = df[['feature1', 'feature2']]
y = df['target']

# Preprocessing: One-hot encoding and scaling
encoder = OneHotEncoder(sparse=False)
scaler = StandardScaler()

X_encoded = encoder.fit_transform(X[['feature2']])
X_scaled = scaler.fit_transform(X[['feature1']])

X_preprocessed = np.hstack([X_scaled, X_encoded])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# Fit logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

print("Model coefficients:", model.coef_)
print("Model intercept:", model.intercept_)

# 4 Plot ROC Curves for each model
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Predict probabilities
probabilities = model.predict_proba(X_test)[:, 1]

# Compute ROC curve and AUC
fpr, tpr, _ = roc_curve(y_test, probabilities)
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "r--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Clustering
# 1 Repeat the above exercise for different values of k
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Example data
X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0]])

# Evaluate clustering for different k values
k_values = [2, 3, 4]
inertia = []
silhouette_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))
    print(f"k = {k}, Inertia: {kmeans.inertia_:.2f}, Silhouette Score: {silhouette_scores[-1]:.2f}")

# Plot Inertia
plt.figure()
plt.plot(k_values, inertia, marker='o', label="Inertia")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Inertia vs. Number of Clusters")
plt.legend()
plt.show()

# Plot Silhouette Scores
plt.figure()
plt.plot(k_values, silhouette_scores, marker='o', label="Silhouette Score")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs. Number of Clusters")
plt.legend()
plt.show()

# 2 What happens if you don't scale your features?
# Without scaling
X_unscaled = np.array([[1, 200], [1, 400], [1, 0],
                       [10, 20], [10, 40], [10, 0]])

# Repeat clustering analysis without scaling
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_unscaled)
    print(f"k = {k}, Inertia (no scaling): {kmeans.inertia_:.2f}")

# 3 Is there a 'right' k? Why or why not?
# Elbow method for selecting k
plt.figure()
plt.plot(k_values, inertia, marker='o', label="Inertia")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal k")
plt.legend()
plt.show()

# Silhouette analysis
plt.figure()
plt.plot(k_values, silhouette_scores, marker='o', label="Silhouette Score")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Analysis for Optimal k")
plt.legend()
plt.show()

# Conclusion: There's no single 'right' k; it depends on domain knowledge and context.
