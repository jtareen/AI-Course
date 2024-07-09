### Logistic Regression Notes

#### 1. Introduction
- **Logistic Regression** is a statistical method used for binary classification problems (i.e., where the outcome can be one of two possible categories).
- It models the probability that a given input belongs to a particular category.

#### 2. Logistic Function (Sigmoid Function)
- The logistic function, also known as the sigmoid function, is defined as:
  \[
  \sigma(z) = \frac{1}{1 + e^{-z}}
  \]
- It outputs values between 0 and 1, which can be interpreted as probabilities.

#### 3. Model Equation
- In logistic regression, the linear combination of input features is transformed using the sigmoid function:
  \[
  \hat{p}(X) = \sigma(\beta_0 + \beta_1 X_1 + \beta_2 X_2 + \cdots + \beta_n X_n)
  \]
  where:
  - \(\hat{p}(X)\) is the predicted probability of the positive class.
  - \(\beta_0\) is the intercept (bias term).
  - \(\beta_1, \beta_2, \ldots, \beta_n\) are the coefficients (weights) for the input features \(X_1, X_2, \ldots, X_n\).

#### 4. Decision Boundary
- The decision boundary is determined by setting a threshold, typically 0.5:
  \[
  \text{Predict 1 (positive class) if } \hat{p}(X) \geq 0.5
  \]
  \[
  \text{Predict 0 (negative class) if } \hat{p}(X) < 0.5
  \]

#### 5. Cost Function
- Logistic regression uses a cost function based on the concept of maximum likelihood estimation:
  \[
  J(\beta) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log(\hat{p}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{p}^{(i)}) \right]
  \]
  where:
  - \(m\) is the number of training examples.
  - \(y^{(i)}\) is the actual label for the \(i\)-th example.
  - \(\hat{p}^{(i)}\) is the predicted probability for the \(i\)-th example.

#### 6. Gradient Descent
- The cost function is minimized using gradient descent, where the coefficients are updated iteratively:
  \[
  \beta_j := \beta_j - \alpha \frac{\partial J(\beta)}{\partial \beta_j}
  \]
  where \(\alpha\) is the learning rate.

#### 7. Assumptions
- **Independence:** The observations are independent of each other.
- **Linearity:** The log-odds of the response variable are a linear combination of the predictor variables.
- **No Multicollinearity:** The predictor variables are not highly correlated with each other.

#### 8. Evaluation Metrics
- **Accuracy:** The proportion of correctly classified instances.
- **Precision:** The proportion of positive predictions that are actually positive.
- **Recall (Sensitivity):** The proportion of actual positives that are correctly predicted.
- **F1 Score:** The harmonic mean of precision and recall.
- **ROC Curve:** Plots true positive rate against false positive rate.
- **AUC (Area Under the Curve):** Measures the entire two-dimensional area underneath the entire ROC curve.

#### 9. Advantages
- Simple and easy to implement.
- Outputs probabilities, which can be useful for decision making.
- Can handle binary as well as multiclass classification (with extensions).

#### 10. Disadvantages
- Assumes a linear relationship between the log-odds and the input features.
- Can struggle with complex relationships in the data.
- Sensitive to outliers.

#### 11. Implementation in Python
Here's a basic implementation using scikit-learn:
```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Sample data
X = np.array([[...], [...], ...])  # feature matrix
y = np.array([...])  # target vector

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC AUC: {roc_auc}")
print(f"Confusion Matrix:\n{conf_matrix}")
```

This covers the basics of logistic regression. Let me know if you need more details on any specific part!