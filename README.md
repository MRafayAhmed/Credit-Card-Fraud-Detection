# Credit-Card-Fraud-Detection
# **Overview**

This project detects fraudulent credit card transactions using a supervised machine learning model (Random Forest Classifier). The dataset used contains anonymized transaction details, including a binary class label (0 = Legit, 1 = Fraud).

# Step-by-Step Explanation
1. Import Libraries
You load essential libraries for:

* Data manipulation: `pandas`, `numpy`
* Visualization: `matplotlib`, `seaborn`
* Modeling: `sklearn`, `imblearn`
2. Load and Inspect the Dataset
```python
data = pd.read_csv('creditcard.csv')
```
You examine the first few rows and class distribution using:
```python
pd.value_counts(data['Class']).plot.bar()
```
Observation: The dataset is highly imbalanced (very few fraud cases).
3. Data Preprocessing
* Normalize the `Amount` column → `normAmount`
* Drop the `Time` and original `Amount` columns
* Split into features `X` and labels `y`
4. Train-Test Split and Oversampling
You split the dataset (70% training, 30% testing), then use **SMOTE** to balance the minority class in the training set:
```python
sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train.ravel())
```
Observation: Class distribution becomes 50-50 after SMOTE.
5. Train Random Forest Classifier
```python
classifier = RandomForestClassifier(...)
classifier.fit(Xtrain, Ytrain)
```
You fit the model using the training data and tune parameters like `n_estimators`, `max_depth`, etc.
6. Feature Importance Visualization
You use:
```python
sns.barplot(x='Feature importance', y='Feature', data=tmp)
```
To see which features influence predictions most.
7. Model Evaluation
You evaluate predictions using:
```python
confusion_matrix(Ytest, predictions)
classification_report(Ytest, predictions)
```
Key metrics shown: **Precision**, **Recall**, **F1-score**
8. Testing Interface (CLI)
You implement a simple command-line interface:
```python
predict_transaction()
```
It prompts the user for 29 input features and prints whether the transaction is likely **fraudulent** or **legit**.


# How to Run the Script
1. Place the CSV file at the specified path (or update the path).
2. Run the entire script in a Jupyter Notebook or Python environment.
3. After training is complete, the script calls:
   ```python
   predict_transaction()
   ```
   You’ll be asked to input 29 values (from a row in the dataset or manually).
4. The model predicts whether it’s **Fraud** or **Legit**.

# Observations

* **SMOTE** significantly improves the model's ability to detect fraud by balancing the classes.
* **Random Forest** performs well for imbalanced classification due to ensemble learning.
* **Feature importance** helps understand which anonymized features are critical for fraud detection.
* The **CLI test** is a simple but effective way to try out predictions manually.

# Optional Tip: Test with Real Data Row
Instead of entering 29 values manually, use:
```python
sample = Xtest.iloc[0].values.reshape(1, -1)
classifier.predict(sample)
