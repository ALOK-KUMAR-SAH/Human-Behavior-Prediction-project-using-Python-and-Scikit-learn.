# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Step 1: Create a dummy dataset
# Features: Age, Income, and Spending Score
data = pd.DataFrame({
    'Age': [25, 32, 47, 54, 23, 33, 40, 52, 29, 36],
    'Income': [30000, 50000, 70000, 80000, 20000, 60000, 75000, 95000, 45000, 52000],
    'Spending_Score': [60, 80, 90, 70, 50, 75, 85, 65, 78, 72],
    'Behavior': ['Positive', 'Negative', 'Positive', 'Positive', 'Negative', 
                 'Negative', 'Positive', 'Positive', 'Negative', 'Positive']
})

# Step 2: Inspect the dataset
print("Dataset:\n", data)

# Step 3: Encode the target variable
label_encoder = LabelEncoder()
data['Behavior'] = label_encoder.fit_transform(data['Behavior'])

# Step 4: Split the data into features (X) and target (y)
X = data[['Age', 'Income', 'Spending_Score']]  # Features
y = data['Behavior']  # Target

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 7: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 8: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy: {:.2f}%".format(accuracy * 100))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 9: Predict behavior for a new data point
new_data = np.array([[30, 55000, 75]])  # Example: Age=30, Income=55000, Spending_Score=75
new_prediction = model.predict(new_data)
predicted_behavior = label_encoder.inverse_transform(new_prediction)
print("\nPredicted Behavior for new data:", predicted_behavior[0])
