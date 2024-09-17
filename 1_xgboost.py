import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score


import seaborn as sns
import matplotlib.pyplot as plt
from joblib import dump


# Load dataset (replace 'path_to_your_dataset' with the actual path)
data = pd.read_csv('parkinsons.data')

# Exclude non-numeric columns or identifiers
features = data.select_dtypes(include=['float64', 'int64']).drop(columns=['name'], errors='ignore')
labels = data['status']

# Scale features
scaler = MinMaxScaler((-1, 1))
scaled_features = scaler.fit_transform(features)

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(scaled_features, labels, test_size=0.2, random_state=7)

# Initialize and train XGBoost classifier
model = XGBClassifier(eval_metric='mlogloss')
model.fit(x_train, y_train)

# Make predictions
y_pred = model.predict(x_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Create confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
confusion_df = pd.DataFrame(conf_matrix, columns=['Predicted Healthy', 'Predicted Parkinsons'],
                             index=['True Healthy', 'True Parkinsons'])

# Display confusion matrix
print('Confusion Matrix:')
print(confusion_df)

# Visualize confusion matrix using seaborn heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='RdYlGn', fmt='d', 
            xticklabels=['Predicted Healthy', 'Predicted Parkinsons'], 
            yticklabels=['True Healthy', 'True Parkinsons'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()

# Calculate precision, recall, and F1-score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-score: {f1}')

scores = cross_val_score(model, scaled_features, labels, cv=5)  # 5-fold cross-validation

# Save the model to a file
dump(model, 'parkinsons_xgboost_model.joblib')
