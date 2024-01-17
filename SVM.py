import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, auc, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

import eli5
from eli5.sklearn import PermutationImportance

import warnings

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
warnings.simplefilter("ignore", category=PendingDeprecationWarning)
warnings.simplefilter("ignore", category=DeprecationWarning)
import shap

# Load the data
# data = pd.read_csv("/Users/bitaetaati/Desktop/Paper/Paper 1 - June 2023/parents_original.csv")

# Remove unnecessary columns
# data = data[data["Grade"].isin(['Third', 'Fourth', 'Fifth', 'Sixth'])]

data = pd.read_csv("/Users/bitaetaati/Desktop/Paper/Paper 1 - June 2023/parents.csv")
data = data.drop(columns=["child_asked_permission", "grade_allowed", "already"])
data = data.dropna()

# Split the data into train and test sets
train_data, test_data, train_target, test_target = train_test_split(data, data["leave_from_school"], test_size=0.2, random_state=1210)
train_data = train_data.dropna()
test_data = test_data.dropna()

# Convert the target variable to numeric binary format
label_encoder = LabelEncoder()
train_target_encoded = label_encoder.fit_transform(train_target).astype('bool_')
test_target_encoded = label_encoder.transform(test_target).astype('bool_')

# Perform one-hot encoding on the input features
train_encoded = pd.get_dummies(train_data.drop(columns=["leave_from_school"]))
test_encoded = pd.get_dummies(test_data.drop(columns=["leave_from_school"]))
#
# Align the test set with the train set columns
test_encoded = test_encoded.reindex(columns=train_encoded.columns, fill_value=0)

# Define the parameter grid for grid search
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'gamma': [0.01, 0.1, 1, 10, 100],
    'degree': [2, 3, 4, 5]
}

# Create the SVM model
model = SVC(kernel="rbf", probability= True)

# Perform grid search with cross-validation
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(train_encoded, train_target_encoded)

# Get the best hyperparameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Print the best hyperparameters
print("Best Hyperparameters:")
print(best_params)

# Train the SVM model with cross-validation
model = best_model
cv_scores = cross_val_score(model, train_encoded, train_target_encoded, cv=5)
model.fit(train_encoded, train_target_encoded)

svm_prob_model = model.predict_proba(test_encoded)[:, 1]
roc_auc = roc_auc_score(test_target_encoded, svm_prob_model)
print("ROC AUC:", roc_auc)


# Plot ROC curve
fpr, tpr, thresholds = roc_curve(test_target_encoded, svm_prob_model)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for SVM model')
plt.legend(loc='lower right')
plt.show()

# Compute accuracy
predictions = best_model.predict(test_encoded)
accuracy = accuracy_score(test_target_encoded, predictions)
print("Accuracy:", accuracy)

# Compute confusion matrix
confusion_mat = confusion_matrix(test_target_encoded, predictions)
print("Confusion Matrix:")
print(confusion_mat)

classification_rep = classification_report(test_target_encoded, predictions)
print("Classification Report:")
print(classification_rep)

# Calculate SHAP values using the best_model and scaled_test_data
explainer = shap.Explainer(best_model.decision_function, train_encoded)
shap_values = explainer(test_encoded)

shap.summary_plot(shap_values, test_encoded, plot="bar", sort=True)



#### pos neg dataframe
mean_shap = np.mean(shap_values.values, axis=0)

# Create a DataFrame with feature names, mean SHAP values, and signs
summary_df = pd.DataFrame({'Feature': test_encoded.columns, 'MeanSHAP': mean_shap, 'Sign': np.sign(mean_shap)})

# Sort the DataFrame by absolute mean SHAP values in descending order
summary_df['AbsMeanSHAP'] = np.abs(summary_df['MeanSHAP'])
summary_df = summary_df.sort_values(by='AbsMeanSHAP', ascending=False).drop(columns='AbsMeanSHAP')

# Display the resulting DataFrame
print(summary_df)