from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve, classification_report, auc
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
warnings.simplefilter("ignore", category=PendingDeprecationWarning)
warnings.simplefilter("ignore", category=DeprecationWarning)
import shap


# # Load the data
# data = pd.read_csv("/Users/bitaetaati/Desktop/Paper/Paper 1 - June 2023/parents_original.csv")
#
# # Remove unnecessary columns
# data = data[data["Grade"].isin(['Third', 'Fourth', 'Fifth', 'Sixth'])]
# data = data.drop(columns=["get_to_school", "Grade"])

data = pd.read_csv("/Users/bitaetaati/Desktop/Paper/Paper 1 - June 2023/parents.csv")
data = data.drop(columns=["child_asked_permission", "grade_allowed", "already"])
data = data.dropna()
data = data.dropna()

train_data, test_data, train_target, test_target = train_test_split(data, data["leave_from_school"], test_size=0.2, random_state=1210)
train_data = train_data.dropna()
test_data = test_data.dropna()

# Convert the target variable to numeric binary format
label_encoder = LabelEncoder()
train_target_encoded = label_encoder.fit_transform(train_target)
test_target_encoded = label_encoder.transform(test_target)

# Perform one-hot encoding on the input features
train_encoded = pd.get_dummies(train_data.drop(columns=["leave_from_school"]))
test_encoded = pd.get_dummies(test_data.drop(columns=["leave_from_school"]))

n_estimators_range = np.arange(10, 1001, 100)
oob_errors = []

for n_estimators in n_estimators_range:
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=1210, oob_score=True)
    rf.fit(train_encoded, train_target_encoded)
    oob_error = 1 - rf.oob_score_
    oob_errors.append(oob_error)

plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, oob_errors, marker='o', linestyle='-')
plt.xlabel('Number of Trees (n_estimators)')
plt.ylabel('OOB Error')
plt.title('OOB Error vs. Number of Trees')
plt.grid(False)  # Set grid to False to remove grid lines

# Highlight the optimal number of trees with a red dot
optimal_n_estimators = n_estimators_range[np.argmin(oob_errors)]

plt.scatter(optimal_n_estimators, min(oob_errors), color='red', label=f'Optimal n_estimators={optimal_n_estimators}', zorder=5)
plt.legend()

plt.show()
# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [optimal_n_estimators],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [None] + list(np.arange(5, 21, 5)),
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

rf = RandomForestClassifier(random_state=1210)
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=100, cv=10, random_state=1210)
random_search.fit(train_encoded, train_target_encoded)

best_rf_model = random_search.best_estimator_
rf_prob_model = best_rf_model.predict_proba(test_encoded)[:, 1]


optimal_parameters = best_rf_model.get_params()
for param, value in optimal_parameters.items():
    print(f"{param}: {value}")
print("Optimal Number of Trees (n_estimators):", optimal_parameters['n_estimators'])


# Get the best model from the grid search
rf_prob = best_rf_model.predict_proba(test_encoded)[:, 1]
predicted_classes = best_rf_model.predict(test_encoded)

# Calculate AUC-ROC score
roc_auc = roc_auc_score(test_target_encoded, rf_prob_model)
print("ROC AUC:", roc_auc)

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(test_target_encoded, rf_prob_model)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Random Forest model')
plt.legend(loc='lower right')
plt.show()


# Compute accuracy
accuracy = accuracy_score(test_target_encoded, predicted_classes)
print("Accuracy:", accuracy)

# Compute confusion matrix
confusion_mat = confusion_matrix(test_target_encoded, predicted_classes)
print("Confusion Matrix:")
print(confusion_mat)


classification_rep = classification_report(test_target_encoded, predicted_classes)
print("Classification Report:")
print(classification_rep)

# Calculate SHAP values using the best_model and scaled_test_data
explainer = shap.TreeExplainer(best_rf_model)
shap_values = explainer.shap_values(test_encoded)


shap.summary_plot(shap_values, test_encoded, sort=True)

#### pos neg dataframe
mean_shap = np.mean(shap_values[0], axis=0)

# Create a DataFrame with feature names, mean SHAP values, and signs
summary_df = pd.DataFrame({'Feature': test_encoded.columns, 'MeanSHAP': mean_shap, 'Sign': np.sign(mean_shap)})

# Sort the DataFrame by absolute mean SHAP values in descending order
summary_df['AbsMeanSHAP'] = np.abs(summary_df['MeanSHAP'])
summary_df = summary_df.sort_values(by='AbsMeanSHAP', ascending=False).drop(columns='AbsMeanSHAP')

# Display the resulting DataFrame
print(summary_df)