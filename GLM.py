import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve, classification_report, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

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
# data = data[data["Distance_from_School"].isin(['Less than 1/4 mile'])]
# data = data.drop(columns=["get_to_school", "Grade"])


data = pd.read_csv("/Users/bitaetaati/Desktop/Paper/Paper 1 - June 2023/parents.csv")
data = data.drop(columns=["child_asked_permission", "grade_allowed", "already"])
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

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'penalty': ['l1', 'l2'],  # Regularization penalty (L1 or L2)
    'C': [0.1, 1.0, 10.0],  # Inverse of regularization strength
    'solver': ['liblinear'],  # Solver algorithm
    'class_weight': [None, 'balanced']  # Class weight (None or 'balanced')
}

glm_fit = LogisticRegression()

# Perform grid search cross-validation
grid_search = GridSearchCV(glm_fit, param_grid, cv=5, scoring='accuracy')
grid_search.fit(train_encoded, train_target_encoded)

# Get the best model from the grid search
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

print("Best Hyperparameters:")
print(best_params)

glm_prob = best_model.predict_proba(test_encoded)[:, 1]

predicted_classes = np.where(glm_prob > 0.5, 1, 0)

confusion = confusion_matrix(test_target_encoded, predicted_classes)
print("Confusion Matrix:")
print(confusion)

accuracy = accuracy_score(test_target_encoded, predicted_classes)
print("Accuracy:", accuracy)

roc_auc = roc_auc_score(test_target_encoded, glm_prob)
print("ROC AUC:", roc_auc)


fpr, tpr, thresholds = roc_curve(test_target_encoded, glm_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Logistic Regression model')
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
#explainer = shap.Explainer(best_model.decision_function, train_encoded)
explainer = shap.Explainer(best_model.predict, train_encoded)
shap_values = explainer(test_encoded)

shap.summary_plot(shap_values, test_encoded)


###Shap Dataframe
mean_abs_shap = np.mean(np.abs(shap_values.values), axis=0)

summary_df = pd.DataFrame({'Feature': test_encoded.columns, 'MeanAbsoluteSHAP': mean_abs_shap})

# Sort the DataFrame by mean absolute SHAP values in descending order
summary_df = summary_df.sort_values(by='MeanAbsoluteSHAP', ascending=False)

# Display the resulting DataFrame
print(summary_df)



#####with neg and pos

mean_shap = np.mean(shap_values.values, axis=0)

# Create a DataFrame with feature names, mean SHAP values, and signs
summary_df = pd.DataFrame({'Feature': test_encoded.columns, 'MeanSHAP': mean_shap, 'Sign': np.sign(mean_shap)})

# Sort the DataFrame by absolute mean SHAP values in descending order
summary_df['AbsMeanSHAP'] = np.abs(summary_df['MeanSHAP'])
summary_df = summary_df.sort_values(by='AbsMeanSHAP', ascending=False).drop(columns='AbsMeanSHAP')

# Display the resulting DataFrame
print(summary_df)