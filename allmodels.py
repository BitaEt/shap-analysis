import numpy as np
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, auc, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
warnings.simplefilter("ignore", category=PendingDeprecationWarning)
warnings.simplefilter("ignore", category=DeprecationWarning)
import shap


def train_classifier_with_grid_search(classifier, param_grid, train_data, train_target, cv=5):
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(classifier, param_grid, cv=cv)
    grid_search.fit(train_data, train_target)

    # Get the best hyperparameters and model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # Print the best hyperparameters
    print("Best Hyperparameters:")
    print(best_params)

    # Train the model with cross-validation
    model = best_model
    cv_scores = cross_val_score(model, train_data, train_target, cv=cv)

    return model, cv_scores


def plot_roc_curve_and_auc(test_target, predicted_probabilities, model_name):
    # Calculate ROC AUC
    roc_auc = roc_auc_score(test_target, predicted_probabilities)

    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(test_target, predicted_probabilities)
    plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name} model')
    plt.legend(loc='lower right')
    plt.show()

    return roc_auc


def print_classification_metrics(model, test_data, test_target):
    # Make predictions using the model
    predictions = model.predict(test_data)

    # Calculate accuracy
    accuracy = accuracy_score(test_target, predictions)
    print("Accuracy:", accuracy)

    # Compute confusion matrix
    confusion_mat = confusion_matrix(test_target, predictions)
    print("Confusion Matrix:")
    print(confusion_mat)

    # Print classification report
    classification_rep = classification_report(test_target, predictions)
    print("Classification Report:")
    print(classification_rep)


def calculate_and_plot_shap_values(model, test_data):
    # Create a callable function that takes a 2D array of samples as input and returns predictions
    def model_callable(input_data):
        return model.predict(input_data)

    # Calculate SHAP values using the callable model and test_data
    explainer = shap.Explainer(model_callable, test_data)
    shap_values = explainer.shap_values(test_data)

    # Plot the summary plot with features sorted based on SHAP values
    shap.summary_plot(shap_values, test_data, plot_type="dot")

    # Create a custom legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Walk = 1', markerfacecolor='red', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Walk = 0', markerfacecolor='blue', markersize=10)]

    # Add the legend to the plot
    plt.legend(handles=legend_elements)

    # Show the plot
    plt.show()

    return shap_values


def train_classifier_with_randomized_search(classifier, param_distributions, train_data, train_target, cv=5, n_iter=10):
    # Perform randomized search with cross-validation
    random_search = RandomizedSearchCV(classifier, param_distributions, cv=cv, n_iter=n_iter)
    random_search.fit(train_data, train_target)

    # Get the best hyperparameters and model
    best_params = random_search.best_params_
    best_model = random_search.best_estimator_

    # Print the best hyperparameters
    print("Best Hyperparameters:")
    print(best_params)

    # Train the model with cross-validation
    model = best_model
    cv_scores = cross_val_score(model, train_data, train_target, cv=cv)

    return model, cv_scores



# Creating test and train dataset

data = pd.read_csv("/Users/bitaetaati/Desktop/Paper/Paper 1 - June 2023/parents copy.csv")
data = data.drop(columns=["child_asked_permission", "grade_allowed", "already"])
data = data.dropna()


# Split the data into train and test sets
train_data, test_data, train_target, test_target = train_test_split(data, data["leave_from_school"], test_size=0.2,
                                                                    random_state=1210)
train_data = train_data.dropna()
test_data = test_data.dropna()

# Convert the target variable to numeric binary format
label_encoder = LabelEncoder()
train_target_encoded = label_encoder.fit_transform(train_target)
test_target_encoded = label_encoder.transform(test_target)


# Perform one-hot encoding on the input features
train_encoded = pd.get_dummies(train_data.drop(columns=["leave_from_school"]))
test_encoded = pd.get_dummies(test_data.drop(columns=["leave_from_school"]))
#
# Align the test set with the train set columns
test_encoded = test_encoded.reindex(columns=train_encoded.columns, fill_value=0)

# Create the SVM model
param_grid_svm = {
    'C': [0.01, 0.1, 1, 10, 100],
    'gamma': [0.01, 0.1, 1, 10, 100],
    'degree': [2, 3, 4, 5]
}
svm = SVC(kernel="rbf", probability=True)
best_svm_model, svm_cv_scores = train_classifier_with_grid_search(svm, param_grid_svm, train_encoded,
                                                                  train_target_encoded)
svm_prob_model = best_svm_model.predict_proba(test_encoded)[:, 1]
#svm_roc_auc = plot_roc_curve_and_auc(test_target_encoded, svm_prob_model, "SVM")
print_classification_metrics(best_svm_model, test_encoded, test_target_encoded)
svm_shap_values = calculate_and_plot_shap_values(best_svm_model, test_encoded)

# Create the LR model
param_grid_lr = {
    'penalty': ['l1', 'l2'],  # Regularization penalty (L1 or L2)
    'C': [0.1, 1.0, 10.0],  # Inverse of regularization strength
    'solver': ['liblinear'],  # Solver algorithm
    'class_weight': [None, 'balanced']  # Class weight (None or 'balanced')
}

lr = LogisticRegression()
best_lr_model, lr_cv_scores = train_classifier_with_grid_search(lr, param_grid_lr, train_encoded,
                                                                train_target_encoded)
lr_prob_model = best_lr_model.predict_proba(test_encoded)[:, 1]
#lr_roc_auc = plot_roc_curve_and_auc(test_target_encoded, lr_prob_model, "LR")
print_classification_metrics(best_lr_model, test_encoded, test_target_encoded)
lr_shap_values = calculate_and_plot_shap_values(best_lr_model, test_encoded)


# Extract coefficients from the logistic regression model
coefficients = best_lr_model.coef_[0]
features = train_encoded.columns

# Sort coefficients by their absolute values
sorted_indices = np.argsort(np.abs(coefficients))

# Plot
plt.figure(figsize=(10, 6))
plt.barh(features[sorted_indices], coefficients[sorted_indices])
plt.xlabel('Coefficient Value')
plt.ylabel('Feature Name')
plt.title('Feature Importance from Logistic Regression Coefficients')
plt.tight_layout()
plt.show()






#RF
param_grid_rf = {
    'n_estimators': np.arange(100, 1001, 100),
    'max_features': ['sqrt', 'log2'],
    'max_depth': [None] + list(np.arange(5, 21, 5)),
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

rf = RandomForestClassifier(random_state=1210)

best_rf_model, rf_cv_scores = train_classifier_with_randomized_search(rf, param_grid_rf, train_encoded,
                                                                      train_target_encoded, cv=5, n_iter=10)

rf_prob_model = best_rf_model.predict_proba(test_encoded)[:, 1]
#rf_roc_auc = plot_roc_curve_and_auc(test_target_encoded, rf_prob_model, "RF")
print_classification_metrics(best_rf_model, test_encoded, test_target_encoded)
rf_shap_values = calculate_and_plot_shap_values(best_rf_model, test_encoded)

#### feature importance

feature_importances_rf = best_rf_model.feature_importances_

# Create a DataFrame to store feature importances and sort them in descending order
feature_importance_df = pd.DataFrame({'Feature': train_encoded.columns, 'Importance': feature_importances_rf})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.bar(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xticks(rotation=90)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Random Forest Feature Importances')
plt.tight_layout()
plt.show()




#####

def plot_roc_curve_and_auc(models, model_names, test_data, test_targets):
    plt.figure(figsize=(8, 6))
    for model, name in zip(models, model_names):
        predicted_probabilities = model.predict_proba(test_data)[:, 1]
        roc_auc = roc_auc_score(test_targets, predicted_probabilities)
        fpr, tpr, _ = roc_curve(test_targets, predicted_probabilities)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Different Models')
    plt.legend(loc='lower right')
    plt.show()

# Create a list of models and their names
models = [best_svm_model, best_lr_model, best_rf_model]
model_names = ['SVM', 'Logistic Regression', 'Random Forest']

# Plot the ROC-AUC curve for all models
plot_roc_curve_and_auc(models, model_names, test_encoded, test_target_encoded)



######Chi SQUARE
# Check if "leave_from_school" is categorical (i.e., has limited distinct values)
if data["leave_from_school"].nunique() <= 10:
    # Perform the chi-square test
    contingency_table = pd.crosstab(data["leave_from_school"], data["period"])  # Replace "other_variable" with the relevant variable you want to compare with "leave_from_school"
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)

    # Interpret the results
    if p_value < 0.05:
        print("The chi-square test is significant. There is a significant association between 'leave_from_school' and the other variable.")
    else:
        print("The chi-square test is not significant. There is no significant association between 'leave_from_school' and the other variable.")
else:
    print("The 'leave_from_school' variable is not categorical or has too many distinct values. Please make sure it is categorical with limited distinct values.")



