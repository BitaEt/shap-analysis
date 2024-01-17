# shap-analysis
shap analysis and statistical modeling (random forest, linear regression and svm) for "Understanding Active Transportation to School Behavior in Socioeconomically Disadvantaged Communities: A Machine Learning and SHAP Analysis Approach". (click [here](https://www.mdpi.com/2071-1050/16/1/48) to read the full paper)

**Data Collection:** The code involves the analysis of surveys created by the National Center for Safe Routes to School. These surveys collect information from parents/guardians about their children's travel behavior, focusing on factors affecting their willingness to walk or bike to school. The dataset used in the research includes responses from 5th to 12th-grade students, gathered from 19 schools between 2009 and 2011.

**Statistical Analysis:** The primary objective is to understand the factors influencing students' choice of walking as their transportation mode to school. Independent variables from the Parent survey are used to predict whether a student walks to school or uses another mode. Different machine learning algorithms like logistic regression, random forest, and support vector machines are employed for this analysis. Feature selection is performed using SHapley Additive exPlanations values (SHAP).

**Model Selection:** Three different models are explored: logistic regression, random forest, and support vector machines. Each model's performance is evaluated, and hyperparameter optimization is conducted to improve model accuracy.

**Model Evaluation Metric:**  Various metrics, including AUC (Area Under the ROC Curve), confusion matrix, accuracy, and error rate, are used to assess the performance of the models. These metrics provide a comprehensive evaluation of how well the models predict the choice of walking to school.

**SHAP Values:**  SHAP values are calculated to understand the impact of individual features on the model's predictions. This helps identify which factors have the most significant influence on the decision to walk to school.
