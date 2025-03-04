# Credit Risk Classification

This repository contains a project that uses machine learning (specifically, **Logistic Regression**) to predict whether a loan is likely to be healthy (0) or high-risk (1).

## Project Structure
credit-risk-classification/
├── Credit_Risk/
│   ├── Resources/
│   │   └── lending_data.csv
│   ├── credit_risk_classification.ipynb
│   └── credit_risk_report.md
├── LICENSE
└── README.md


- **Credit_Risk**:  
  Contains the primary Jupyter Notebook (`credit_risk_classification.ipynb`) where the data is loaded, the Logistic Regression model is trained, and results are generated.  
  - **Resources**:  
    Contains the `lending_data.csv` file that holds the loan information.
  - **credit_risk_report.md**:  
    Provides a structured overview of the analysis, including methodology, results, and conclusions.

## Overview of the Analysis

1. **Purpose of the Analysis**  
   The goal is to determine if a Logistic Regression model can accurately predict whether a loan is healthy (0) or high-risk (1). This involves training on a dataset of historical lending data and evaluating how well the model performs on unseen test data.

2. **Data**  
   - **Source**: `lending_data.csv` in the `Resources` folder.  
   - **Details**: 77,536 loan records, including features like loan size, interest rate, borrower income, debt-to-income ratio, number of accounts, derogatory marks, and total debt.  
   - **Target Variable**: `loan_status` (0 for healthy, 1 for high-risk).

3. **Machine Learning Workflow**  
   - **Data Preparation**: Load the CSV, create a DataFrame, and clean/inspect the data.  
   - **Feature/Label Separation**:  
     - **Features (X)**: All columns except `loan_status`.  
     - **Label (y)**: The `loan_status` column.  
   - **Train-Test Split**: Split data into training and testing sets to evaluate model performance.  
   - **Model Selection**: Logistic Regression from scikit-learn.  
   - **Model Training**: Fit the model on the training set.  
   - **Prediction**: Use the trained model to predict on the test set.  
   - **Evaluation**: Evaluate with metrics like accuracy, precision, recall, confusion matrix, and classification report.

---

## Results

Using the Logistic Regression model, the following metrics were observed (example values; replace with your actual results):

- **Accuracy**: 0.99  
- **Precision**:  
  - Class 0 (healthy): 1.00  
  - Class 1 (high-risk): 0.84  
- **Recall**:  
  - Class 0 (healthy): 0.99  
  - Class 1 (high-risk): 0.94

These results indicate that the model is quite effective, especially in identifying healthy loans. While high-risk predictions also perform well, further tuning or additional data could improve the precision for Class 1.

---

## Conclusion

Based on these metrics, the Logistic Regression model demonstrates strong performance. However, before deploying this model in a production environment, it is recommended to:

- Evaluate on different or larger datasets.
- Consider resampling or other techniques to address any class imbalance.
- Explore additional models or hyperparameter tuning to further refine performance.

---
## Troubleshooting: Convergence Warning in Logistic Regression

During model training, you might encounter the following warning:

ConvergenceWarning: lbfgs failed to converge (status=1): STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.


### What This Warning Means
- **Explanation:**  
  The warning indicates that the LBFGS optimizer, which is used by default in scikit-learn's LogisticRegression, did not fully converge within the default iteration limit. This means that while the model has reached the iteration cap, it might not have found the optimal set of weights.
- **Impact:**  
  Despite this warning, our model still performs very well (with 99% accuracy, strong precision, and recall). However, addressing the warning can lead to cleaner output and potentially even better performance.

### How to Address the Warning
1. **Increase the Maximum Iterations:**
   ```python
   lr_model = LogisticRegression(random_state=1, max_iter=1000)
Scale Your Data:
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
lr_model = LogisticRegression(random_state=1)
lr_model.fit(X_train_scaled, y_train)
y_pred = lr_model.predict(X_test_scaled)
Try a Different Solver:
lr_model = LogisticRegression(random_state=1, solver='liblinear')
Adjust the Tolerance:
lr_model = LogisticRegression(random_state=1, tol=1e-4)
Conclusion
While the convergence warning does not severely impact model performance, it is good practice to address it—especially through data scaling, which generally benefits model training overall.


This section will serve as a useful reference for anyone reviewing your project, illustrating both your problem-solving skills and attention to detail.

---

## How to Use

1. **Clone this Repository**  
   ```bash
   git clone https://github.com/Ebauer286/credit-risk-classification.git

2. **Install Dependencies**
Make sure you have a Python environment (3.7+) and install the necessary libraries:
pip install -r requirements.txt
(If you have a requirements.txt file. Otherwise, ensure pandas, numpy, and scikit-learn are installed.)

3. **Run the Notebook**
From the Credit_Risk folder, open the Jupyter Notebook:
jupyter notebook credit_risk_classification.ipynb

## Instructions

2. **Execute the cells in order to:**
   - Load and explore the dataset.
   - Split data into training and testing sets.
   - Train and evaluate the Logistic Regression model.
   - View the evaluation metrics.

3. **Review the Report**  
   For a detailed explanation of the analysis and results, see `credit_risk_report.md` in the `Credit_Risk` folder.

---

## License

This project is licensed under the MIT License. Feel free to modify or distribute this code as needed, but please provide attribution to the original source.

---

## Contact

- **Author:** Elizabeth Bauer 
- **LinkedIn:** https://www.linkedin.com/in/elizabeth-bauer-827a00a6/ 
- **Email:** [Your Email](ebauer286@gmail.com)

Feel free to open an issue or submit a pull request for any improvements or suggestions.### Tips for Organizing Your Repository

1. **Ensure all data files are in a `Resources` folder** to keep the main directory tidy.
2. **Add a `.gitignore`** to avoid committing large or sensitive files.
3. **Include a `requirements.txt`** if you want to list the specific Python dependencies and their versions.
4. **Keep your notebooks clean and well-documented**—remove any unnecessary or debugging cells before committing.
