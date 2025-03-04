# Module 12 Report Template

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.
  The purpose of this analysis was to determine if the Linear Regression machine learning model is better suited for predicting healthy loans or high risk loans using the given dataset.
  
* Explain what financial information the data was on, and what you needed to predict.
    The dataset provided consisted of information on 77,536 loans. Within those loans, we were provided specific data on: loan size, interest rate, borrower income, debt to income ratio, number of accounts, derogartory mark, and total debt. We were left to do an analysis on the loan status and determine of the 77,536 loans provided how many were healthy loans vs. high risk loans.
  
* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
  The prediction is the labels that are created from the data features (variable X) to determine the risk assesment for the loan status (variable y).
  
* Describe the stages of the machine learning process you went through as part of this analysis.
  Step 1: Importing the csv data set and creating the data frame so we can evaluate the data features and columns.
  Step 2: Splitting the data frame into two variables X (data features) and y (loan status). Labels are created in order for us to be able to predict the risk assesment if the loans are healthy (0) or high risk (1)
  Step 3: Splitting the features and label data in the training and testing datasets by using the train_test_split function.
  Step 4: Import LogisticRegression machine learning model from SKLearn.
  Step 5: Run the model
  Step 6: Use the training data to fit the model to make predictions based on the features from the test dataset.
  Step 7: Evaluate the predictions by using a confusion_matrix from SKLearn and classification_report from SKLearn.
  
* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any other algorithms).
  Methods used: train_test_split from SKLearn, confusion_matrix from SKLearn, classification_report from SKLearn

## Results

Using bulleted lists, describe the accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
    * Description of Model 1 Accuracy, Precision, and Recall scores.
    -Accuray score: .99
    -Precision:
        -Class 0: 1.00
        -Class 1: .84
    -Recall:
        -Class 0: .99
        -Class 1: .94  

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:

* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

If you do not recommend any of the models, please justify your reasoning.

Overall, the Logistic Regression model performed well, especially in predicting outcomes for Class 0 (healthy loans), where both precision and recall were extremely close to perfect.

For Class 1 (high-risk loans), the model's precision is reported as 0.84 (noting a slight discrepancy from the 0.85 in the results) while the recall is 0.91. This suggests that the model may produce more false positives than false negatives for high-risk loans.

Given this information, the Logistic Regression model appears to do a great job at predicting both healthy and high-risk loans based on the features used for training. However, it is recommended that the model be further evaluated against different datasets before being implemented in production.
