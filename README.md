You can access our app by following this link [P2P_Lending_Flask_App](http://p2plendingteama.ap-south-1.elasticbeanstalk.com/).
---
# Predictive Modelling Using Social Profile in Online P2P Lending Platform

Online peer-to-peer (P2P) lending markets enable individual consumers to borrow from and lend money to one another directly. We study the borrower-, loan-, and social-related determinants of performance predictability in an online P2P lending market by conceptualizing financial and social strength to predict whether borrowers could be funded with lower interest and whether lenders would be timely paid.

## Understanding the Dataset
The Prosper dataset contains all the transaction and member data since its inception in November 2005. This is a considerable volume of information that encloses approximately (by December 2008) 6 million bids, 900,000 members, 4,000 groups, and 350,000 listings. In order to facilitate the analysis of the data, the dataset was filtered to contain all the loans created in the calendar year 2007, all the listings created in the calendar year 2007, the bids created for these listings, the subset of members that created these listings and bids, and finally.

The Prosper loan dataset contains 113,937 loans with 81 variables on each loan, including loan amount, borrower rate (or interest rate), current loan status, borrower income, and many others.

## Preprocessing and Sentiment Analysis
**Handling Null Values**: Some variables, such as TotalProsperLoans, have a lot of null values, along with other variables associated with the Prosper history of the debtor.

**Removing Redundant Variables**: Many variables, like ListingKey and ListingNumber, seem to be administrative identifiers and are redundant. They won't be of much use to us.

**Handling Categorical Variables**: There are 17 variables of type object that will probably need some attention before they're ready for some of our classification models.

**Rescaling Variables**: Variables have a wide variety of ranges, indicating a need for rescaling to make them more amenable to some of our classifiers.

**Data Formatting**: Renamed columns with spaces, converted date columns to datetime, and converted categorical columns to category type.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
rescaledValidationX = scaler.transform(X_valid)
```

**fit_transform()** is used on the training data so that we can scale the training data and also learn the scaling parameters of that data. Here, the model built by us will learn the mean and variance of the features of the training set. These learned parameters are then used to scale our test data.

**transform()** uses the same mean and variance as it is calculated from our training data to transform our test data. Thus, the parameters learned by our model using the training data will help us to transform our test data. As we do not want to be biased with our model, we want our test data to be a completely new and a surprise set for our model.

## Exploratory Data Analysis (EDA)
**Introduction:**

- **merged_data** dataset comprises 113,066 rows and 77 columns.
- The dataset comprises continuous variables and float data types.

### **Univariate Analysis:**

**Variable Distribution and Data Operations**: We created a new credit rating variable by merging two existing rating variables, which were mutually exclusive by the loan origination date. We further created a credit risk variable, categorizing the rating into three groups: low, medium, or high risk. The loan original amount is multimodal, with the most popular loaned amounts being 4k, 15k, and $10k. From the two similar interest rates variables, I chose the borrower annual percentage rate.

**Unusual Distributions and Data Operations**: We deleted two loans with their closed date earlier than the loan origination date. We reduced the number of income ranges by one by merging USD 0 with Not Employed into the same category. We created an active loans dummy to quickly filter all active or non-active loan statuses from the data. We looked at a couple of right-skewed variables using a logarithmic scale to better see where the bulk of the data points lie.

### **Bivariate Analysis:**

**Relationships Observed:** Borrower APR is negatively correlated with the loan original amount, monthly loan payment, and credit score range lower, which makes sense. Borrowers who can borrow larger amounts are likely more solvent, thus getting lower interest rates and having higher credit scores, allowing them to afford higher monthly payments. The interconnectedness of the main features is also shown by the positive correlation between loan original amount, monthly loan payment, and credit score.

The credit rating clearly separates borrowers into different interest rate groups, which is what we would expect from a reasonable lending business. High-risk borrowers, for example, get loans for much lower amounts than medium or low-risk borrowers.

**Interesting Relationships Between Other Features:**
Higher-income borrowers and homeowners borrow higher amounts for lower rates compared to lower-income borrowers.
Loans with higher interest rates default more often.
Unemployed borrowers pay the highest interest among all employment statuses.
Higher loan amounts are borrowed for longer terms.

### **Multivariate Analysis:**

**Relationships Observed**:

The credit risk category is a major factor that segments borrowers into distinguishable groups. It strongly determines the interest rate and the borrowed amount. Borrowed amounts are increasing overall in the last few years, while the average interest rate is decreasing, especially for the medium and high-risk groups.

**Interesting Interactions**:

House ownership or income range are also significant for borrower segmentation, but only until combined with credit risk. This is because these variables feed into the credit score/credit rating/credit risk calculations. Once a borrower is assigned a credit rating/credit risk, these variables make no difference anymore.

# Descriptive Statistics

We generated descriptive statistics of the dataset to understand the central tendency, dispersion and distribution of the data. We also Created the correlation matrix using seaborn heatmap to visually identify relationships and patterns in the dataset.

# Features Creation

We describe the creation of new features for loan data analysis. These features help in understanding the financial implications for borrowers and lenders, and provide insights into the eligibility and repayment capabilities of the borrowers. The features created include Equated Monthly Installments (EMI), Eligible Loan Amount (ELA), modified loan status, and Preferred Return on Investment (PROI). Additionally, we calculate the Return on Investment (ROI) to evaluate the financial performance of the loans.

### Equated Monthly Installments (EMI)

**Components:**
- **LoanTenure**: Loan tenure
- **LP_CustomerPrincipalPayments**: Principal repayment
- **BorrowerRate**: Interest rate

**Calculation Procedure:**
For each row in the dataset:
1. Calculate `result_1 = P * r * (1 + r)^n`
2. Calculate `result_2 = (1 + r)^n - 1`
3. Calculate `EMI = result_1 / result_2`

The EMI feature represents the fixed monthly payment made by the borrower, which includes both principal and interest components. This calculation is crucial for assessing the borrower's ability to repay the loan over the specified tenure.

### Eligible Loan Amount (ELA)

**Components:**
- **LoanOriginalAmount**: Applied amount
- **BorrowerRate**: Interest rate
- **LoanTenure**: Loan tenure
- **StatedMonthlyIncome**: Income

**Calculation Procedure:**
For each row in the dataset:
1. Calculate: `Total Payment Due = (A + (A * r)) * n`
2. Calculate: `Max allowable amount = I * 12 * 30%`
3. If `Total Payment Due <= Max allowable amount`
   - Then `ELA = AppliedAmount`
   - Else `ELA = Max allowable amount`

The ELA feature determines the maximum loan amount a borrower is eligible for based on their income and loan terms. This ensures that the loan amount does not exceed the borrower's repayment capacity.


### Preferred Return on Investment (PROI)

**Calculation Procedure:**
1. Calculate the interest amount: `InterestAmount = LoanOriginalAmount * BorrowerRate * LoanTenure /12`
2. Calculate the total amount: `TotalAmount = InterestAmount + LoanOriginalAmount`
3. Calculate ROI: `ROI =(InterestAmount/TotalAmount) / (LoanTenure/12)`
4. Initialize PROI with the median of ROI.
5. Adjust PROI based on various conditions:
   - Increase if `LP_CustomerPrincipalPayments <= 1000`
   - Decrease if `LP_CustomerPrincipalPayments > 2000` and `LP_CustomerPrincipalPayments <= 10500`
   - Decrease if `LP_CustomerPrincipalPayments > 10500`
   - Increase if `ProsperRating (Alpha)` is 'C' or 'D'
   - Decrease if `ProsperRating (Alpha)` is 'G'
   - Decrease if `LoanOriginalAmount <= 2000`
   - Increase if `LoanOriginalAmount > 19500` and `LoanOriginalAmount <= 25500`
   - Increase if `LoanOriginalAmount > 25500`
   - Increase if `LoanCurrentDaysDelinquent >= 50`
   - Decrease if `MonthlyLoanPayment <= 90`
   - Increase if `MonthlyLoanPayment > 360` and `MonthlyLoanPayment <= 750`

The PROI feature provides a refined measure of the expected return on investment for each loan, considering various factors like principal payments, credit rating, loan amount, delinquency days, and monthly payments. This helps in evaluating the profitability and risk associated with each loan.

### Loan Status Modification

**Modification Logic:**
- The loan status is updated based on the closed date and delinquency days.
- If the loan has a closed date and the current days delinquent exceed 180 days, the loan status is set to 1.
- Otherwise, the loan status is set to 0.

This modification helps in accurately reflecting the current status of the loan, identifying loans that are significantly delinquent and have been closed.

# Data Encoding

We applied label encoding on categorical variables namely 'ProsperRating', 'ListingCategory', 'BorrowerState', 'Occupation', 'EmploymentStatus', 'IncomeRange', 'LoanOriginationQuarter', 'MemberKey', 'CreditRating', 'CreditRisk' to convert them to numerical features. This process maps each unique category to a unique integer.

# Feature Selection 

Feature selection process is to identify the most significant features that influence the target variable, `Loan Status, EMI, ELA, PROI`. By selecting the top features, we aim to improve the predictive power of our models and streamline the data for more efficient processing.

## Process Overview

The feature selection process involved the following steps:
1. Splitting the data into features (`X`) and the target variable (`y`).
2. Performing Mutual Information Analysis to compute the mutual information between each feature and the target variable.
3. Creating a DataFrame to store and sort mutual information scores.
4. Selecting the top features based on the highest mutual information scores.

### Splitting Data

- **Target Variable**: `Loan Status, EMI, ELA, PROI`
- **Features**: All columns except `Loan Status, EMI, ELA, PROI` were considered as features.

### Mutual Information Analysis

Mutual Information Analysis was used to measure the dependency between each feature and the target variable. Mutual information is a measure of the mutual dependence between two variables. It quantifies the amount of information obtained about one variable through the other variable.

#### Steps:

1. **Compute Mutual Information**: Mutual information scores were computed between each feature in `X` and the target variable `y`.
2. **Create DataFrame**: A DataFrame was created to store the features and their corresponding mutual information scores.
3. **Select Top Features**: The top features with the highest mutual information scores were selected.

### Results

The mutual information scores for each feature were computed and sorted to determine the most significant features influencing the target variable. The top features were selected based on their mutual information scores.

#### Top Features

Based on the mutual information scores, the top features selected are:

1. **StatedMonthlyIncome**
2. **LoanOriginalAmount**
3. **LoanTenure**
4. **ProsperScore**
5. **MonthlyLoanPayment**
6. **BorrowerRate**
7. **LoanCurrentDaysDelinquent**
8. **ProsperRating (numeric)**
9. **IncomeVerifiable**
10. **DebtToIncomeRatio**
11. **LP_CustomerPrincipalPayments**

These features were identified as the most significant in predicting the target variable `Loan Status, EMI, ELA, PROI`.

## Model Building

### Metrics considered for Model Evaluation
**Accuracy, Precision, Recall, and F1 Score**
- **Accuracy**: What proportion of actual positives and negatives is correctly classified?
- **Precision**: What proportion of predicted positives are truly positive?
- **Recall**: What proportion of actual positives is correctly classified?
- **F1 Score**: Harmonic mean of Precision and Recall.

### Binary Classification Problem

### Logistic Regression
- Logistic Regression helps find how probabilities are changed with actions.
- The function is defined as P(y) = 1 / (1 + e^-(A+Bx)).
- Logistic regression involves finding the **best fit S-curve** where A is the intercept and B is the regression coefficient. The output of logistic regression is a probability score.

### Naive Bayes
- Naive Bayes classifiers are a family of simple probabilistic classifiers based on applying Bayes' theorem with strong (naive) independence assumptions between the features.
- The probability of a class given a feature is calculated by multiplying the conditional probabilities of the feature values for the class.
- **Advantages**: It is easy to implement and can handle a large number of features.
- **Disadvantages**: It assumes that all features are independent, which is often not the case in real-world applications.

### Gradient Boosting Tree Model
- Gradient Boosting is a machine learning technique for regression and classification problems, which builds a model in a stage-wise fashion from weak learners like decision trees.
- It combines the predictions of several base estimators to improve robustness over a single estimator.
- **Boosting**: This method focuses on training new models to correct the errors made by previous models.
- **Advantages**: It can handle a variety of data types and is very powerful.
- **Disadvantages**: It can be prone to overfitting if not properly tuned.

### Linear Discriminant Analysis (LDA)
- Linear Discriminant Analysis uses the information from both the selection and target features to create a new axis and projects the data onto the new axis in such a way as to **minimize the variance and maximize the distance between the means of the two classes**.
- Both LDA and PCA are linear transformation techniques: LDA is supervised whereas PCA is unsupervised – PCA ignores class labels. LDA chooses axes to maximize the distance between points in different categories.
- PCA performs better in cases where the number of samples per class is less. Whereas LDA works better with large datasets having multiple classes; class separability is an important factor while reducing dimensionality.
- Linear Discriminant Analysis fails when the covariances of the X variables are a function of the value of Y.


### Model Evaluation and Selection

#### Model Accuracies
- **Logistic Regression**: 0.9445514054678475
- **Naive Bayes**: 0.7637658837119754
- **Gradient Boosting Tree (GBT)**: 0.9532152483634964
- **Linear Discriminant Analysis (LDA)**: 0.8623411628802464

#### Model Selection
Based on the evaluation metrics, we selected the Gradient Boosting Tree (GBT) model for the following reasons:

1. **Highest Accuracy**: The GBT model achieved the highest accuracy of 0.9532, outperforming the other models.
2. **Robustness**: GBT is known for its robustness as it combines the predictions of several base estimators, improving the overall performance and reducing the risk of overfitting.
3. **Handling Data Types**: GBT can handle a variety of data types and can model complex relationships in the data, making it suitable for our classification problem.
4. **Model Improvement**: GBT focuses on correcting the errors made by previous models, leading to continuous improvement and better performance.

Given these advantages, the GBT model is the most suitable choice for our binary classification task.

### Multiclass Classification Problem

Target varible were categorized into 5 categories as follows : 'Very Low','Low', 'Medium', 'High', 'Very High'

### AdaBoost Classifier

We used the AdaBoost classifier to predict each of the multiclass targets. The steps involved are:

1. **Model Training**: AdaBoost models were trained on the prepared features.
2. **Model Evaluation**: Accuracy was used as the primary evaluation metric.

**AdaBoost Performance Metrics:**
- **EMI**: 0.9218
- **ELA**: 0.7911
- **PROI**: 0.8693
- **Overall Accuracy**: 0.8607

### Gradient Boosting Classifier
We also employed Gradient Boosting Classifier for the multiclass targets, following similar steps:

1. **Model Training**: Gradient Boosting models were trained on the prepared features.
2. **Model Evaluation**: Accuracy was used as the primary evaluation metric.

**Gradient Boosting Performance Metrics:**
- **EMI**: 0.9848
- **ELA**: 0.9755
- **PROI**: 0.9896
- **Overall Accuracy**: 0.9833

### Shapes of Training and Testing Sets
- **Training Set**:
  - Features (X_train): (20775, 30)
  - Targets (y_train): (20775, 4)
- **Testing Set**:
  - Features (X_test): (5194, 30)
  - Targets (y_test): (5194, 4)

### Final Model Selection
Based on the evaluation, we selected the Gradient Boosting Regressor for multiclass targets due to its superior performance metrics and robustness in handling complex data relationships.   

### Multiclass Regression Model
For each multiclass target, we used a Gradient Boosting Regressor to predict the target values. The steps involved are:

1. **Feature Union**: Combined original features with the probabilities from the binary classifier using FeatureUnion.
2. **Polynomial Features**: Generated polynomial features to capture interactions.
3. **Scaling**: StandardScaler was used to standardize the combined features.
4. **Model Training**: Gradient Boosting Regressor models were trained on the prepared features.

### Performance Metrics

For each multiclass target, the performance metrics are as follows:

- **EMI**:
  - Mean Squared Error (MSE): `overall_mse['EMI']`
  - R² Score: `overall_r2['EMI']`
  - Accuracy: 0.9255
- **ELA**:
  - Mean Squared Error (MSE): `overall_mse['ELA']`
  - R² Score: `overall_r2['ELA']`
  - Accuracy: 0.9141
- **PROI**:
  - Mean Squared Error (MSE): `overall_mse['PROI']`
  - R² Score: `overall_r2['PROI']`
  - Accuracy: 0.7156

#### Overall Accuracy
The overall accuracy for the multiclass targets is: 0.8518

# Pipeline

The outline the process and results of building predictive models for both binary classification (`Loan Status`) and multiclass regression (`EMI`, `ELA`, `PROI`). The pipeline includes data preparation, model training, evaluation, and saving/loading of pipelines for deployment.

## Process Overview

### Binary Classification Pipeline

1. **Data Preparation**:
   - Features (`X_binary`) and target variable (`y_binary`) are defined.
   - Top features based on mutual information are selected (`top_features_binary`).
   - Data is split into training and testing sets.

2. **Model Training**:
   - Features are scaled using `StandardScaler`.
   - Logistic Regression model (`binary_clf`) is used for binary classification.

3. **Pipeline Creation**:
   - Pipeline (`binary_pipeline`) is created with scaling and binary classifier.

4. **Model Evaluation**:
   - Accuracy of the binary classification model is calculated.

5. **Results**:
   - Accuracy of the binary classification model is 99.97%.

### Multiclass Regression Pipeline

1. **Data Preparation**:
   - Features (`X_multi`) and target variables (`y_multi`) for `EMI`, `ELA`, and `PROI` are defined.
   - Top features based on mutual information are selected (`top_features_multi`).
   - Data is split into training and testing sets.

2. **Model Training**:
   - Gradient Boosting Regressors are used for each multiclass target (`EMI`, `ELA`, `PROI`).

3. **Pipeline Fitting**:
   - Pipelines for each target are fitted and evaluated using Mean Squared Error (MSE) and R-squared (R^2).
   
4. **Model Evaluation**:
   - Accuracy of the Multiclass Regression model is calculated.


### Combined Pipeline

1. **Pipeline Overview**:
   - A combined pipeline integrates binary probabilities into the multiclass regression models.
   - Features include original features and binary probabilities from the binary classifier.

2. **Custom Transformer**:
   - `BinaryProbTransformer` is defined to extract probabilities from the binary classifier.

3. **Model Training**:
   - Gradient Boosting Regressors are used for each multiclass target (`EMI`, `ELA`, `PROI`).
   - Combined pipeline includes feature union, polynomial features, scaling, and regressor.

4. **Pipeline Fitting**:
   - Pipelines for each target are fitted and evaluated using Mean Squared Error (MSE) and R-squared (R^2).


5. **Results**:
   - **Combined Pipeline Multiclass Regression Report:**
     - **EMI:**
       - Mean Squared Error: 72.0185
       - R^2 Score: 0.9999
     - **ELA:**
       - Mean Squared Error: 3097205.8294
       - R^2 Score: 0.9992
     - **PROI:**
       - Mean Squared Error: 0.0000
       - R^2 Score: 0.9960

   - **Overall Mean Squared Error and R^2 for each multiclass target:**
     - EMI: MSE=72.0185, R^2=0.9999
     - ELA: MSE=3097205.8294, R^2=0.9992
     - PROI: MSE=0.0000, R^2=0.9960

6. **Predictions**:
   - Predictions are made using the saved pipelines for each target.

7. **Comparison**:
   - Comparison DataFrame (`comparison_df`) is created to compare actual vs. predicted values.

8. **Model Persistence**:
   - Pipelines are saved in the pickle file as (`combined_pipeline_<target>.pkl`) for each multiclass target.


The summary the process of building and evaluating predictive models for binary classification (`Loan Status`) and multiclass regression (`EMI`, `ELA`, `PROI`). The models are trained, evaluated, and saved for deployment, aiming to predict loan default status and other financial metrics accurately. The results demonstrate the effectiveness of the models in predicting these outcomes based on selected features and model configurations.

# Deployment
You can access our app by following this link [P2P_Lending_Flask_App](http://p2plendingteama.ap-south-1.elasticbeanstalk.com/).

## Flask
- Flask is a tool that lets you create applications for your machine learning model using simple Python code.
- We wrote Python code for our app using Flask; the app asks the user to enter the following data: loan amount, loan tenure, interest rate, monthly income, monthly loan payment, prosper rating, prosper score, debt to income ratio, income verification, customer principal payments, loan current days delinquent, income range, and employment status.
- The output of our app will be predictions for EMI, ELA, and PROI, where:
  - **EMI** represents the Expected Monthly Installment.
  - **ELA** represents the Estimated Loan Amount.
  - **PROI** represents the Preferred Rate of Interest.
- The app runs on localhost for testing.
- To deploy it on the internet, we deployed it to AWS.

## AWS
- We deployed our Flask app on AWS Elastic Beanstalk and integrated it with AWS CodePipeline and GitHub for continuous deployment. This way, we can share our app on the internet with others.

- We prepared the necessary files to deploy our app successfully:

- `requirements.txt`: Contains the libraries that must be downloaded to run the app file (`application.py`) successfully.
- `application.py`: Contains the Python code of a Flask web app.
- `combined_pipeline_EMI.pkl`, `combined_pipeline_ELA.pkl`, and `combined_pipeline_PROI.pkl`: Contain our trained models for different targets.
- `static` and `templates`: Folders containing static files and HTML templates for the web app.

## Conclusion
- The study provided valuable insights into the predictive power of various borrower, loan, and social-related features in an online P2P lending market. Our predictive models showcased the importance of credit risk, loan amount, interest rate, and borrower income in determining the likelihood of loan default. By understanding these determinants, stakeholders can make more informed decisions, ultimately improving the efficiency and effectiveness of P2P lending platforms.