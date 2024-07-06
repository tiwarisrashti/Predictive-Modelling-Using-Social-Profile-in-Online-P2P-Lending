from flask import Flask, render_template, request, jsonify
import dill
import numpy as np
import pandas as pd
import os
import logging

application = Flask(__name__, static_folder='static')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load all the trained pipelines
multiclass_targets = ['EMI', 'ELA', 'PROI']
pipelines = {}
for target in multiclass_targets:
    file_path = f'combined_pipeline_{target}.pkl'
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Pipeline file not found: {file_path}")
        
        with open(file_path, 'rb') as f:
            pipelines[target] = dill.load(f)
        logger.info(f"Successfully loaded pipeline for {target}")
    except Exception as e:
        logger.error(f"Error loading pipeline for {target}: {str(e)}")
        pipelines[target] = None

@application.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@application.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract inputs from the form
        loan_amount = float(request.form['emiInput'])
        loan_tenure = int(request.form['elaInput'])
        interest_rate = float(request.form['InterestRateInput'])
        monthly_income = float(request.form['statemonthlyIncomeInput'])
        monthly_loan_payment = float(request.form['monthlyloanpaymentInput'])
        prosper_rating = int(request.form['ProsperRatingInput'])
        prosper_score = int(request.form['ProsperScoreInput'])
        debt_to_income_ratio = float(request.form['DebtToIncomeRatioInput'])
        income_verification = request.form['incomeVerification'] == 'Yes'
        lp_customer_principal_payments = float(request.form['LP_CustomerPrincipalPaymentsInput'])
        loan_current_days_delinquent = float(request.form['LoanCurrentDaysDelinquentInput'])
        income_range = request.form['IncomeRangeInput']
        employment_status = request.form['EmploymentStatusInput']

        # Input validation
        if not (0 < loan_amount <= 1000000):
            return jsonify({'error': 'Loan amount out of acceptable range'}), 400
        if not (1 <= loan_tenure <= 360):
            return jsonify({'error': 'Loan tenure out of acceptable range'}), 400
        if not (0 <= interest_rate <= 100):
            return jsonify({'error': 'Interest rate out of acceptable range'}), 400
        if not (0 <= debt_to_income_ratio <= 10.01):
            return jsonify({'error': 'Debt to income ratio out of acceptable range'}), 400

        # Prepare the input as a DataFrame
        data = {
            'StatedMonthlyIncome': [monthly_income],
            'LoanOriginalAmount': [loan_amount],
            'LoanTenure': [loan_tenure],
            'ProsperScore': [prosper_score],
            'MonthlyLoanPayment': [monthly_loan_payment],
            'BorrowerRate': [interest_rate / 100],
            'LoanCurrentDaysDelinquent': [loan_current_days_delinquent],
            'ProsperRating (numeric)': [prosper_rating],
            'IncomeVerifiable': [income_verification],
            'DebtToIncomeRatio': [debt_to_income_ratio],
            'LP_CustomerPrincipalPayments': [lp_customer_principal_payments],
        }
        df = pd.DataFrame(data)
        logger.debug(f"Input DataFrame: {df}")

        # Make predictions for each target using respective pipelines
        predictions = {}
        for target in multiclass_targets:
            pipeline = pipelines[target]
            if pipeline is not None:
                try:
                    prediction = pipeline.predict(df)
                    logger.info(f"Raw prediction for {target}: {prediction}")
                    
                    if isinstance(prediction[0], (int, float)):
                        if target == 'PROI':
                            predictions[target] = f"{float(prediction[0]) * 100:.2f}%"
                        else:
                            predictions[target] = round(float(prediction[0]), 2)
                    elif isinstance(prediction[0], str):
                        predictions[target] = prediction[0]  # Keep it as a string if it's already a string
                    else:
                        raise ValueError(f"Unexpected prediction type for {target}: {type(prediction[0])}")
                    
                except Exception as e:
                    logger.error(f"Error making prediction for {target}: {str(e)}")
                    predictions[target] = f"Error: {str(e)}"
            else:
                logger.error(f"Pipeline for {target} is None")
                predictions[target] = 'Error: Pipeline not loaded'

        # Prepare the user data to display
        user_data = {
            'Loan Amount': loan_amount,
            'Loan Tenure': loan_tenure,
            'Interest Rate': interest_rate,
            'Monthly Income': monthly_income,
            'Monthly Loan Payment': monthly_loan_payment,
            'Prosper Rating': prosper_rating,
            'Prosper Score': prosper_score,
            'Debt To Income Ratio': debt_to_income_ratio,
            'Income Verification': 'Yes' if income_verification else 'No',
            'LP Customer Principal Payments': lp_customer_principal_payments,
            'Loan Current Days Delinquent': loan_current_days_delinquent,
            'Income Range': income_range,
            'Employment Status': employment_status
        }
        # Get current working directory
        
        print("User Data:", user_data)
        print("Final predictions:", predictions)
    
        logger.info(f"Prediction made for input: {user_data}")
        return render_template('index.html', prediction=predictions, user_data=user_data)
    except ValueError as ve:
        logger.error(f"Invalid input: {str(ve)}")
        return jsonify({'error': f"Invalid input: {str(ve)}"}), 400
    except KeyError as ke:
        logger.error(f"Missing input field: {str(ke)}")
        return jsonify({'error': f"Missing input field: {str(ke)}"}), 400
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        return jsonify({'error': f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    application.run(debug=False, host='0.0.0.0', port=5000)
