from flask import Flask, render_template, request, redirect, url_for
from joblib import load
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load the trained XGBoost model
loaded_model = load('parkinsons_xgboost_model.joblib')

# Landing page
@app.route('/', methods=['GET'])
def landing():
    # Render the landing page template
    return render_template('index.html')

# Login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Placeholder logic for checking login credentials
        username = request.form['username']
        password = request.form['password']

        # Placeholder check for login credentials
        if username == 'testuser' and password == 'testpassword':
            # Redirect to data collection on successful login
            return redirect(url_for('data_collection'))
        else:
            # Redirect back to login page on failed login
            return redirect(url_for('login'))

    return render_template('login.html')

# Signup page and handling signup form submission (placeholder logic)
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        # Placeholder logic for processing signup data (replace with actual signup logic)
        username = request.form['username']
        password = request.form['password']
        # Process the signup data here
        
        # Redirect to data collection after successful signup
        return redirect(url_for('data_collection'))
    return render_template('signup.html')

# Data collection page and handling form submission for prediction
@app.route('/data_collection', methods=['GET', 'POST'])
def data_collection():
    if request.method == 'POST':
        print("Form submitted")  # Add this line for debugging

        # Placeholder logic for data collection (replace with your actual logic)
        name = request.form['name']
        mdvp_fo = float(request.form['mdvp_fo'])
        mdvp_fhi = float(request.form['mdvp_fhi'])
        mdvp_flo = float(request.form['mdvp_flo'])
        mdvp_jitter = float(request.form['mdvp_jitter'])
        mdvp_jitter_abs = float(request.form['mdvp_jitter_abs'])
        mdvp_rap = float(request.form['mdvp_rap'])
        mdvp_ppq = float(request.form['mdvp_ppq'])
        jitter_ddp = float(request.form['jitter_ddp'])
        mdvp_shimmer = float(request.form['mdvp_shimmer'])
        mdvp_shimmer_db = float(request.form['mdvp_shimmer_db'])
        shimmer_apq3 = float(request.form['shimmer_apq3'])
        shimmer_apq5 = float(request.form['shimmer_apq5'])
        mdvp_apq = float(request.form['mdvp_apq'])
        shimmer_dda = float(request.form['shimmer_dda'])
        nhr = float(request.form['nhr'])
        hnr = float(request.form['hnr'])
        status = float(request.form['status'])
        rpde = float(request.form['rpde'])
        dfa = float(request.form['dfa'])
        spread1 = float(request.form['spread1'])
        spread2 = float(request.form['spread2'])
        d2 = float(request.form['d2'])
        ppe = float(request.form['ppe'])
        # Add the remaining form fields according to your dataset columns

        # Prepare the collected data in the format expected by the model
        input_data = pd.DataFrame({
            'MDVP:Fo(Hz)': [mdvp_fo],
            'MDVP:Fhi(Hz)': [mdvp_fhi],
            'MDVP:Flo(Hz)': [mdvp_flo],
            'MDVP:Jitter(%)': [mdvp_jitter],
            'MDVP:Jitter(Abs)': [mdvp_jitter_abs],
            'MDVP:RAP': [mdvp_rap],
            'MDVP:PPQ': [mdvp_ppq],
            'Jitter:DDP': [jitter_ddp],
            'MDVP:Shimmer': [mdvp_shimmer],
            'MDVP:Shimmer(dB)': [mdvp_shimmer_db],
            'Shimmer:APQ3': [shimmer_apq3],
            'Shimmer:APQ5': [shimmer_apq5],
            'MDVP:APQ': [mdvp_apq],
            'Shimmer:DDA': [shimmer_dda],
            'NHR': [nhr],
            'HNR': [hnr],
            'status': [status],
            'RPDE': [rpde],
            'DFA': [dfa],
            'spread1': [spread1],
            'spread2': [spread2],
            'D2': [d2],
            'PPE': [ppe]
            # Add the remaining columns based on your dataset
        })

        app.logger.info(request.form)

        # Use the loaded XGBoost model to make predictions
        prediction = loaded_model.predict(input_data)

        # Determine the result based on the prediction
        result = "Parkinson's Disease Present" if prediction[0] == 1 else "Parkinson's Disease Absent"

        # Return the result to the user
        return render_template('prediction_result.html', result=result)

        # # Placeholder: Perform prediction (replace this with your model prediction logic)
        # prediction = "Yes"  # Change this placeholder prediction result

        # # Render the result page with the prediction result
        # return render_template('prediction_result.html', prediction=prediction)

    return render_template('data_collection.html')  # Render the data collection form

if __name__ == "__main__":
    app.run(debug=True)
