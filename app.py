import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('diabetes_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Set app title and description
st.title('AI-Powered Diabetes Prediction System')
st.write("""
This app predicts the likelihood of diabetes based on patient health metrics.
Enter the patient's details and click 'Predict' to see the result.
""")

# Create input fields for user data
st.header('Patient Information')
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, value=0)
    glucose = st.number_input('Glucose Level (mg/dL)', min_value=0, max_value=200, value=100)
    blood_pressure = st.number_input('Blood Pressure (mm Hg)', min_value=0, max_value=130, value=70)
    skin_thickness = st.number_input('Skin Thickness (mm)', min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input('Insulin Level (mu U/ml)', min_value=0, max_value=850, value=80)
    bmi = st.number_input('BMI (kg/m²)', min_value=0.0, max_value=70.0, value=25.0, step=0.1)
    diabetes_pedigree = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.5, step=0.01)
    age = st.number_input('Age (years)', min_value=1, max_value=120, value=30)

# Create a button for prediction
if st.button('Predict Diabetes Risk'):
    # Prepare input data for prediction
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                           insulin, bmi, diabetes_pedigree, age]])
    
    # Make prediction
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)
    
    # Display results
    st.subheader('Prediction Result')
    if prediction[0] == 1:
        st.error('High risk of diabetes (Probability: {:.2f}%)'.format(prediction_proba[0][1]*100))
        st.write("Recommendation: Please consult a doctor for further evaluation and possible intervention.")
    else:
        st.success('Low risk of diabetes (Probability: {:.2f}%)'.format(prediction_proba[0][0]*100))
        st.write("Recommendation: Maintain healthy lifestyle with regular checkups.")

# Add some information about diabetes
st.markdown("""
### About Diabetes
Diabetes is a chronic disease that occurs when the pancreas is no longer able to make insulin, 
or when the body cannot make good use of the insulin it produces. Early detection can help 
prevent serious complications.
""")

# Add footer
st.markdown("---")
st.write("Developed as part of Engineering Project - AI-Powered Healthcare Solutions")