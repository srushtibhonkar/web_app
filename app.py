import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np


data = pickle.load(open('./Diabetes/model_diabetes.pkl','rb'))
model_diabetes = data['model']
scaler_diabetes = data['scaler']

data_1 = pickle.load(open('./Heart/model_heart.pkl','rb'))
model_heart = data_1['model']

data_2 = pickle.load(open('./Parkinsons/model_parkinson.pkl','rb'))
model_parkinsons = data_2['model']
scaler_parkinson = data_2['scaler']

data_3 = pickle.load(open('./autism/autism.pkl','rb'))
model_autism = data_3['model']
scaler_autism = data_3['scaler']

with st.sidebar: 
    selected = option_menu('Multiple Disease Prediction System',                          
                          ['Diabetes Prediction',
                           'Heart Disease Prediction',
                           'Parkinsons Prediction'],
                          icons=['activity','heart','person'],
                          default_index=0)


if (selected == 'Diabetes Prediction'):
    # page title
    st.title('Diabetes Prediction using ML')
    
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
        
    with col2:
        Glucose = st.text_input('Glucose Level')
    
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    
    with col2:
        Insulin = st.text_input('Insulin Level')
    
    with col3:
        BMI = st.text_input('BMI value')
    
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    
    with col2:
        Age = st.text_input('Age of the Person')

    diab_diagnosis = ''

    if st.button('Diabetes Test Result'):
    	X = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    	X = scaler_diabetes.transform(X)
    	diab_prediction = model_diabetes.predict(X)
    	diab_diagnosis = "The person is Diabetic" if (diab_prediction[0]) else "The person is not Diabetic"
    
    st.success(diab_diagnosis)




# Heart Disease Prediction Page
if (selected == 'Heart Disease Prediction'):
    
    # page title
    st.title('Heart Disease Prediction using ML')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age')
        
    with col2:
        sex = st.text_input('Sex')
        
    with col3:
        cp = st.text_input('Chest Pain types')
        
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
        
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
        
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
        
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
        
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
        
    with col3:
        exang = st.text_input('Exercise Induced Angina')
        
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
        
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')
        
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')
        
    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
        
        
     
     
    # code for Prediction
    heart_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Heart Disease Test Result'):
        Y = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,thalach,exang,oldpeak,slope,ca,thal]],dtype=np.float64)
        heart_prediction = model_heart.predict(Y)                          
        if (heart_prediction[0] == 1):
        	heart_diagnosis = 'The person is having heart disease'
        else:
        	heart_diagnosis = 'The person does not have any heart disease'

    st.success(heart_diagnosis)
        
    
    

# Parkinson's Prediction Page
if (selected == "Parkinsons Prediction"):
    
    # page title
    st.title("Parkinson's Disease Prediction using ML")
    
    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
        
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
        
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
        
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')
        
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
        
    with col1:
        RAP = st.text_input('MDVP:RAP')
        
    with col2:
        PPQ = st.text_input('MDVP:PPQ')
        
    with col3:
        DDP = st.text_input('Jitter:DDP')
        
    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')
        
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
        
    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')
        
    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')
        
    with col3:
        APQ = st.text_input('MDVP:APQ')
        
    with col4:
        DDA = st.text_input('Shimmer:DDA')
        
    with col5:
        NHR = st.text_input('NHR')
        
    with col1:
        HNR = st.text_input('HNR')
        
    with col2:
        RPDE = st.text_input('RPDE')
        
    with col3:
        DFA = st.text_input('DFA')
        
    with col4:
        spread1 = st.text_input('spread1')
        
    with col5:
        spread2 = st.text_input('spread2')
        
    with col1:
        D2 = st.text_input('D2')
        
    with col2:
        PPE = st.text_input('PPE')
        
    
    
    # code for Prediction
    parkinsons_diagnosis = ''
    
    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):
    	Z = np.array([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]],dtype=np.float64)
    	Z = scaler_parkinson.transform(Z)
    	parkinsons_prediction = model_parkinsons.predict(Z)
    	parkinsons_diagnosis = "The person has Parkinson's disease" if (parkinsons_prediction[0]) else "The person does not have Parkinson's disease"

    st.success(parkinsons_diagnosis)



# Heart Disease Prediction Page
if (selected == 'Autism Disease Prediction'):
    
    # page title
    st.title('Autism Disease Prediction using ML')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('A1')
        
    with col2:
        sex = st.text_input('A2')
        
    with col3:
        cp = st.text_input('A3')
        
    with col1:
        trestbps = st.text_input('Age_Mons')
        
    with col2:
        chol = st.text_input('Sex')
        
    with col3:
        fbs = st.text_input('Ethnicity')
        
    with col1:
        restecg = st.text_input('Jaundice')
        
    with col2:
        thalach = st.text_input('Family_mem_with_ASD')
        
    with col3:
        exang = st.text_input('Who completed the test')
          
        
     
     
    # code for Prediction
    autism_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Autism Disease Test Result'):
        Y = np.array([[A1, A2, A3, Age_Mons, Sex, Ethnicity, Jaundice, Family_mem_with_ASD, Who completed the test]],dtype=np.float64)
        autism_prediction = model_autism.predict(Y)                          
        if (autism_prediction[0] == 1):
        	autism_diagnosis = 'The person is having heart disease'
        else:
        	autism_diagnosis = 'The person does not have any heart disease'

    st.success(autism_diagnosis)