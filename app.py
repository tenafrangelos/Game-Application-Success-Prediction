from flask import Flask, jsonify, request, render_template
import os
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, BaggingRegressor, RandomForestRegressor
import streamlit as st
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import re
import joblib

#app = Flask(__name__)

# Load the dataset and split it into training and test sets
data = pd.read_csv('D:/FCIS/Pattern/project/games-regression-dataset.csv')
x_train, x_test, y_train, y_test = train_test_split(data.drop('Average User Rating', axis=1), data['Average User Rating'], test_size=0.20, random_state=42)

def date(data,x):
    data[x] = pd.to_datetime(data[x], dayfirst=True,infer_datetime_format=True)
    data[x[0]+'_Year'] = pd.DatetimeIndex(data[x]).year
    data[x[0]+'_Month'] = pd.DatetimeIndex(data[x]).month
    data[x[0]+'_Day'] = pd.DatetimeIndex(data[x]).day
    data[x]=data[x].map(datetime.toordinal)

# Define the list of columns to display in the app
cols_to_display = ['URL', 'ID', 'Name', 'Subtitle', 'Icon URL', 'User Rating Count', 'Price', 'In-app Purchases', 
                   'Description', 'Developer', 'Age Rating', 'Languages', 'Size', 'Primary Genre', 'Genres', 
                   'Original Release Date', 'Current Version Release Date', 'Average User Rating']

# Define the list of models and their names
model_names = ['encoder', 'le', 'MLB', 'multiLB', 'scaler', 'lr', 'mr', 'pr', 'rr', 'lr_lasso', 'en', 'br', 'gbr', 'bgr', 'rfr', 'xgb']
models = []

def load_models():
    # Load the models from their respective joblib files
    for name in model_names:
        joblib_file = f"{name}.joblib"
        joblib_file = os.path.join('D:/FCIS/Pattern/project/', joblib_file)
        model = joblib.load(joblib_file)
        models.append(model)

# Define a function to preprocess the input data

def preprocess_input(data, encoder, le, MLB, multiLB, top_features, scaler):
    print(data['Languages'].isnull())
    if(data['Languages'].isnull().iloc[0]):
        data['Languages'] = data.fillna('EN')

    '''
    if(data['Subtitle'].isnull().iloc[0]):
        data['Subtitle'] = data['Subtitle'].fillna(data['Name'])  
    '''

    data['size_Q1'] = data['Size'].apply(lambda x: 1 if x < 2.751732e+07 else 0)
    data['size_Q2'] = data['Size'].apply(lambda x: 1 if 2.751732e+07 <= x < 6.740582e+07 else 0)
    data['size_Q3'] = data['Size'].apply(lambda x: 1 if 6.740582e+07 <= x < 1.592689e+08 else 0)
    data['size_Q4'] = data['Size'].apply(lambda x: 1 if x >= 1.592689e+08 else 0)

    data['Num_words_description'] = data['Description'].apply(lambda x: len(re.findall('(\w+)',str(x))))

    if(data['Subtitle'].isnull().iloc[0]):
        data['Subtitle'] = data['Subtitle'].fillna('')
    data['subtitle_yes_no'] = data['Subtitle'].apply(lambda x: 0 if x == '' else 1)

    if(data['In-app Purchases'].isnull().iloc[0]):
        data['In-app Purchases'] = data['In-app Purchases'].fillna('0.0') #No data will be taken as $0
    data['In-app Purchases'] = data['In-app Purchases'].apply(lambda x: '0.0' if x == '0' else x) #for consistency
    data['In-app Purchases'] = data['In-app Purchases'].apply(lambda x: x.split(', ')) #turn string into list

    # Prices range from 0 to 99.99. Split into 4 quadrants
    data['In-App-Q1'] = data['In-app Purchases'].apply(lambda x: 1 if any(float(i) < 25 for i in x) else 0) #prices from 0 to 24.99
    data['In-App-Q2'] = data['In-app Purchases'].apply(lambda x: 1 if any(25 <= float(i) < 50 for i in x) else 0) #prices from 25 to 49.99
    data['In-App-Q3'] = data['In-app Purchases'].apply(lambda x: 1 if any(50 <= float(i) < 75 for i in x) else 0) #prices from 50 to 74.99
    data['In-App-Q4'] = data['In-app Purchases'].apply(lambda x: 1 if any(75 <= float(i) < 100 for i in x) else 0) #prices from 75 to 99.99

    X_test_encoded = encoder.transform(data[['Primary Genre']])
    feature_names = encoder.get_feature_names(['Primary Genre'])
    X_test_encoded_df = pd.DataFrame(X_test_encoded.toarray(), columns=feature_names)
    data = pd.concat([data.reset_index(drop=True), X_test_encoded_df], axis=1)

    data["Age Rating"] = le.transform(data["Age Rating"])
    
    date(data,'Original Release Date')
    date(data,'Current Version Release Date')
    data['DifferentDate'] = data['Current Version Release Date'] - data['Original Release Date']
    
    print(data.shape)

    data['Languages'] = data["Languages"].str.split(", ")
    x_Languages_encoded = MLB.transform(data["Languages"])
    new_feature_names = ['Languages_' + name for name in MLB.classes_]
    data = pd.concat([data.reset_index(drop=True), pd.DataFrame(x_Languages_encoded, columns=new_feature_names)], axis=1)    

    data['Genres'] = data["Genres"].str.split(", ")
    x_test_developer_encoded = multiLB.transform(data["Genres"])
    new_feature_names = ['Genres_' + name for name in multiLB.classes_]
    data = pd.concat([data.reset_index(drop=True), pd.DataFrame(x_test_developer_encoded, columns=new_feature_names)], axis=1)    

    data = data.drop(columns=['URL','ID','Name','Subtitle','Icon URL','In-app Purchases','Description','Developer','Languages', 'Primary Genre','Genres'],axis=1)
    
    data= data[top_features]
    #print(data)
    data_norm = scaler.transform(data)
    return data_norm,data.columns


# Define a function to make predictions on the input data
def make_prediction(data, Y):
    # Apply the trained regression models to the input data
    predictions = []
    mse_scores = []
    r2_scores = []
    for i in range(5, len(models)):
        model_name = model_names[i]
        model = models[i]
        prediction = model.predict(data)
        mse = mean_squared_error(Y, prediction)
        r2 = r2_score(Y, prediction)
        predictions.append((model_name, prediction))
        mse_scores.append(mse)
        r2_scores.append(r2)
    
    return predictions, mse_scores, r2_scores


# Define the app layout
def app():
    # Set the app title
    st.title('Game Application Success Prediction')

    #Load Models
    load_models()

    # Define the input options
    input_option = st.selectbox('Select input option', ('Manual input', 'CSV file'))

    # Define the input text boxes for manual input
    if input_option == 'Manual input':
        url = st.text_input('URL')
        id = st.text_input('ID')
        name = st.text_input('Name')
        subtitle = st.text_input('Subtitle')
        icon_url = st.text_input('Icon URL')
        user_rating_count = st.text_input('User Rating Count')
        price = st.text_input('Price')
        in_app_purchases = st.text_input('In-app Purchases')
        description = st.text_input('Description')
        developer = st.text_input('Developer')
        age_rating = st.text_input('Age Rating')
        languages = st.text_input('Languages')
        size = st.text_input('Size')
        primary_genre = st.text_input('Primary Genre')
        genres = st.text_input('Genres')
        original_release_date = st.text_input('Original Release Date')
        current_version_release_date = st.text_input('Current Version Release Date')

        # Define the submit button for manual input
        if st.button('Submit'):
            # Combine the input data into a DataFrame
            input_data = pd.DataFrame({
                'URL': [url],
                'ID': [id],
                'Name': [name],
                'Subtitle': [subtitle],
                'Icon URL': [icon_url],
                'User Rating Count': [user_rating_count],
                'Price': [price],
                'In-app Purchases': [in_app_purchases],
                'Description': [description],
                'Developer': [developer],
                'Age Rating': [age_rating],
                'Languages': [languages],
                'Size': [size],
                'Primary Genre': [primary_genre],
                'Genres': [genres],
                'Original Release Date': [original_release_date],
                'Current Version Release Date': [current_version_release_date]
            })

            # Preprocess the input data
            processed_data = preprocess_input(input_data)

            # Make a prediction on the input data
            predictions = make_prediction(processed_data)

            # Display the predictions
            st.write('Average User Rating:')
            st.write(prediction[0])

    # Define the file uploader for CSV input
    elif input_option == 'CSV file':
        uploaded_file = st.file_uploader('Upload CSV file', type='csv')

        # Define the submit button for CSV input
        if uploaded_file is not None and st.button('Submit'):
            # Load the data from the uploaded CSV file
            data = pd.read_csv(uploaded_file)
            X = data.drop('Average User Rating', axis=1)
            Y = data['Average User Rating']

            # Preprocess the input data
            processed_data = preprocess_input(X, models[0], models[1], models[2], models[3], models[4])

            # Make a prediction on the input data
            predictions, mse_scores, r2_scores = make_prediction(processed_data,Y)

            # Display the prediction
            for i, (name, prediction) in enumerate(predictions):
                st.write(f"Model {i+1}: {name}")
                st.write(f"Prediction: {prediction}")
                st.write(f"Mean Squared Error: {mse_scores[i]}")
                st.write(f"R2 score: {r2_scores[i]}")

if __name__ == '__main__':
    app()