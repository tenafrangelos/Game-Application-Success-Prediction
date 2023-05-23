import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,Normalizer,OneHotEncoder
from sklearn.model_selection import train_test_split
from datetime import datetime
import time
import re
from requests.exceptions import RequestException
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
import streamlit as st
import joblib,pickle
import os

# Load the dataset and split it into training and test sets
data = pd.read_csv('D:/FCIS/Pattern/project/games-classification-dataset.csv')
x_train, x_test, y_train, y_test = train_test_split(data.drop('Rate', axis=1), data['Rate'], test_size=0.20, random_state=42)

def date(data,x):
    data[x] = pd.to_datetime(data[x], dayfirst=True,infer_datetime_format=True)
    data[x[0]+'_Year'] = pd.DatetimeIndex(data[x]).year
    data[x[0]+'_Month'] = pd.DatetimeIndex(data[x]).month
    data[x[0]+'_Day'] = pd.DatetimeIndex(data[x]).day
    data[x]=data[x].map(datetime.toordinal)


# Define the list of models and their names
model_names = ['c_encoder', 'c_MLB', 'c_multiLB', 'c_scaler', 'SVM', 'Logistic Regression', 'KNN', 'Decision Tree', 'Random Forest']
models = []

def load_models():
    # Load the models from their respective joblib files
    for name in model_names:
        pickle_file = f"{name}.pkl"
        pickle_file = os.path.join('D:/FCIS/Pattern/project/', pickle_file)
        model = pickle.load(open(pickle_file,'rb'))
        models.append(model)

    pickle_in = open("D:/FCIS/Pattern/project/gbc_grid.sav","rb")
    gbc_grid = pickle.load(pickle_in)
    models.append(gbc_grid)
    model_names.append('gbc_grid')



# Define a function to preprocess the input data


def preprocess_input(data, encoder, MLB, multiLB, top_features, scaler):
    
    if(data['Languages'].isnull().sum()):
        data['Languages'] = data['Languages'].fillna('EN')

    data['size_Q1'] = data['Size'].apply(lambda x: 1 if x < 2.751732e+07 else 0)
    data['size_Q2'] = data['Size'].apply(lambda x: 1 if 2.751732e+07 <= x < 6.740582e+07 else 0)
    data['size_Q3'] = data['Size'].apply(lambda x: 1 if 6.740582e+07 <= x < 1.592689e+08 else 0)
    data['size_Q4'] = data['Size'].apply(lambda x: 1 if x >= 1.592689e+08 else 0)

    data['Num_words_description'] = data['Description'].apply(lambda x: len(re.findall('(\w+)',str(x))))

    if(data['Subtitle'].isnull().iloc[0]):
        data['Subtitle'] = data['Subtitle'].fillna('')
    data['subtitle_yes_no'] = data['Subtitle'].apply(lambda x: 0 if x == '' else 1)

    if(data['In-app Purchases'].isnull().sum()):
        data['In-app Purchases'] = data['In-app Purchases'].fillna('0.0') #No data will be taken as $0
    data['In-app Purchases'] = data['In-app Purchases'].apply(lambda x: '0.0' if x == '0' else x) #for consistency
    data['In-app Purchases'] = data['In-app Purchases'].apply(lambda x: x.split(', ')) #turn string into list

    # Prices range from 0 to 99.99. Split into 4 quadrants
    data['In-App-Q1'] = data['In-app Purchases'].apply(lambda x: 1 if any(float(i) < 25 for i in x) else 0) #prices from 0 to 24.99
    data['In-App-Q2'] = data['In-app Purchases'].apply(lambda x: 1 if any(25 <= float(i) < 50 for i in x) else 0) #prices from 25 to 49.99
    data['In-App-Q3'] = data['In-app Purchases'].apply(lambda x: 1 if any(50 <= float(i) < 75 for i in x) else 0) #prices from 50 to 74.99
    data['In-App-Q4'] = data['In-app Purchases'].apply(lambda x: 1 if any(75 <= float(i) < 100 for i in x) else 0) #prices from 75 to 99.99
    
    data['Primary Genre'] = encoder.transform(data['Primary Genre'])

    scale_mapper = {"4+":4, "9+":9, "12+":12, "17+":17}
    data["Age Rating"] = data["Age Rating"].replace(scale_mapper)
    
    date(data,'Original Release Date')
    date(data,'Current Version Release Date')
    data['DifferentDate'] = data['Current Version Release Date'] - data['Original Release Date']

    print(data.shape)

    data['Languages'] = data["Languages"].str.split(", ")
    for i in data["Languages"]:
        if(type(i)== float):
            print(i)
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
    return data_norm


# Define a function to make predictions on the input data
def make_prediction(data, Y):
    #data = data.reshape((len(data), 1))
    
    # Apply the trained regression models to the input data
    predictions = []
    mse_scores = []
    r2_scores = []
    
    for i in range(4, len(models)):
        print(i)
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
def app2():
    # Set the app title
    st.title('Game Application Success Prediction')

    #Load Models
    load_models()
    pickle_file = "top_feature_indices.pkl"
    top = pickle.load(open(pickle_file,'rb'))
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
            st.write('Rate:')
            st.write(prediction[0])

    # Define the file uploader for CSV input
    elif input_option == 'CSV file':
        uploaded_file = st.file_uploader('Upload CSV file', type='csv')

        # Define the submit button for CSV input
        if uploaded_file is not None and st.button('Submit'):
            # Load the data from the uploaded CSV file
            data = pd.read_csv(uploaded_file)
            X = data.drop('Rate', axis=1)
            Y = data['Rate'].values
            Y = pd.Series(Y)

            scale_mapper = {"Low":1, "Intermediate":2, "High":3}

            Y = Y.replace(scale_mapper)

            processed_data = preprocess_input(X, models[0], models[1], models[2], top, models[3])
            processed_data = np.array(processed_data)

            # Make a prediction on the input data
            predictions, mse_scores, r2_scores = make_prediction(processed_data, Y)

            # Display the prediction
            for i, (name, prediction) in enumerate(predictions):
                st.write(f"Model {i+1}: {name}")
                st.write(f"Prediction: {prediction}")
                st.write(f"Mean Squared Error: {mse_scores[i]}")
                st.write(f"R2 score: {r2_scores[i]}")

if __name__ == '__main__':
    app2()