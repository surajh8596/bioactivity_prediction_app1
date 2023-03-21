import streamlit as st
import pandas as pd
from PIL import Image
from matplotlib import image
import os
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression


#resourses path
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
dir_of_interest = os.path.join(FILE_DIR, "resourses")

# Sidebar
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input file", type=['txt'])
    st.sidebar.markdown("""
[Example input file](https://raw.githubusercontent.com/dataprofessor/bioactivity-prediction-app/main/example_acetylcholinesterase.txt)
""")
    
# Page title
st.markdown("""
# Bioactivity Prediction App (Acetylcholinesterase)""")

# Logo image
IMAGE_PATH = os.path.join(dir_of_interest, "logo.png")
img = image.imread(IMAGE_PATH)
st.image(img)

if st.sidebar.button('Predict'):
    #DATA PATH
    DATA_PATH = os.path.join(dir_of_interest, "Data.csv")
    df = pd.read_csv(DATA_PATH)
    df.drop("Unnamed: 0", axis=1, inplace=True)
    #Showing dataset on page
    st.header("BIO-ACTIVITY DATASETS:")
    st.dataframe(data=df, use_container_width=True)

    #X, Y features
    X = df.drop(['pIC50'], axis=1)
    Y = df.iloc[:,-1]

    #Removing low varinace Input features
    def remove_low_variance(input_data, threshold=0.1):
        selection = VarianceThreshold(threshold)
        selection.fit(input_data)
        return input_data[input_data.columns[selection.get_support(indices=True)]]
    X = remove_low_variance(X, threshold=0.1)
    st.header("IMPORTANT INPUT FEATURES FOR TRAINING THE MODEL:")
    st.dataframe(data=X, use_container_width=True)

    #Training the model using Random Forest regressor
    model = LinearRegression()
    model.fit(X, Y)

    #Input data from user
    load_data = pd.read_table(uploaded_file, sep=' ', header=None)

    #Prediction
    prediction=model.predict(X)
    st.header('PREDICTION OUTPUT')
    prediction_output = pd.Series(prediction, name='pIC50')
    molecule_name = pd.Series(load_data[1], name='molecule_name')
    final = pd.concat([molecule_name, prediction_output], axis=1)[:load_data.shape[0]]
    st.dataframe(data=final, use_container_width=True)

else:
    st.info('Upload input data in the sidebar to start!')