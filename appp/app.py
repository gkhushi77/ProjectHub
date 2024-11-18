import streamlit as st
import pandas as pd
import os
from streamlit_extras.add_vertical_space import add_vertical_space
from ydata_profiling import *
from ydata_profiling.profile_report import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from ydata_profiling.compare_reports import compare
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load data

def load_data():
    return pd.read_csv("data.csv")

df = load_data()

# Encode categorical features
label_encoder = LabelEncoder()
df['domain_encoded'] = label_encoder.fit_transform(df['domain'])

# Define features and target
X = df[['domain_encoded', 'scale']]  # Assuming 'scale' column contains the difficulty level
y_project_name = df['name']
y_brief = df['brief']

# Split the dataset
X_train, X_test, y_train_project_name, y_test_project_name, y_train_brief, y_test_brief = train_test_split(X, y_project_name, y_brief, test_size=0.2, random_state=42)

# Initialize and train the models
model_project_name = RandomForestClassifier()
model_project_name.fit(X_train, y_train_project_name)

model_brief = RandomForestClassifier()
model_brief.fit(X_train, y_train_brief)

input_domain_encoded = None
input_difficulty = None


# Import necessary modules for preprocessing and recommendation system

def main():
    page = st.sidebar.selectbox("**Select a page**", ["Home", "ML Preprocessing", "Recommendation System"])

    if page == "Home":
        st.title("ProjectHub")
        st.write("Welcome to the realm of student solutions!")
        st.image("P_hub.jpg",width=200)
        st.write("ProjectHub offers a multifaceted solution to the challenges faced by students in project development endeavors. Through a user-centric approach, ProjectHub provides a suite of innovative features aimed at enhancing the project development. ProjectHub inspires creativity and guides learners towards projects that resonate with their unique goals. This platform stands as a valuable resource, empowering students to excel in their educational journey. ")
       
        
    elif page == "ML Preprocessing":
        st.title("ML Preprocessing")
        st.write("Welcome to ML Preprocessing section!")
    
        with st.sidebar:
            st.title("AutoML")
            choice = st.radio("Options", ["Upload","Profiling"])
            st.info("This application allows you to build an automated ML pipeline using Streamlit, Pandas Profiling and PyCaret. ")
            
        if os.path.exists("sourcedata.csv"):
            df = pd.read_csv("sourcedata.csv", index_col=None)

        if choice=="Upload":
            st.title("Upload Your Data for Profiling!")
            file = st.file_uploader("Upload Your Dataset Here")
            if file:
                df = pd.read_csv(file, index_col=None)
                df.to_csv("sourcedata.csv", index=None)
                st.dataframe(df)
       
        if choice=="Profiling":
            st.title("Automated Exploratory Data Analysis")
            profile_report = ProfileReport(df)
            st_profile_report(profile_report)
        
        
    elif page == "Recommendation System":
        # Display content for the Recommendation System page
        st.title("Recommendation System")
        # Add recommendation system functionality
        
        st.write("**Enter your preferences:**")
        input_domain = st.selectbox("Select the domain",[
            "","full stack development", "Machine Learning", "Game Development", "Web Development"])
        input_difficulty = st.selectbox("Enter the difficulty level",["",1,2,3])

        if st.button("Get Recommendation"):
            input_domain_encoded = label_encoder.transform([input_domain])[0]
            input_difficulty = int(input_difficulty)
        
        # Make predictions
        predicted_project_name = model_project_name.predict([[input_domain_encoded, input_difficulty]])
        predicted_brief = model_brief.predict([[input_domain_encoded, input_difficulty]])

        st.write("**Recommended Project Name:**", predicted_project_name[0])
        st.write("**Brief:**", predicted_brief[0] + "\n"+"\n"+"**For more recommendations, you can click - Get Recommendation**" )
        

if __name__ == "__main__":
    main()

