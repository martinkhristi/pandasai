import streamlit as st
import pandas as pd
# import seaborn as sns  # Removed seaborn dependency
import matplotlib.pyplot as plt
import plotly.express as px
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import specific modules for language models
from langchain_groq.chat_models import ChatGroq
from pandasai.llm.openai import OpenAI
from pandasai import SmartDataframe

# Set up Streamlit page
st.set_page_config(page_title="Data Analysis Platform")
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: #F5F5F5;
        color: #333333;
    }
    .stButton button {
        background-color: #0072C6;
        color: #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Define functions to load language models
def load_groq_llm():
    return ChatGroq(model_name="mixtral-8x7b-32768", api_key=os.getenv('GROQ_API_KEY'))

def load_openai_llm():
    return OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Sidebar for user inputs
st.sidebar.title("Settings")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")
llm_choice = st.sidebar.selectbox("Select Language Model", ("Groq", "OpenAI"))

# Main application content starts here
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # General Information
    st.subheader("General Information")
    st.write(f"Shape of the dataset: {data.shape}")
    st.write(f"Data Types:\n{data.dtypes}")


