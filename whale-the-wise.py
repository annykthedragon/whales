import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import urllib

def get_file_content_as_string(path):
    url = 'http://localhost:8503/C:/Users/ak281/OneDrive/Desktop/WhaleTheWise/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")
    
def main():
    readme_text = st.markdown(get_file_content_as_string("about.md"))

    st.sidebar.title("Меню")
    app_mode = st.sidebar.selectbox("Что хотите?",
    ["О проекте", "Независимый оценщик", "О команде"])
    if app_mode == "О проекте":
        st.sidebar.success('Для запуска оценщика нажмите "Независимый оценщик".')
    elif app_mode == "Независимый оценщик":
        readme_text.empty()
    elif app_mode == "О команде":
        readme_text.empty()



main()