import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import os, urllib, cv2
import streamlit.components.v1 as components
import pydeck as pdk
from urllib.error import URLError

@st.cache(show_spinner=False)
def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/annykthedragon/whales/master/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")

def main():
    readme_text = st.markdown(get_file_content_as_string("about.md"))
    readme2_text = st.markdown(get_file_content_as_string("us.md"))

    st.sidebar.title("Меню")
    app_mode = st.sidebar.selectbox("Что хотите?",
    ["О проекте", "Независимый оценщик", "О команде"])
    if app_mode == "О проекте":
        st.sidebar.success('Для оценки имущества нажмите "Независимый оценщик".')
        readme2_text.empty()
    elif app_mode == "Независимый оценщик":
        readme_text.empty()
        readme2_text.empty()
        asset = st.sidebar.selectbox("Оцениваемое имущество:",
        ["Транспортное средство", "Недвижимость"])
        if asset == "Транспортное средство":
            car_br = st.multiselect("Выберите марку машины:", 
            ["BMV", "Audi", "Toyota", "Lada", "Mercedes"])
        elif asset == "Недвижимость":
            est_place = st.multiselect("Город:", 
            ["Екатеринбург", "Москва", "Санкт-Петербург", "Сочи", "Краснодар"])
            est_s = st.slider("Площадь (кв.м): ", min_value=10,   
                       max_value=10000, value=35, step=1)
    elif app_mode == "О команде":
        readme_text.empty()
        st.sidebar.success('Для оценки имущества нажмите "Независимый оценщик".')





main()
