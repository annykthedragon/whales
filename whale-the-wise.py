import streamlit as st
import pandas as pd
from sklearn import datasets
import os, urllib
import streamlit.components.v1 as components
import pydeck as pdk
from urllib.error import URLError
import numpy as np
from geopy.distance import geodesic 
import math
from sklearn.metrics import mean_absolute_error, r2_score, median_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import warnings
import time
warnings.filterwarnings('ignore')

@st.cache(show_spinner=False)
def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/annykthedragon/whales/master/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")

def main():
    readme_text = st.markdown(get_file_content_as_string("about.md"))
    readme2_text = st.markdown(get_file_content_as_string("us.md"))

    st.sidebar.title("ПАНЕЛЬ УПРАВЛЕНИЯ")
    app_mode = st.sidebar.selectbox("Что хотите?",
    ["О проекте", "Независимый оценщик", "О команде"])
    if app_mode == "О проекте":
        st.sidebar.success('Для оценки имущества нажмите "Независимый оценщик".')
        readme2_text.empty()



    elif app_mode == "Независимый оценщик":
        readme_text.empty()
        readme2_text.empty()
        asset = st.sidebar.selectbox("Оцениваемое имущество:",
        ["Недвижимость", "Транспортное средство"])


        if asset == "Недвижимость":
            st.write("""# Оценка стоимости недвижимости""")
            placeholder1 = st.empty()
            placeholder2 = st.empty()
            st.sidebar.warning("""Пока в сервисе можно оценить стоимость имущества только в тестовом городе - Москве.""")

            #Получаем данные пользователя
            def user_input_features():
                latitude = st.sidebar.text_input('Широта', '55.858817')
                longitude = st.sidebar.text_input('Долгота', '37.638755')
                wallsMaterial = st.sidebar.selectbox('Тип дома',
                ['None', 'block', 'brick', 'monolith', 'monolithBrick', 'old', 'panel', 'stalin', 'wood'])
                floorNumber = st.sidebar.slider('Этаж', 1, 30, 3)
                floorsTotal = st.sidebar.slider('Всего этажей в доме', 1, 30, 5)
                if floorNumber > floorsTotal:
                    st.sidebar.error("Номер этажа не должен превышать количество этажей.")
                totalArea = st.sidebar.slider('Площадь (кв.м.)', 10, 200, 40)
                kitchenArea = st.sidebar.slider('Площадь кухни (кв.м.)', 5, 30, 10)
                if kitchenArea > totalArea:
                    st.sidebar.error("Площадь кухни не может превышать площадь квартиры.")

                fldata = {'wallsMaterial': wallsMaterial,
                        'floorNumber': floorNumber,
                        'floorsTotal': floorsTotal,
                        'totalArea': totalArea,
                        'kitchenArea': kitchenArea,
                        'latitude': latitude,
                        'longitude': longitude}
                apt = pd.DataFrame(fldata, index=[0])

                return apt

            #Создаем датафрейм с параметрами квартиры
            flat = user_input_features()

            #Вычисляем столбцы с категорийными признаками, затем заменяем их на числа
            fl_columns = flat.columns[flat.dtypes == 'object']
            labelencoder = LabelEncoder()
            for column in fl_columns:
                flat[column] = labelencoder.fit_transform(flat[column])

            #Оцениваем расстояние до центра
            def get_azimuth(latitude, longitude):
            
                rad = 6372795

                llat1 = city_center_coordinates[0]
                llong1 = city_center_coordinates[1]
                llat2 = latitude
                llong2 = longitude

                lat1 = llat1*math.pi/180.
                lat2 = llat2*math.pi/180.
                long1 = llong1*math.pi/180.
                long2 = llong2*math.pi/180.

                cl1 = math.cos(lat1)
                cl2 = math.cos(lat2)
                sl1 = math.sin(lat1)
                sl2 = math.sin(lat2)
                delta = long2 - long1
                cdelta = math.cos(delta)
                sdelta = math.sin(delta)

                y = math.sqrt(math.pow(cl2*sdelta,2)+math.pow(cl1*sl2-sl1*cl2*cdelta,2))
                x = sl1*sl2+cl1*cl2*cdelta
                ad = math.atan2(y,x)

                x = (cl1*sl2) - (sl1*cl2*cdelta)
                y = sdelta*cl2
                z = math.degrees(math.atan(-y/x))

                if (x < 0):
                    z = z+180.

                z2 = (z+180.) % 360. - 180.
                z2 = - math.radians(z2)
                anglerad2 = z2 - ((2*math.pi)*math.floor((z2/(2*math.pi))) )
                angledeg = (anglerad2*180.)/math.pi
                
                return round(angledeg, 2)

            #Выясняем точность выбранной модели
            def mean_absolute_percentage_error(y_true, y_pred): 
                y_true, y_pred = np.array(y_true), np.array(y_pred)
                return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


            def median_absolute_percentage_error(y_true, y_pred): 
                y_true, y_pred = np.array(y_true), np.array(y_pred)
                return np.median(np.abs((y_true - y_pred) / y_true)) * 100

            def print_metrics(prediction, val_y):
                val_mae = mean_absolute_error(val_y, prediction)
                median_AE = median_absolute_error(val_y, prediction)
                r2 = r2_score(val_y, prediction)

                st.write('### Показатели качества модели:')
                st.write('R\u00b2: {:.2}'.format(r2))
                st.write('Средняя абсолютная ошибка: {:.3} %'.format(mean_absolute_percentage_error(val_y, prediction)))
                st.write('Медианная абсолютная ошибка: {:.3} %'.format(median_absolute_percentage_error(val_y, prediction)))
            
            if st.sidebar.button('Рассчитать'):

                with placeholder1.container():
                    st.info('Загружаем датасет и делаем первичную обработку...')

                #При помощи библиотеки pandas считываем csv-файл и преобразуем его в формат датафрейма (таблицы)
                file_path = 'https://raw.githubusercontent.com/annykthedragon/whales/master/realest-2020-moscow.csv'
                df = pd.read_csv(file_path)

                with placeholder2.container():
                    st.success('Датасет загружен.')
                
                #Создаем новый столбец Стоимость 1 кв.м путем построчного деления стоимостей квартир на их общие площади
                df['priceMetr'] = df['price']/df['totalArea']

                #Задаем широту и долготу центра города (В тестовом формате - Москва) и рассчитываем для каждой квартиры расстояние от центра и азимут 
                city_center_coordinates = [55.7522, 37.6156]
                df['distance'] = list(map(lambda x, y: geodesic(city_center_coordinates, [x, y]).meters, df['latitude'], df['longitude']))
                df['azimuth'] = list(map(lambda x, y: get_azimuth(x, y), df['latitude'], df['longitude']))

                #Выбираем из датафрейма только те квартиры, которые расположены не дальше 40 км от центра города с панельными стенами
                df = df.loc[(df['distance'] < 40000)] 

                #Округляем значения стоблцов Стоимости метра, расстояния и азимута
                df['priceMetr'] = df['priceMetr'].round(0)
                df['distance'] = df['distance'].round(0)
                df['azimuth'] = df['azimuth'].round(0)


                with placeholder1.container():
                    st.info('Удаляем выбросы...')
                #Вычисляем строки со значениями-выбросами 
                first_quartile = df.quantile(q=0.25)
                third_quartile = df.quantile(q=0.75)
                IQR = third_quartile - first_quartile
                outliers = df[(df > (third_quartile + 1.5 * IQR)) | (df < (first_quartile - 1.5 * IQR))].count(axis=1)
                outliers.sort_values(axis=0, ascending=False, inplace=True)

                #Удаляем из датафрейма 3000 строк, подходящих под критерии выбросов
                outliers = outliers.head(3000)
                df.drop(outliers.index, inplace=True)

                with placeholder1.container():
                    st.info('Превращаем категорийные признаки в числовые...')
                #Вычисляем столбцы с категорийными признаками, затем заменяем их на числа
                categorical_columns = df.columns[df.dtypes == 'object']
                labelencoder = LabelEncoder()
                for column in categorical_columns:
                    df[column] = labelencoder.fit_transform(df[column])
                    print(dict(enumerate(labelencoder.classes_)))

                with placeholder1.container():
                    st.info('Создаем целевую переменную, делим датасет на выборки...')
                #Назначаем целевой переменной цену 1 кв. метра
                y = df['priceMetr']

                #Создаем список признаков, на основании которых будем строить модели
                features = [
                            'wallsMaterial', 
                            'floorNumber', 
                            'floorsTotal', 
                            'totalArea', 
                            'kitchenArea',
                            'distance',
                            'azimuth'
                        ]

                #Создаем датафрейм, состоящий из признаков, выбранных ранее
                X = df[features]

                #Проводим случайное разбиение данных на выборки для обучения (train) и валидации (val), по умолчанию в пропорции 0.75/0.25
                train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
                with placeholder2.container():
                    st.success('Данные обработаны.')

    
                with placeholder1.container():
                    st.info('Начинаем обучение (модель Random forest)...')

                with st.spinner('На это уйдёт некоторое время...'):
                    #Создаем регрессионную модель случайного леса 
                    rf_model = RandomForestRegressor(n_estimators=2000, 
                                                    n_jobs=-1,  
                                                    bootstrap=False,
                                                    criterion='mse',
                                                    max_features=3,
                                                    random_state=1,
                                                    max_depth=55,
                                                    min_samples_split=5
                                                    )

                    #Проводим подгонку модели на обучающей выборке 
                    rf_model.fit(train_X, train_y)

                    #Вычисляем предсказанные значения цен на основе валидационной выборки
                    rf_prediction = rf_model.predict(val_X).round(0)

                    #Вычисляем и печатаем величины ошибок при сравнении известных цен квартир из валидационной выборки с предсказанными моделью
                    print_metrics(rf_prediction, val_y)

                with placeholder2.container():
                    st.success('Машинное обучение прошло успешно.')


                with placeholder1.container():
                    st.info('Оцениваем недвижимость клиента...')
                
                #Рассчитываем недостающие параметры квартиры - расстояние от центра города и азимут
                flat['distance'] = list(map(lambda x, y: geodesic(city_center_coordinates, [x, y]).meters, flat['latitude'], flat['longitude']))
                flat['azimuth'] = list(map(lambda x, y: get_azimuth(x, y), flat['latitude'], flat['longitude']))
                flat['distance'] = flat['distance'].round(0)
                flat['azimuth'] = flat['azimuth'].round(0)

                #Удаляем ненужные столбцы с широтой и долготой
                flat = flat.drop('latitude', axis=1)
                flat = flat.drop('longitude', axis=1)

                #Вычисляем предсказанное значение стоимости 
                rf_prediction_flat = rf_model.predict(flat).round(0)
                
                with placeholder2.container():
                    st.success(f'Стоимость 1 кв.м: {int(rf_prediction_flat[0].round(-3))} рублей')

                #Полученное знаечение и умножаем на общую площадь квартиры
                price = rf_prediction_flat*flat['totalArea'][0]
                

                #Печатаем предсказанное значение цены предложения
                placeholder1.empty()
                st.write(f'## Ожидаемая цена недвижимости: {int(price[0].round(-3))} рублей')
                st.balloons()
            


        elif asset == "Транспортное средство":
            st.write("""# Оценка стоимости транспортных средств""")
            placeholder1 = st.empty()
            placeholder2 = st.empty()

            #Получаем данные пользователя
            def user_input_features_с():
                Car_type = st.sidebar.selectbox('Тип транспортного средства',
                ['Compact MPV', 'Convertible', 'Coupe', 'Hardtop Coupe', 
                'Hatchback', 'Liftback', 'Limousine', 'Microvan', 
                'Minivan', 'Pickup', 'Roadster', 'SUV', 
                'Sedan', 'Speedster', 'Station Wagon', 'Van'])
                mark = st.sidebar.selectbox('Марка',
                ['Acura ', 'Alfa Romeo ', 'Audi ', 'BMW ', 'Bentley ', 'Cadillac ', 
                'Changan ', 'Chery ', 'Chevrolet ', 'Chrysler ', 'Citroen ', 
                'DW Hower ', 'Daewoo ', 'Daihatsu ', 'Daimler ', 'Datsun ', 'Dodge ', 
                'DongFeng ', 'FAW ', 'Fiat ', 'Ford ', 'GAZ ', 'GMC ', 'Geely ', 
                'Genesis ', 'Great Wall ', 'Haval ', 'Hawtai ', 'Honda ', 'Hummer ', 
                'Hyundai ', 'ICH', 'Infiniti ', 'Iran Khodro ', 'Isuzu ', 'Jaguar ', 
                'Jeep ', 'Kia ', 'LADA (VAZ) ', 'Lamborghini ', 'Land Rover ',
                'Lexus ', 'Lifan ', 'Lincoln ', 'Luxgen ', 'LyAZ ', 'MINI ', 'Maserati ',
                'Maybach ', 'Mazda ', 'Mercedes-Benz ', 'Mitsubishi ', 'Moskvich ', 
                'Nissan ', 'Opel ', 'Peugeot ', 'Porsche ', 'Race car ', 'Ravon ', 'Renault ', 
                'Rolls-Royce ', 'Rover ', 'SEAT ', 'Saab ', 'Scion ', 'Skoda ', 'Smart ', 
                'SsangYong ', 'Subaru ', 'Suzuki ', 'TagAZ ', 'Tesla ', 'Toyota ', 
                'Volkswagen ', 'Volvo ', 'Vortex ', 'YAZ ', 'ZAZ ', 'ZX '])
                Model = st.sidebar.text_input('Модель', 'Vito')  
                Year = st.sidebar.slider('Год выпуска', 1927, 2021, 2018)  
                Box = st.sidebar.selectbox('Коробка передач',
                ['automatic', 'mechanics', 'robot', 'variator'])
                Engine = st.sidebar.selectbox('Двигатель',
                ['SYG', 'diesel', 'electro', 'gasoline', 'hybrid'])
                Mileage = st.sidebar.slider('Пробег', 0, 1000000, 111000)

                сdata = {'Car_type': Car_type,
                        'mark': mark,
                        'Model': Model,
                        'Year': Year,
                        'Box': Box,
                        'Engine': Engine,
                        'Mileage': Mileage}
                car = pd.DataFrame(сdata, index=[0])

                return car

            #Создаем датафрейм с параметрами квартиры
            transport = user_input_features_с()
            print(transport)

            #Вычисляем столбцы с категорийными признаками, затем заменяем их на числа
            car_columns = transport.columns[transport.dtypes == 'object']
            labelencoder = LabelEncoder()
            for column in car_columns:
                transport[column] = labelencoder.fit_transform(transport[column])


            #Выясняем точность выбранной модели
            def mean_absolute_percentage_error(y_true, y_pred): 
                y_true, y_pred = np.array(y_true), np.array(y_pred)
                return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


            def median_absolute_percentage_error(y_true, y_pred): 
                y_true, y_pred = np.array(y_true), np.array(y_pred)
                return np.median(np.abs((y_true - y_pred) / y_true)) * 100

            def print_metrics(prediction, val_y):
                val_mae = mean_absolute_error(val_y, prediction)
                median_AE = median_absolute_error(val_y, prediction)
                r2 = r2_score(val_y, prediction)

                st.write('### Показатели качества модели:')
                st.write('R\u00b2: {:.2}'.format(r2))
                st.write('Средняя абсолютная ошибка: {:.3} %'.format(mean_absolute_percentage_error(val_y, prediction)))
                st.write('Медианная абсолютная ошибка: {:.3} %'.format(median_absolute_percentage_error(val_y, prediction)))
            
            if st.sidebar.button('Рассчитать'):

                with placeholder1.container():
                    st.info('Загружаем датасет и делаем первичную обработку...')

                #При помощи библиотеки pandas считываем csv-файл и преобразуем его в формат датафрейма (таблицы)
                file_path = 'https://raw.githubusercontent.com/annykthedragon/whales/master/cars_dataset.csv'
                df_c = pd.read_csv(file_path)
                keep_col = ['Car_type',
                            'mark', 
                            'Model', 
                            'Year', 
                            'Box',
                            'Engine',
                            'Mileage',
                            'Price']
                df = df_c[keep_col]

                with placeholder2.container():
                    st.success('Датасет загружен.')
                

                with placeholder1.container():
                    st.info('Удаляем выбросы...')
                #Вычисляем строки со значениями-выбросами 
                first_quartile = df.quantile(q=0.25)
                third_quartile = df.quantile(q=0.75)
                IQR = third_quartile - first_quartile
                outliers = df[(df > (third_quartile + 1.5 * IQR)) | (df < (first_quartile - 1.5 * IQR))].count(axis=1)
                outliers.sort_values(axis=0, ascending=False, inplace=True)

                #Удаляем из датафрейма 3000 строк, подходящих под критерии выбросов
                outliers = outliers.head(3000)
                df.drop(outliers.index, inplace=True)

                with placeholder1.container():
                    st.info('Превращаем категорийные признаки в числовые...')
                #Вычисляем столбцы с категорийными признаками, затем заменяем их на числа
                categorical_columns = df.columns[df.dtypes == 'object']
                labelencoder = LabelEncoder()
                for column in categorical_columns:
                    df[column] = labelencoder.fit_transform(df[column])
                    print(dict(enumerate(labelencoder.classes_)))

                with placeholder1.container():
                    st.info('Создаем целевую переменную, делим датасет на выборки...')
                #Назначаем целевой переменной цену 1 кв. метра
                y = df['Price']

                #Создаем список признаков, на основании которых будем строить модели
                features_c = [
                            'Car_type', 
                            'mark', 
                            'Model', 
                            'Year', 
                            'Box',
                            'Engine',
                            'Mileage'
                        ]

                #Создаем датафрейм, состоящий из признаков, выбранных ранее
                X = df[features_c]

                #Проводим случайное разбиение данных на выборки для обучения (train) и валидации (val), по умолчанию в пропорции 0.75/0.25
                train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
                with placeholder2.container():
                    st.success('Данные обработаны.')

                with placeholder1.container():
                    st.info('Начинаем обучение (модель Random forest)...')

                with st.spinner('На это уйдёт некоторое время...'):
                    #Создаем регрессионную модель случайного леса 
                    rf_model = RandomForestRegressor(n_estimators=2000, 
                                                    n_jobs=-1,  
                                                    bootstrap=False,
                                                    criterion='mse',
                                                    max_features=3,
                                                    random_state=1,
                                                    max_depth=55,
                                                    min_samples_split=5
                                                    )

                    #Проводим подгонку модели на обучающей выборке 
                    rf_model.fit(train_X, train_y)

                    #Вычисляем предсказанные значения цен на основе валидационной выборки
                    rf_prediction = rf_model.predict(val_X).round(0)

                    #Вычисляем и печатаем величины ошибок при сравнении известных цен квартир из валидационной выборки с предсказанными моделью
                    print_metrics(rf_prediction, val_y)

                with placeholder2.container():
                    st.success('Машинное обучение прошло успешно.')


                with placeholder1.container():
                    st.info('Оцениваем транспортное средство клиента...')
                
                #Вычисляем предсказанное значение стоимости 
                rf_prediction_transport = rf_model.predict(transport).round(0)

                price_с = rf_prediction_transport
                

                #Печатаем предсказанное значение цены предложения
                placeholder1.empty()

                st.write(f'## Ожидаемая цена транспортного средства: {int(price_с[0].round(-3))} рублей')
                st.balloons()


    elif app_mode == "О команде":
        readme_text.empty()
        st.sidebar.success('Для оценки имущества нажмите "Независимый оценщик".')





main()
