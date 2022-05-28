import pandas as pd
import numpy as np
import joblib
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.disable_eager_execution()
import tensorflow_hub as hub
from tensorflow.keras.models import load_model
from keras.models import model_from_json

def transformar(datos_entrada):
    
    # 1. Dataframe de los datos de entrada
    df_datos = pd.DataFrame(datos_entrada, index=[0])
    
    # Unir el título con la trama para que sea un mismo texto para el encoder
    df_datos['title_plot'] = df_datos['title'] + ' - ' + df_datos['plot']
    
    # Eliminar columnas que se unieron
    df_datos.drop(columns=['title','plot'], inplace=True)

    # Importación el módulo TF Hub del Universal Sentence Encoder
    tf.disable_eager_execution()
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
    embed = hub.Module(module_url)

    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        sentences_embeddings = session.run(embed(df_datos['title_plot']))    
    
    x_test_embed = pd.DataFrame(sentences_embeddings)
    x_test_embed.index = df_datos.index
    
    # Concateno los embedding realizados con la tabla original para traer el año
    df_datos_2 = pd.concat([df_datos, x_test_embed], axis=1)
    df_datos_2.drop(columns=['title_plot'], inplace=True)

    # Una vez tengo el DF con el año y los embedding, se escala todo (el año es el que lo requiere)
    #Scaler
    scaler = joblib.load(os.path.dirname(__file__) + '/scaler.pkl')

    df_datos_2 = scaler.transform(df_datos_2)
    
    # Modelo cargado
    json_file = open(os.path.dirname(__file__) +'/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # cargar pesos al nuevo modelo
    loaded_model.load_weights(os.path.dirname(__file__) +'/model.h5')
    print("Cargado modelo desde disco.")
    loaded_model.compile(optimizer = 'adam', loss='binary_crossentropy', metrics = ['accuracy'])


    # Predicciones
    y_pred_genres = loaded_model.predict(df_datos_2)[0]

    # dar formato a predicciones
    cols = ['p_Action', 'p_Adventure', 'p_Animation', 'p_Biography', 'p_Comedy', 'p_Crime', 'p_Documentary', 'p_Drama', 'p_Family',
        'p_Fantasy', 'p_Film-Noir', 'p_History', 'p_Horror', 'p_Music', 'p_Musical', 'p_Mystery', 'p_News', 'p_Romance',
        'p_Sci-Fi', 'p_Short', 'p_Sport', 'p_Thriller', 'p_War', 'p_Western']
    
    res = pd.DataFrame(y_pred_genres).T
    res.columns = cols
    
    return res.to_dict()
    
    