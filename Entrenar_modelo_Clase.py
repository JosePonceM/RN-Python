import cv2
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense #importamos las librerias necesarias para el procesamiento de imagenes, manejo de archivos y construccion del modelo de red neuronal
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

#Ruta de la carpeta con imagenes
RUTA_DATOS = r"C:\Users\JoseP11\Documents\Sistemas Inteligentes 8vo\RecNumeros\img_num" 
#Ancho de las imagenes
IMAGEN_ANCHO = 128
IMAGEN_ALTO = 128

#Listas para imagenes y etiquetas
imagenes = [] 
etiquetas = [] 

#Ciclo para recorrer las carpetas
for nombre_carpeta in os.listdir(RUTA_DATOS): 
    ruta_carpeta = os.path.join(RUTA_DATOS, nombre_carpeta)
    #Verificar si la ruta es una carpeta
    if os.path.isdir(ruta_carpeta): 

        #Ciclo para recorrer las imagenes de la carpeta
        for nombre_imagen in os.listdir(ruta_carpeta): 
            ruta_imagen = os.path.join(ruta_carpeta, nombre_imagen)
            #Leer imagen actual
            imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
            #Verificación de que se leyó la imagen
            if imagen is not None:
                imagen = cv2.resize(imagen, (IMAGEN_ANCHO, IMAGEN_ALTO))
                imagen = imagen / 255.0 
                #Agregar imagen y etiqueta a sus listas
                imagenes.append(imagen) 
                etiquetas.append(int(nombre_carpeta))

#Conversión de listas a arreglos de las imagenes y etiquetas
imagenes = np.array(imagenes).reshape(-1, IMAGEN_ALTO, IMAGEN_ANCHO, 1)
etiquetas = to_categorical(etiquetas, num_classes= 10)

#Divide los datos en grupos de entrenamiento y prueba
x_entrenamiento, x_prueba,y_entrenamiento, y_prueba =train_test_split(imagenes, etiquetas, test_size=0.2, random_state=42) 

modelo = Sequential()
#Definición de multiples capas para el proceso de imagenes del modelo
modelo.add(Conv2D(32,kernel_size=(3, 3), 
    activation='relu', 
    input_shape=(IMAGEN_ALTO, IMAGEN_ANCHO, 1))
) 

modelo.add(MaxPooling2D(pool_size=(2, 2)))
modelo.add(Conv2D(64,kernel_size=(3, 3), activation='relu'))
modelo.add(MaxPooling2D(pool_size=(2, 2)))
modelo.add(Flatten())
modelo.add(Dense(128, activation='relu'))
modelo.add(Dense(10, activation='softmax'))

modelo.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
modelo.fit(x_entrenamiento, 
    y_entrenamiento, 
    epochs=10, 
    validation_data=(x_prueba, y_prueba)
)
#Guardado del entrenamiento en el archivo .keras
ruta_modelo = input("Escribe la ruta donde quieres guardar el modelo .keras: ")
modelo.save(ruta_modelo)
print("Modelo guardado")