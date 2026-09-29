import cv2
import numpy as np
from tensorflow.keras.models import load_model

#Solicitar ruta donde se encuentra el modelo entrenado
ruta_busqueda = input("Ingresa la ruta donde el modelo buscará el modelo entrenado: ")
modelo = load_model(ruta_busqueda)

#Tamaño de las imágenes
IMAGEN_ANCHO = 128
IMAGEN_ALTO = 128

#Inicializar la cámara
camara = cv2.VideoCapture(0)

#Ciclo para capturar imágenes con la camara
while True:
    correcto, frame = camara.read()
    if not correcto:
        break

    #Definición del area para reconocer números
    x_inicio = 100
    y_inicio = 100
    ancho = 200
    alto = 200

    #Dibujar rectángulo y obtener solo los patrones de ese recuadro
    cv2.rectangle(frame, (x_inicio, y_inicio), (x_inicio+ancho, y_inicio+alto), (255,0,0), 2)
    recorte = frame[y_inicio:y_inicio+alto, x_inicio:x_inicio+ancho]

    #Darle a la imagen escala de grises, redimensionarla y normalizarla
    gris = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
    redimensionado = cv2.resize(gris, (IMAGEN_ANCHO, IMAGEN_ALTO))
    normalizado = redimensionado / 255.0

    #Formato para la predicción
    entrada = normalizado.reshape(1, IMAGEN_ALTO, IMAGEN_ANCHO, 1)

    #Hacer predicción y obtener el numero que cree que es
    prediccion = modelo.predict(entrada)
    numero = np.argmax(prediccion)

    #Muestrar el numero detectado en la pantalla con su etiqueta
    texto = f"Numero: {numero}"
    cv2.putText(frame, texto, (x_inicio, y_inicio-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Reconovimiento de numero", frame)

    #Cerrar al presionar la tecla ESC
    if cv2.waitKey(1) == 27:
        break

#Cerrar la camara y terminar el proceso
camara.release()
cv2.destroyAllWindows()