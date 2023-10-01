#NOMBRE: JHONNIER ALEXIS AQUITE LIZ
#MODELO PARA IDENTIFICAR PERSONAS Y EMOCIONES. 

# IMPORTAMOS LA LIBRERIAS NECESARIAS. 
import cv2 #para detectar rostro
import numpy as np # buscar el valor minimo 
import face_recognition as fr # para codificar la imgs
import os # para el id de las imgs
import random
from datetime import datetime

import mediapipe as mp
import math #funcion para calcular longitudes.
import tkinter as tk




path = 'Personal'
imags = []
clas = []
lista = os.listdir(path)

#variables auxiliares
comp1 = 100
emocion_actual = None  # Variable para rastrear la emoción actual

def actualizar_emocion():
    global emocion_actual
    emocion_label.config(text=f"Su emoción es: {emocion_actual}")
    root.after(1000, actualizar_emocion)  # Actualizar cada segundo


# Leemos los rostros del directorio
for lis in lista:
    # Leemos las imágenes de los rostros
    imgdb = cv2.imread(f'{path}/{lis}')
    # Almacenamos la imagen
    imags.append(imgdb)
    # Almacenamos el nombre
    clas.append(os.path.splitext(lis)[0])
print(clas)


# Función de codificación de rostros
def codrostros(imags):
    listacod = []
    #Iteramos 
    for img in imags:
        # se corrije el color 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #codificamos las caracteristicas mas importantes de las imgs
        cod = fr.face_encodings(img)[0]
        #guardamos la codificacion en el nuevo arreglo 
        listacod.append(cod)

    return listacod

def registro(nombre):
    #abrimos el archivo en modo lectura y escrotura
    with open('registro.csv', 'a') as h:
    
        info = datetime.now() #importamos el modulo datetime
        #formatiamos la fecha
        fecha = info.strftime('%Y:%m:%d')
        #formatiamos la hora
        hr = info.strftime('%H:%M:%S')
        #Escribimos los datos 
        h.write(f'\n{nombre},{fecha},{hr}')
        print(info)

#llamamos a la funcion de codificacion de rostros con el parametro de imags
rostroscod = codrostros(imags)

#iniciamos la videoCapture 
cap = cv2.VideoCapture(0)

# Creamos la función que nos construye la malla facial
mapa_dibujo = mp.solutions.drawing_utils
# Ajusta la configuración de la malla.
confi_dibujo = mapa_dibujo.DrawingSpec(thickness=1, circle_radius=1)  

# creamos un obtjeto para guadar la malla, con la funcion face_mesh.
mpmalla_facial = mp.solutions.face_mesh 
# asignamos el numero de rostros.
Malla_facial = mpmalla_facial.FaceMesh(max_num_faces=1)  

# comenzamos con el modelo.
while True:
    #capturamos de la video.. todos los frames
    ret, frame = cap.read() 
    # reducimos la img para una mejor precision
    frame2 = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    #convertimos a colo RGB por el formato cv2
    rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    #buscamos los rostros para posteriormente hacer una similitud
    # a la bd
    faces = fr.face_locations(rgb)
    #codificamos los rostro
    facescod = fr.face_encodings(rgb, faces)

    # Guarda la emoción actual anterior
    emocion_actual_anterior = emocion_actual  

    #iteramos para cada uno de los rostros que pueda detectar.
    for facecod, faceloc in zip(facescod, faces):
        # comparamos los rostros en tiempor real con la BD
        comparacion = fr.compare_faces(rostroscod, facecod)
        # calculamos la similitud
        # entre mas pequeño sea la similitud mas aceptable
        similitud = fr.face_distance(rostroscod, facecod)

        print(f"similitud del rostro {similitud}")
        # Buscamos el valor más bajo
        min = np.argmin(similitud)

        #si comparacion esta en el mas bajo
        if comparacion[min]:
            #asignamos el nombre
            nombre = clas[min].upper()
            print(nombre)
            #extraemos las cordenadas 
            yi, xf, yf, xi = faceloc
            #escalamos
            yi, xf, yf, xi = yi * 4, xf * 4, yf * 4, xi * 4
            #extraemos los indice.
            indice = comparacion.index(True)
            #comparamos
            if comp1 != indice:
                #nos permite cambiar de colores segun pasa el programa
                r = random.randrange(0, 255, 50)
                g = random.randrange(0, 255, 50)
                b = random.randrange(0, 255, 50)

                comp1 = indice
            # dibujamos el rectangulo y identificamos a la persona con su id
            if comp1 == indice:
                #cv2.rectangle(frame, (xi, yi), (xf, yf), (r, g, b), 3)
                #cv2.rectangle(frame, (xi, yf - 35), (xf, yf), (r, g, b), cv2.FILLED)
                cv2.putText(frame, nombre, (xi + 6, yf - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                registro(nombre)

    #Deteccion de emociones
    #Despues de aver capturados todos los frame 
    #hacemos una correccion de color
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Observamos los resultados
    resultado = Malla_facial.process(frameRGB)

    # creamos las siguientes array para guardar los puntos de interes 
    puntox = []
    puntoy = []
    lista = []
    r = 5
    t = 3
    # if si detecta rostro:
    if resultado.multi_face_landmarks:
        #si detecta rostro se guardara en la variable rostro 
        for rostro in resultado.multi_face_landmarks:
            # hacemos las conexiones de los puntos.
            # por lo que tenemos las coordenadas de los rostros detectados
            mapa_dibujo.draw_landmarks(frame, rostro, mpmalla_facial.FACEMESH_TESSELATION, confi_dibujo, confi_dibujo)

            # Ahora vamos a extraer los puntos del rostro detectado
            for id, puntos in enumerate(rostro.landmark):
                #sacamos el alto y el ancho de la proporcion de la img
                al, an, c = frame.shape
                #multiplicamos la proporcion * el ancho y alto lo que
                # nos da la cordenada en X y Y
                x, y = int(puntos.x * an), int(puntos.y * al)
                # guardamos todos las cordenadas
                puntox.append(x)
                puntoy.append(y)
                # guardamos el id con las respectivas cordenadas, lo cual tendria 
                # una cantidad de 468 puntos de interes
                lista.append([id, x, y])
                #si hay 468 puntos de interes de la lista
                if len(lista) == 468:
                    # preguntamos por la ceja derecha. 
                    # preguntaremos por el punto 65 y sus cordenadas.
                    # estos seran nuestros puntos claves
                    x1, y1 = lista[65][1:]
                    # preguntaremos por el punto 158 y sus cordenadas.
                    x2, y2 = lista[158][1:]
                    # obtenmos la cordenada en x y y.
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    # aqui podemos ver en la malla el comportamiento de la longitud
                    #cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 0), t)
                    #cv2.circle(frame, (x1, y1), r, (0, 0, 0), cv2.FILLED)
                    #cv2.circle(frame, (x2, y2), r, (0, 0, 0), cv2.FILLED)
                    #cv2.circle(frame, (cx, cy), r, (0, 0, 0), cv2.FILLED)

                    # obtenemos la longitud 
                    longitud1 = math.hypot(x2 - x1, y2 - y1)

                    print(f"La longitud 1 es : {longitud1}")

                    # Ceja izquierda
                    x3, y3 = lista[295][1:]
                    x4, y4 = lista[385][1:]
                    cx2, cy2 = (x3 + x4) // 2, (y3 + y4) // 2

                    longitud2 = math.hypot(x4 - x3, y4 - y3)
                    print(f"La longitud 2 es : {longitud2}")

                    # Boca
                    x5, y5 = lista[78][1:]
                    x6, y6 = lista[308][1:]
                    cx3, cy3 = (x5 + x6) // 2, (y5 + y6) // 2

                    longitud3 = math.hypot(x6 - x5, y6 - y5)
                    print(f"La longitud 3 es : {longitud3}")

                    # Boca y apertura
                    x7, y7 = lista[13][1:]
                    x8, y8 = lista[14][1:]
                    cx4, cy4 = (x7 + x8) // 2, (y7 + y8) // 2

                    longitud4 = math.hypot(x8 - x7, y8 - y7)
                    print(f"La longitud 3 es : {longitud4}")

                    # Clasificación de emociones
                    # Bravo
                    if longitud1 < 19 and longitud2 < 19 and longitud3 > 80 and longitud3 < 109 and longitud4 < 5:
                        emocion = 'Molesto'
                    # Feliz
                    elif longitud1 > 20 and longitud1 < 30 and longitud2 > 20 and longitud2 < 30 and longitud3 > 100 and longitud4 > 12:
                        emocion = 'Feliz'
                    # Asombrado
                    elif longitud1 > 25 and longitud2 > 25 and longitud3 > 80  and longitud3 <90 and longitud4 > 20:
                        emocion = 'Asombrado'
                    # Triste
                    elif longitud1 > 20 and longitud1 < 35 and longitud2 > 20 and longitud2 < 35 and longitud3 > 80 and longitud3 < 90 and longitud4 < 5:
                        emocion = 'Triste'
                    else:
                        emocion = 'Normal'  # Emoción neutral si no se cumple ninguna condición

                    # Si la emoción cambió, actualiza la emoción actual
                    if emocion_actual != emocion:
                        emocion_actual = emocion

                     
    # Si la emoción actual cambió desde la emoción anterior, muestra la nueva emoción
    if emocion_actual != emocion_actual_anterior:
        cv2.putText(frame, f"Su emocion es: {emocion_actual}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Mostramos frames
    cv2.imshow("Reconocimiento facial y emociones", frame)

    t = cv2.waitKey(1)
    if t == 27:
         break

# Crear la ventana de la interfaz gráfica
root = tk.Tk()
root.title("Reconocimiento Facial y Emociones")

# Agregar una etiqueta para mostrar la emoción actual
emocion_label = tk.Label(root, text="", font=("Helvetica", 16))
emocion_label.pack()

# Iniciar la función para actualizar la emoción
actualizar_emocion()

# Ejecutar el bucle principal de la interfaz gráfica
root.mainloop()


cap.release()
cv2.destroyAllWindows()