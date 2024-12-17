import cv2
import numpy as np

# Inicialización de la captura de video
cap = cv2.VideoCapture(0)
canvas = np.zeros((480, 640, 3), dtype="uint8")

# Configuración para detectar el color de la piel
lower_skin = np.array([0, 20, 70], dtype=np.uint8)
upper_skin = np.array([20, 255, 255], dtype=np.uint8)

prev_x, prev_y = 0, 0  # Posición anterior del dedo
drawing = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Máscara para detectar la piel
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Filtros para mejorar la detección
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = cv2.dilate(mask, None, iterations=2)
    mask = cv2.erode(mask, None, iterations=1)

    # Encontrar contornos de la mano
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Seleccionar el contorno más grande (asumimos que es la mano)
        max_contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(max_contour)

        # Encontrar los defectos de convexidad para detectar la punta del dedo
        if len(hull) > 3:  # Se necesitan al menos 4 puntos para hallar los defectos
            defects = cv2.convexityDefects(max_contour, cv2.convexHull(max_contour, returnPoints=False))

            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(max_contour[s][0])
                    end = tuple(max_contour[e][0])
                    far = tuple(max_contour[f][0])

                    # Detectamos la punta del dedo índice sin restricciones de la mitad de la pantalla
                    index_x, index_y = start

                    # Activar o desactivar el dibujo
                    if drawing:
                        if prev_x == 0 and prev_y == 0:
                            prev_x, prev_y = index_x, index_y
                        cv2.line(canvas, (prev_x, prev_y), (index_x, index_y), (255, 0, 0), 5)
                        prev_x, prev_y = index_x, index_y
                    else:
                        prev_x, prev_y = 0, 0

                    # Romper el ciclo después de encontrar la punta
                    break

    # Mostrar la combinación de la cámara y el canvas
    frame = cv2.addWeighted(frame, 1, canvas, 0.5, 0)
    cv2.imshow("Paint con Dedo Índice", frame)

    # Tecla para activar o desactivar el dibujo
    if cv2.waitKey(1) & 0xFF == ord('d'):
        drawing = not drawing

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
