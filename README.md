# reconhecimento_facial
import cv2

# Carregar o modelo pré-treinado para detecção de faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializar a câmera (0 é o índice da câmera padrão)
cap = cv2.VideoCapture(0)

while True:
    # Ler um frame da câmera
    ret, frame = cap.read()

    # Converter o frame para escala de cinza para a detecção de faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar faces no frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Desenhar retângulos ao redor das faces detectadas
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Mostrar o frame com as detecções
    cv2.imshow('Reconhecimento Facial', frame)

    # Sair do loop quando pressionar a tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
