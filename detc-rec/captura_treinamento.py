import cv2
import numpy as np

config_path = "classificadores/yolov3.cfg"
weights_path = "classificadores/yolov3.weights"
names_path = "classificadores/coco.names"

net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

with open(names_path, "r") as f:
    classes = f.read().strip().split("\n")

camera = cv2.VideoCapture(0)

amostra = 1
numero_amostras = 25

ids = input("Digite o ID do usuário: ")

largura, altura = 220, 220
conf_threshold = 0.5
nms_threshold = 0.4

print("Capturando as faces...")

while True:
    conectado, imagem = camera.read()
    if not conectado:
        print("Erro ao acessar a câmera!")
        break

    altura_frame, largura_frame = imagem.shape[:2]
    blob = cv2.dnn.blobFromImage(imagem, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    caixas = []
    confidences = []
    class_ids = []

    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > conf_threshold and class_id == 0:
                center_x, center_y, w, h = (obj[0:4] * np.array([largura_frame, altura_frame, largura_frame, altura_frame])).astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                caixas.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(caixas, confidences, conf_threshold, nms_threshold)

    for i in indices.flatten():
        x, y, w, h = caixas[i]
        cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 0, 255), 2)

        regiao = imagem[y:y + h, x:x + w]
        if np.average(regiao) > 110:
            face = cv2.resize(regiao, (largura, altura))
            cv2.imwrite(f"imagens/treinamento/pessoa.{ids}.{amostra}.jpg", face)
            print(f"Foto {amostra} salva.")
            amostra += 1

    cv2.imwrite(f"temp_frame_{amostra}.jpg", imagem)

    if cv2.waitKey(1) & 0xFF == ord('q') or amostra > numero_amostras:
        break

camera.release()
cv2.destroyAllWindows()
print("Faces capturadas com sucesso.")
