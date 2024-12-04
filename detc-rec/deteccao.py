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

imagem = cv2.imread("imagens/outras/carro1.jpg")

altura_frame, largura_frame = imagem.shape[:2]
blob = cv2.dnn.blobFromImage(imagem, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)

detections = net.forward(output_layers)

caixas = []
confidences = []
class_ids = []

conf_threshold = 0.5
nms_threshold = 0.4

for detection in detections:
    for obj in detection:
        scores = obj[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > conf_threshold:
            center_x, center_y, w, h = (obj[0:4] * np.array([largura_frame, altura_frame, largura_frame, altura_frame])).astype("int")
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            caixas.append([x, y, int(w), int(h)])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indices = cv2.dnn.NMSBoxes(caixas, confidences, conf_threshold, nms_threshold)

total_detectado = len(indices)
print(f"Total Detectado > {total_detectado}")
print("\nMatriz das posições")
print("| eixo_x |  eixo_y |  largura | altura |")
print("-" * 40)

for i in indices.flatten():
    x, y, w, h = caixas[i]
    class_name = classes[class_ids[i]]
    print(f"|   {x}   |   {y}   |    {w}   |   {h}   |")
    print(f"Classe detectada: {class_name}")
    print("-" * 40)

    cv2.rectangle(imagem, (x, y), (x + w, y + h), (255, 0, 255), 2)
    cv2.putText(imagem, f"{class_name} ({confidences[i]:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

cv2.imshow("Objetos detectados", imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()
