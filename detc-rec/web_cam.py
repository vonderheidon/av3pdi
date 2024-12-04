import cv2
import numpy as np
import smtplib

def enviarEmail():
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.ehlo()
    server.starttls()

    user = ""
    senha = ""
    para = ""
    msg = "UM ROSTO FOI DETECTADO"

    try:
        server.login(user, senha)
        print("Logando no servidor SMTP...")
    except:
        print("Erro: Email ou senha incorretos, ou permissão negada.")

    server.sendmail(user, para, msg)
    print("Email enviado.")
    server.close()

config_path = "classificadores/yolov3.cfg"
weights_path = "classificadores/yolov3.weights"
names_path = "classificadores/coco.names"

net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

with open(names_path, "r") as f:
    classes = f.read().strip().split("\n")

video = cv2.VideoCapture(0)

print("Iniciando detecção de rostos com YOLO...")

while True:
    conexao, quadro = video.read()
    if not conexao:
        print("Erro ao acessar a câmera.")
        break

    altura_frame, largura_frame = quadro.shape[:2]
    blob = cv2.dnn.blobFromImage(quadro, 1 / 255.0, (416, 416), swapRB=True, crop=False)
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

            if confidence > conf_threshold and class_id == 0:
                center_x, center_y, w, h = (obj[0:4] * np.array([largura_frame, altura_frame, largura_frame, altura_frame])).astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                caixas.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(caixas, confidences, conf_threshold, nms_threshold)

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = caixas[i]
            cv2.rectangle(quadro, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(quadro, f"Pessoa ({confidences[i]:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print("Rosto detectado!")

    else:
        print("Nenhum rosto detectado.")

    cv2.imshow("Detecção com YOLO", quadro)

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
