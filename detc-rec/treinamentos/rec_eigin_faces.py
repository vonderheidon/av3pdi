#coding: utf-8
import cv2

camera = cv2.VideoCapture(0)
detector_face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
reconhecimento_face = cv2.face.EigenFaceRecognizer_create()
reconhecimento_face.read("classificadorEigen.yml")
largura, altura = 220, 220
tag = cv2.FONT_HERSHEY_COMPLEX_SMALL


while (True):
	
	conectado, imagem = camera.read()
	
	imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
	faces_detectadas = detector_face.detectMultiScale(imagem_cinza, scaleFactor=1.5, minSize=(30, 30))
	
	for (x,y,l,a) in faces_detectadas:
		imagem_face = cv2.resize(imagem_cinza[y:y + a, x:x + l], (largura, altura))
		cv2.rectangle(imagem, (x, y),(x + l, y + a),(0,0,255), 2)

		#reconhecimento
		ids, autenticador = reconhecimento_face.predict(imagem_face)
		if ids == 1:
			nome = 'SEU_NOME'
		else:
			nome = 'Não encontrado'
		cv2.putText(imagem, nome, (x, y+(a+30)), tag, 2, (0,0,255))
		cv2.putText(imagem, str(autenticador), (x,y + (a+50)), tag, 1,(0,255,0))
		
	
	cv2.imshow("Face", imagem)
	
	if cv2.waitKey(1) == ord('q'):
		break
	
camera.release()
cv2.destroyAllWindows()
