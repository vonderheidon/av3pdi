import cv2
import os
import numpy as np

eigen_face = cv2.face.EigenFaceRecognizer_create()
# eigen_face = cv2.face.EigenFaceRecognizer_create(num_components=50, threshold=2)
fisher_face = cv2.face.FisherFaceRecognizer_create()
# fisher_face = cv2.face.FisherFaceRecognizer_create(num_components=50, threshold=2)
lpbh_face = cv2.face.LBPHFaceRecognizer_create()

def imagem_nomes():
    caminhos = [os.path.join('imagens/treinamento', f) for f in os.listdir('imagens/treinamento')]

    faces = []
    ids = []

    for caminhoImagem in caminhos:
        imagemFace = cv2.cvtColor(cv2.imread(caminhoImagem), cv2.COLOR_BGR2GRAY)
        identificacao = int(os.path.split(caminhoImagem)[-1].split('.')[1])

        ids.append(identificacao)
        faces.append(imagemFace)
    return np.array(ids), faces

ids, faces = imagem_nomes()

print("Treinando...")

eigen_face.train(faces, ids)
eigen_face.write('treinamentos/classificadorEigen.yml')

fisher_face.train(faces, ids)
fisher_face.write('treinamentos/classificadorFisher.yml')

lpbh_face.train(faces, ids)
lpbh_face.write('treinamentos/classificadorLBPH.yml')

print("Treinamento Finalizado")
