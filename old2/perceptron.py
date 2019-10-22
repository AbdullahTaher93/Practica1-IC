import function_rn
import numpy as np
from urllib import request
import gzip
import pickle
import random
from PIL import Image
# 10 neuronas, cada nerurona con una salida ente 1 y 0 (salida no vector)

#Lectura de MNIST
filename = [
["training_images","train-images-idx3-ubyte.gz"],
["test_images","t10k-images-idx3-ubyte.gz"],
["training_labels","train-labels-idx1-ubyte.gz"],
["test_labels","t10k-labels-idx1-ubyte.gz"]
]

mnist = {}
for name in filename[:2]:
    with gzip.open(name[1], 'rb') as f:
        mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
for name in filename[-2:]:
    with gzip.open(name[1], 'rb') as f:
        mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
x_train, t_train, x_test, t_test = mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]


##ESTRUCTURAS
##PERCEPTRON, 1 capa (+ input), 10 neuronas con 28x28 pesos (conexiones)
#MNIST_Image[pixel]
#MNIST_Label;
"""
1. Set all inputs to 0
2. Set all weights to a random value -1 1
3. Set output to 0 
"""

input={}
weight = [ [0 for columna in range(784)] for fila in range (10)]
output={}
LEARNING_RATE = 0.05

for i in range(784):#TODO  ad bias
    input[i]=0

for i in range(10):
    output[i]=0
    for j in range(784):#TODO  ad bias
        weight[i][j]=random.uniform(-1, 1)  #peso aleatorio entre -1 y 1
########################################################################
# TRAINING
modo='training'
for n in range(len(t_train)):
    if n%10000==0:
        print("Entenando en la iteración entrenamiento ",n)
    target_numero=t_train[n]
    numbrerv=function_rn.number2vector(target_numero)
    inputImg=function_rn.normalizaImg(x_train[n])

    #son 10 neuronas
    for i in range(10):
        weight_neuro=weight[i]
        target_binario_neuro=numbrerv[i]
        output[i]=function_rn.trainNeuron(inputImg,weight_neuro,target_binario_neuro,LEARNING_RATE,modo)
        weight[i]=weight_neuro #actualiza pesos
#valor utilizado en las correcciones de los pesos

########################################################################
# TEST
modo='test'
cont_nok=0
for n in range(len(t_test)):
    if n%10000==0:
        print("Entenando en la iteración test ",n)
    target_numero=t_test[n]
    numbrerv=function_rn.number2vector(target_numero)
    inputImg=function_rn.normalizaImg(x_test[n])
    #son 10 neuronas
    for i in range(10):
        weight_neuro=weight[i]
        target_binario_neuro=numbrerv[i]
        output[i]=function_rn.trainNeuron(inputImg,weight_neuro,target_binario_neuro,LEARNING_RATE,modo)
        if output[i]!=target_binario_neuro:
            cont_nok=cont_nok+1
            break

print('--------------------------------------')
print('Total test:',len(t_test))
print('errores:',cont_nok)
print('porcentaje errores:',cont_nok*100/len(t_test))

"""   
img = np.invert(Image.open("4.png").convert('L')).ravel()
img=function_rn.normalizaImg(img)
for i in range(10):
    weight_neuro=weight[i]
    if i==4:
        target_binario_neuro=1
    else:   
        target_binario_neuro=0
    output[i]=function_rn.trainNeuron(img,weight_neuro,target_binario_neuro,LEARNING_RATE)
    print(output[i])
 """