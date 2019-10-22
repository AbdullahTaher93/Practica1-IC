#import _pickle as cPickle
#import mnist
import numpy as np
from urllib import request
import gzip
import pickle
import random
from PIL import Image
# 10 neuronas, cada nerurona con una salida ente 1 y 0 (salida no vector)
# #pasa un entero a su representación en vector 
def number2vector(numero):
    #print("en print def ",numero)
    for i in range(10):
    #    print(i)
        if i==numero:
            numbrerv[i]=1 
        else:
            numbrerv[i]=0
   # print(numbrerv)    
    return numbrerv
#Lectura de MNIST

#entrena. 
def trainNeuron(input,weight_neuro, target):
    output=0
    for j in range (784):
        output= output+ (input[j] * weight_neuro[j])
        #normalizar output
    output= output/784

    estado=activation_function(output)
    err=getCellError(target,estado)
    updateCellWeights(weight_neuro, err, input)
    return output

def getCellError(target,estado):
    err=target-estado  
    return err


def updateCellWeights(weight_neuro, err, input):
    LEARNING_RATE = 0.05 
    for i in range(784):
        weight_neuro[i] = weight_neuro[i] + (LEARNING_RATE * input[i] * err)
        #print ("*******",weight_neuro[i], input[i], err)  
        
def activation_function(output):
    #binario 
    if output >0:
        return 1
    else:   
        return 0


#normalizar input
def normalizaImg(vector):
    vector_t={}
    for i in range(784):
        vector_t[i]=vector[i]/256 #256:rgb max
    return vector_t

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
"""
with open("mnist.pkl", 'wb') as f:
    pickle.dump(mnist,f)
print("Save complete.")
"""
 #
 # mnist.init()
x_train, t_train, x_test, t_test = mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]


##ESTRUCTURAS
##PERCEPTRON 10 celulas con 28x28 pesos, una capa+input
#MNIST_Image[pixel]
#MNIST_Label;
#Cell[input] (array 28x28)
#Cell[weight] (array 28x28)
#Cell[output] (double)
"""
struct MNIST_ImageFileHeader{
    uint32_t magicNumber;
    uint32_t maxImages;
    uint32_t imgWidth;
    uint32_t imgHeight;
};

struct MNIST_LabelFileHeader{
    uint32_t magicNumber;
    uint32_t maxLabels;
};
"""
#Layer[cell] (array 10)
#targerVector=(array 10) (Vector.val)
"""
1. Set all inputs to 0
2. Set all weights to a random value 0-1
3. Set output to 0 
"""

input={}
weight = [ [0 for columna in range(784)] for fila in range (10)]
output={}


for i in range(784):#TODO  ad bias
    input[i]=0

for i in range(10):
    output[i]=0
    for j in range(784):#TODO  ad bias
        weight[i][j]=random.uniform(-1, 1)  #peso aleatorio entre -1 y 1

for n in range(len(t_train)): #(len(t_train)):
    target_numero=t_train[n]
    numbrerv={}
    numbrerv=number2vector(target_numero)
    inputImg=normalizaImg(x_train[n])

    #son 10 neuronas
    for i in range(10):
        weight_neuro=weight[i]
        #print(weight_neuro[i])
        target_binario_neuro=numbrerv[i]
        #print(i,target_binario_neuro,target_numero)
        output[i]=trainNeuron(inputImg,weight_neuro,target_binario_neuro)
        weight[i]=weight_neuro #actualiza pesos
        
        #print(weight_neuro[i])
        #print("##########################")
    #print((weight))
    
img = np.invert(Image.open("4.png").convert('L')).ravel()
img=normalizaImg(img)
for i in range(10):
    weight_neuro=weight[i]
    #print(weight_neuro)
    if i==4:
        target_binario_neuro=1
    else:   
        target_binario_neuro=0
    output[i]=trainNeuron(img,weight_neuro,target_binario_neuro)
    print(output[i])
 