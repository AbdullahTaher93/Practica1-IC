import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from PIL import Image
#input_data.py descargará los archvios MNIST en la carpeta MNIST_data/ que creará de ser necesario
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  # y labels are oh-encoded
#one_hot trae la data en arreglo con valores de 0 y 1 (binario)

#the batch size refers to how many training examples we are using at each step......No es lo mismo que  n_test = mnist.test.num_examples  # 10,000????
batch_size = 128
batch_x, batch_y = mnist.train.next_batch(batch_size)

#n_input = 784  # input layer (28x28 pixels)
print('Dimensions: %s x %s' % (batch_x.shape[0], batch_x.shape[1]))
print('\n1st row', batch_x[0])
print (mnist.train.num_examples)
print('\n1st etiq', batch_y[0])
print('\n')
i=1
for row in batch_x:
    print('\n')
    for element in row:
        print(element,end=' ')
        if i%128==0:
            print('\n')
        i=i+1
        
