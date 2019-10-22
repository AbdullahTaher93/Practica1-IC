import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from PIL import Image
#input_data.py descargará los archvios MNIST en la carpeta MNIST_data/ que creará de ser necesario
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  # y labels are oh-encoded
#one_hot trae la data en arreglo con valores de 0 y 1 (binario)


#TODO: INVERTIGAR KERAS

n_train = mnist.train.num_examples  # 55,000
n_validation = mnist.validation.num_examples  # 5000
n_test = mnist.test.num_examples  # 10,000

#ARQUITECTURA
n_input = 784  # input layer (28x28 pixels)
n_hidden1 = 512  # 1st hidden layer
n_hidden2 = 256  # 2nd hidden layer
n_hidden3 = 128  # 3rd hidden layer
n_output = 10  # output layer (0-9 digits)

#HYPEPARAMETERS CONSULTAR!!!!!
#learning_rate how much the parameters will adjust at each step of the learning process. 
# These adjustments are a key component of training: after each pass through the network we tune the weights slightly to try and reduce the loss. 
# Larger learning rates can converge faster, but also have the potential to overshoot the optimal values as they are updated
learning_rate = 1e-4
#The number of iterations refers to how many times we go through the training step
n_iterations = 1000
#the batch size refers to how many training examples we are using at each step......No es lo mismo que  n_test = mnist.test.num_examples  # 10,000????
batch_size = 128 #cuantos datos extrae de MNIST en cada iteración for
#the dropout variable represents a threshold (limit) at which we eliminate some units at random. 
# We will be using dropout in our final hidden layer to give each unit a 50% chance of being eliminated at every training step. 
# This helps prevent overfitting.(sobreajuste????)
dropout = 0.5

#Tensors..Array
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_output])
#The keep_prob tensor is used to control the dropout rate, and we initialize it as a placeholder rather than an immutable variable because 
# we want to use the same tensor both for training (when dropout is set to 0.5) and testing (when dropout is set to 1.0).
keep_prob = tf.placeholder(tf.float32)

#arreglos de pesos por capa, inicializado con valores random cercanos a cero (+ y -). stddev=0.1 derivación estandar. 
#por defecto media igual 0
#valores se modifican automaticamente en el entrenamiento. 
weights = {
    'w1': tf.Variable(tf.truncated_normal([n_input, n_hidden1], stddev=0.1)),
    'w2': tf.Variable(tf.truncated_normal([n_hidden1, n_hidden2], stddev=0.1)),
    'w3': tf.Variable(tf.truncated_normal([n_hidden2, n_hidden3], stddev=0.1)),
    'out': tf.Variable(tf.truncated_normal([n_hidden3, n_output], stddev=0.1)),
}
"""arreglo de bias por capa con valor 0.1
valores se modifican automaticamente en el entrenamiento. """
biases = {
    'b1': tf.Variable(tf.constant(0.1, shape=[n_hidden1])), 
    'b2': tf.Variable(tf.constant(0.1, shape=[n_hidden2])),
    'b3': tf.Variable(tf.constant(0.1, shape=[n_hidden3])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_output]))
}

#matmul: producto matrices

#????????? X no se ha asignado valor aun???   X*W+b
layer_1 = tf.add(tf.matmul(X, weights['w1']), biases['b1'])
layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
layer_3 = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])
#??????? nn.dropout
#At the last hidden layer, we will apply a dropout operation using our keep_prob value of 0.5.
layer_drop = tf.nn.dropout(layer_3, keep_prob)
output_layer = tf.matmul(layer_3, weights['out']) + biases['out']


#LossFunction  cross-entropy, also known as log-loss, 
cross_entropy = tf.reduce_mean( #reduce media
    tf.nn.softmax_cross_entropy_with_logits(
        labels=Y, logits=output_layer
        ))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#method of evaluating the accuracy (exactitud)
correct_pred = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))#exactitud

#Graph
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# train on mini batches
#The number of iterations refers to how many times we go through the training step
for i in range(n_iterations):
    #batch_x arrego de 128x784 en donde cada fila es una entrada (imagen). batch_y es arrego de etiqueta 128x10. cada fila representa un digito de resultado esperado
    #cuantos datos extrae de MNIST en cada iteración for
    batch_x, batch_y = mnist.train.next_batch(batch_size) 
    sess.run(train_step, feed_dict={
        X: batch_x, Y: batch_y, keep_prob: dropout
        })

    # print loss and accuracy (per minibatch)
    if i % 100 == 0:
        minibatch_loss, minibatch_accuracy = sess.run(
            [cross_entropy, accuracy],
            feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0}
            )
        print(
            "Iteration",
            str(i),
            "\t| Loss =",
            str(minibatch_loss),
            "\t| Accuracy =",
            str(minibatch_accuracy)
            )

test_accuracy = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1.0})
print("\nAccuracy on test set:", test_accuracy)

#PRUEBA IMAGEN LOCAL

img = np.invert(Image.open("4.png").convert('L')).ravel()
prediction = sess.run(tf.argmax(output_layer, 1), feed_dict={X: [img]})
print ("Prediction for test image:", np.squeeze(prediction))
sess.close()
