# Practica1-IC

## Description  

The purpose of this Practice is solving a pattern recognition problem using artificial neural networks, we will evaluate by using  several types of neural networks to solve a  Handwritten Digit Recognition problem. using  [MNIST]( http://yann.lecun.com/exdb/mnist/) database.

### Neural Networks introduction 

Neural networks are used as a method of deep learning, one of the many subfields of artificial intelligence. They were first proposed around 70 years ago as an attempt at simulating the way the human brain works, though in a much more simplified form. Individual ‘neurons’ are connected in layers, with weights assigned to determine how the neuron responds when signals are propagated through the network. Previously, neural networks were limited in the number of neurons they were able to simulate, and therefore the complexity of learning they could achieve. But in recent years, due to advancements in hardware development, we have been able to build very deep networks, and train them on enormous datasets to achieve breakthroughs in machine intelligence.

These breakthroughs have allowed machines to match and exceed the capabilities of humans at performing certain tasks. One such task is object recognition. Though machines have historically been unable to match human vision, recent advances in deep learning have made it possible to build neural networks which can recognize objects, faces, text, and even emotions.

### Neural Network Types

![NNstypes](https://github.com/AbdullahTaher93/Practica1-IC/blob/master/images/NNTypes.png)

We can read more about NNs types [here](https://towardsdatascience.com/the-mostly-complete-chart-of-neural-networks-explained-3fb6f2367464).



###  Description of 'MINST' Database 

The [MNIST dataset](http://yann.lecun.com/exdb/mnist/) is an acronym that stands for the Modified National Institute of Standards and Technology dataset.

It is a dataset of 60,000 small square 28×28 pixel grayscale images of handwritten single digits between 0 and 9.
The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively.
It is a widely used and deeply understood dataset and, for the most part, is “solved.” Top-performing models are deep learning convolutional neural networks that achieve a classification accuracy of above 99%, with an error rate between 0.4 %and 0.2% on the hold out test dataset.


#### Implementation

Now, we will create a simple program by python3 using [Keras API](https://keras.io/),Keras is a high-level neural networks API, written in Python and capable of running on top of [TensorFlow](https://github.com/tensorflow/tensorflow), CNTK, or Theano. It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research.




The first thing we have to do it is Loading the database.

    # example of loading the mnist dataset
    from keras.datasets import mnist
    from matplotlib import pyplot
    # load dataset
    (trainX, trainy), (testX, testy) = mnist.load_data()


But,before that we have to install Keras and tensorflow:

    sudo pip install Keras
    sudo pip install tensorflow


We can show the database summary using.

    # summarize loaded dataset
    print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
    print('Test: X=%s, y=%s' % (testX.shape, testy.shape))

![summary](https://github.com/AbdullahTaher93/Practica1-IC/blob/master/images/summary.jpg)

A plot of the first nine images in the dataset is also we can create showing the natural handwritten nature of the images to be classified.

    # plot first few images
    for i in range(9):
        # define subplot
        pyplot.subplot(330 + 1 + i)
        # plot raw pixel data
        pyplot.imshow(trainX[i], cmap=pyplot.get_cmap('gray'))
    # show the figure
    pyplot.show()


![nineimages](https://github.com/AbdullahTaher93/Practica1-IC/blob/master/images/nineImages.png)


### Model Evaluation Methodology

#### 1.Baseline Model



