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


## Model Evaluation Methodology

#### 1.Baseline Model

The first step is to develop a baseline model.

This is critical as it both involves developing the infrastructure for the test harness so that any model we design can be evaluated on the dataset, and it establishes a baseline in model performance on the problem, by which all improvements can be compared.

The design of the test harness is modular, and we can develop a separate function for each piece. This allows a given aspect of the test harness to be modified or inter-changed, if we desire, separately from the rest.

We can develop this test harness with five key elements. They are the loading of the dataset, the preparation of the dataset, the definition of the model, the evaluation of the model, and the presentation of results.

After Loading database,we can load the images and reshape the data arrays to have a single color channel.becasue we know we know that that the images all have the same square size of 28×28 pixels, and that the images are grayscale.

    # reshape dataset to have a single channel
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))

We also know that there are 10 classes and that classes are represented as unique integers.
We can, therefore, use a one hot encoding for the class element of each sample, transforming the integer into a 10 element binary vector with a 1 for the index of the class value, and 0 values for all other classes. We can achieve this with the to_categorical() utility function.

    # one hot encode target values
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)

#### Prepare Pixel Data

We know that the pixel values for each image in the dataset are unsigned integers in the range between black and white, or 0 and 255.

We do not know the best way to scale the pixel values for modeling, but we know that some scaling will be required.

A good starting point is to normalize the pixel values of grayscale images, e.g. rescale them to the range [0,1]. This involves first converting the data type from unsigned integers to floats, then dividing the pixel values by the maximum value.

    # scale pixels
    def prep_pixels(train, test):
        # convert from integers to floats
        train_norm = train.astype('float32')
        test_norm = test.astype('float32')
        # normalize to range 0-1
        train_norm = train_norm / 255.0
        test_norm = test_norm / 255.0
        # return normalized images
        return train_norm, test_norm

Next, we need to define a baseline convolutional neural network model for the problem.

For the convolutional front-end, we can start with a single convolutional layer with a small filter size (3,3) and a modest number of filters (32) followed by a max pooling layer. The filter maps can then be flattened to provide features to the classifier.

Given that the problem is a multi-class classification task, we know that we will require an output layer with 10 nodes in order to predict the probability distribution of an image belonging to each of the 10 classes. This will also require the use of a softmax activation function. Between the feature extractor and the output layer, we can add a dense layer to interpret the features, in this case with 100 nodes.

All layers will use the ReLU activation function and the He weight initialization scheme, both best practices.

We will use a conservative configuration for the stochastic gradient descent optimizer with a learning rate of 0.01 and a momentum of 0.9. The categorical cross-entropy loss function will be optimized, suitable for multi-class classification, and we will monitor the classification accuracy metric, which is appropriate given we have the same number of examples in each of the 10 classes.

    # define cnn model
    def define_model():
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(10, activation='softmax'))
        # compile model
        opt = SGD(lr=0.01, momentum=0.9)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

#### Evaluation

The model will be evaluated using five-fold cross-validation. The value of k=5 was chosen to provide a baseline for both repeated evaluation and to not be so large as to require a long running time. Each test set will be 20% of the training dataset, or about 12,000 examples, close to the size of the actual test set for this problem.

We will train the baseline model for a modest 10 training epochs with a default batch size of 32 examples. The test set for each fold will be used to evaluate the model both during each epoch of the training run, so that we can later create learning curves, and at the end of the run, so that we can estimate the performance of the model. As such, we will keep track of the resulting history from each run, as well as the classification accuracy of the fold.

The evaluate_model() function below implements these behaviors, taking the defined model and training dataset as arguments and returning a list of accuracy scores and training histories that can be later summarized.

    # evaluate a model using k-fold cross-validation
    def evaluate_model(model, dataX, dataY, n_folds=5):
        scores, histories = list(), list()
        # prepare cross validation
        kfold = KFold(n_folds, shuffle=True, random_state=1)
        # enumerate splits
        for train_ix, test_ix in kfold.split(dataX):
            # select rows for train and test
            trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
            # fit model
            history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
            # evaluate model
            _, acc = model.evaluate(testX, testY, verbose=0)
            print('> %.3f' % (acc * 100.0))
            # stores scores
            scores.append(acc)
            histories.append(history)
        return scores, histories

#### Present Results

First, the diagnostics involve creating a line plot showing model performance on the train and test set during each fold of the k-fold cross-validation. These plots are valuable for getting an idea of whether a model is overfitting, underfitting, or has a good fit for the dataset.

We will create a single figure with two subplots, one for loss and one for accuracy. Blue lines will indicate model performance on the training dataset and orange lines will indicate performance on the hold out test dataset. The summarize_diagnostics() function below creates and shows this plot given the collected training histories.


    # plot diagnostic learning curves
    def summarize_diagnostics(histories):
        for i in range(len(histories)):
            # plot loss
            pyplot.subplot(211)
            pyplot.title('Cross Entropy Loss')
            pyplot.plot(histories[i].history['loss'], color='blue', label='train')
            pyplot.plot(histories[i].history['val_loss'], color='orange', label='test')
            # plot accuracy
            pyplot.subplot(212)
            pyplot.title('Classification Accuracy')
            pyplot.plot(histories[i].history['acc'], color='blue', label='train')
            pyplot.plot(histories[i].history['val_acc'], color='orange', label='test')
        pyplot.show()

The summarize_performance() function below implements this for a given list of scores collected during model evaluation.

    # summarize model performance
    def summarize_performance(scores):
        # print summary
        print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
        # box and whisker plots of results
        pyplot.boxplot(scores)
        pyplot.show()

We need a function that will drive the test harness.

This involves calling all of the define functions.

    # run the test harness for evaluating a model
    def run_test_harness():
        # load dataset
        trainX, trainY, testX, testY = load_dataset()
        # prepare pixel data
        trainX, testX = prep_pixels(trainX, testX)
        # define model
        model = define_model()
        # evaluate model
        scores, histories = evaluate_model(model, trainX, trainY)
        # learning curves
        summarize_diagnostics(histories)
        # summarize estimated performance
        summarize_performance(scores)
    # entry point, run the test harness
    run_test_harness()

Running the example prints the classification accuracy for each fold of the cross-validation process. This is helpful to get an idea that the model evaluation is progressing.

We can see two cases where the model achieves perfect skill and one case where it achieved lower than 99% accuracy. These are good results.


![acc](https://github.com/AbdullahTaher93/Practica1-IC/blob/master/images/acc.jpg)

Next, a diagnostic plot is shown, giving insight into the learning behavior of the model across each fold.

In this case, we can see that the model generally achieves a good fit, with train and test learning curves converging. There is no obvious sign of over- or underfitting.

![AccFigure](https://github.com/AbdullahTaher93/Practica1-IC/blob/master/images/AccFigure.jpeg)


Next, a summary of the model performance is calculated. We can see in this case, the model has an estimated skill of about 99.6%, which is impressive, although it has a high standard deviation of about half a percent.


Accuracy: mean=99.668 std=0.591, n=5


The Python File you can find it [here](https://github.com/AbdullahTaher93/Practica1-IC/blob/master/ICPRACTICA1/testing.py)

### Improvement to Learning

[Batch normalization](https://en.wikipedia.org/wiki/Batch_normalization) is a technique designed to automatically standardize the inputs to a layer in a deep learning neural network.

Once implemented, batch normalization has the effect of dramatically accelerating the training process of a neural network, and in some cases improves the performance of the model via a modest regularization effect.

Batch normalization can be used after convolutional and fully connected layers. It has the effect of changing the distribution of the output of the layer, specifically by standardizing the outputs. This has the effect of stabilizing and accelerating the learning process.

We can update the model definition to use batch normalization after the activation function for the convolutional and dense layers of our baseline model. The updated version of define_model() function with batch normalization is listed below.

    # define cnn model
    def define_model():
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
        model.add(BatchNormalization())
        model.add(Dense(10, activation='softmax'))
        # compile model
        opt = SGD(lr=0.01, momentum=0.9)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

Running the example again reports model performance for each fold of the cross-validation process.

We can see perhaps a small drop in model performance as compared to the baseline across the cross-validation folds.


![AccFigure3](https://github.com/AbdullahTaher93/Practica1-IC/blob/master/images/AccFigure3.jpg)

A plot of the learning curves is created, in this case showing that the speed of learning (improvement over epochs) does not appear to be different from the baseline model.

The plots suggest that batch normalization, at least as implemented in this case, does not offer any benefit.

![AccFigure4](https://github.com/AbdullahTaher93/Practica1-IC/blob/master/images/AccFigure4.png)


The Python File you can find it [here](https://github.com/AbdullahTaher93/Practica1-IC/blob/master/ICPRACTICA1/batch.py)


#### Increase in Model Depth

There are Two common ways, changing the capacity of the feature extraction part of the model or changing the capacity or function of the classifier part of the model. Perhaps the point of biggest influence is a change to the feature extractor.

We can increase the depth of the feature extractor part of the model, following a VGG-like pattern of adding more convolutional and pooling layers with the same sized filter, while increasing the number of filters. In this case, we will add a double convolutional layer with 64 filters each, followed by another max pooling layer.

    # define cnn model
    def define_model():
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(10, activation='softmax'))
        # compile model
        opt = SGD(lr=0.01, momentum=0.9)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

Running the example reports model performance for each fold of the cross-validation process.

The per-fold scores may suggest some improvement over the baseline.

![AccFigure5](https://github.com/AbdullahTaher93/Practica1-IC/blob/master/images/AccFigure5.jpg)


![AccFigure6](https://github.com/AbdullahTaher93/Practica1-IC/blob/master/images/AccFigure6.png)

The Python File you can find it [here](https://github.com/AbdullahTaher93/Practica1-IC/blob/master/ICPRACTICA1/depthIncrease.py)

_
## Bibliography
_____________________________________________________________________

1- https://www.digitalocean.com/community/tutorials/how-to-build-a-neural-network-to-recognize-handwritten-digits-with-tensorflow


2- https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/

3- https://towardsdatascience.com/the-mostly-complete-chart-of-neural-networks-explained-3fb6f2367464

4- https://github.com/tensorflow/tensorflow




