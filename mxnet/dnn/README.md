#### MXNet on GPU : Hand-written Digit Recognition
The [MNIST](http://yann.lecun.com/exdb/mnist/) dataset is used to train a neural network to recognize hand-written digits.

This program does the following:
- creates a Convolutional Neural Network (CNN) using MNIST (`mxnet gluon`)
- loads the MNIST dataset and splits it into training and validation sets
- trains the network over 10 epochs using `gluon` APIs that support GPUs
- tests the accuracy of the network on the validation set

##### Running the program

To run the program, execute the following command:

`python3 mnist_gpu.py`

##### Expected Output

The output messages indicate:

1. Both GPU instances are being utilized in training the network 

![gpus](https://github.com/techyugadi/dnn_gpu/blob/master/img/dnn17.png)

3. Training the network over 10 epoch, (batch size of 128 per epoch) and resulting in increasing accuracy over the previous one

![epochs](https://github.com/techyugadi/dnn_gpu/blob/master/img/dnn18.png)

4. While in training phase, the `nvtop` output indicates GPU utilization

![nvtop](https://github.com/techyugadi/dnn_gpu/blob/master/img/dnn19.png)

5. Completion of the training phase, and accuracy metrics over the validation set.

![accuracy](https://github.com/techyugadi/dnn_gpu/blob/master/img/dnn20.png)
