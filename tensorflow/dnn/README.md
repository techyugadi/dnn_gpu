#### Tensorflow on GPU : Image Recognition
The [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset is used to train a neural network to recognize and classify images into 10 different classes (e.g., cat, dog, airplane, ship, etc.).

This program does the following:
- loads the CIFAR-10 dataset and splits it into training and validation sets
- creates a Convolutional Neural Network (CNN) using Tensorflow (`tensorflow.keras`), several layers deep
- wraps the CNN in 'MirroredStrategy' scope - one of the strategies that Tensorflow offers for training on GPUs
- trains the network over 10 epochs
- tests the accuracy of the network on the validation set

##### Running the program

Copy the program `gpurun.py` to a location which can be volume mounted into the Tensorflow docker. Then run a command (depending on *your* intended volume mount) like this: \
`docker run --gpus all -it -v ~/workspace:/workspace -shm-size=1g --ulimit memlock=-1 nvcr.io/nvidia/tensorflow:20.01-tf2-py3`

You will be dropped into a shell within the container. From the directory where the program was volume-mounted, run: \
`python3 cifar_gpu.py` 

![gpurun](https://github.com/techyugadi/dnn_gpu/blob/master/img/dnn09.png)

##### Expected Output

The output messages indicate:

1. Construction and cascading of the layers of the CNN

![cnn](https://github.com/techyugadi/dnn_gpu/blob/master/img/dnn11.png)

2. Both GPU instances are being utilized in training the network 

![gpus](https://github.com/techyugadi/dnn_gpu/blob/master/img/dnn10.png)

3. Training the network over 10 epoch, each epoch taking as input around 50,000 images, and resulting in increasing accuracy over the previous one

![epochs](https://github.com/techyugadi/dnn_gpu/blob/master/img/dnn13.png)

4. While in training phase, the `nvtop` output indicates GPU utilization

![nvtop](https://github.com/techyugadi/dnn_gpu/blob/master/img/dnn12.png)

5. Completion of the training phase, and accuracy metrics over the validation set.

![accuracy](https://github.com/techyugadi/dnn_gpu/blob/master/img/dnn14.png)

Note: the accuracy can possibly be improved by further fine-tuning the layers of the CNN, and/or training over many more epochs. Here we have restricted the test to a smaller duration.
