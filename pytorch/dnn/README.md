#### PyTorch on GPU : Image Recognition
The [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset is used to train a neural network to recognize and classify images into 10 different classes (e.g., cat, dog, airplane, ship, etc.).

This program does the following:
- loads the CIFAR-10 dataset and splits it into training and validation sets
- creates a Convolutional Neural Network (CNN) using PyTorch (`torch.nn` package), and sets it to run on GPU device
- trains the network over 10 epochs
- performs a quick sanity test on the trained model, with only four test images
- tests the accuracy of the network on the validation set

##### Running the program

To run the program, execute the following command:

`python3 cifar_torch.py`

##### Expected Output

The output messages indicate:

1. GPU device (`cuda:0`) is being utilized in training the network 

![gpus](https://github.com/techyugadi/dnn_gpu/blob/master/img/dnn23.png)

2. Training the network over 20 epoch, and resulting in increasing accuracy over the previous one (decreasing loss or error)

![epochs](https://github.com/techyugadi/dnn_gpu/blob/master/img/dnn24.png)

3. While in training phase, the `nvtop` output indicates GPU utilization

![nvtop](https://github.com/techyugadi/dnn_gpu/blob/master/img/dnn25.png)

4. A quick test on the trained model with four test images, and comparison of the 'ground truth' with predicted image labels.

![accuracy](https://github.com/techyugadi/dnn_gpu/blob/master/img/dnn26.png)

5. Completion of the training phase, and accuracy metrics over the validation set.

![accuracy](https://github.com/techyugadi/dnn_gpu/blob/master/img/dnn27.png)
