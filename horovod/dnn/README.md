#### Using Horovod

The purpose of including this program is to test whether our environment is corrctly configired to run a PyTorch program *with Horovod*.

This program recognizes hand-written digits using the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset.

##### Running the program

To run the program we will use the **horovodrun** util : \
`horovodrun -np 2 -H localhost:2 --gloo --verbose python pytorch_mnist.py`

##### Expected Output

The program trains a Convolutional Neural Network over several epochs and tests the accuracy of the model against a validation set. The accuracy metric is displayed.

![accuracy](https://github.com/techyugadi/dnn_gpu/blob/master/img/dnn28.png)

We can also verify using `nvtop` that both GPU instances in our environment are utilized in the training phase.

![gpus](https://github.com/techyugadi/dnn_gpu/blob/master/img/dnn29.png)
