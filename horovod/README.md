#### Using Horovod
[Horovod](https://github.com/horovod/horovod) is a distributed training framework for Deep Neural Networks. It helps us run programs written in various deep learning frameworks like TensorFlow, Keras, PyTorch, MXNet, etc., in a **multi-server multi-GPU** environment:
- with minimal code changes
- with appreciable performance gains

In this repository, we have an example of running a PyTorch program (image recognition using MNIST dataset), using Horovod.

To install Horovod, we need to install one or more of the underlying Deep Learning Frameworks like TensorFlow, PyTorch, etc. Then we can write our programs in one of these Deep Learning Frameworks, make a few changes, and submit it to Horovod. So, the prerequisites for installing those frameworks (like TensorFlow) will apply. \
`pip install tensorflow-gpu, mxnet-cu101, keras, pytorch`

That apart, the NCCL communications library is highly recommended for better performance with Horovod. 

If we plan to use MPI libraries which Horovod can leverage, then MPI installation will be necessary. Otherwise, we may use the `Gloo` communications library that is bundled with Horovod.

Also, most programs using Horovod may need the Python `future` package which can be installed as follows: \
`pip install future`

##### Horovod Installation
Once the prerequisites have been addressed, we can install Horovod Python package using `pip`.

To install Horovod, run: \
`HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_GPU_BROADCAST=NCCL HOROVOD_WITH_TORCH=1 pip install --no-cache-dir horovod`
