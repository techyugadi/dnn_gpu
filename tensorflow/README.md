#### Tensorflow on GPU
This repository contains code to jumpstart Deep Neural Networks on a GPU Platform. The GPU platform chosen in this case is NVIDIA Tesla Dual T4. 

When a DNN is trained on a GPU platform, the training time may be shorter by orders of magnitude. But the right programmatic interfaces offered by the deep learning library must be invoked to ensure optimal utilization of the GPU resources.

##### Deep Learning Frameworks
This repository contains a few sample programs for each of the following four Deep Learning frameworks:
- Tensorflow
- MXNet
- PyTorch
- Horovod

##### Sample Applications
Two types of sample programs are available in this repository:
- Quick *pre-flight checks* to ensure that the Deep Learning framework is *actually* recognizing the GPU instances and utilizing them for computational steps that *can* exploit one or more GPU instances.
- Actual *Deep Learning examples*, like Image Classification, etc., programmed using GPU-enabled features of each library. 

##### Pre-requisites
The above sample programs were tested on an *Ubuntu 18.04 Desktop* environment running on NVIDIA hardware. Although a GUI is not essential for training a DNN, it may help in a development phase, for working with Python notebooks or other IDEs, displaying performance stats (GPU utilization) graphically, etc.
- To verify the availability of NVIDIA GPU instances, run the command: `nvidia-smi`.

Docker (Community Edition) was installed to run one or two Deep Learning frameworks in a container.

![Checking Pre-requisites](https://github.com/techyugadi/dnn_gpu/blob/master/img/dnn01.png) \
**Check Prerequisites**

We also installed a tool called `nvtop` that displays GPU utilization on NVIDIA platform, graphically. A summary of steps to build this tool from [source](https://github.com/Syllo/nvtop) is as follows:  

`apt install cmake libncurses5-dev libncursesw5-dev git` \
`git clone https://github.com/Syllo/nvtop.git`  \
`cd nvtop; mkdir build; cd build; cmake`   
`make`  \
`make install`  \
Note: The above set of commands must be run as `root`.

Now if we run `nvtop` command, the following output is displayed:

![nvtop](https://github.com/techyugadi/dnn_gpu/blob/master/img/dnn02.png) \
**nvtop**

To run sample programs directly on the host OS (instead of containers), at a minimum, Python must be installed. Anaconda and support for virtual environments are nice-to-have, but not essential.
