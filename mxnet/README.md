#### MXNet on GPU
To install [MXNet](https://mxnet.apache.org/) on NVDIA GPU platform, a few prerequisites have to be satisfied:
- One should verify that NVIDIA driver for the installed graphics hardware is also installed (run `dpkg -l | grep nvidia-driver`)
- A library named CUDA that enables massively parallel computing, must be installed (this can be verified by running `ldconfig -p | grep cuda`)
- Accordingly, the CUDA binaries and shared libraries must be on the host PATH and LD_LIBRARY_PATH respectively
- NVIDIA Deep Neural Network library (called CuDNN) that runs on top of CUDA, must be installed (this can be verified on Ubuntu, by running `dpkg -l | grep cudnn`)

Many cloud-hosted GPU platforms for running deep learning workloads, come bundled with the above prerequisite packages. Otherwise, these libraries have to be installed manually. \
For example, even a quick check using `nvidia-smi` command may indicate pre-inst
alled drivers / libraries:

![install](https://github.com/techyugadi/dnn_gpu/blob/master/img/dnn21.png)

##### MXNet Installation
Once the prerequisites have been addressed, we can install MXNet Python package using `pip` or `pip3`. 

There is a [getting started webpage](https://mxnet.apache.org/get_started?) that helps us choose the correct Python package to install.

![install](https://github.com/techyugadi/dnn_gpu/blob/master/img/dnn15.png)

Accordingly, we can run the following command to install the latest version of MXNet: \
`pip3 install mxnet-cu101`
