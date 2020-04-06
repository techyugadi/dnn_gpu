#### PyTorch on GPU
To install [PyTorch](https://pytorch.org/) on NVDIA GPU platform, a few prerequisites have to be satisfied:
- One should verify that NVIDIA driver for the installed graphics hardware is also installed 
- A toolkit named CUDA that enables massively parallel computing on GPUs, must be installed

Many cloud-hosted GPU platforms for running deep learning workloads, come bundled with the above prerequisite packages. Otherwise, these libraries have to be installed manually. \
For example, even a quick check using `nvidia-smi` command may indicate pre-installed drivers / libraries:

![install](https://github.com/techyugadi/dnn_gpu/blob/master/img/dnn21.png)

##### PyTorch Installation
Once the prerequisites have been addressed, we can install PyTorch Python package using `pip` or `pip3`. 

There is a [getting started webpage](https://pytorch.org/get-started/locally/) that helps us choose the correct Python package to install.

![install](https://github.com/techyugadi/dnn_gpu/blob/master/img/dnn22.png)

Accordingly, we can run the following command to install the latest version of PyTorch: \
`pip3 install torch torchvision`
