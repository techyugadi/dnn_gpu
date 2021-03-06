#### Tensorflow on GPU
To install [Tensorflow](https://www.tensorflow.org/) on NVDIA GPU platform, a few prerequisites have to be satisfied:
- One should verify that NVIDIA driver for the installed graphics hardware is also installed (run `dpkg -l | grep nvidia-driver`)
- A library named CUDA that enables massively parallel computing, must be installed (this can be verified by running `ldconfig -p | grep cuda`)
- Accordingly, the CUDA binaries and shared libraries must be on the host PATH and LD_LIBRARY_PATH respectively
- NVIDIA Deep Neural Network library (called CuDNN) that runs on top of CUDA, must be installed (this can be verified on Ubuntu, by running `dpkg -l | grep cudnn`)
- A library named NCCL to support communication between multiple GPUs using shared memory
Tensorflow-gpu runs on top of the above NVIDIA libraries.

Many cloud-hosted GPU platforms for running deep learning workloads, come bundle
d with the above prerequisite packages. Otherwise, these libraries have to be in
stalled manually. \
For example, even a quick check using `nvidia-smi` command may indicate pre-inst
alled drivers / libraries:

![install](https://github.com/techyugadi/dnn_gpu/blob/master/img/dnn21.png)

##### Tensorflow Docker 
Alternatively, a Tensorflow docker image built by NVIDIA, taking care of all prerequisites, is available.

The docker image is: `nvcr.io/nvidia/tensorflow:20.01-tf2-py3`. \
This docker image is based on Tensorflow v2, that supports Python3.

The command to run this docker image is:

`docker run --gpus all -it -v ~/workspace:/workspace -shm-size=1g --ulimit memlock=-1 nvcr.io/nvidia/tensorflow:20.01-tf2-py3`

Note: This is the correct command for docker v19.03 and above.

This command will drop us into a shell inside the container. We should volume mount our Deep Learning programs and data into a suitable directory inside the container (e.g., `workspace`) and then run commands to start off machine learning workloads on Tensorflow.
