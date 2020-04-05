#### Tensorflow on GPU : Pre-flight Check
This program prints the number of GPUs available for the Tensorflow runtime to utilize.

If it detects that GPUs *are* available, it performs a simple matrix multiplication task, making use of all GPUs available.

##### Running the program

Copy the program `gpurun.py` to a location which can be volume mounted into the Tensorflow docker. Then run a command (based on *your* intended volume mount) like this: \
`docker run --gpus all -it -v ~/workspace:/workspace -shm-size=1g --ulimit memlock=-1 nvcr.io/nvidia/tensorflow:20.01-tf2-py3

You will be dropped into a shell within the container. From the directory where the program was volume-mounted, run: \
`python gpurun.py`

##### Expected Output

1. Firstly, it shows Tensorflow version and the number of GPUs available(*2*)

2. It indicates that the matrix multiplication function (which is GPU-enabed) is actually executed over multiple GPUs.

3. While the computation is in progress, `nvtop` will indicate spikes in usage of both GPUs. (Ideally, if we had observed `nvtop` just before starting the program, it would have shown a flat line.)

4. Finally the result of the matrix operation is printed.
To install Tensorflow on NVDIA GPU platform, a few prerequisites have to be satisfied:
- One should also verify that NVIDIA driver for the installed graphics hardware is also installed (run `dpkg -l | grep nvidia-driver`)
- A library named CUDA that enables massively parallel computing, must be installed (this can be verified by running `ldconfig -p | grep cuda`)
- Accordingly, the CUDA binaries and shared libraries must be on the host PATH and LD_LIBRARY_PATH respectively
- NVIDIA Deep Neural Network library that runs on top of CUDA, must be installed (this can be verified on Ubuntu, by running `dpkg -l | grep cudnn`)
- A library named NCCL to support communicatio between multiple GPUs using shared memory
Tensorflow-gpu runs on top of the above NVIDIA libraries.

##### Tensorflow Docker 
Alternatively, a Tensorflow docker image built by NVIDIA, taking care of all prerequisites, is available.

The docker image is: `nvcr.io/nvidia/tensorflow:20.01-tf2-py3`. \
This docker image is based on Tensorflow v2, that supports Python3.

The command to run this docker image is:

`docker run --gpus all -it -v ~/workspace:/workspace -shm-size=1g --ulimit memlock=-1 nvcr.io/nvidia/tensorflow:20.01-tf2-py3`

Note: This is the correct command for docker v19.03 and above.

This command will drop us into a shell inside the container. We should volume mount our Deep Learning programs and data into a suitable directory inside the container (e.g., `workspace`) and then run commands to start off machine learning workloads on Tensorflow.
