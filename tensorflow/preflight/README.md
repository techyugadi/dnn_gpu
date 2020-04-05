#### Tensorflow on GPU : Pre-flight Check
This program prints the number of GPUs available for the Tensorflow runtime to utilize.

If it detects that GPUs *are* available, it performs a simple matrix multiplication task, making use of all GPUs available.

##### Running the program

Copy the program `gpurun.py` to a location which can be volume mounted into the Tensorflow docker. Then run a command (depending on *your* intended volume mount) like this: \
`docker run --gpus all -it -v ~/workspace:/workspace -shm-size=1g --ulimit memlock=-1 nvcr.io/nvidia/tensorflow:20.01-tf2-py3

You will be dropped into a shell within the container. From the directory where the program was volume-mounted, run: \
`python gpurun.py`

![gpurun](https://github.com/techyugadi/dnn_gpu/blob/master/img/dnn03.png)

##### Expected Output

1. Firstly, it shows Tensorflow version and the number of GPUs available(*2*)

![tfver](https://github.com/techyugadi/dnn_gpu/blob/master/img/dnn04.png)

![numgpus](https://github.com/techyugadi/dnn_gpu/blob/master/img/dnn05.png)

2. It indicates that the matrix multiplication function (which is GPU-enabed) is actually executed over multiple GPUs.

![matmult](https://github.com/techyugadi/dnn_gpu/blob/master/img/dnn06.png)

3. While the computation is in progress, `nvtop` will indicate spikes in usage of both GPUs. (Ideally, if we had observed `nvtop` just before starting the program, it would have shown a flat line.)

![nvtop](https://github.com/techyugadi/dnn_gpu/blob/master/img/dnn07.png)

4. Finally the result of the matrix operation is printed.

![result](https://github.com/techyugadi/dnn_gpu/blob/master/img/dnn08.png)
