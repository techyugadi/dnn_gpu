#### MXNet on GPU : Pre-flight Check
This program prints the number of GPUs available for the MXNet runtime to utilize.

If it detects that GPUs *are* available, it performs a simple matrix operation, invoking a function call that can succeed only on a GPU platform. As long as thois operation returns the expected matrix result, without emitting any error message, we can tell that MXNet is detecting and using the GPUs correctly.

##### Running the program

Simply run the program `mxgpu.py` : \
`python3 mxgpu.py`

##### Expected Output

The program prints the version of MXNet being used, the number of GPUs available, and the result of the matrix operation (obtained by invoking a function that runs only on GPUs).

![preflight](https://github.com/techyugadi/dnn_gpu/blob/master/img/dnn16.png)
