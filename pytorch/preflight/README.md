#### PyTorch on GPU : Pre-flight Check
This program prints the number of GPUs available for the PyTorch runtime to utilize.

If it detects that GPUs *are* available, it prints the GPU device name and the number of GPUs.

##### Running the program

Simply run the program `torch_dryrun.py` : \
`python3 torch_dryrun.py`

##### Expected Output

The program prints the version of PyTorch being used, the GPU device id (`cuda:0`) and number of GPUs available (2).
