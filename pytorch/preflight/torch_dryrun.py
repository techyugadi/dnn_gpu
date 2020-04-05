import torch
print("PyTorch version: ", torch.__version__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

if torch.cuda.is_available():
  print("Number of GPUs: ", torch.cuda.device_count())
