import mxnet as mx

print("mxnet version: ", mx.__version__)
print("\n")

print("Number of GPUs: ", mx.context.num_gpus())
print("\n")

a = mx.nd.ones((2,3), mx.gpu())
b = a * 2 + 1

print(b.asnumpy())
