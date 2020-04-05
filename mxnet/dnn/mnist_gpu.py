import mxnet as mx
from mxnet import nd, gluon, autograd
net = gluon.nn.Sequential(prefix='cnn_')
with net.name_scope():
    net.add(gluon.nn.Conv2D(channels=20, kernel_size=3, activation='relu'))
    net.add(gluon.nn.MaxPool2D(pool_size=(2,2), strides=(2,2)))
    net.add(gluon.nn.Conv2D(channels=50, kernel_size=5, activation='relu'))
    net.add(gluon.nn.MaxPool2D(pool_size=(2,2), strides=(2,2)))
    net.add(gluon.nn.Flatten())
    net.add(gluon.nn.Dense(128, activation="relu"))
    net.add(gluon.nn.Dense(10))

loss = gluon.loss.SoftmaxCrossEntropyLoss()

GPU_COUNT = mx.context.num_gpus()
ctx = [mx.gpu(i) for i in range(GPU_COUNT)]
net.collect_params().initialize(ctx=ctx)

from mxnet.test_utils import get_mnist
mnist = get_mnist()
batch = mnist['train_data'][0:GPU_COUNT*2, :]
data = gluon.utils.split_and_load(batch, ctx)

weight = net.collect_params()['cnn_conv0_weight']

def forward_backward(net, data, label):
    with autograd.record():
        losses = [loss(net(X), Y) for X, Y in zip(data, label)]
    for l in losses:
        l.backward()

label = gluon.utils.split_and_load(mnist['train_label'][0:4], ctx)
forward_backward(net, data, label)
for c in ctx:
    print('=== grad of channel 0 of the first conv2d on {} ==={}'.format(
        c, weight.grad(ctx=c)[0]))
        
from mxnet.io import NDArrayIter
from time import time

def train_batch(batch, ctx, net, trainer):
    # split the data batch and load them on GPUs
    data = gluon.utils.split_and_load(batch.data[0], ctx)
    label = gluon.utils.split_and_load(batch.label[0], ctx)
    # compute gradient
    forward_backward(net, data, label)
    # update parameters
    trainer.step(batch.data[0].shape[0])

def valid_batch(batch, ctx, net):
    data = batch.data[0].as_in_context(ctx[0])
    pred = nd.argmax(net(data), axis=1)
    return nd.sum(pred == batch.label[0].as_in_context(ctx[0])).asscalar()

def run(num_gpus, batch_size, lr):
    # the list of GPUs will be used
    ctx = [mx.gpu(i) for i in range(num_gpus)]
    print('Running on {}'.format(ctx))

    # data iterator
    mnist = get_mnist()
    train_data = NDArrayIter(mnist["train_data"], mnist["train_label"], batch_size)
    valid_data = NDArrayIter(mnist["test_data"], mnist["test_label"], batch_size)
    print('Batch size is {}'.format(batch_size))

    net.collect_params().initialize(force_reinit=True, ctx=ctx)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
    for epoch in range(10):
        # train
        start = time()
        train_data.reset()
        for batch in train_data:
            train_batch(batch, ctx, net, trainer)
        nd.waitall()  # wait until all computations are finished to benchmark the time
        print('Epoch %d, training time = %.1f sec'%(epoch, time()-start))

        # validating
        valid_data.reset()
        correct, num = 0.0, 0.0
        for batch in valid_data:
            correct += valid_batch(batch, ctx, net)
            num += batch.data[0].shape[0]
        print('         validation accuracy = %.4f'%(correct/num))

run(GPU_COUNT, 64*GPU_COUNT, .3)
