from lasagnekit.datasets.mnist import MNIST
from lasagnekit.datasets.helpers import split

def load_mnist(training_subset=None ,valid_ratio=0.16667):
    c, w, h = 1, 28, 28
    def preprocess(data):
        data = data * 2 - 1
        return data.reshape((data.shape[0], c, w, h))

    train_full = MNIST(which='train')
    train_full.load()
    train_full.X = preprocess(train_full.X)

    train, valid = split(train_full, test_size=valid_ratio) # 10000 examples in validation set

    if training_subset is not None:
        nb = int(training_subset * len(train.X))
        print('training on a subset of training data of size : {}'.format(nb))
        train.X = train.X[0:nb]
        train.y = train.y[0:nb]

    test = MNIST(which='test')
    test.load()
    test.X = preprocess(test.X)
    return train, valid, test
