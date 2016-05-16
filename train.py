import os
import matplotlib.pyplot as plt
from collections import OrderedDict
import numpy as np
import theano
import theano.tensor as T
from lasagne import layers, objectives, updates
from helpers import iterate_minibatches
from tabulate import tabulate
from time import time
import model

def train_and_validate(train, valid, test, 
                       model_name='ciresan_4',
                       hp=None):
    if hp is None:
        hp = {}
    c, w, h = train.X.shape[1:]
    nb_outputs = len(set(train.y))

    valid_hp = [
        'learning_rate',
        'learning_rate_decay',
        'momentum',
        'batchsize',
        'nb_epochs',
        'augment',
        'augment_params',
    ]
    assert set(hp.keys()).issubset(set(valid_hp))
  
    learning_rate = theano.shared(np.array(hp.get('learning_rate', 0.01)).astype(np.float32))
    momentum = hp.get('momentum', 0)
    batchsize = hp.get('batchsize', 128)

    X = T.tensor4()
    y = T.ivector()
    
    CLS = getattr(model, model_name)
    net = CLS(
        w=w, h=w, c=c, 
        nb_outputs=nb_outputs)

    print('Compiling the net...')

    y_pred = layers.get_output(net, X)
    y_pred_detm = layers.get_output(net, X, deterministic=True)

    loss = objectives.categorical_crossentropy(y_pred, y).mean()

    loss_detm = objectives.categorical_crossentropy(y_pred, y).mean()
    y_acc_detm = T.eq(y_pred_detm.argmax(axis=1), y).mean()

    loss_fn = theano.function([X, y], loss_detm)
    acc_fn = theano.function([X, y], y_acc_detm)

    params = layers.get_all_params(net, trainable=True)
    grad_updates = updates.momentum(loss, params, learning_rate=learning_rate, momentum=momentum)
    train_fn = theano.function([X, y], loss, updates=grad_updates)
    
    nb_epochs = hp.get('nb_epochs', 1000)
    data_augment = hp.get('augment', False)
    augment_params = hp.get('augment_params', {})
    
    lr_decay = hp.get('learning_rate_decay', 1)
    print('Training...')
    history = []
    for epoch in range(1, nb_epochs + 1):
        data_aug_time = []
        train_time = []

        # Data augmentation
        t = time()
        if data_augment:
            train_X_full = augment(train.X, **augment_params)
        else:
            train_X_full = train.X
        data_aug_time.append(time() - t)

        train_y_full = train.y
        for train_X, train_y in iterate_minibatches(train_X_full, train_y_full, batchsize):
            # Train one mini=batch
            t = time()
            train_fn(train_X, train_y)
            train_time.append(time() - t)
        stats = OrderedDict()
        stats['train_loss'] = float(loss_fn(train.X, train.y))
        stats['valid_loss'] = float(loss_fn(valid.X, valid.y))
        stats['test_loss'] = float(acc_fn(test.X, test.y))
        stats['train_acc'] = float(acc_fn(train.X, train.y))
        stats['valid_acc'] = float(acc_fn(valid.X, valid.y))
        stats['test_acc'] = float(acc_fn(test.X, test.y))
        stats['data_aug_time'] = float(np.sum(data_aug_time))
        stats['train_time'] = float(np.sum(train_time))
        stats['epoch'] = epoch

        history.append(stats)
        print(tabulate([stats], headers="keys"))

        lr = learning_rate.get_value()
        lr = lr * lr_decay
        learning_rate.set_value(np.array(lr).astype(np.float32))
    return history
