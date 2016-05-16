from lasagne import layers, init
from lasagne.nonlinearities import tanh, softmax
import theano.tensor as T

def tanh_lecun(x):
    A =  1.7159
    B = 0.6666
    return A * T.tanh(B * x)

def ciresan_4(w=32, h=32, c=1, nb_outputs=10):
    nonlin = tanh_lecun
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_hid = layers.DenseLayer(l_in, 2500, nonlinearity=nonlin, W=init.HeUniform(), name="hid1")
    l_hid = layers.DenseLayer(l_hid, 2000, nonlinearity=nonlin, W=init.HeUniform(), name="hid2")
    l_hid = layers.DenseLayer(l_hid, 1000, nonlinearity=nonlin, W=init.HeUniform(), name="hid3")
    l_hid = layers.DenseLayer(l_hid, 1000, nonlinearity=nonlin, W=init.HeUniform(), name="hid3")
    l_hid = layers.DenseLayer(l_hid, 500, nonlinearity=nonlin, W=init.HeUniform(), name="hid4")
    l_out = layers.DenseLayer(l_hid, 10, nonlinearity=softmax, W=init.HeUniform(), name="output")
    return l_out
