import numpy as np
from fast_layers import *


def binarize_forward(x):
    out = np.ones_like(x, dtype=np.float32)
    out[x < 0] = -1.
    alpha = np.mean(np.abs(x))
    return (out, alpha), (x, alpha)


def binarize_backward(dout, cache):
    x, alpha = cache
    dx = x.copy()
    dx[x < -1.] = 0
    dx[x > 1.] = 0
    dout *= 1. / x.shape[0] + dx * alpha
    return dout


def bin_fc_forward(x, w):
    x = binarize_forward(x)
    out = np.dot(x[0][0], w[0][0]) * (x[0][1] * w[0][1])
    return out, (x[1], w[1])


def bin_fc_backward(dout, cache):
    xb, wb = cache
    dx = np.dot(dout, wb[0].T)
    dw = np.dot(xb[0].T, dout)
    dx = binarize_backward(dx, xb)
    dw = binarize_backward(dw, wb)
    return dx, dw


def bin_conv_forward(x, w, params):
    x = binarize_forward(x)
    out, cache = conv_forward_fast(x[0][0], w[0][0], np.zeros(w[0][0].shape[0]), params)
    out *= x[0][1] * w[0][1]
    return out, (x[1], w[1], cache)


def bin_conv_backward(dout, cache):
    x, w, conv_cache = cache
    dx, dw, _ = conv_backward_fast(dout, conv_cache)
    dx = binarize_backward(dx, x)
    dw = binarize_backward(dw, w)
    return dx, dw


def conv_params_binary(x):
    c = x.shape[0]
    bx = np.zeros_like(x)
    alpha = np.zeros(c)
    for i in xrange(c):
        (bx[i], alpha[i]), cache = binarize_forward(bx[i])
    return bx, alpha


def conv_params_binary_backward(dout, cache):

    pass


def fc_params_binary(x):
    return binarize_forward(x)
