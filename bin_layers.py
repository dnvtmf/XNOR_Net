from fast_layers import *


def binarize_forward_2d(x):
    out = np.ones_like(x, dtype=np.float32)
    out[x < 0] = -1.
    alpha = np.mean(np.abs(x))
    return out, x, alpha


def binarize_backward_2d(dout, cache):
    bx, x, alpha = cache
    dx = x.copy()
    dx[x < -1.] = 0
    dx[x > 1.] = 0
    dout *= 1. / x.shape[0] + dx * alpha
    return dout


def bin_fc_forward(x, w):
    x = binarize_forward_2d(x)
    out = np.dot(x[0], w[0]) * (x[2] * w[2])
    return out, (x, w)


def bin_fc_backward(dout, cache):
    x, w = cache
    dx = np.dot(dout, w[0].T)
    dw = np.dot(x[0].T, dout)
    dx = binarize_backward_2d(dx, x)
    dw = binarize_backward_2d(dw, w)
    return dx, dw


def binarize_forward_3d(x, axis=0):
    out = np.ones_like(x, dtype=np.float32)
    out[x < 0] = -1.
    alpha = np.mean(np.abs(x), axis=axis)
    return (out, alpha), (x, alpha)


def binarize_backward_3d(dout, cache):
    x, alpha = cache
    dx = x.copy()
    dx[x < -1.] = 0
    dx[x > 1.] = 0
    dout *= 1. / x.shape[0] + dx * alpha
    return dout


def bin_conv_forward(x, w, params):
    x = binarize_forward_2d(x)
    out, cache = conv_forward_fast(x[0], w[0], np.zeros(w[0].shape[0]), params)
    out *= x[2] * w[2]
    return out, (x, w, cache)


def bin_conv_backward(dout, cache):
    x, w, conv_cache = cache
    dx, dw, _ = conv_backward_fast(dout, conv_cache)
    dx = binarize_backward_2d(dx, x)
    dw = binarize_backward_2d(dw, w)
    return dx, dw
