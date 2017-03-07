from fast_layers import *
import numpy as np
from scipy.signal import convolve


def binarize_forward_2d(x):
    out = np.ones_like(x, dtype=np.float32)
    out[x < 0.] = -1.
    alpha = np.mean(np.abs(x))
    return out, x, alpha


def binarize_backward_2d(dout, cache):
    bx, x, alpha = cache
    dx = np.ones_like(x)
    dx[x < -1.0] = 0.
    dx[x > 1.0] = 0.
    dout *= 1. / np.prod(x.shape) + dx * alpha
    return dout


def bin_fc_forward(x, bw):
    bx = binarize_forward_2d(x)
    out = np.dot(bx[0], bw[0]) * (bx[2] * bw[2])
    return out, (bx, bw)


def bin_fc_backward(dout, cache):
    bx, bw = cache
    dx = np.dot(dout, bw[0].T)
    dw = np.dot(bx[0].T, dout)
    dx = binarize_backward_2d(dx, bx)
    dw = binarize_backward_2d(dw, bw)
    return dx, dw


def sign_forward(x):
    out = np.ones_like(x)
    out[x < 0.] = -1.
    return out, x


def sign_backward(dout, cache):
    x = cache
    dout[x < 1.0] = 0.
    dout[x > 1.0] = 0.
    return dout


def binarize_forward_filters(x):
    out, cache = sign_forward(x)
    alpha = np.mean(np.abs(x), axis=(1, 2, 3)).reshape((1, -1, 1, 1))
    return out, x, alpha, cache


def binarize_backward_filters(dout, da, cache):
    bx, x, alpha, sc = cache
    dalpha = np.sum(da, axis=(0, 2, 3)) * (1. / np.prod(x.shape[1:]))
    dx = sign_backward(dout, sc) + dalpha.reshape(-1, 1, 1, 1) * bx
    return dx


def binarize_forward_4d(x, w, h):
    out, cache = sign_forward(x)
    a = np.mean(np.abs(x), axis=(0, 1), keepdims=True)
    v = np.ones((1, 1, w, h)) * (1. / (w * h))
    k = convolve(a, v, mode='same')
    return out, x, k, cache, v


def binarize_backward_4d(dout, dk, cache):
    bx, x, k, sc, v = cache
    dx = sign_backward(dout, sc)
    dk = np.sum(dk, axis=(0, 1), keepdims=True)
    dx += np.rot90(convolve(np.rot90(dk, 2), v, mode='same'), 2) * bx
    return dx


def bin_conv_forward(x, bw, params):
    bx = binarize_forward_4d(x, bw[1].shape[2], bw[1].shape[3])
    out, cache = conv_forward_fast(bx[0], bw[0], np.zeros(bw[0].shape[0]), params)
    out1 = out * bw[2]
    out2 = out1 * bx[2]
    return out2, (bx, bw, cache, out, out1)


def bin_conv_backward(dout, cache):
    bx, bw, conv_cache, out, out1 = cache
    dk = dout * out1
    dout *= bx[2]
    da = dout * out
    dout *= bw[2]
    dx, dw, _ = conv_backward_fast(dout, conv_cache)
    dx = binarize_backward_4d(dx, dk, bx)
    dw = binarize_backward_filters(dw, da, bw)
    return dx, dw
