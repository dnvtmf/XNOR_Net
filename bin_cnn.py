from bin_layers import *
from layer_utils import *


class CNN(object):
    """
    A three-layer convolutional network with the following architecture:

    any architecture with layers in {conv, relu , pool, affine, softmax/svm}

    The network operates on mini-batches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, arch, input_dim=(3, 32, 32), weight_scale=1e-3,
                 classifier='softmax', reg=0.0, dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - arch: the architecture of the convolution network
            a list of layers
            - norm: batch normalization
            - conv: the layer of convolution
            - pool: the layer of max pool
                 default: pool_height=2, pool_width=2, stride=2
            - fc: the layer of full connection
                - hidden_dim: the size of affine nodes
            - flat: convert the 3-D tensor to a 1-D tensor
        - input_dim: Tuple (C, H, W) giving size of input data
        - classifier: use softmax(default) or svm classifier
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy data type to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.arch = []
        self.binary = False
        self.bin_params = {}
        self.num_layers = len(arch)
        if classifier == 'svm':
            self.classifier = svm_loss
        elif classifier == 'softmax':
            self.classifier = softmax_loss
        else:
            assert 0, 'Unknown classifier'
        dims = input_dim
        print 'image'
        for i in xrange(self.num_layers):
            self.arch.append(arch[i].copy())
            layer = self.arch[i]
            if layer['name'] == 'conv':
                assert len(dims) == 3, 'conv layer should before fc layer'
                C = layer['num_filters']
                HF, WF = layer['filter_size']
                layer.setdefault('conv_param', {})
                HP = layer['conv_param'].setdefault('pad', (HF - 1) / 2)
                WP = layer['conv_param'].setdefault('pad', (WF - 1) / 2)
                S = layer['conv_param'].setdefault('stride', 1)
                self.params['W%d' % i] = weight_scale * np.random.randn(C, dims[0], HF, WF)
                self.bin_params['W%d' % i] = np.zeros((C, dims[0], HF, WF))
                dims = (C, (dims[1] + 2 * HP - HF) / S + 1, (dims[2] + 2 * WP - WF) / S + 1)
                print '-->conv(%d x (%d, %d))' % (C, HF, WF)
            elif layer['name'] == 'fc':
                D, C = dims[0], layer['hidden_dim']
                self.params['W%d' % i] = weight_scale * np.random.randn(D, C)
                self.bin_params['W%d' % i] = np.zeros((D, C))
                dims = (C,)
                print '-->fc(%d)' % C
            elif layer['name'] == 'norm':
                C = dims[0]
                self.params['gamma%d' % i] = np.ones(C)
                self.params['beta%d' % i] = np.zeros(C)
                layer['bn_param'] = {'mode': 'train'}
                layer['spatial'] = True if len(dims) != 1 else False
                print '-->norm'
            elif layer['name'] == 'pool':
                assert len(dims) == 3, 'can not put pool layer after affine layer'
                layer.setdefault('pool_param', {})
                PH = layer['pool_param'].setdefault('pool_height', 2)
                PW = layer['pool_param'].setdefault('pool_width', 2)
                S = layer['pool_param'].setdefault('stride', 2)
                dims = (dims[0], (dims[1] - PH) / S + 1, (dims[2] - PW) / S + 1)
            elif layer['name'] == 'flat':
                dims = (np.prod(dims[1:]),)
                print '-->flat'
            else:
                assert 0, 'Unknown layer'

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def loss(self, images, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        """
        out = images
        cache = {}
        params = self.params
        bin_params = self.bin_params
        mode = 'test' if y is None else 'train'
        # binary the weights if it is not binary
        if not self.binary:
            self.binary = True
            for l in xrange(self.num_layers):
                if self.arch[l]['name'] == 'conv':
                    bin_params['W%d' % l] = conv_params_binary(params['W%d' % l])
                elif self.arch[l]['name'] == 'fc':
                    bin_params['W%d' % l] = fc_params_binary(params['W%d' % l])
        for l in xrange(self.num_layers):
            layer = self.arch[l]
            if layer['name'] == 'conv':
                out, cache['conv%d' % l] = bin_conv_forward(out, bin_params['W%d' % l], layer['conv_param'])
            elif layer['name'] == 'fc':
                out, cache['fc%d' % l] = bin_fc_forward(out, bin_params['W%d' % l])
            elif layer['name'] == 'pool':
                out, cache['pool%d' % l] = max_pool_forward_fast(out, layer['pool_param'])
            elif layer['name'] == 'norm':
                layer['bn_param']['mode'] = mode
                if layer['spatial']:
                    out, cache['norm%d' % l] = spatial_batchnorm_forward(
                        out, params['gamma%d' % l], params['beta%d' % l], layer['bn_param'])
                else:
                    out, cache['norm%d' % l] = batchnorm_forward(
                        out, params['gamma%d' % l], params['beta%d' % l], layer['bn_param'])
            elif layer['name'] == 'flat':
                out, cache['flat%d' % l] = flat_forward(out)
            else:
                assert 0, 'ghost'

        scores = out
        if y is None:
            return scores
        self.binary = False
        grads = {}
        loss, dout = self.classifier(scores, y)
        for l in xrange(self.num_layers - 1, -1, -1):
            layer = self.arch[l]
            if layer['name'] == 'conv':
                dout, grads['W%d' % l], grads['b%d' % l] = bin_conv_backward(
                    dout, cache['conv%d' % l])
            elif layer['name'] == 'fc':
                dout, grads['W%d' % l] = bin_fc_backward(dout, cache['fc%d' % l])
            elif layer['name'] == 'pool':
                dout = max_pool_backward_fast(dout, cache['pool%d' % l])
            elif layer['name'] == 'norm':
                if layer['spatial']:
                    dout, grads['gamma%d' % l], grads['beta%d' % l] = \
                        spatial_batchnorm_backward(dout, cache['norm%d' % l])
                else:
                    dout, grads['gamma%d' % l], grads['beta%d' % l] = \
                        batchnorm_backward(dout, cache['norm%d' % l])
            elif layer['name'] == 'flat':
                dout = flat_backward(dout, cache['flat%d' % l])
            else:
                assert 0, 'ghost'
        return loss, grads
