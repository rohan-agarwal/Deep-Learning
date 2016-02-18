"""

Junk file for Theano experimentation

"""

import numpy
from PIL import Image
import theano
from theano import tensor as T
from theano.tensor.nnet import conv

# rng = numpy.random.RandomState(23455)

# # instantiate 4D tensor for input
# input = T.tensor4(name='input')

# # variables for kernel size
# k_num = 38
# k_h = 1
# k_w = 121 

# # random initialization for layer 1 weights
# w_shp = (k_num, 3, k_h, k_w)
# w_bound = numpy.sqrt(k_num * k_h * k_w)
# W = theano.shared(numpy.asarray(
#     rng.uniform(
#         low=-1.0 / w_bound,
#         high=1.0 / w_bound,
#         size=w_shp),
#     dtype=input.dtype), name='W')

# b_shp = k_num
# b = theano.shared(numpy.zeros(b_shp), name='b')

# # build symbolic expression that computes the convolution of input with
# # filters in w
# conv_out = conv.conv2d(input, W)

# #   dimshuffle('x', 2, 'x', 0, 1)
# #   This will work on 3d tensors with no broadcastable
# #   dimensions. The first dimension will be broadcastable,
# #   then we will have the third dimension of the input tensor as
# #   the second of the resulting tensor, etc. If the tensor has
# #   shape (20, 30, 40), the resulting tensor will have dimensions
# #   (1, 40, 1, 20, 30). (AxBxC tensor is mapped to 1xCx1xAxB tensor)
# output = T.nnet.sigmoid(conv_out + b.dimshuffle('x', 0, 'x', 'x'))

# # create theano function to compute filtered images
# f = theano.function([input], output)

# # open random image of dimensions 639x516
# img = Image.open("C:\Users\Rohan\Documents\GitHub\Deep-Learning\Data\clean\one.png")
# # dimensions are (height, width, channel)
# img = numpy.asarray(img, dtype='float32') / 256.
# i_h = img.shape[0]
# i_w = img.shape[1]
# # put image in 4D tensor of shape (1, 3, height, width)
# img_ = img.transpose(2, 0, 1).reshape(1, 3, i_w, i_h)
# filtered_img = f(img_)

# print filtered_img.shape


class ConvLayer(object):

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):

        assert image_shape[1] == filter_shape[1]
        self.input = input

        fan_in = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))

        W_bound = numpy.sqrt(6. / (fan_in + fan_out))

        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        self.params = [self.W, self.b]

        self.input = input

x = T.matrix('x')
y = T.ivector('y')

print '... building the model'

layer0_input = x.reshape((batch_size, 1, 28, 28))


