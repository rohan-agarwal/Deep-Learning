"""

Kernel Operations for ODCNN

"""


import numpy
import theano
from theano import tensor as T
from theano.tensor.nnet import conv
from PIL import Image
import glob


class KernelLayer(object):

    def __init__(self, rng, input, filter_shape, image_shape):

        assert image_shape[1] == filter_shape[1]
        self.input = input

        fan_in = numpy.prod(filter_shape[1:])
        W_bound = numpy.sqrt(6. / fan_in)
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

rng = numpy.random.RandomState(23455)
img = Image.open("C:\Users\Rohan\Documents\GitHub\Deep-Learning\Data\clean\one.png")

layer0 = KernelLayer(
        rng,
        input=img,
        image_shape=(batch_size, 1, 28, 28),
        filter_shape=(nkerns[0], 1, 5, 5),
    )


def load_data():

    def pic2np(gpath):
        image_list = []
        for f in glob.glob(gpath):
            im = Image.open(f)
            im = numpy.asarray(im, dtype='float32') / 256.
            im = im.ravel()
            image_list.append(im)
        return image_list

    data_x = pic2np(
        'C:\Users\Rohan\Documents\GitHub\Deep-Learning\Data\\blur\*.jpg')
    data_y = pic2np(
        'C:\Users\Rohan\Documents\GitHub\Deep-Learning\Data\clean\*.png')

    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX))

    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX))

    return shared_x, shared_y
