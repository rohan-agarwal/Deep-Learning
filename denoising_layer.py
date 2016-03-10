import theano
import theano.tensor as T

import os
import numpy
import timeit

from load_data import *
from utils import *


class denoising_layer(object):

    def __init__(
        self,
        numpy_rng,
        corrupted_input,
        input,
        n_visible=800*600,
        n_hidden=100,
        W=None,
        bhid=None,
        bvis=None
    ):

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if not W:
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:
            bvis = theano.shared(
                value=numpy.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        if not bhid:
            bhid = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )

        self.W = W
        self.b = bhid
        self.b_prime = bvis
        self.W_prime = self.W.T

        if input is None:
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        if corrupted_input is None:
            self.c = T.dmatrix(name='input')
        else:
            self.c = corrupted_input

        self.params = [self.W, self.b, self.b_prime]

        self.z = T.nnet.sigmoid(0)

    def get_hidden_values(self, input):
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self, learning_rate):
        tilde_x = self.c
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        self.z = z
        # cross entropy
        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        cost = T.mean(L)
        gparams = T.grad(cost, self.params)
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        return (cost, updates)


def test_dA(learning_rate=0.1, training_epochs=5,
            batch_size=1, output_folder='dA_plots'):

    datasets = load_data()
    train_set_c, train_set_x = datasets[0]
    n_train_batches = train_set_c.get_value(borrow=True).shape[0] / batch_size

    index = T.lscalar()
    x = T.matrix('x')
    c = T.matrix('c')

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    rng = numpy.random.RandomState(123)

    da = denoising_layer(
        numpy_rng=rng,
        corrupted_input=c,
        input=x,
        n_visible=800 * 600,
        n_hidden=200
    )

    cost, updates = da.get_cost_updates(learning_rate)

    train_da = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            c: train_set_c[index * batch_size: (index + 1) * batch_size],
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )

    start_time = timeit.default_timer()

    for epoch in xrange(training_epochs):
        c = []
        for batch_index in xrange(n_train_batches):
            j = train_da(batch_index)
            c.append(j)

        print 'Training epoch %d, cost ' % epoch, numpy.mean(c)

    end_time = timeit.default_timer()

    training_time = (end_time - start_time)

    print "training time: " + str(training_time)

    image = Image.fromarray(tile_raster_images(
        X=da.W.get_value(borrow=True).T,
        img_shape=(600, 800), tile_shape=(10, 10),
        tile_spacing=(1, 1)))
    image.save('filters')

    os.chdir('../')

if __name__ == '__main__':
    test_dA()
