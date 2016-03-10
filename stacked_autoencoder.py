import theano
import theano.tensor as T
theano.config.on_unused_input = "warn"

import numpy
import os
import sys

from denoising_layer import *


class stacked_autoencoder(object):

    def __init__(
        self,
        numpy_rng,
        n_ins=800 * 600,
        hidden_layers_sizes=[100, 100],
    ):

        self.dA_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        self.c = T.matrix('c')
        self.x = T.matrix('x')

        for i in xrange(self.n_layers):

            input_size = n_ins

            if i == 0:
                layer_input = self.c
            else:
                layer_input = self.dA_layers[-1].z

            clear_image = self.x

            dA_layer = denoising_layer(numpy_rng,
                                       corrupted_input=layer_input,
                                       input=clear_image,
                                       n_visible=input_size,
                                       n_hidden=hidden_layers_sizes[i])

            self.dA_layers.append(dA_layer)
            self.params.extend(dA_layer.params)

        self.final_output = self.dA_layers[-1].z

        self.finetune_cost = - \
            T.sum(self.x * T.log(self.final_output) + (1 - self.x)
                  * T.log(1 - self.final_output), axis=1)

    def pretraining_functions(self, train_set_c, train_set_x, batch_size):

        index = T.lscalar('index')
        learning_rate = T.scalar('lr')

        batch_begin = index * batch_size
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for dA in self.dA_layers:

            cost, updates = dA.get_cost_updates(learning_rate)

            fn = theano.function(
                inputs=[
                    index,
                    theano.Param(learning_rate, default=0.1)
                ],
                outputs=cost,
                updates=updates,
                givens={
                    self.c: train_set_c[batch_begin: batch_end],
                    self.x: train_set_x[batch_begin: batch_end]
                }
            )
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

    def build_finetune_functions(self, datasets, batch_size, learning_rate):

        (train_set_c, train_set_x) = datasets[0]
        (valid_set_c, valid_set_x) = datasets[1]
        (test_set_c, test_set_x) = datasets[2]

        n_valid_batches = valid_set_c.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        n_test_batches = test_set_c.get_value(borrow=True).shape[0]
        n_test_batches /= batch_size

        index = T.lscalar('index')

        gparams = T.grad(self.finetune_cost, self.params)

        updates = [
            (param, param - gparam * learning_rate)
            for param, gparam in zip(self.params, gparams)
        ]

        train_fn = theano.function(
            inputs=[index],
            outputs=self.finetune_cost,
            updates=updates,
            givens={
                self.c: train_set_c[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.x: train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='train'
        )

        test_score_i = theano.function(
            [index],
            self.finetune_cost,
            givens={
                self.c: test_set_c[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.x: test_set_x[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='test'
        )

        valid_score_i = theano.function(
            [index],
            self.finetune_cost,
            givens={
                self.c: valid_set_c[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.x: valid_set_x[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='valid'
        )

        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]

        return train_fn, valid_score, test_score


def test_SdA(finetune_lr=0.1, pretraining_epochs=5,
             pretrain_lr=0.1, training_epochs=5, batch_size=1):

    datasets = load_data()

    train_set_c, train_set_x = datasets[0]
    valid_set_c, valid_set_x = datasets[1]
    test_set_c, test_set_x = datasets[2]

    n_train_batches = train_set_c.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size

    numpy_rng = numpy.random.RandomState(89677)
    print '... building the model'

    sda = stacked_autoencoder(
        numpy_rng=numpy_rng,
        n_ins=800 * 600,
        hidden_layers_sizes=[100, 100]
    )

    print '... getting the pretraining functions'
    pretraining_fns = sda.pretraining_functions(
        train_set_c, train_set_x, batch_size)

    print '... pre-training the model'
    start_time = timeit.default_timer()

    for i in xrange(sda.n_layers):
        for epoch in xrange(pretraining_epochs):
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                                            lr=pretrain_lr))
            print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print numpy.mean(c)

    end_time = timeit.default_timer()

    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    print '... getting the finetuning functions'
    train_fn, validate_model, test_model = sda.build_finetune_functions(
        datasets=datasets,
        batch_size=batch_size,
        learning_rate=finetune_lr
    )

    print '... finetuning the model'

    patience = 10 * n_train_batches
    patience_increase = 2.

    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = validate_model()
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                if this_validation_loss < best_validation_loss:

                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    test_losses = test_model()
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%, '
            'on iteration %i, '
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., best_iter + 1, test_score * 100.)
    )
    print >> sys.stderr, ('The training code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))


test_SdA()
