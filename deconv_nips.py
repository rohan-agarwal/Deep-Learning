"""

Recreating the ODCNN created by Li Xu
http://lxu.me/projects/dcnn/

"""

from nips_kernellayer import *
import os
import sys
import timeit

learning_rate = 0.1
batch_size = 500
n_epochs = 200
index = T.lscalar()
x = T.matrix('x')
y = T.ivector('y')

rng = numpy.random.RandomState(23455)

x_data, y_data = load_data()

# instantiate 4D tensor for input
input = T.tensor4(name='input')

print '...building the model'

# "nkern": number of kernels
nkerns = [38, 38, 512, 512]

# image tensor layout:  [mini-batch size, number of input
# feature maps, image height, image width]
# weight matrix layout: [number of feature maps at layer m, number of
# feature maps at layer m-1, filter height, filter width]

layer0_input = x.reshape((batch_size, 3, 600, 800))

layer0 = KernelLayer(
    rng,
    input=layer0_input,
    image_shape=(batch_size, 3, 600, 800),
    filter_shape=(nkerns[0], 3, 1, 121)
)

layer1 = KernelLayer(
    rng,
    input=layer0.output,
    image_shape=(batch_size, nkerns[0], 600, 800),
    filter_shape=(nkerns[1], nkerns[0], 1, 121)
)

layer2 = KernelLayer(
    rng,
    input=layer0.output,
    image_shape=(batch_size, nkerns[0], 600, 680),
    filter_shape=(nkerns[1], nkerns[0], 121, 1)
)

layer3 = KernelLayer(
    rng,
    input=layer0.output,
    image_shape=(batch_size, nkerns[1], 480, 680),
    filter_shape=(nkerns[2], nkerns[1], 16, 16)
)

layer4 = KernelLayer(
    rng,
    input=layer0.output,
    image_shape=(batch_size, nkerns[2], 480, 680),
    filter_shape=(nkerns[3], nkerns[2], 16, 16)
)

cost = T.sum(layer0.output)

test_model = theano.function(
    [index],
    layer0.output,
    givens={
        x: x_data[index * batch_size: (index + 1) * batch_size],
        y: y_data[index * batch_size: (index + 1) * batch_size]
    }
)

validate_model = theano.function(
    [index],
    layer0.output,
    givens={
        x: x_data[index * batch_size: (index + 1) * batch_size],
        y: y_data[index * batch_size: (index + 1) * batch_size]
    }
)

params = layer0.params
grads = T.grad(cost, layer0.params)

updates = [
    (param_i, param_i - learning_rate * grad_i)
    for param_i, grad_i in zip(params, grads)
]

train_model = theano.function(
    [index],
    cost,
    updates=updates,
    givens={
        x: x_data[index * batch_size: (index + 1) * batch_size],
        y: y_data[index * batch_size: (index + 1) * batch_size]
    }
)

print '... training'
# early-stopping parameters
patience = 10000  # look as this many examples regardless
patience_increase = 2
improvement_threshold = 0.995
validation_frequency = min(n_train_batches, patience / 2)

best_validation_loss = numpy.inf
best_iter = 0
test_score = 0.
start_time = timeit.default_timer()

epoch = 0
done_looping = False

while (epoch < n_epochs) and (not done_looping):
    epoch = epoch + 1
    for minibatch_index in xrange(n_train_batches):

        iter = (epoch - 1) * n_train_batches + minibatch_index

        if iter % 100 == 0:
            print 'training @ iter = ', iter
        cost_ij = train_model(minibatch_index)

        if (iter + 1) % validation_frequency == 0:

            # compute zero-one loss on validation set
            validation_losses = [validate_model(i) for i
                                 in xrange(n_valid_batches)]
            this_validation_loss = numpy.mean(validation_losses)
            print('epoch %i, minibatch %i/%i, validation error %f %%' %
                  (epoch, minibatch_index + 1, n_train_batches,
                   this_validation_loss * 100.))

            # if we got the best validation score until now
            if this_validation_loss < best_validation_loss:

                # improve patience if loss improvement is good enough
                if this_validation_loss < best_validation_loss *  \
                   improvement_threshold:
                    patience = max(patience, iter * patience_increase)

                # save best validation score and iteration number
                best_validation_loss = this_validation_loss
                best_iter = iter

                # test it on the test set
                test_losses = [
                    test_model(i)
                    for i in xrange(n_test_batches)
                ]
                test_score = numpy.mean(test_losses)
                print(('     epoch %i, minibatch %i/%i, test error of '
                       'best model %f %%') %
                      (epoch, minibatch_index + 1, n_train_batches,
                       test_score * 100.))

        if patience <= iter:
            done_looping = True
            break

end_time = timeit.default_timer()
print('Optimization complete.')
print('Best validation score of %f %% obtained at iteration %i, '
      'with test performance %f %%' %
      (best_validation_loss * 100., best_iter + 1, test_score * 100.))
print >> sys.stderr, ('The code for file ' +
                      os.path.split(__file__)[1] +
                      ' ran for %.2fm' % ((end_time - start_time) / 60.))
