import numpy
import pylab
from PIL import Image
import theano
from theano import tensor as T
from theano.tensor.nnet import conv

rng = numpy.random.RandomState(23455)

# instantiate 4D tensor for input
input = T.tensor4(name='input')

# variables for kernel size
k_num = 38
k_h = 1
k_w = 121

# random initialization for layer 1 weights
w_shp = (k_num, 3, k_h, k_w)
w_bound = numpy.sqrt(k_num * k_h * k_w)
W = theano.shared(numpy.asarray(
    rng.uniform(
        low=-1.0 / w_bound,
        high=1.0 / w_bound,
        size=w_shp),
    dtype=input.dtype), name='W')

b_shp = k_num
b = theano.shared(numpy.zeros(b_shp), name='b')

# build symbolic expression that computes the convolution of input with
# filters in w
conv_out = conv.conv2d(input, W)

#   dimshuffle('x', 2, 'x', 0, 1)
#   This will work on 3d tensors with no broadcastable
#   dimensions. The first dimension will be broadcastable,
#   then we will have the third dimension of the input tensor as
#   the second of the resulting tensor, etc. If the tensor has
#   shape (20, 30, 40), the resulting tensor will have dimensions
#   (1, 40, 1, 20, 30). (AxBxC tensor is mapped to 1xCx1xAxB tensor)
output = T.nnet.sigmoid(conv_out + b.dimshuffle('x', 0, 'x', 'x'))

# create theano function to compute filtered images
f = theano.function([input], output)

# open random image of dimensions 639x516
img = Image.open("C:\Users\Rohan\Documents\GitHub\Deep-Learning\Data\one.png")
# dimensions are (height, width, channel)
img = numpy.asarray(img, dtype='float32') / 256.
# put image in 4D tensor of shape (1, 3, height, width)
img_ = img.transpose(2, 0, 1).reshape(1, 3, 600, 800)
filtered_img = f(img_)
print filtered_img.shape

# plot original image and first and second components of output
# pylab.subplot(1, 3, 1)
# pylab.axis('off')
# pylab.imshow(img)
# pylab.gray()
# recall that the convOp output (filtered image) is actually a "minibatch",
# of size 1 here, so we take index 0 in the first dimension:
# for i in range(k_num):
#     pylab.subplot((i % 3)+1, k_num+1, i+2)
#     pylab.axis('off')
#     pylab.imshow(filtered_img[0, i, :, :])
# pylab.show()
