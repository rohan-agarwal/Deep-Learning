import glob
from PIL import Image
import numpy
import theano

blurry_path = 'C:\Users\Rohan\Documents\GitHub\Deep-Learning\Data\\blur\\'
clean_path = 'C:\Users\Rohan\Documents\GitHub\Deep-Learning\Data\clean\\'


def load_data():

    def pic2np(gpath):
        image_list = []
        for f in glob.glob(gpath):
            im = Image.open(f)
            im = numpy.asarray(im, dtype='float32') / 256.
            im = im.ravel()
            image_list.append(im)
        return image_list

    def shared_dataset(data):

        shared_data = theano.shared(numpy.asarray(data,
                                                  dtype=theano.config.floatX))

        return shared_data

    data_x = pic2np(blurry_path + '*.jpg')
    data_y = pic2np(clean_path + '*.png')

    size = len(data_x)
    cut1 = int(.6 * size)
    cut2 = int(.8 * size)
    train_x = data_x[0:cut1]
    test_x = data_x[cut2:size]
    train_y = data_y[0:cut1]
    test_y = data_y[cut2:size]
    valid_x = data_x[cut1:cut2]
    valid_y = data_y[cut1:cut2]

    train_set_x = shared_dataset(train_x)
    train_set_y = shared_dataset(train_y)
    test_set_x = shared_dataset(test_x)
    test_set_y = shared_dataset(test_y)
    valid_set_x = shared_dataset(valid_x)
    valid_set_y = shared_dataset(valid_y)

    rval = [(train_set_x, train_set_y),
            (valid_set_x, valid_set_y), (test_set_x, test_set_y)]

    return rval
