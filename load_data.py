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
            im = im.split()[1]
            im = numpy.array(im, dtype='float32') / 256.
            im = im.ravel()
            image_list.append(im)
        return image_list

    def shared_dataset(data):

        shared_data = theano.shared(numpy.asarray(data,
                                                  dtype=theano.config.floatX),
                                    borrow=True)

        return shared_data

    data_c = pic2np(blurry_path + '*.jpg')
    data_x = pic2np(clean_path + '*.png')

    size = len(data_c)
    cut1 = int(.6 * size)
    cut2 = int(.8 * size)

    train_c = data_c[0:cut1]
    valid_c = data_c[cut1:cut2]
    test_c = data_c[cut2:size]

    train_x = data_x[0:cut1]
    valid_x = data_x[cut1:cut2]
    test_x = data_x[cut2:size]

    train_set_c = shared_dataset(train_c)
    train_set_x = shared_dataset(train_x)

    valid_set_c = shared_dataset(valid_c)
    valid_set_x = shared_dataset(valid_x)

    test_set_c = shared_dataset(test_c)
    test_set_x = shared_dataset(test_x)

    rval = [(train_set_c, train_set_x),
            (valid_set_c, valid_set_x), (test_set_c, test_set_x)]

    return rval

load_data()
