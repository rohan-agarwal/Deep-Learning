"""

Recreating the ODCNN created by Li Xu
http://lxu.me/projects/dcnn/

"""

from nips_kernellayer import *
from PIL import Image


print '...building the model'

img = Image.open("C:\Users\Rohan\Documents\GitHub\Deep-Learning\Data\one.png")
img = numpy.asarray(img, dtype='float32') / 256.

img_h = img.shape[0]
img_w = img.shape[1]


