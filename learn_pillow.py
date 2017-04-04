# -*- coding: utf-8 -*-

from PIL import Image
import numpy

import ipdb; ipdb.set_trace()

image = Image.open('messi5.jpg')
image.show()
r, g, b = image.split()
histogram = image.histogram()
exif = image._getexif()
cropped = image.crop((0, 80, 200, 400))
