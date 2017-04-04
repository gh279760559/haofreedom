"""This is trying to implent the 2D edge detection using our own method."""

from PIL import Image
import numpy as np
from routines import segment_image
import vision
# import ipdb
from routines import utilities

# %%load and show the image
image = Image.open('messi5.jpg')
image.show()

# %%set the image to array for rest calculation
a = np.array(image)
# ipdb.set_trace()
# %%use server method SE/HED to get the segment image
boundary_tuple = segment_image(a)
# boundary_tuple = segment_image(a, use_hed_model=True)
boundary_img_dic = boundary_tuple[1].get("boundary_image")
boundary_img = utilities.image_string_to_numpy(boundary_img_dic[1])
boundary = Image.fromarray(boundary_img * 255)  # 0-1
boundary.show()
