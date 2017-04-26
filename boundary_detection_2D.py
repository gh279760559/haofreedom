from PIL import Image
import numpy as np
from routines import segment_image
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# import vision
import ipdb
from routines import utilities
import scipy.io as sio
import os
import sys
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement

image = Image.open('1.png')
imgar = np.array(image)
boundary_tuple = segment_image(imgar)
boundary_img_dic = boundary_tuple[1].get("boundary_image")
boundary_img = utilities.image_string_to_numpy(boundary_img_dic[1])
boundary = Image.fromarray(boundary_img * 255)  # 0-1
boundary.show()
