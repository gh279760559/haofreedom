"""This is trying to implent the 2D edge detection using our own method."""

from PIL import Image
import numpy as np
from routines import segment_image
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# import vision
# import ipdb
from routines import utilities
import scipy.io as sio
import os
import sys
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement

# %% to start: get everything done.
# set the parameters
datasetNum = '93'
plyfile = "093.ply"  # set the parameters for testing
rang = 100  # set the parameters for testing
parentdir = "/home/hao/MyCode/3d-mesh-segmentation/data/"+datasetNum+"/"
imagedir = parentdir+"image"
depthdir = parentdir+"depth"
# set the image index you want to calculate
imgind = 500  # 1 means the 1st image
# x = lines[0]

# the intinsic variables
cx = 320
cy = 240
fx = 544.47329
fy = 544.47329
# get the trajactory log file
f = open(parentdir+"trajectory.log", "r")
lines = f.readlines()
lines = list(map(lambda s: s.strip(), lines))  # remove \n
ind = 0
for i in np.arange(len(lines)):
    if(ind % 5 != 0):
        tmp = [float(i) for i in lines[i].split()]
        if(ind == 1):
            trajactory = tmp
        else:
            trajactory = np.vstack((trajactory, tmp))
    ind = ind + 1


# get all images
ind1 = 0
allimg = []
for file in sorted(os.listdir(imagedir)):
    if file.endswith(".png"):
        allimgtmp = os.path.join(imagedir, file)
        if ind1 == 0:
            allimg.insert(0, allimgtmp)
        else:
            allimg.append(allimgtmp)
        # print(allimg)  # for test
        ind1 = ind1 + 1

# same thing: get all depths
ind2 = 0
alldepth = []
for file in sorted(os.listdir(depthdir)):
    if file.endswith(".png"):
        alldepthtmp = os.path.join(depthdir, file)
        if ind2 == 0:
            alldepth.insert(0, alldepthtmp)
        else:
            alldepth.append(alldepthtmp)
        # print(allimg)  # for test
        ind2 = ind2 + 1

# test
# ipdb.set_trace()
if(ind1 != ind2):
    # print("img and depth numbers not match")
    sys.exit("img and depth numbers not match")

# %% get the 3D point relevant to the world coordinates
# load and show the image
image = Image.open(allimg[imgind])
imgdepth = Image.open(alldepth[imgind])

# load the trajectory
startpos = 4 * (imgind-1)
cameramat = trajactory[startpos:startpos+3, :]

# set the image to array for rest calculation
imgar = np.array(image)
imgdepthar = np.array(imgdepth)
# ipdb.set_trace()

# use server method SE/HED to get the segment image
boundary_tuple = segment_image(imgar)
# boundary_tuple = segment_image(a, use_hed_model=True)
boundary_img_dic = boundary_tuple[1].get("boundary_image")
boundary_img = utilities.image_string_to_numpy(boundary_img_dic[1])


# Now def the function which is used to calculate the 3D points
# NOTICE: possible can be improved in the future


def cal_3dpt(ptr, ptc, ptd, cx, cy, fx, fy):
    """
    3d pts and depth point calculation.

    Keyword arguments:
    ptr, ptc the row and column of the 2d pt; ptd is the corresponding
    depth map; cx, cy, fx, fy are the intrinsic parameters...
    Notice: can be improved in future for all pts at one time.
    """
    pt3 = []
    ptd = ptd/1000
    u = (ptc - cx) * ptd / fx
    v = (ptr - cy) * ptd / fy
    pt3 = [u, v, ptd]
    return pt3


# get the pixel position of the boundary and the depth value
# Now apply the depth to the 3d point
'''Note: the loop can be improved later!!'''
zxc = 0
ccc = 0
ptr_index = ptc_index = ptd_index = pt3cam = []
for i in np.arange(boundary_img.shape[0]):
    for j in np.arange(boundary_img.shape[1]):
        zxc = zxc + 1
        # if(boundary_img[i, j] != 0):  # draw all boundaries
        if(boundary_img[i, j] >= 0.1):  # darw remarkable boundaries
            ccc = ccc + 1
            ptr_index.append(i)
            ptc_index.append(j)
            ptd_index.append(imgdepthar[i, j])
            tmp3d = cal_3dpt(i, j, imgdepthar[i, j], cx, cy, fx, fy)
            if(ccc == 1):
                pt3cam = tmp3d
            else:
                pt3cam = np.vstack((pt3cam, tmp3d))

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot(pt3cam[:, 0], pt3cam[:, 1], pt3cam[:, 2], '.b', ms='0.3')
# ax.set_axis_off()

# Now apply the camera matrix
''' the loop can be improved'''
pt3world = np.zeros(pt3cam.shape)
for i in np.arange(pt3cam.shape[0]):
    pt3world[i, :] = np.dot(cameramat[:, 0:3], pt3cam[i, :]) + cameramat[:, 3]

# %% test
# show the image
image.show()  # for test

# show the boundary img
boundary = Image.fromarray(boundary_img * 255)  # 0-1
boundary.show()
# save it
sio.savemat(parentdir+'boundary'+datasetNum+'.mat',
            {'boundary_img': boundary_img})

# read the ply file
# plydata = PlyData.read(parentdir+plyfile)
# np.save(parentdir+datasetNum, plydata)
# x = y = z = r = g = b = vertex_index = []
# x = plydata.elements[0].data['x']
# y = plydata.elements[0].data['y']
# z = plydata.elements[0].data['z']
# r = plydata.elements[0].data['red']
# g = plydata.elements[0].data['green']
# b = plydata.elements[0].data['blue']
# vertex_index = plydata.elements[1].data['vertex_indices']
plydata = np.load(parentdir+datasetNum+'.npy')

x = y = z = r = g = b = vertex_index = []
x = plydata[0].data['x']
y = plydata[0].data['y']
z = plydata[0].data['z']
r = plydata[0].data['red']
g = plydata[0].data['green']
b = plydata[0].data['blue']
# vertex_index = plydata[0].data['vertex_indices']
x1 = x[::rang]
y1 = y[::rang]
z1 = z[::rang]
r1 = r[::rang]
g1 = g[::rang]
b1 = b[::rang]
rgb = np.column_stack((r1, g1, b1)) / 255.
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(x1, y1, z1, s=0.5, c=rgb)
ax.legend()
# for i in range(len(x)):
#     plt.scatter(x[i], y[i], z[i], c=[r[i], g[i], b[i]])
plt.show()
plt.hold(True)
ax.scatter(pt3world[:, 0], pt3world[:, 1], pt3world[:, 2], facecolor='0', s=2)
# ax.plot(pt3cam[:, 0], pt3cam[:, 1], pt3cam[:, 2], '.b', ms='0.5')
ax.set_axis_off()
ax.legend()
plt.show()
