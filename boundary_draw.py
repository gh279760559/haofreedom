"""The 2D edge detection using our own method."""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os
import boundary_detection_2D
import argparse
from pathlib import Path
from plyfile import PlyData
import json


def build_parser():
    """Input arguments."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        '-f', '--filename',
        help='Name of json file to load',
        type=str,
        default='parameters.json'
        )

    #
    # parser.add_argument(
    #     '-d', '--dataset_number',
    #     type=str,
    #     help='input dataset Number (from ScenNN)',
    #     default='93'
    # )
    # parser.add_argument(
    #     '-p', '--plyfile',
    #     type=str,
    #     help='input ply file (from ScenNN)',
    #     default='093.ply'
    # )
    # parser.add_argument(
    #     '--data_directory',
    #     type=str,
    #     help='the data directory',
    #     default='data'
    # )
    # parser.add_argument(
    #     '--image_directory',
    #     type=str,
    #     help='the image directory',
    #     default='image'
    # )
    # parser.add_argument(
    #     '--depth_directory',
    #     type=str,
    #     help='the depth directory',
    #     default='depth'
    # )
    # parser.add_argument(
    #     '--rang',
    #     type=int,
    #     help='it may take a while to plot all points, instead set the rang',
    #     default='100'
    # )
    # parser.add_argument(
    #     '--image_index',
    #     type=int,
    #     help='set the image index you want to calculate',
    #     default='1'
    # )
    # parser.add_argument(
    #     '--cx',
    #     type=int,
    #     help='the intinsic variables cx',
    #     default='320'
    # )
    # parser.add_argument(
    #     '--cy',
    #     type=int,
    #     help='the intinsic variables cy',
    #     default='240'
    # )
    # parser.add_argument(
    #     '--fx',
    #     type=float,
    #     help='the intinsic variables fx',
    #     default='544.47329'
    # )
    # parser.add_argument(
    #     '--fy',
    #     type=float,
    #     help='the intinsic variables fy',
    #     default='544.47329'
    # )
    # parser.add_argument(
    #     '-t', '--threshold',
    #     type=float,
    #     help='the threshold variables t',
    #     default='0.1'
    # )

    return parser


def line_to_list_of_float(line):
    """Split the line and return each value."""
    return [float(value) for value in line.split()]


def get_trajectory(parentdir):
    """Compute the trajectory."""
    with open(os.path.join(parentdir, 'trajectory.log'), "r") as f:
        lines = f.read().splitlines()
    trajectory = np.array([line_to_list_of_float(line)
                          for index, line in enumerate(lines)
                          if index % 5])
    return trajectory


def loadImg_name(image_directory):
    """Load all imgs from the directory."""
    all_img = [os.path.join(image_directory, file)
               for file in sorted(os.listdir(image_directory))
               if os.path.splitext(file)[1] == '.png']
    return all_img


def calulate_3Dpt(pt2D, ptd, cx, cy, fx, fy, threshold):
    """
    3d pts and depth point calculation.

    Keyword arguments:
    pt2D 2 by N 2D points;
    ptd is the corresponding depth map;
    cx, cy, fx, fy are the intrinsic parameters.
    threshold is the value for the boundary
    """
    pt2D_row, pt2D_column = np.where(pt2D > threshold)
    ptd = ptd[pt2D > threshold] / 1000
    u3d = (pt2D_column - cx) * ptd / fx
    v3d = (pt2D_row - cy) * ptd / fy
    pt3 = np.array([u3d, v3d, ptd])
    return pt3


def load_npy(parentdir, dataset_number, image_index, plyfile):
    """Load the npy file, if not exist will build it."""
    npyfile = Path(parentdir + 'boundary' + dataset_number +
                   'img' + str(image_index)+'.mat')
    filename = '{}.npy'.format(dataset_number)
    if npyfile.is_file():
        # read the ply file
        plydata = np.load(os.path.join(parentdir, filename))
    else:
        # plydata = PlyData.read(parentdir + plyfile)
        plydata = PlyData.read(os.path.join(parentdir, plyfile))
        np.save(os.path.join(parentdir, filename), plydata)

    x, y, z, r, g, b = (plydata[0].data[key]
                        for key in ('x', 'y', 'z', 'red', 'green', 'blue'))
    vertices = np.column_stack((x, y, z))
    rgb = np.column_stack((r, g, b))
    return vertices, rgb


def test_result(image_array, boundary_image, parentdir, dataset_number,
                image_index, rang, plyfile, pt3cam, pt3world):
    """Test results."""
    # %% test
    # show the image
    boundary_detection_2D.image_show(image_array)
    # show the boundary img
    boundary = Image.fromarray(boundary_image * 255)  # 0-1
    boundary.show()

    # save it
    npyfile = Path(parentdir + 'boundary' + dataset_number +
                   'img' + str(image_index)+'.mat')
    if npyfile.is_file() is False:
        scipy.io.savemat(parentdir + 'boundary' + dataset_number +
                         'img' + str(image_index)+'.mat',
                         {'boundary_image': boundary_image})

    vertices, rgb = load_npy(parentdir, dataset_number, image_index, plyfile)
    vertices = vertices[::rang, :]
    # ipdb.set_trace()
    rgb = rgb[::rang, :] / 255

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(pt3cam[0, :], pt3cam[1, :], pt3cam[2, :], '.b', ms='0.5')
    ax.legend()
    plt.show()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=0.5, c=rgb)
    ax.legend()
    plt.show()
    plt.hold(True)
    ax.scatter(pt3world[:, 0], pt3world[:, 1], pt3world[:, 2],
               facecolor='0', s=2)
    # ax.plot(pt3cam[:, 0], pt3cam[:, 1], pt3cam[:, 2], '.b', ms='0.5')
    ax.set_axis_off()
    ax.legend()
    plt.show()


def main():
    """Main function."""
    args = build_parser().parse_args()
    assert os.path.isfile(args.filename), (
        'File {} does not exist!'.format(args.filename)
    )
    with open(args.filename) as fd:
        data = json.load(fd)

    dataset_number, plyfile = (
        data[key] for key in
        ('dataset_number', 'plyfile')
        )
    data_directory, image_directory, depth_directory = (
        data[key] for key in
        ('data_directory', 'image_directory', 'depth_directory')
        )
    rang, image_index = (data[key] for key in ('rang', 'image_index'))
    cx, cy, fx, fy = (data[key] for key in ('cx', 'cy', 'fx', 'fy'))
    threshold = data['t']

    upperdir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    parentdir = os.path.join(
                            upperdir, data_directory, dataset_number
                            )
    image_directory = os.path.join(parentdir, image_directory)
    depth_directory = os.path.join(parentdir, depth_directory)
    trajectory = get_trajectory(parentdir)
    all_image = loadImg_name(image_directory)
    all_depth = loadImg_name(depth_directory)
    # %% 3D points calculation from the boundary image
    # load and array the image
    image_array = boundary_detection_2D.read_to_array(
                                                    all_image[image_index]
                                                     )
    imagedepth_array = boundary_detection_2D.read_to_array(
                       all_depth[image_index])
    # use server method SE/HED to get the segment image
    boundary_image = boundary_detection_2D.detect_boundary(image_array)
    # load the trajectory
    startpos = 4 * (image_index - 1)
    cameramat = trajectory[startpos:startpos+3, :]
    # get the pixel position of the boundary and the depth value
    # Now apply the depth to the 3d point
    pt3cam = calulate_3Dpt(boundary_image, imagedepth_array, cx, cy,
                           fx, fy, threshold)

    # Now apply the camera matrix
    pt3world = np.dot(cameramat[:, 0:3], pt3cam).T + cameramat[:, 3]
    test_result(image_array, boundary_image, parentdir, dataset_number,
                image_index, rang, plyfile, pt3cam, pt3world)


if __name__ == '__main__':
    main()
