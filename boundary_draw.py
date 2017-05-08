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
    return parser


def line_to_list_of_float(line):
    """Split the line and return each value."""
    return [float(value) for value in line.split()]


def get_trajectory(parent_directory):
    """Compute the trajectory."""
    with open(os.path.join(parent_directory, 'trajectory.log'), "r") as f:
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


def load_npy(parent_directory, dataset_number, image_index, plyfile):
    """Load the npy file, if not exist will build it."""
    filename = '{}.npy'.format(dataset_number)
    npyfile = Path(os.path.join(parent_directory, filename))
    if npyfile.is_file():
        # read the ply file
        plydata = np.load(os.path.join(parent_directory, filename))
    else:
        plydata = PlyData.read(os.path.join(parent_directory, plyfile))
        np.save(os.path.join(parent_directory, filename), plydata)
    x, y, z, r, g, b = (plydata[0].data[key]
                        for key in ('x', 'y', 'z', 'red', 'green', 'blue'))
    vertices = np.column_stack((x, y, z))
    rgb = np.column_stack((r, g, b))
    return vertices, rgb


def test_result(image_array, boundary_image, parent_directory, dataset_number,
                image_index, rang, plyfile, pt3_camera, pt3_world):
    """Test results."""
    # %% test
    # show the image
    boundary_detection_2D.image_show(image_array)
    # show the boundary img
    boundary = Image.fromarray(boundary_image * 255)  # 0-1
    boundary.show()

    # save it
    filename = '{}.mat'.format('boundary' + str(image_index))
    npyfile = Path(os.path.join(parent_directory, filename))
    if npyfile.is_file() is False:
        scipy.io.savemat(
            os.path.join(parent_directory, filename),
            {'boundary_image': boundary_image})
    vertices, rgb = load_npy(
                    parent_directory, dataset_number, image_index, plyfile
                            )
    vertices = vertices[::rang, :]
    rgb = rgb[::rang, :] / 255

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(
        pt3_camera[0, :], pt3_camera[1, :], pt3_camera[2, :], '.b', ms='0.5'
           )
    plt.show()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=0.5, c=rgb)
    plt.show()
    plt.hold(True)
    ax.scatter(pt3_world[:, 0], pt3_world[:, 1], pt3_world[:, 2],
               facecolor='0', s=2)
    # ax.plot(
    #     pt3_camera[:, 0], pt3_camera[:, 1], pt3_camera[:, 2], '.b', ms='0.5'
    #        )
    ax.set_axis_off()
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
        data[key] for key in ('dataset_number', 'plyfile')
        )
    data_directory, image_directory, depth_directory = (
        data[key]
        for key in ('data_directory', 'image_directory', 'depth_directory')
        )
    rang, image_index = (data[key] for key in ('rang', 'image_index'))
    cx, cy, fx, fy = (data[key] for key in ('cx', 'cy', 'fx', 'fy'))
    threshold = data['t']

    upper_directory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    parent_directory = os.path.join(
                            upper_directory, data_directory, dataset_number
                            )
    image_directory = os.path.join(parent_directory, image_directory)
    depth_directory = os.path.join(parent_directory, depth_directory)
    trajectory = get_trajectory(parent_directory)
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
    camera_matrix = trajectory[startpos:startpos+3, :]
    # get the pixel position of the boundary and the depth value
    # Now apply the depth to the 3d point
    pt3_camera = calulate_3Dpt(
            boundary_image, imagedepth_array, cx, cy, fx, fy, threshold
                              )

    # Now apply the camera matrix
    pt3_world = np.dot(
                camera_matrix[:, 0:3], pt3_camera).T + camera_matrix[:, 3]
    test_result(image_array, boundary_image, parent_directory, dataset_number,
                image_index, rang, plyfile, pt3_camera, pt3_world)


if __name__ == '__main__':
    main()
