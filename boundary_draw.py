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


def build_parser():
    """Input arguments."""
    # __doc__ automatically contains the docstring for the file.
    parser = argparse.ArgumentParser(description='all data')

    parser.add_argument(
        nargs='?',
        const=1,
        type=str,
        help='dataset Number',
        dest='datasetNum',
        default='93'
    )
    parser.add_argument(
        nargs='?',
        const=1,
        type=str,
        help='ply file',
        dest='plyfile',
        default='093.ply'
    )
    parser.add_argument(
        nargs='?',
        const=1,
        type=str,
        help='data directory',
        dest='datadir',
        default='data'
    )
    parser.add_argument(
        nargs='?',
        const=1,
        type=str,
        help='image directory',
        dest='imagedir',
        default='image'
    )
    parser.add_argument(
        nargs='?',
        const=1,
        type=str,
        help='depth directory',
        dest='depthdir',
        default='depth'
    )
    parser.add_argument(
        nargs='?',
        const=1,
        type=int,
        help='set the parameters for testing',
        dest='rang',
        default='100'
    )
    parser.add_argument(
        nargs='?',
        const=1,
        type=int,
        help='set the image index you want to calculate',
        dest='imgind',
        default='1'
    )
    parser.add_argument(
        nargs='?',
        const=1,
        type=int,
        help='the intinsic variables cx',
        dest='cx',
        default='320'
    )
    parser.add_argument(
        nargs='?',
        const=1,
        type=int,
        help='the intinsic variables cy',
        dest='cy',
        default='240'
    )
    parser.add_argument(
        nargs='?',
        const=1,
        type=float,
        help='the intinsic variables fx',
        dest='fx',
        default='544.47329'
    )
    parser.add_argument(
        nargs='?',
        const=1,
        type=float,
        help='the intinsic variables fy',
        dest='fy',
        default='544.47329'
    )
    parser.add_argument(
        nargs='?',
        const=1,
        type=float,
        help='the threshold variables t',
        dest='threshold',
        default='0.1'
    )

    return parser


def line_to_list_of_float(line):
    """Split the line and return each value."""
    return [float(value) for value in line.split()]


def get_trajactory(parentdir):
    """Compute the trajactory."""
    with open(os.path.join(parentdir, 'trajectory.log'), "r") as f:
        lines = f.read().splitlines()
    trajactory = np.array([line_to_list_of_float(line)
                          for index, line in enumerate(lines)
                          if index % 5])
    return trajactory


def loadImg_name(imagedir):
    """Load all imgs from the directory."""
    allimg = [os.path.join(imagedir, file)
              for file in sorted(os.listdir(imagedir))
              if os.path.splitext(file)[1] == '.png']
    return allimg


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


def test_result(image_array, boundary_image,
                parentdir, args, pt3cam, pt3world):
    """Test results."""
    # %% test
    # show the image
    boundary_detection_2D.image_show(image_array)
    # show the boundary img
    boundary = Image.fromarray(boundary_image * 255)  # 0-1
    boundary.show()
    # save it
    scipy.io.savemat(parentdir + 'boundary' + args.datasetNum +
                     'img' + str(args.imgind)+'.mat',
                     {'boundary_image': boundary_image})
    npyfile = Path(parentdir + '/' + args.datasetNum + '.npy')
    if npyfile.is_file():
        plydata = np.load(parentdir + '/' + args.datasetNum + '.npy')
    else:
        # read the ply file
        plydata = PlyData.read(parentdir + args.plyfile)
        np.save(parentdir + '/' + args.datasetNum, plydata)
        plydata = np.load(parentdir + '/' + args.datasetNum + '.npy')

    x = plydata[0].data['x']
    y = plydata[0].data['y']
    z = plydata[0].data['z']
    r = plydata[0].data['red']
    g = plydata[0].data['green']
    b = plydata[0].data['blue']
    # vertex_index = plydata[0].data['vertex_indices']
    x1 = x[::args.rang]
    y1 = y[::args.rang]
    z1 = z[::args.rang]
    r1 = r[::args.rang]
    g1 = g[::args.rang]
    b1 = b[::args.rang]
    rgb = np.column_stack((r1, g1, b1)) / 255.

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(pt3cam[0, :], pt3cam[1, :], pt3cam[2, :], '.b', ms='0.5')
    ax.legend()
    plt.show()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(x1, y1, z1, s=0.5, c=rgb)
    ax.legend()
    # for i in range(len(x)):
    #     plt.scatter(x[i], y[i], z[i], c=[r[i], g[i], b[i]])
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
    upperdir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    parentdir = os.path.join(upperdir, args.datadir, args.datasetNum)
    imagedir = os.path.join(parentdir, args.imagedir)
    depthdir = os.path.join(parentdir, args.depthdir)
    trajactory = get_trajactory(parentdir)
    allimg = loadImg_name(imagedir)
    alldepth = loadImg_name(depthdir)
    # %% 3D points calculation from the boundary image
    # load and array the image
    image_array = boundary_detection_2D.read_to_array(allimg[args.imgind])
    imagedepth_array = boundary_detection_2D.read_to_array(
                       alldepth[args.imgind])
    # use server method SE/HED to get the segment image
    boundary_image = boundary_detection_2D.detect_boundary(image_array)
    # load the trajectory
    startpos = 4 * (args.imgind - 1)
    cameramat = trajactory[startpos:startpos+3, :]
    # get the pixel position of the boundary and the depth value
    # Now apply the depth to the 3d point
    pt3cam = calulate_3Dpt(boundary_image, imagedepth_array, args.cx, args.cy,
                           args.fx, args.fy, args.threshold)

    # Now apply the camera matrix

    pt3world = np.dot(cameramat[:, 0:3], pt3cam).T + cameramat[:, 3]
    test_result(image_array, boundary_image,
                parentdir, args, pt3cam, pt3world)


if __name__ == '__main__':
    main()
