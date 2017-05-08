"""Map 3D points back to 2D img."""

import numpy as np
import matplotlib.pyplot as plt
import os
import boundary_detection_2D
import boundary_draw
import json


# map 3 to 2
def map_pt3to_image(pt3, cx, cy, fx, fy):
    """
    Use to map the 3D pt to 2D img.

    Keyword arguments:
    pt3 3 by N 3d points;
    cx, cy, fx, fy are the intrinsic parameters.
    """
    pt3u = pt3[0, :]
    pt3v = pt3[1, :]
    pt3w = pt3[2, :]
    leng = pt3.shape[1]
    fx = np.matlib.repmat(fx, 1, leng)
    fy = np.matlib.repmat(fy, 1, leng)
    cx = np.matlib.repmat(cx, 1, leng)
    cy = np.matlib.repmat(cy, 1, leng)

    x = fx*pt3u/pt3w+cx
    y = fy*pt3v/pt3w+cy
    pt2 = np.vstack((y, x))
    return pt2


def main():
    """Main function."""
    args = boundary_draw.build_parser().parse_args()
    assert os.path.isfile(args.filename), (
        'File {} does not exist!'.format(args.filename)
    )
    with open(args.filename) as fd:
        data = json.load(fd)

    row, column = (
        data[key] for key in ('row', 'column')
    )

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
    trajectory = boundary_draw.get_trajectory(parent_directory)
    # load the trajectory
    start_position = 4 * (image_index - 1)
    camera_matrix = trajectory[start_position:start_position+3, :]
    rot = camera_matrix[:, 0:3]
    tau = camera_matrix[:, 3]
    # remove the points wrong direction
    # camera direction
    vertices, facevertexcdata = boundary_draw.load_npy(
                        parent_directory, dataset_number, image_index, plyfile
                                                      )
    vertices = vertices.T

    camd = np.array([np.dot(rot.T, np.array([0, 0, 1]))]).T
    camds = np.matlib.repmat(camd, 1, vertices.shape[1])

    pt3_tmp1 = vertices[:, np.einsum('ij,ij->j', vertices, camds) > 0]
    facevertexcdata = facevertexcdata.T
    facevertexcdata1 = facevertexcdata[
        :, np.einsum('ij,ij->j', vertices, camds) > 0
    ]

    # map to 2d img
    taus = np.matlib.repmat(tau, pt3_tmp1.shape[1], 1)
    pt3_tmp2, resid, rank, s = np.linalg.lstsq(rot, pt3_tmp1-taus.T)
    pt2tmp1 = map_pt3to_image(pt3_tmp2, cx, cy, fx, fy)

    # find the row>=pt2[0]>0, column>=pt2[1]>0
    ind = np.logical_and.reduce(
        (pt2tmp1[0, :] > 0, pt2tmp1[0, :] <= row, pt2tmp1[1, :] > 0,
            pt2tmp1[1, :] <= column))
    pt21 = pt2tmp1[:, ind]
    pt2_color = facevertexcdata1[:, ind]
    # %% test
    all_image = boundary_draw.loadImg_name(image_directory)
    all_depth = boundary_draw.loadImg_name(depth_directory)

    image_array = boundary_detection_2D.read_to_array(
                                                    all_image[image_index]
                                                     )

    imagedepth_array = boundary_detection_2D.read_to_array(
                                                    all_depth[image_index]
                                                           )
    boundary_image = boundary_detection_2D.detect_boundary(image_array)
    pt3_camera = boundary_draw.calulate_3Dpt(
            boundary_image, imagedepth_array, cx, cy, fx, fy, threshold
                                        )
    pt3_world = np.dot(rot, pt3_camera).T + tau
    boundary_draw.test_result(
                image_array, boundary_image, parent_directory, dataset_number,
                image_index, rang, plyfile, pt3_camera, pt3_world)

    fig = plt.figure()
    plt.scatter(pt21[1, :], -pt21[0, :], s=0.5, c=pt2_color.T/pt2_color.max())
    ax = fig.gca()
    ax.set_axis_off()
    ax.legend()
    plt.show()


if __name__ == '__main__':
    main()
