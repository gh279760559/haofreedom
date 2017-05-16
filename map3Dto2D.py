"""Map 3D points back to 2D img."""

import numpy as np
import matplotlib.pyplot as plt
import boundary_detection_2D
import boundary_draw


# map 3 to 2
def map_pt3to_image(pt3, cx, cy, fx, fy):
    """
    Use to map the 3D pt to 2D img.

    Arguments:
    pt3 3 by N 3d points;
    cx, cy, fx, fy are the intrinsic parameters.
    """
    pt3u = pt3[0, :]
    pt3v = pt3[1, :]
    pt3w = pt3[2, :]
    leng = pt3.shape[1]
    fx = np.tile(fx, (1, leng))
    fy = np.tile(fy, (1, leng))
    cx = np.tile(cx, (1, leng))
    cy = np.tile(cy, (1, leng))

    x = fx * pt3u / pt3w + cx
    y = fy * pt3v / pt3w + cy
    pt2 = np.vstack((y, x))
    return pt2


def load_trajectory(trajectory, image_index):
    """Load trajectory from the dataset."""
    start_position = 4 * (image_index - 1)
    camera_matrix = trajectory[start_position:start_position+3, :]
    rot = camera_matrix[:, 0:3]
    tau = camera_matrix[:, 3]
    return rot, tau


def remove_wrong_direction(camera_rot, vertices, facevertexcdata):
    """Remove the wrong face direction between the pt3 and camera."""
    camd = np.array([np.dot(camera_rot.T, np.array([0, 0, 1]))]).T
    camds = np.tile(camd, (1, vertices.shape[1]))
    # do something in Matlab like pt3_tmp1 = vertices(:,dot(vertices,camds)>=0)
    index = np.einsum('ij,ij->j', vertices, camds) > 0
    pt3_infront = vertices[:, index]
    face_color = facevertexcdata[
        :, np.einsum('ij,ij->j', vertices, camds) > 0
    ]
    return pt3_infront, face_color, index


def map_3dto2d(pt3, rotate, transform, cx, cy, fx, fy):
    """Map the 3D points to 2D image."""
    taus = np.tile(transform, (pt3.shape[1], 1))
    pt3_tmp2, _, _, _ = np.linalg.lstsq(rotate, pt3-taus.T)
    pt2 = map_pt3to_image(pt3_tmp2, cx, cy, fx, fy)
    return pt2


def crop_2d(pt2, facevertexcdata, row, column):
    """Crop the mapped 2D point so that it is in the image."""
    ind = np.logical_and.reduce(
        (pt2[0, :] > 0, pt2[0, :] <= row, pt2[1, :] > 0,
            pt2[1, :] <= column))
    pt21 = pt2[:, ind]
    pt2_color = facevertexcdata[:, ind]
    return pt21, pt2_color, ind


def test_result(
    parent_directory, all_image, all_depth, image_index,
    cx, cy, fx, fy, threshold, rot, tau, rang,
    dataset_number, plyfile, pt2, pt2_color
                ):
    """Test the results use some plots."""
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
    plt.scatter(pt2[1, :], -pt2[0, :], s=0.5, c=pt2_color.T/pt2_color.max())
    ax = fig.gca()
    ax.set_axis_off()
    ax.legend()
    plt.show()


def main():
    """Main function."""
    [
        row, column, dataset_number, plyfile, data_directory_name,
        image_directory_name, depth_directory_name, rang,
        image_index, cx, cy, fx, fy, threshold
    ] = boundary_draw.load_allstuff()

    parent_directory, all_image, trajectory = boundary_draw.load_img(
        data_directory_name, dataset_number, image_directory_name
                                    )
    _, all_depth, _ = boundary_draw.load_img(
        data_directory_name, dataset_number, depth_directory_name
                        )
    vertices, facevertexcdata, faces = boundary_draw.load_npy(
                        parent_directory, dataset_number, image_index, plyfile
    )
    # load the trajectory
    rot, tau = load_trajectory(trajectory, image_index)

    # remove the points wrong direction
    # camera direction
    vertices = vertices.T
    facevertexcdata = facevertexcdata.T
    pt3_tmp1, facevertexcdata1, index = remove_wrong_direction(
        rot, vertices, facevertexcdata)

    # map to 2d img
    pt2tmp1 = map_3dto2d(pt3_tmp1, rot, tau, cx, cy, fx, fy)

    # find the row>=pt2[0]>0, column>=pt2[1]>0
    pt21, pt2_color, ind = crop_2d(pt2tmp1, facevertexcdata1, row, column)

    return pt21, pt2_color, faces, index, ind

    # # %% test
    test_result(
        parent_directory, all_image, all_depth,
        image_index, cx, cy, fx, fy, threshold, rot, tau, rang,
        dataset_number, plyfile, pt21, pt2_color)


if __name__ == '__main__':
    main()
