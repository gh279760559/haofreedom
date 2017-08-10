"""denseCRF in 3D space."""
import numpy as np
from plyfile import PlyData
from skimage import color
from pathlib import Path
import argparse
import os.path
import copy
import sys
import ipdb


def build_parser():
    """Some setting."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        '-i', '--mesh-filepath',
        nargs='*',
        help='Ply text file specifying a mesh.',
        dest='mesh_filepath',
        required=True
    )

    parser.add_argument(
        '-o', '--output-path',
        help='Output path.',
        dest='output_path',
        required=True
    )

    parser.add_argument(
        '-n', '--n-segments',
        help=(
            'Number of segments desired.'
            'Used as label count for DCRF.'
        ),
        dest='n_segs',
        type=int,
        default=2  # Four walls, ceiling, floor.
    )

    parser.add_argument(
        '--p',
        help='Whether save as ply.',
        dest='whether_saveply',
        default=False
    )

    return parser


def load_npz(mesh_filepath, if_test):
    """Load the npy file, if not exist will build it."""
    filename = os.path.basename(mesh_filepath)
    filename_no_format = os.path.splitext(filename)[0]
    # parent directories
    directories = os.path.dirname(os.path.realpath(mesh_filepath))
    npyfile = Path(directories + '/' + filename_no_format + '.npz')
    if npyfile.is_file():
        print('the mesh NPY format exists, now loading...')
        # read the ply file
        npzfile = np.load(npyfile)
        vertices_pos = npzfile['vertices']
        vertices_color = npzfile['rgb']
        vertices_color = color.rgb2lab(
            vertices_color[np.newaxis, ...]).squeeze()
        # following can be used as well.
        # vertices_color = 0.21*vertices_color[:, 0] + \
        #     0.72*vertices_color[:, 1] \
        #     + 0.07*vertices_color[:, 2]
        # vertices_color = np.mean(vertices_color, axis=1)
        # vertices_color = vertices_color[..., np.newaxis]
        vertices_normals = npzfile['vertices_normals']

        label = npzfile['label']

        if(if_test == 1):
            plydata = PlyData.read(
                directories + '/' + filename_no_format+'.ply')

    else:
        print('the mesh NPY format not exist, now load ply and save')

        plydata = PlyData.read(directories + '/' + filename_no_format+'.ply')

        x, y, z, r, g, b, nx, ny, nz = (plydata['vertex'].data[key]
                                        for key in ('x', 'y', 'z',
                                                    'red', 'green', 'blue',
                                                    'nx', 'ny', 'nz'))
        # if use CloudCompare do this,
        # label = plydata['vertex'].data['scalar_alpha']
        # else do this
        label = plydata['vertex'].data['label']
        vertices_pos = np.column_stack((x, y, z))
        vertices_normals = np.column_stack((nx, ny, nz))
        vertices_color = np.column_stack((r, g, b))
        saved_name = directories + '/' + filename_no_format + '.npz'

        np.savez(
            saved_name, vertices=vertices_pos, rgb=vertices_color,
            vertices_normals=vertices_normals, label=label)

    if(if_test == 1):
        return [vertices_pos, vertices_color, vertices_normals,
                label, plydata]
    else:
        return [vertices_pos, vertices_color, vertices_normals, label]


def groundtruth_calculation(labels, simulating_point_index):
    """Return the index of points that have same color."""
    labels_number = labels[simulating_point_index]
    indx = np.where(labels == labels_number)
    return indx


def generate_empty_list(list_size):
    """Generate the N-D empty list with size N=list_size."""
    tmp = []
    for i in range(list_size):
        tmp.append([])
    return tmp


def main(args):
    """Main function."""
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    print('Loading and parameters setting...')
    simulating_point_index = []
    # 093
    # simulating_point_index.append([972119, 975828, 1140442, 491021, 1204595])
    # 207
    # simulating_point_index.append([313301, 587101, 948647, 733043, 367594])
    # 207-cut_version:try.ply
    simulating_point_index.append([69070])
    # 207-cut_version:111.ply
    # simulating_point_index.append([136099, 101948])
    # can add more
    # 207-density_reduced_version:112.ply
    # simulating_point_index.append([30052])
    # simulating_point_index.append
    # simulating_point_index.append
    # simulating_point_index.append
    # simulating_point_index.append
    if_test = 1

    print("load the npy file or save as npyfile...")
    positions, colors, normals, data_obj, label_color = [], [], [], [], []
    mesh_filepath_nums = len(args.mesh_filepath)
    dataset_numbers = mesh_filepath_nums
    # test

    if(dataset_numbers != len(simulating_point_index)):
        sys.exit(
            "the dataset numbers are not \
            matching the simulating numbers, line 284")
    else:
        dataset_numbers = np.int(dataset_numbers)
    for i in range(0, mesh_filepath_nums):
            [positions_tmp, colors_tmp,
                normals_tmp, label_tmp,
                dataobj_tmp] = load_npz(args.mesh_filepath[i], if_test)
            label_color.append(label_tmp)
            positions.append(positions_tmp)
            colors.append(colors_tmp)
            normals.append(normals_tmp)
            data_obj.append(dataobj_tmp)

    print('now get the groundtruth labelling...')
    # get all simulating click areas!
    print("get all simulating click areas!")
    indx_gt = generate_empty_list(dataset_numbers)
    for i in range(dataset_numbers):
        for j in range(len(simulating_point_index[i])):
            indx_tmp = groundtruth_calculation(
                label_color[i], simulating_point_index[i][j])

            indx_gt[i].append(indx_tmp)
    # test by change the color to black and save as ply file
    if(args.whether_saveply):
        for i in range(dataset_numbers):
            for j in range(len(simulating_point_index[i])):
                data = copy.deepcopy(data_obj[i])
                print("testing...")
                print("set the color value according to index...")
                data['vertex']['red'][indx_gt[i][j]] = 0
                data['vertex']['green'][indx_gt[i][j]] = 0
                data['vertex']['blue'][indx_gt[i][j]] = 0
                data = PlyData(data)
                print("saving ply...{}".format(simulating_point_index[i][j]))
                out_filename = \
                    'test_groundTruth_at_simulatingindex{}.ply'.format(
                        simulating_point_index[i][j]
                    )
                data.write(os.path.join(args.output_path, out_filename))

                out_filename1 = \
                    'test_groundTruth_at_simulatingindex{}.npz'.format(
                        simulating_point_index[i][j]
                    )
                np.savez(
                  os.path.join(
                        args.output_path, out_filename1), indx=indx_gt)
    return data_obj


if __name__ == '__main__':
    args = build_parser().parse_args()
    print(args)
    plydata = main(args)
