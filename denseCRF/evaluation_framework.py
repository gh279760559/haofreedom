"denseCRF in 3D space."
import numpy as np
# import matplotlib.pyplot as plt
# import boundary_detection_2D
# import boundary_draw
# from sklearn.preprocessing import PolynomialFeatures
import pydensecrf.densecrf as dcrf
from plyfile import PlyData
from skimage import color
from math import floor
from pathlib import Path
from scipy.spatial import KDTree
from sklearn.preprocessing import normalize
from skimage import color
from joblib import Parallel, delayed
import argparse
import os.path
import copy
import time
import sys
import ipdb


def build_parser():
    """Some setting."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        '-m', '--mesh-filepath',
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
        '--debug',
        help='Print additional debugging information.',
        dest='debug',
        action='store_true',
        default=False
    )

    return parser


def load_npz(mesh_filepath, if_test):
    """Load the npy file, if not exist will build it."""
    npyfile = Path(mesh_filepath)
    if npyfile.is_file():
        # read the ply file
        npzfile = np.load(mesh_filepath)
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
        if(if_test == 1):
            plydata = npzfile['plydata']
        # ipdb.set_trace()
    else:
        print('the mesh NPY format not exist, now load ply and save')
        plydata = PlyData.read(mesh_filepath[:-3]+'ply')
        x, y, z, r, g, b, nx, ny, nz = (plydata['vertex'].data[key]
                                        for key in ('x', 'y', 'z',
                                                    'red', 'green', 'blue',
                                                    'nx', 'ny', 'nz'))
        vertices_pos = np.column_stack((x, y, z))
        vertices_normals = np.column_stack((nx, ny, nz))
        vertices_color = np.column_stack((r, g, b))
        np.savez(
            mesh_filepath[:-4], vertices=vertices_pos, rgb=vertices_color,
            vertices_normals=vertices_normals,
            plydata=plydata)
    if(if_test == 1):
        return [vertices_pos, vertices_color, vertices_normals, plydata]
    else:
        return [vertices_pos, vertices_color, vertices_normals]

# not use at the moment.
# def feature_norm(features, normmethod=2, debug=False):
#     """Feature normlization."""
#     if normmethod == 1:
#         # one way to normalise: standardization
#         features -= np.mean(features, axis=0)
#         # if debug:
#         #     print_stats(features)
#         features /= np.std(features, axis=0)
#         # if debug:
#         #     print_stats(features)
#     elif normmethod == 2:
#         # ipdb.set_trace()
#         # another way: scaling to unit
#         if(features.shape[1] == 1):
#             features = normalize(features, axis=0)
#         else:
#             features = normalize(features, axis=1)
#     elif normmethod == 3:
#         features -= np.mean(features, axis=0)
#         features /= np.std(features, axis=0)
#         if(features.shape[1] == 1):
#             features = normalize(features, axis=0)
#         else:
#             features = normalize(features, axis=1)
#     else:
#         import sys
#         sys.exit('normmethod para is not right (either 1 or 2)')
#     return features


def label_simulation_area(label_length,
                          simulating_point_index,
                          segment_nums):
    """Simulating click areas and return labels."""
    click_simulation_area = simulating_point_index
    # manually set the labels
    labels_numeric = np.zeros(label_length).astype(np.int32)
    labels_numeric[click_simulation_area] = 1

    prob = 1e-19
    labels = np.zeros((segment_nums, label_length))+prob
    labels = -np.log(labels)

    # suppose click means 1st label
    labels[1, click_simulation_area] = -np.log((1-prob)/segment_nums)
    return labels, labels_numeric


def run_inference(model, steps, debug=False):
    """Inference."""
    if debug:
        Q, tmp1, tmp2 = model.startInference()
        for i in range(steps):
            print("KL-divergence at {}: {}".format(i, model.klDivergence(Q)))
            model.stepInference(Q, tmp1, tmp2)
    else:
        Q = model.inference(steps)
    return Q


def unif(range):
    """Return random value between range."""
    return np.random.uniform(*range)


def get_random_params():
    """Set the range for all parameters."""
    return {
        'posi_weight': unif([0, 1]),
        'normal_weight': unif([0, 1]),
        'color_weight': unif([0, 1]),
        'gaussian_weight': unif([0, 10])
    }


def score_get(prediction, indx):
    """Calculate the score."""
    score = np.array([])
    detected_indx = np.where(prediction != 0)[0]
    union_part = np.union1d(detected_indx, indx)
    intersection_part = np.intersect1d(detected_indx, indx)
    score = np.append(
        score, len(intersection_part) / len(union_part))
    return score, detected_indx


def eval_para(U, face_num, paras, seg_nums,
              positions, normals, colors,
              bilateral_weight, whether_use_bilateral):
    """Tune parameters."""
    d = dcrf.DenseCRF(face_num, seg_nums)
    # print("set Unary...")
    d.setUnaryEnergy(U)
    # print("add pairwise Energy...")
    features_gaussian = np.concatenate(
        (positions/paras['posi_weight'],
         normals/paras['normal_weight'],
         colors/paras['color_weight']
         ), axis=1)
    if(whether_use_bilateral == 1):
        features_bilateral = np.concatenate(
            (positions/paras['posi_weight'],
             normals/paras['normal_weight']), axis=1)
    d.addPairwiseEnergy(
        np.ascontiguousarray(features_gaussian.T, dtype=np.float32),
        compat=paras['gaussian_weight'], kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC
    )
    if(whether_use_bilateral == 1):
        d.addPairwiseEnergy(
            np.ascontiguousarray(features_bilateral.T, dtype=np.float32),
            compat=bilateral_weight, kernel=dcrf.DIAG_KERNEL,
            normalization=dcrf.NORMALIZE_SYMMETRIC
        )
    # print("inference...")
    Q = run_inference(d, 15)
    prediction = np.argmax(Q, axis=0)
    return prediction


def groundtruth_calculation(colors, simulating_point_index):
    """Return the index of points that have same color."""
    color_on_index = colors[simulating_point_index, :]
    tmp1 = np.where(colors[:, 0] == color_on_index[0])
    tmp2 = np.where(colors[:, 1] == color_on_index[1])
    tmp3 = np.where(colors[:, 2] == color_on_index[2])
    tmp4 = np.intersect1d(tmp1, tmp2)
    indx = np.intersect1d(tmp4, tmp3)
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
    whether_use_bilateral = 0
    simulating_point_index = []
    # 093
    simulating_point_index.append([972119, 975828, 1140442, 491021, 1204595])
    # 207
    simulating_point_index.append([313301, 587101, 948647, 733043, 367594])
    # can add more
    # simulating_point_index.append
    # simulating_point_index.append
    # simulating_point_index.append
    # simulating_point_index.append
    bilateral_weight = 1
    if_test = 0

    print("load the npy file or save as npyfile...")
    print("make sure 1,3,5... is true color one and 2,4,6... is label one")
    positions, colors, normals, data_obj, label_color = [], [], [], [], []
    mesh_filepath_nums = len(args.mesh_filepath)
    dataset_numbers = mesh_filepath_nums/2
    # test
    if(dataset_numbers != len(simulating_point_index)):
        sys.exit(
            "the dataset numbers are not \
            matching the simulating numbers, line 284")
    elif(mesh_filepath_nums % 2 != 0):
        sys.exit("check the mesh_filepath arguments numbers, line 286")
    else:
        dataset_numbers = np.int(dataset_numbers)
    for i in range(0, mesh_filepath_nums):
        if(i % 2 == 0):
            if(if_test):
                [positions_tmp, colors_tmp,
                    normals_tmp, _] = load_npz(args.mesh_filepath[i], if_test)
            else:
                [positions_tmp, colors_tmp,
                    normals_tmp] = load_npz(args.mesh_filepath[i], if_test)
            positions.append(positions_tmp)
            colors.append(colors_tmp)
            normals.append(normals_tmp)
        elif(i % 2 == 1):
            [_, label_color_tmp, _] = load_npz(args.mesh_filepath[i], if_test)
            label_color.append(label_color_tmp)
            if(if_test):
                [_, _, _,
                    dataobj_tmp] = load_npz(args.mesh_filepath[i], if_test)
                data_obj.append(dataobj_tmp)
        else:
            sys.exit("error 297.!")

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
    if(if_test):
        for i in range(dataset_numbers):
            for j in range(len(simulating_point_index[i])):
                data = copy.deepcopy(data_obj[i])
                print("testing...")
                print("set the color value according to index...")
                data[0]['red'][indx_gt[0][j]] = 0
                data[0]['green'][indx_gt[0][j]] = 0
                data[0]['blue'][indx_gt[0][j]] = 0
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

    # possibly normalize the features?
    face_num = np.array([])
    for i in range(dataset_numbers):
        if (len(positions[i]) == len(colors[i]) == len(normals[i])):
            face_num_tmp = len(positions[i])
            if(if_test):
                if(int(face_num_tmp) - face_num_tmp != 0):
                    sys.exit('wrong line 357.')
            face_num = np.append(face_num, face_num_tmp)

        else:
            sys.exit("the length is different, check code 361!")

    print("get simulated labels...")

    # indx = generate_empty_list(dataset_numbers)
    Unary = generate_empty_list(dataset_numbers)
    for i in range(dataset_numbers):
        for j in range(len(simulating_point_index[i])):
            import random
            picked_val = random.choice(indx_gt[i][j])
            labels, labels_numeric = label_simulation_area(
                face_num[i].astype(np.int), picked_val,
                args.n_segs)
            Unary[i].append(labels.astype(np.float32))

    print("Now tuning...")
    # ipdb.set_trace()
    loop_time = 1000
    satisfied_percentage = 0.9
    eval_result = 0
    loop_indx = 1
    score = np.array([])
    paras = np.array([])
    debug = False
    while eval_result < satisfied_percentage or loop_indx < loop_time:
        # can be improved using parralle joblib
        # n_segs = args.n_segs
        # parallelise = Parallel(8)
        # task_it = (delayed(eval_para)(
        #     U[j], face_num, n_segs,
        #     positions, normals, colors,
        #     bilateral_weight, whether_use_bilateral
        # ) for j in range(len(simulating_point_index)))
        # tmp = parallelise(task_it)
        # paras_tmp1, prediction_tmp1 = zip(*tmp)
        score = generate_empty_list(dataset_numbers)
        detected_indx = generate_empty_list(dataset_numbers)
        paras = get_random_params()
        print('Now we start the loop {}'.format(loop_indx))
        for i in range(dataset_numbers):
            for j in range(len(simulating_point_index[i])):
                prediction_tmp = eval_para(
                    Unary[i][j], face_num[i].astype(np.int),
                    paras, args.n_segs,
                    positions[i], normals[i], colors[i],
                    bilateral_weight, whether_use_bilateral)
                score_tmp, detected_indx_tmp = score_get(
                    prediction_tmp, indx_gt[i][j])
                score[i].append(score_tmp)
                detected_indx[i].append(detected_indx_tmp)
        ipdb.set_trace()
        score_avg = np.mean(score)
        print("the score is {}".format(score_avg))
        print("the paras is {}".format(paras))
        score = np.append(score, score_avg)
        paras = np.append(paras, paras)
        if(debug):
            saved_name = 'results_{}'.format(
                simulating_point_index
            )
            saved_name1 = 'resultsParas_{}'.format(
                simulating_point_index
            )
            np.save(saved_name, score)
            np.save(saved_name1, paras)
        loop_indx += 1


if __name__ == '__main__':
    args = build_parser().parse_args()
    print(args)
    main(args)
