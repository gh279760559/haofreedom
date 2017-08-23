"""denseCRF in 3D space."""
import numpy as np
import argparse
import os.path
import copy
import random
import json
import xml.etree.ElementTree as etree
from plyfile import PlyData
from skimage import color
from pathlib import Path
from operator import attrgetter

import pydensecrf.densecrf as dcrf


def build_parser():
    """Input arguments."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        '-f', '--filename',
        help='Name of json file to load',
        type=str,
        default='parameters.json'
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
        '-t', '--if-test',
        help=(
            'if save each result as ply'
        ),
        dest='if_test',
        type=int,
        default=0  # Four walls, ceiling, floor.
    )

    parser.add_argument(
        '-b', '--if-bilateral',
        help=(
            'whether using bilateral pairwise'
        ),
        dest='whether_use_bilateral',
        type=int,
        default=1  # Four walls, ceiling, floor.
    )

    return parser


def load_json():
    """Load from json file."""
    args = build_parser().parse_args()
    assert os.path.isfile(args.filename), (
        'File {} does not exist!'.format(args.filename)
    )
    with open(args.filename) as fd:
        data = json.load(fd)
    mesh_filepath = []
    xml_filepath = []
    object_num_used = []
    output_path = []
    ply_num = len(data)
    for i in (range(ply_num)):
        mesh_filepath.append(
            path_join(data[i]["mesh_path"], data[i]["ply_name"]))
        xml_filepath.append(
            path_join(data[i]["mesh_path"], data[i]["xml_name"]))
        object_num_used.append(data[i]["object_num_used"])
        output_path.append(data[i]["output_path"])
    return [
        mesh_filepath, xml_filepath, output_path, object_num_used, ply_num
        ]


def load_xml(xml_data, object_num_used, if_sorted):
    """Load xml file and get area and id."""
    xmlData = etree.parse(xml_data)
    root = xmlData.getroot()
    if(if_sorted == 0):
        for node in root.findall("*"):
            node[:] = sorted(node, key=attrgetter("area"))
    label_id = np.array([])
    for i in range(object_num_used):
        label_id = np.append(label_id, np.int(root[i].attrib['id']))
    label_id = label_id.astype(int)
    return label_id


def path_join(*args):
    """Easy use path join."""
    return os.path.join(*args)


def file_measurement(mesh_filepath):
    """Some basic file measurement."""
    filename = os.path.basename(mesh_filepath)
    filename_no_format = os.path.splitext(filename)[0]
    directories = os.path.dirname(os.path.realpath(mesh_filepath))
    npzfile_name = filename_no_format + '.npz'
    plyfile_name = filename_no_format + '.ply'
    return [directories, filename_no_format, npzfile_name, plyfile_name]


def column_stack(*args):
    """Easy use np column stack."""
    return np.column_stack(*args)


def load_npz(mesh_filepath, output_path):
    """Load the npy file, if not exist will build it."""
    [
        directories, filename_no_format, npzfile_name, plyfile_name
        ] = file_measurement(mesh_filepath)
    npzfile_path = path_join(output_path, npzfile_name)
    if Path(npzfile_path).is_file():
        print('the mesh NPY format exists, now loading...')
        # read the ply file
        npzfile = np.load(npzfile_path)
        vertices_pos = npzfile['vertices']
        vertices_color = npzfile['rgb']
        vertices_color = color.rgb2lab(
            vertices_color[np.newaxis, ...]).squeeze()
        vertices_normals = npzfile['vertices_normals']
        label = npzfile['label']
        plydata = PlyData.read(
            path_join(directories, plyfile_name))
    else:
        print('the mesh NPY format not exist, now load ply and save')

        plydata = PlyData.read(path_join(directories, plyfile_name))
        if("label" in plydata['vertex'].data.dtype.names):
            label = plydata['vertex'].data['label']
        else:
            print("the ply is modified version (not from sceneNN\n" +
                  "check what is the data name of label saved\n" +
                  "now we try if it is from CloudCompare)")
            label = plydata['vertex'].data['scalar_label']
        vertices_pos = column_stack((
            plydata['vertex'].data['x'],
            plydata['vertex'].data['y'],
            plydata['vertex'].data['z'],
        ))
        vertices_normals = column_stack((
            plydata['vertex'].data['nx'],
            plydata['vertex'].data['ny'],
            plydata['vertex'].data['nz'],
        ))
        vertices_color = column_stack((
            plydata['vertex'].data['red'],
            plydata['vertex'].data['green'],
            plydata['vertex'].data['blue'],
        ))
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        np.savez(
            npzfile_path,
            vertices=vertices_pos, rgb=vertices_color,
            vertices_normals=vertices_normals, label=label)

    return [vertices_pos, vertices_color, vertices_normals,
            label, plydata]


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


def get_random_params(whether_use_bilateral):
    """Set the range for all parameters."""
    if(whether_use_bilateral):
        return {
            'posi_weight': np.random.uniform(0, 1),
            'normal_weight': np.random.uniform(0, 1),
            'color_weight': np.random.uniform(0, 1),
            'gaussian_weight': np.random.uniform(0, 10),
            'bilateral_weight': np.random.uniform(0, 10)
            }
    else:
        return {
            'posi_weight': np.random.uniform(0, 1),
            'normal_weight': np.random.uniform(0, 1),
            'color_weight': np.random.uniform(0, 1),
            'gaussian_weight': np.random.uniform(0, 1)
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
              whether_use_bilateral):
    """Tune parameters."""
    d = dcrf.DenseCRF(face_num, seg_nums)
    # print("set Unary...")
    d.setUnaryEnergy(U)
    # print("add pairwise Energy...")

    if(whether_use_bilateral == 1):
        features_gaussian = np.concatenate(
            (positions/paras['posi_weight'],
             colors/paras['color_weight']
             ), axis=1)
        features_bilateral = np.concatenate(
            (positions/paras['posi_weight'],
             normals/paras['normal_weight']), axis=1)
    else:
        features_gaussian = np.concatenate(
            (positions/paras['posi_weight'],
             normals/paras['normal_weight'],
             colors/paras['color_weight']
             ), axis=1)
    d.addPairwiseEnergy(
        np.ascontiguousarray(features_gaussian.T, dtype=np.float32),
        compat=paras['gaussian_weight'], kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC
    )
    if(whether_use_bilateral == 1):
        d.addPairwiseEnergy(
            np.ascontiguousarray(features_bilateral.T, dtype=np.float32),
            compat=paras['gaussian_weight'], kernel=dcrf.DIAG_KERNEL,
            normalization=dcrf.NORMALIZE_SYMMETRIC
        )
    # print("inference...")
    Q = run_inference(d, 3)
    prediction = np.argmax(Q, axis=0)
    return prediction


def groundtruth_calculation(labels, label_id):
    """Return the index of points that have same color."""
    indx = np.where(labels == label_id)[0]
    return indx


def generate_empty_list(list_size):
    """Generate the N-D empty list with size N=list_size."""
    tmp = []
    for i in range(list_size):
        tmp.append([])
    return tmp


def save_to_ply(ply_data, indx, output_path):
    """Set the color of the index area to 0."""
    data = copy.deepcopy(ply_data)
    data['vertex']['red'] = 255
    data['vertex']['green'] = 0
    data['vertex']['blue'] = 0
    data['vertex']['red'][indx] = 0
    data['vertex']['green'][indx] = 255
    data['vertex']['blue'][indx] = 0
    data = PlyData(data)
    output_path_ply = output_path + '.ply'
    data.write(output_path_ply)
    output_path_npz = output_path + '.npz'
    np.savez(output_path_npz, indx=indx)


def main(args):
    """Main function."""
    print('Loading the json file...')
    [
        mesh_filepath, xml_filepath, output_path, object_num_used, ply_num
    ] = load_json()
    print('get the lable id...')
    label_id = []
    for i in range(ply_num):
        label_id.append(
            load_xml(xml_filepath[i], object_num_used[i], 0))
    whether_use_bilateral = 1
    # bilateral_weight = 1
    if_test = 0
    # for the training
    loop_time = 100
    satisfied_percentage = 0.9

    print("load the npy file or save as npyfile...")
    positions, colors, normals, data_obj, label = [], [], [], [], []
    # test
    assert ply_num == len(label_id), (
        "the dataset numbers are not matching the simulating numbers," +
        " it happened at line 402")
    # test_end

    for i in range(0, ply_num):
            [positions_tmp, colors_tmp,
                normals_tmp, label_tmp,
                dataobj_tmp] = load_npz(
                    mesh_filepath[i], output_path[i])
            label.append(label_tmp)
            positions.append(positions_tmp)
            colors.append(colors_tmp)
            normals.append(normals_tmp)
            data_obj.append(dataobj_tmp)
    print('now get the groundtruth labelling...')
    # get all simulating click areas!
    print("get all simulating click areas!")
    indx_gt = generate_empty_list(ply_num)
    for i in range(ply_num):
        for j in range(len(label_id[i])):
            indx_tmp = groundtruth_calculation(
                label[i], label_id[i][j])
            indx_gt[i].append(indx_tmp)

    # test by change the color to black and save as ply file
    if(if_test):
        print("Now test by save as ply and label_id area to black...")
        for i in range(ply_num):
            for j in range(len(label_id[i])):
                out_filename = 'test_groundTruth_at_label{}'.format(
                    label_id[i][j])
                output_name = path_join(output_path[i], out_filename)
                save_to_ply(
                    data_obj[i], indx_gt[i][j], output_name)

    # possibly normalize the features?
    # get face_num
    face_num = np.array([])
    for i in range(ply_num):
        assert len(positions[i]) == len(colors[i]) == len(normals[i]), (
            "postion, color and normal length are different, line 387"
        )
        face_num_tmp = len(positions[i])
        assert int(face_num_tmp) - face_num_tmp == 0, (
            "face_num_tmp is not integer line 393"
        )
        face_num = np.append(face_num, face_num_tmp)

    print("get simulated labels...")

    print("Now tuning...")

    eval_result = 0
    loop_indx = 0
    score = np.array([])
    paras = np.array([])
    # this setting is a tricky thing
    # want to stop the running and save the results by
    # manually change the argument
    prediction = generate_empty_list(loop_time)
    labels_numeric = generate_empty_list(loop_time)
    print(
        "Now we start training, the satisfied percentage is {}".format(
            satisfied_percentage))
    print("The maximum loop time is {}".format(loop_time))
    while eval_result < satisfied_percentage and loop_indx < loop_time:
        # can be improved using parralle joblib
        paras_tmp = get_random_params(whether_use_bilateral)
        score_tmp1 = generate_empty_list(ply_num)
        print("totally {} plys".format(ply_num))
        for i in range(ply_num):
            print('Now we start the data {} in loop {}'.format(
                i+1, loop_indx+1))
            for j in range(len(label_id[i])):
                picked_val = random.choice(indx_gt[i][j])
                labels, labels_numeric_tmp = label_simulation_area(
                    face_num[i].astype(np.int), picked_val,
                    args.n_segs)
                Unary = labels.astype(np.float32)
                prediction_tmp = eval_para(
                    Unary, face_num[i].astype(np.int),
                    paras_tmp, args.n_segs,
                    positions[i], normals[i], colors[i],
                    whether_use_bilateral)
                score_tmp, _ = score_get(
                    prediction_tmp, indx_gt[i][j])
                score_tmp1[i].append(score_tmp)
                prediction[loop_indx].append(prediction_tmp)
                labels_numeric[loop_indx].append(labels_numeric_tmp)
                out_filename = 'parameters{}labels{}'.format(
                    paras_tmp, label_id[i][j])
                output_name = path_join(output_path[i], out_filename)
                tmp = np.where(prediction_tmp != 0)[0]
                if(if_test):
                    print("in test mode, save the prediction as ply" +
                          "currently in loop {} ply {} object {},".format(
                              loop_indx+1, i+1, j+1) +
                          "max is: loop {}, ply {}".format(loop_time, ply_num)
                          + " and object {} in this ply".format(
                              len(label_id[i])))
                    save_to_ply(data_obj[i], tmp, output_name)
        score_avg = np.mean(score_tmp1)
        print("the score is {}".format(score_avg))
        print("the paras is {}".format(paras_tmp))
        score = np.append(score, score_avg)
        paras = np.append(paras, paras_tmp)
        loop_indx += 1

    print('now save the scores and paras...')
    saved_name = 'results_{}'.format(
        label_id
    )
    saved_name1 = 'resultsParas_{}'.format(
        label_id
    )
    np.save(saved_name, score)
    np.save(saved_name1, paras)
    print('now save the best one as ply...')
    indx_tmp = np.argmax(score)
    prediction_ply = prediction[indx_tmp]
    indx = 0
    for i in range(ply_num):
        for j in range(len(label_id[i])):
            tmp = np.where(prediction_ply[indx] != 0)[0]
            out_filename = 'scores{}labels{}plynum{}'.format(
                score[indx_tmp], label_id[i][j], i+1)
            save_to_ply(data_obj[i], tmp, out_filename)
            indx += 1

    return [score, paras, prediction, data_obj]


if __name__ == '__main__':
    args = build_parser().parse_args()
    print(args)
    [score, paras, prediction, data_obj] = main(args)
