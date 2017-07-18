"""Simulating human click and use denseCRF to do segmentation."""
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
# from scipy.spatial.distance import cdist
# from sklearn.cluster import MiniBatchKMeans
import pydensecrf.densecrf as dcrf
from plyfile import PlyData
from skimage import color
from math import floor
from numbers import Number  # for pydensecrf
import numpy as np
import argparse
import os.path
import copy
import time
import ipdb


def build_parser():
    """Some setting."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        '-m', '--mesh-filepath',
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
            'Used as K for K-Means and label count for DCRF.'
        ),
        dest='n_segs',
        type=int,
        default=2  # Four walls, ceiling, floor.
    )

    parser.add_argument(
        '-p', '--proportion-keep',
        help='Specify the volume of labels to pass to DCRF.',
        dest='proportion_keep',
        type=float,
        default=0.2
    )

    parser.add_argument(
        '--debug',
        help='Print additional debugging information.',
        dest='debug',
        action='store_true',
        default=False
    )

    return parser


def check_args(args):
    """Input checking."""
    assert os.path.isfile(args.mesh_filepath), 'Mesh file must exist.'

    assert 1.0 >= args.proportion_keep > 0.0, (
        'Keep proportion defined in (0, 1].'
    )

    assert args.n_segs > 0, 'Number of segments defined in positive integers.'


def load_mesh(mesh_filepath):
    """Initially load mesh."""
    data = PlyData.read(mesh_filepath)

    faces, vertices = data['face'], data['vertex']

    return faces, vertices, data


# Given a face, get the vertices which bound it
def get_face_vertices(face, vertices):
    """Get the vertices according to the face."""
    component_vertex_indices = face['vertex_indices']
    return list(map(lambda idx: vertices[idx], component_vertex_indices))


# Get the arithmetic mean for a set of attributes and a face
def get_face_position_mean(face, vertices, keys):
    """Calculate the position means according to the vertices."""
    face_vertices = get_face_vertices(face, vertices)
    vertex_coords = np.asarray(
        [
            face_vertex[key]
            for face_vertex in face_vertices
            for key in keys
        ]
    )
    # # # # # # # ipdb.set_trace()
    vertex_coords = np.reshape(vertex_coords, (len(face_vertices), len(keys)))
    # # # # # ipdb.set_trace()
    return np.mean(vertex_coords, axis=0)


# Get the arithmetic mean for a set of attributes and a face
def get_face_normal_mean(face, vertices, keys):
    """Calculate the normal means according to the vertices."""
    face_vertices = get_face_vertices(face, vertices)
    vertex_coords = np.asarray(
        [
            face_vertex[key]
            for face_vertex in face_vertices
            for key in keys
        ]
    )
    # # # # # # # ipdb.set_trace()
    vertex_coords = np.reshape(vertex_coords, (len(face_vertices), len(keys)))
    # # # # # ipdb.set_trace()
    return np.mean(vertex_coords, axis=0)


# Function gets normal of face via cross product of vectors in plane
def get_face_normal(face, vertices, keys):
    """Get positions according to face and vertices."""
    keys = ['nx', 'ny', 'nz']
    face_vertices = get_face_vertices(face, vertices)
    vertex_coords = np.asarray(
        [
            face_vertex[key]
            for face_vertex in face_vertices
            for key in keys
        ]
    )
    vertex_coords = np.reshape(vertex_coords, (len(face_vertices), len(keys)))
    return np.cross(vertex_coords[1] - vertex_coords[0],
                    vertex_coords[2] - vertex_coords[0])


def get_face_gray(face, vertices, keys):
    """Calculate the rgb means according to the vertices."""
    face_vertices = get_face_vertices(face, vertices)
    vertex_coords = np.asarray(
        [
            face_vertex[key]
            for face_vertex in face_vertices
            for key in keys
        ]
    )
    vertex_coords = np.reshape(vertex_coords, (len(face_vertices), len(keys)))
    # # # # # # # ipdb.set_trace()
    colorValue = 0.21*np.mean(
        vertex_coords[:, 0]) \
        + 0.72*np.mean(vertex_coords[:, 1]) + 0.07*np.mean(vertex_coords[:, 2])
    return colorValue


# Get the arithmetic mean for a set of attributes and a face
def get_face_rgb_mean(face, vertices, keys):
    """Calculate the color means according to the vertices."""
    face_vertices = get_face_vertices(face, vertices)
    vertex_coords = np.asarray(
        [
            face_vertex[key]
            for face_vertex in face_vertices
            for key in keys
        ]
    )
    # # # # # # # ipdb.set_trace()
    vertex_coords = np.reshape(vertex_coords, (len(face_vertices), len(keys)))
    # # # # # ipdb.set_trace()
    return np.mean(vertex_coords, axis=0)


# Wrapper to get centre of face
def get_face_pos_centroid(face, vertices):
    """Get positions according to face and vertices."""
    keys = 'xyz'
    return get_face_position_mean(face, vertices, keys)


# Wrapper to get vertex colour
def get_face_colour_centroid(face, vertices):
    """Get colors according to face and vertices."""
    keys = ['red', 'green', 'blue']
    return get_face_gray(face, vertices, keys)


# Wrapper to get face normal, by average of vertices
def get_face_normal_as_mean(face, vertices):
    """Get mean normals according to face and vertices."""
    keys = ['nx', 'ny', 'nz']
    return get_face_normal_mean(face, vertices, keys)


# Depending on what features we use (normals, position etc),
# the final feature vector will be a different length.
#
# Can compute polynomial features from the basic features.
def get_features(
        faces, vertices, pos_para, normal_para, color_para, poly_deg=1
        ):
    """Feature calculation."""
    # A bit dirty.
    # Maps face to feature vector.
    def get_face_feature(face):
        return np.r_[
            # get_face_normal(face, vertices),
            get_face_pos_centroid(face, vertices),
            get_face_normal_as_mean(face, vertices),
            # get_face_colour_centroid(face, vertices)
        ]
    # loop
    features = np.asarray(list(map(get_face_feature, faces)))
    # ipdb.set_trace()
    features_centre_position = features[:, 0:3]/pos_para
    features_centre_normal = features[:, 3:6]/normal_para
    # features_centre_color = features[:, 6]/color_para
    # # ipdb.set_trace()
    # make it n by 1 so that for concatenate
    # features_centre_color = features_centre_color[..., np.newaxis]
    # features_gaussian = features_centre_normal
    # features_gaussian = features_centre_position
    features_gaussian = np.concatenate(
        (features_centre_position, features_centre_normal), axis=1
    )
    # features_bilateral = np.concatenate(
    #     (features_centre_position, features_centre_normal), axis=1
    # )
    features_bilateral = features_centre_position
    # ipdb.set_trace()
    # features_bilateral = features_centre_normal
    # # # # # # ipdb.set_trace()

    if poly_deg == 1:
        return features, features_gaussian, features_bilateral
    else:
        poly = PolynomialFeatures(poly_deg, include_bias=False)
        feature = poly.fit_transform(features)
        features_gaussian = poly.fit_transform(features_gaussian)
        features_bilateral = poly.fit_transform(features_bilateral)
        # # # # # # ipdb.set_trace()
        return feature, features_gaussian, features_bilateral


def print_stats(feats):
    """Print stuff."""
    print("Mean")
    print(np.mean(feats, axis=0))
    print("Min")
    print(np.min(feats, axis=0))
    print("Max")
    print(np.max(feats, axis=0))
    print("Std")
    print(np.std(feats, axis=0))


def feature_norm(features, debug=False):
    """Feature normlization."""
    features -= np.mean(features, axis=0)
    if debug:
        print_stats(features)

    features /= np.std(features, axis=0)
    if debug:
        print_stats(features)

    return features


# Get 'negative log probs' for p(y|f) where y is a labelling and f is a face.
# Not true probabilities, computed from one-hot encoding of clustered labels
def get_unary_potentials(labels):
    """Unary potential."""
    encoder = OneHotEncoder(sparse=False)
    one_hot_labels = encoder.fit_transform(labels.reshape(-1, 1)).T

    eps = 1e-12
    one_hot_labels = -np.log(one_hot_labels + eps)
    return np.ascontiguousarray(one_hot_labels, dtype=np.float32)


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


def get_n_linear_colours_rgb(n):
    """Get n linear spaced rgb colours (for label visualisation)."""
    # Evenly spaces hues
    hues = np.linspace(0, 1, n + 1)

    # Saturation and value at full
    pad_s_v = np.ones(n + 1)

    # Concatenate and reshape to form swatch image
    padded = np.stack((hues, pad_s_v, pad_s_v))
    hsv = padded[..., None].T

    # Convert swatch image to RGB
    rgb = color.hsv2rgb(hsv).squeeze()[:n, :]

    # Squeeze to list of rgb values
    return rgb


def get_visualisation(num_classes, prediction):
    """visualisation."""
    colours_for_vis = get_n_linear_colours_rgb(num_classes)
    return list(map(lambda i: colours_for_vis[i], prediction))


def get_labelled_mesh(data, labels):
    """Get label mesh."""
    data = copy.deepcopy(data)
    vertices = data['vertex']
    faces = data['face']
    channels = ['red', 'green', 'blue']
    for face_idx, face in enumerate(faces):
        face_vertices = get_face_vertices(face, vertices)
        for vert_index in range(0, 2):
            for channel_idx, channel in enumerate(channels):
                face_vertices[vert_index][channel] = floor(
                    labels[face_idx][channel_idx] * 255
                )
    return data


def vis_mask(visualisation, faces_to_discard):
    """Mask."""
    visualisation = copy.deepcopy(visualisation)
    for i in faces_to_discard:
        visualisation[i] = [1., 1., 1.]
    return visualisation


def get_init_and_labelled_meshes(prediction, data_obj, labels_numeric, n_segs):
    """Name show."""
    # labels_numeric[:] = 0
    # labels_numeric[:len(labels_numeric)//2] = 1
    pred_vis = get_visualisation(n_segs, prediction)
    labelled_data = get_labelled_mesh(data_obj, pred_vis)

    init_vis = get_visualisation(n_segs, labels_numeric)
    # # # # ipdb.set_trace()
    # init_vis = vis_mask(init_vis, [])
    init_data = get_labelled_mesh(data_obj, copy.deepcopy(init_vis))
    # # # # ipdb.set_trace()
    return init_data, labelled_data


def create_pairwise_gaussian(sdims, shape):
    """From pydensecrf.

    Util function that create pairwise gaussian potentials. This works for all
    image dimensions. For the 2D case does the same as
    `DenseCRF2D.addPairwiseGaussian`.

    Parameters
    ----------
    sdims: list or tuple
        The scaling factors per dimension. This is referred to `sxy` in
        `DenseCRF2D.addPairwiseGaussian`.
    shape: list or tuple
        The shape the CRF has.

    """
    # create mesh
    # # ipdb.set_trace()
    hcord_range = [range(s) for s in shape]
    mesh = np.array(np.meshgrid(*hcord_range, indexing='ij'), dtype=np.float32)

    # scale mesh accordingly
    for i, s in enumerate(sdims):
        mesh[i] /= s
    return mesh.reshape([len(sdims), -1])


def create_pairwise_bilateral(sdims, schan, img, chdim=-1):
    """From pydensecrf.

    Util function that create pairwise bilateral potentials. This works for
    all image dimensions. For the 2D case does the same as
    `DenseCRF2D.addPairwiseBilateral`.

    Parameters
    ----------
    sdims: list or tuple
        The scaling factors per dimension. This is referred to `sxy` in
        `DenseCRF2D.addPairwiseBilateral`.
    schan: list or tuple
        The scaling factors per channel in the image. This is referred to
        `srgb` in `DenseCRF2D.addPairwiseBilateral`.
    img: numpy.array
        The input image.
    chdim: int, optional
        This specifies where the channel dimension is in the image. For
        example `chdim=2` for a RGB image of size (240, 300, 3). If the
        image has no channel dimension (e.g. it has only one channel) use
        `chdim=-1`.

    """
    # Put channel dim in right position
    if chdim == -1:
        # We don't have a channel, add a new axis
        im_feat = img[np.newaxis].astype(np.float32)
    else:
        # Put the channel dim as axis 0, all others stay relatively the same
        im_feat = np.rollaxis(img, chdim).astype(np.float32)

    # scale image features per channel
    # Allow for a single number in `schan` to broadcast across all channels:
    if isinstance(schan, Number):
        im_feat /= schan
    else:
        for i, s in enumerate(schan):
            im_feat[i] /= s

    # create a mesh
    cord_range = [range(s) for s in im_feat.shape[1:]]
    mesh = np.array(np.meshgrid(*cord_range, indexing='ij'), dtype=np.float32)

    # scale mesh accordingly
    for i, s in enumerate(sdims):
        mesh[i] /= s

    feats = np.concatenate([mesh, im_feat])
    return feats.reshape([feats.shape[0], -1])



def main(args):
    """Main function."""
    check_args(args)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    print('Loading')
    # theta_alpha, beta, gamma setting, depending on the structure of
    # the pairwise setting.
    # 0.05, 0.1, 20
    pos_para, normal_para, color_para = 1, 1, 1
    weight_gaussian = 30
    weight_bilateral = 10
    inference_time = 15
    poly_deg = 1  # try 2 will polynomial
    simulating_point_index = 22000
    faces, vertices, data_obj = load_mesh(args.mesh_filepath)

    features, features_gaussian, features_bilateral = get_features(
        faces, vertices, pos_para, normal_para, color_para, poly_deg
    )

    # normalization
    # # ipdb.set_trace()
    features = feature_norm(features, debug=False)
    # features_gaussian = feature_norm(features_gaussian, debug=False)
    # features_bilateral = feature_norm(features_bilateral, debug=False)

    # manually set the labels
    labels_numeric = np.zeros(features.shape[0]).astype(np.int32)
    # ipdb.set_trace()
    from scipy.spatial import KDTree
    tree = KDTree(features)
    _, ind_tmp = tree.query(features[simulating_point_index], k=100)
    click_simulation_area = np.append(simulating_point_index, ind_tmp)
    labels_numeric[click_simulation_area] = 1
    # click_simulation_area = np.array(range(22000, 22200))
    # labels_numeric[click_simulation_area] = 1

    prob = 1e-19
    labels = np.zeros((args.n_segs, features.shape[0]))+prob
    labels = -np.log(labels)
    # labels[:] = -np.log(1 / (args.n_segs + 0.) + 1e-8)

    # suppose click means 1st label
    labels[0, click_simulation_area] = -np.log((1-prob)/args.n_segs)
    # labels[1] = 0
    # labels[1, 1000] = -np.log(1)
    U = labels.astype(np.float32)

    # labels = get_unary_potentials(labels_numeric)
    # U = unary_from_labels(
    #    labels_numeric, args.n_segs, gt_prob=0.99, zero_unsure=False
    # )
    # U = labels
    # # # ipdb.set_trace()

    # set unary
    # HAS_UNK = 1
    # # # # # ipdb.set_trace()
    # U = get_unary_potentials(labels)
    # U = unary_from_labels(
    #    labels, args.n_segs, gt_prob=0.7, zero_unsure=HAS_UNK
    # )

    # Now start CRF
    print('CRF')
    d = dcrf.DenseCRF(faces.count, args.n_segs)

    # # # # # # ipdb.set_trace()
    d.setUnaryEnergy(U)

    # d.addPairwiseEnergy(
    #     np.ascontiguousarray(features.T, dtype=np.float32),
    #     compat=0.0003
    # )
    for i in range(1, 100):
        weight_gaussian = i
        print('lets try weight_gaussian = {}'.format(weight_gaussian))
        d.addPairwiseEnergy(
            np.ascontiguousarray(features_gaussian.T, dtype=np.float32),
            compat=weight_gaussian, kernel=dcrf.DIAG_KERNEL,
            normalization=dcrf.NORMALIZE_SYMMETRIC
        )  # 10
        # #
        d.addPairwiseEnergy(
            np.ascontiguousarray(features_bilateral.T, dtype=np.float32),
            compat=weight_bilateral, kernel=dcrf.DIAG_KERNEL,
            normalization=dcrf.NORMALIZE_SYMMETRIC
        )  # 3

        Q = run_inference(d, inference_time)

        prediction = np.argmax(Q, axis=0)

        print('Generating visualisations')
        # # # # ipdb.set_trace()
        init_data, labelled_data = get_init_and_labelled_meshes(
            prediction, data_obj, labels_numeric, args.n_segs
        )

        # # # ipdb.set_trace()
        filename = os.path.basename(args.mesh_filepath)
        stamp = str(int(time.time()))
        out_filename = '{}_crf_{}_{}.ply'.format(
            os.path.splitext(filename)[0], stamp, args.n_segs
        )
        labelled_data.write(os.path.join(args.output_path, out_filename))
        print('Written {}'.format(out_filename))

        out_filename = '{}_init_{}_{}.ply'.format(
            os.path.splitext(filename)[0], stamp, args.n_segs
        )
        init_data.write(os.path.join(args.output_path, out_filename))
        print('Written {}'.format(out_filename))


if __name__ == '__main__':
    args = build_parser().parse_args()
    main(args)
