"""Map 3D points back to 2D img."""

import numpy as np
import map3Dto2D


def set_values_found_to_index(matrix_original, values_to_find):
    """Set value found to index."""
    temp = ~np.in1d(
        matrix_original, values_to_find).reshape(matrix_original.shape)
    matrix_output = np.searchsorted(
        values_to_find, matrix_original[~np.any(temp, axis=1), :]
        )

    return matrix_output


def generate_face_2D(faces, index, ind):
    """Get the new face index on 2D image."""
    # get the face index
    index1 = index.ravel().nonzero()[0]
    intersection = index1[ind]
    # intersection = np.array(sorted(intersection))

    temp = np.in1d(faces, intersection).reshape(faces.shape)
    row_index, _ = np.where(temp)
    row_unique = np.unique(row_index)
    facefinal = faces[row_unique, :]
    facefinal = set_values_found_to_index(facefinal, intersection)
    facefinal = facefinal[(facefinal <= len(intersection)).all(axis=1)]
    return facefinal


def main():
    """Main function."""
    pt21, pt2_color, faces, index, ind = map3Dto2D.main()
    face_2D = generate_face_2D(faces, index, ind)
    return pt21, pt2_color, face_2D


if __name__ == '__main__':
    pt21, pt2_color, face_2D = main()
