"""Compare 2 ply file ."""
from plyfile import PlyData
import argparse
import os.path
import sys
import numpy as np
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

    return parser


def merge_ply(mesh_file1_path, mesh_file2_path):
    """Merge 2 plys so that the ply contains the true color with labelling."""
    print("loading the first one...")
    # ipdb.set_trace()
    plydata = PlyData.read(mesh_file1_path)
    print("loading the second one...")
    plydata1 = PlyData.read(mesh_file2_path)
    # ipdb.set_trace()
    if('label' in plydata['vertex'].data.dtype.names):
        print("calculating...")
        plydata['vertex']['red'] = plydata1['vertex']['red']
        plydata['vertex']['green'] = plydata1['vertex']['green']
        plydata['vertex']['blue'] = plydata1['vertex']['blue']
        filename = os.path.basename(mesh_file1_path)
        out_filename_tmp = '{}_true_color_with_labelling.ply'.format(
            os.path.splitext(filename)[0]
        )
        out_filename = os.path.join(args.output_path, out_filename_tmp)
        print("writing to ply...")
        plydata.write(out_filename)
    elif('label' in plydata1['vertex'].data.dtype.names):
        print("calculating...")
        plydata1['vertex']['red'] = plydata['vertex']['red']
        plydata1['vertex']['green'] = plydata['vertex']['green']
        plydata1['vertex']['blue'] = plydata['vertex']['blue']
        filename = os.path.basename(mesh_file2_path)
        out_filename_tmp = '{}_true_color_with_labelling.ply'.format(
            os.path.splitext(filename)[0]
        )
        out_filename = os.path.join(args.output_path, out_filename_tmp)
        print("writing to ply...")
        plydata1.write(out_filename)
    else:
        sys.exit("****Check the -m files after!****")


def main(args):
    """Main function."""
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    print("Merging...")
    merge_ply(args.mesh_filepath[0], args.mesh_filepath[1])
    print("Done...")


if __name__ == '__main__':
    print("Make sure the files after -m are:")
    print("1. 2 file names that contain ABSOLUTE directories")
    print("2. must be ply!")
    args = build_parser().parse_args()
    main(args)
