"""use function from server to extract the boundary from 2D images and show."""
from PIL import Image
import numpy as np
from routines import segment_image
from routines import utilities
import argparse


def read_to_array(image_directory):
    """Load the image and change it to array.

    Arguments:
    image_directory: the full string of the image directory.

    Return: the image array.
    """
    image = Image.open(image_directory)
    image_array = np.array(image)
    return image_array


def detect_boundary(image_array):
    """Detect boundary on the image.

    Arguments:
    image_array: the image array.

    Return: the boundary image.
    """
    boundary_tuple = segment_image(image_array)
    boundary_image_dic = boundary_tuple[1]['boundary_image']
    boundary_image = utilities.image_string_to_numpy(boundary_image_dic[1])
    return boundary_image


def image_show(image_array):
    """Show the image.

    Arguments:
    image_array: the image array.

    No return.
    """
    # change from the range 0-1 to 0-255 for display.
    boundary_show = Image.fromarray(image_array * 255)
    boundary_show.show()


def build_parser():
    """Input arguments."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        type=str,
        help='A image to work on',
        dest='imagename'
    )

    return parser


def main():
    """Main function."""
    args = build_parser().parse_args()
    image_array = read_to_array(args.imagename)
    boundary_image = detect_boundary(image_array)
    image_show(boundary_image)


if __name__ == '__main__':
    main()
