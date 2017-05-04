"""use function from server to extract the boundary from 2D images and show."""
from PIL import Image
import numpy as np
from routines import segment_image
from routines import utilities

# image = Image.open('image02113.png')
# imgar = np.array(image)
# boundary_tuple = segment_image(imgar)
# boundary_image_dic = boundary_tuple[1]['boundary_image']
# boundary_image = utilities.image_string_to_numpy(boundary_image_dic[1])
# boundary_show = Image.fromarray(boundary_image * 255)  # 0-1
# boundary_show.show()


def read_to_array(image_directory):
    """Load the image and change it to array.

    Keyword arguments:
    image_directory: the full string of the image directory.

    Return: the image array.
    """
    image = Image.open(image_directory)
    image_array = np.array(image)
    return image_array


def detect_boundary(image_array):
    """Detect boundary on the image.

    Keyword arguments:
    image_array: the image array.

    Return: the boundary image.
    """
    boundary_tuple = segment_image(image_array)
    # Generally use brackets to get values out of dictionaries.
    # i.e. boundary_tuple[1]['boundary_image'].
    # Doing that will throw an exception if the key is not present
    # (specifically a KeyError), whereas get will return None,
    # and may be harder to debug.
    boundary_image_dic = boundary_tuple[1]['boundary_image']
    boundary_image = utilities.image_string_to_numpy(boundary_image_dic[1])
    return boundary_image


def image_show(image_array):
    """Show the image.

    Keyword arguments:
    image_array: the image array.

    No return.
    """
    boundary_show = Image.fromarray(image_array * 255)  # 0-1
    boundary_show.show()


def main():
    """Main function."""
    imagedir = 'image02113.png'
    image_array = read_to_array(imagedir)
    boundary_image = detect_boundary(image_array)
    image_show(boundary_image)


if __name__ == '__main__':
    main()
