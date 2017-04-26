"""This is trying to implent the 2D edge detection using our own method."""

from urllib.request import urlretrieve
from hed.hed_model import HEDModel
from scipy.misc import imresize
from PIL import Image
import numpy as np
import sys
import os
# import ipdb


def get_gradient_image(image, hed_model):
    hed_pred_out = hed_model.predict(image)
    # Fuse layer as the output
    gradient_image = hed_pred_out[-1].squeeze()
    gradient_image = imresize(gradient_image, image.shape[:2])
    return gradient_image


def initialise_hed():
    # Load pretrained HED model of NYUD in Lasagne

    models_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'models'
    )
    hed_model_file = os.path.join(
        models_dir,
        'model_hed_caffe.pkl'
    )

    if not os.path.isfile(hed_model_file):
        print('Model file does not exist')
        hed_key = 'HED_MODEL_URL'
        if hed_key in os.environ.keys():
            print('Downloading HED model')
            if not os.path.isdir(models_dir):
                os.makedirs(models_dir)
            urlretrieve(
                os.environ[hed_key],
                hed_model_file
            )
        else:
            print(
                '{} is not set in your environment, '
                'cannot download model. Exiting.'.format(hed_key)
            )
            sys.exit(1)

    hed_model = HEDModel(hed_model_file)

    return hed_model


def main():
    # %% set the environment variable
    os.environ['HED_MODEL_URL'] = \
        ('http://digitalbridge-gpu-models.s3-accelerate.amazonaws.com'
            '/v1/model_hed_caffe.pkl')
    # ipdb.set_trace()
    # %%load and show the image
    image = Image.open('messi5.jpg')
    image.show()
    image = np.array(image)

    hed_model = initialise_hed()
    prediction = get_gradient_image(image, hed_model)
    boundary = Image.fromarray(prediction * 255)
    boundary.show()


if __name__ == '__main__':
    main()
