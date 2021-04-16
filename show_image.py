import argparse
import os

import h5py
import torch

from preprocess import (
    DATA_PATH,
    IMAGES_FILENAME,
    show_image,
)


def main(args):
    images = h5py.File(os.path.join(DATA_PATH, IMAGES_FILENAME[args.split]), "r")
    image_data = images[str(args.image_id)][()]

    image = torch.FloatTensor(image_data)

    features_scale_factor = 255
    image = image / features_scale_factor

    show_image(image)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", default="test", type=str, help="dataset split to use",
    )
    parser.add_argument(
        "--image-id", type=int, required=True,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
