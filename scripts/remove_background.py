"""
This script will be used to remove noisy background from cars images to
improve the quality of our data and get a better model.
The main idea is to use a vehicle detector to extract the car
from the picture, getting rid of all the background, which may cause
confusion to our CNN model.
We must create a new folder to store this new dataset, following exactly the
same directory structure with its subfolders but with new images.
"""

import argparse
import os

from utils.detection import get_vehicle_coordinates
from utils.utils import walkdir
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Train your model.")
    parser.add_argument(
        "data_folder",
        type=str,
        help=(
            "Full path to the directory having all the cars images. Already "
            "splitted in train/test sets. E.g. "
            "`/home/app/src/data/car_ims_v1/`."
        ),
    )
    parser.add_argument(
        "output_data_folder",
        type=str,
        help=(
            "Full path to the directory in which we will store the resulting "
            "cropped pictures. E.g. `/home/app/src/data/car_ims_v2/`."
        ),
    )

    args = parser.parse_args()

    return args


def main(data_folder, output_data_folder):
    """
    Parameters
    ----------
    data_folder : str
        Full path to train/test images folder.

    output_data_folder : str
        Full path to the directory in which we will store the resulting
        cropped images.
    """

    os.makedirs(output_data_folder, exist_ok=True)
    for dirpath, dirs, files in os.walk(data_folder):
        for filename in files:
            img = cv2.imread(os.path.join(dirpath,filename))
            x1, y1, x2, y2 = get_vehicle_coordinates(img)
            roi = img[y1:y2,x1:x2]
            classes,_= os.path.split(dirpath)
            _,class_ = os.path.split(classes)
            _,sub_classes =os.path.split(dirpath)
            new_path = os.path.join(output_data_folder, class_, sub_classes)
            os.makedirs(new_path, exist_ok=True)
            path_roi = os.path.join(new_path, filename)
            cv2.imwrite(path_roi, roi)
        
if __name__ == "__main__":
    args = parse_args()
    main(args.data_folder, args.output_data_folder)
