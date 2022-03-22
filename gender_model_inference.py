import logging
import argparse
import numpy as np
from time import time
from datetime import timedelta
from model_inference_base import ClassificationModelInference
from libs_internal.io.h5 import (load_h5_group_vals, load_h5_group_names)

logger = logging.getLogger('luigi-interface')
logger.setLevel(logging.INFO)


class GenderModelInference(ClassificationModelInference):
    """
    Class to make inference on the shoes vs feet classification model by
    providing the preprocess video file and the keypoints file.
    """

    def __init__(self, model_file, preprocess_file, keypoint_file,
                 output_filename):
        """
        Initialization of the GenderModelInference model.

        Parameters:
            model_file: (str) input model file name
            preprocess_file: (str) preprocessing file name
            keypoint_file: (str) keypoint file name
            output_filename: (str) output filename

        """
        super().__init__(model_file, preprocess_file, keypoint_file,
                         output_filename)
        # the labels that are associated to the model output,
        # by default, set "female" for index=0 and "male" for index=1
        self.class_labels = ["female", "male"]
        # if true, then use keypoints to crop the initial image.
        # This helps to reduce the noise from background
        self.imgKeypointCrop = True

    def make_inference(self):
        """
        Given the input image, return the classification inference result.
        """
        keypoints = None
        if self.imgKeypointCrop:
            frame_names = load_h5_group_names(self.preprocess_file,
                                              tag="data")
            # get the keypoints np.array for the given frame keypoints is a
            # 2D np.array like [[x1, y1, _], [x2, y2, _], ...]
            keypoints = load_h5_group_vals(self.keypoint_file, tag="data",
                                           names=frame_names[0:1])[0]

        # get the first frame in the video
        img_raw = self._read_frame(keypoints)
        if img_raw is None:
            logger.error("No image found. Return dummpy prediction.")
            return [0., 0.]
        return self.inference(img_raw, self.class_labels)


def main():
    t_start = time()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", "-m", default="model_shoes-vs-feet.h5", type=str,
        required=False, help="Your model h5 file.",
    )
    parser.add_argument(
        "--keypoint-file", "-k", default=None, type=str, required=True,
        help="Your 2D keypoint h5 file.",
    )
    parser.add_argument(
        "--input", "-i", default=None, type=str, required=True,
        help="Your input frames h5 file.",
    )
    parser.add_argument(
        "--output", "-o", default=None, type=str, required=True,
        help="Your output frames h5 file.",
    )
    args = parser.parse_args()
    t = GenderModelInference(
        args.model, args.input, args.keypoint_file, args.output)
    t.make_inference()

    tdif = time() - t_start
    logger.info("Time used: %s" % str(timedelta(seconds=tdif)))


if __name__ == "__main__":
    main()
