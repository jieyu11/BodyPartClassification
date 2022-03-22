import logging
import argparse
import numpy as np
from time import time
from datetime import timedelta
from model_inference_base import ClassificationModelInference
from libs_internal.io.h5 import (load_h5_group_vals, load_h5_group_names)

logger = logging.getLogger('luigi-interface')
logger.setLevel(logging.INFO)


class ShoesFeetModelInference(ClassificationModelInference):
    """
    Class to make inference on the shoes vs feet classification model by
    providing the preprocess video file and the keypoints file.
    """

    def __init__(self, model_file, preprocess_file, keypoint_file,
                 output_filename):
        """
        Initialization of the ShoesFeetModelInference model.

        Parameters:
            model_file: (str) input model file name
            preprocess_file: (str) preprocessing file name
            keypoint_file: (str) keypoint file name
            output_filename: (str) output filename
        """
        super().__init__(model_file, preprocess_file, keypoint_file,
                         output_filename)
        # the labels that are associated to the model output,
        # by default, set "shoes" for index=0 and "feet" for index=1
        self.class_labels = ["shoes", "feet"]
        # if true, then use keypoints to crop the initial image.
        # This helps to reduce the noise from background
        self.imgKeypointCrop = True

    def _get_keypoints(self):
        """
        Get the keypoints nparray for knees and ankles. The keypoints map e.g.:
            0: nose          1: left_eye       2: right_eye      3: left_ear
            4: right_ear     5: left_shoulder  6: right_shoulder 7: left_elbow
            8: right_elbow   9: left_wrist    10: right_wrist   11: left_hip
            12: right_hip   13: left_knee     14: right_knee    15: left_ankle
            16: right_ankle
        """
        try:
            # get the keypoints names np.array, e.g. ["nose", "left_eye"]
            logger.info("Getting keypoints in: %s" % self.keypoint_file)
            kp_names = load_h5_group_vals(self.keypoint_file, tag="info",
                                          names=["twoD_keypoints"])[0]
            # convert the binary string to string
            kp_names = [kp.decode('ascii') for kp in kp_names]
            frame_names = load_h5_group_names(self.preprocess_file,
                                              tag="data")
            # keypoints is a 2D np.array like [[x1, y1, _], [x2, y2, _], ...]
            keypoints = load_h5_group_vals(self.keypoint_file, tag="data",
                                           names=frame_names[0:1])[0]
            kp_list = ["left_knee", "right_knee", "left_ankle", "right_ankle"]
            assert all([k in kp_names for k in kp_list]), \
                "Must have %s in keypoints" % str(kp_list)
            # keypoint index for those in the list above
            kp_index = [i for i, kp in enumerate(kp_names) if kp in kp_list]
            logger.info("keypoints index %s found for %s" %
                        (str(kp_index), str(kp_list)))

            # only keep the keypoints relevant to shoes-vs-feet
            keypoints = keypoints[kp_index]
            return keypoints
        except Exception:
            logger.error("Fail to retrieve the keypoints!")
            return None

    def make_inference(self):
        """
        Given the input image, return the classification inference result.
        """
        keypoints = None
        if self.imgKeypointCrop:
            # get the keypoints np.array for the left/right knees and ankles
            # the 2D np.array like [[x1, y1, _], [x2, y2, _], ...]
            keypoints = self._get_keypoints()

        # Get the first frame in the video, if keypoints (knees and ankles) is
        # not None, then use expansion_factor=0.5 to expand the focus area.
        # img_raw = self._read_frame(keypoints, expansion_factor=0.5)
        img_raw = self._read_frame(keypoints, expansion_factor=1.0)
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
    t = ShoesFeetModelInference(
        args.model, args.input, args.keypoint_file, args.output)
    t.make_inference()

    tdif = time() - t_start
    logger.info("Time used: %s" % str(timedelta(seconds=tdif)))


if __name__ == "__main__":
    main()
