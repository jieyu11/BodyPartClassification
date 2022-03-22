import cv2
import h5py
import logging
import numpy as np
from tensorflow.keras.models import load_model
from libs_internal.io.h5 import (load_h5_group_vals, load_h5_group_names)

logger = logging.getLogger('luigi-interface')
logger.setLevel(logging.INFO)


class ClassificationModelInference:
    """
    Class to make inference on a generic binary classification model by
    providing the preprocess video file and the keypoints file.
    """

    def __init__(self, model_file, preprocess_file, keypoint_file,
                 output_filename):
        """
        Initialization of ClassificationModelInference class.

        Parameters:
            model_file: (str) input model file name
            preprocess_file: (str) preprocessing file name
            keypoint_file: (str) keypoint file name
            output_filename: (str) output filename
        """
        self.model = load_model(model_file)
        assert self.model, "Fail to load model: %s" % model_file
        logger.info("Loading the input model file: %s" % model_file)
        self.preprocess_file = preprocess_file
        self.keypoint_file = keypoint_file
        logger.info("Preprocess file: %s" % self.preprocess_file)
        logger.info("Keypoint file: %s" % self.keypoint_file)
        self.imgSize = 256
        logger.info("Image size converted to %d x %d for model inference" %
                    (self.imgSize, self.imgSize))
        self.output_filename = output_filename
        logger.info("Output filename: %s" % output_filename)

    def _keypointcropped_image(self, img, keypoints, expansion_factor):
        """
        Read the input frame image from preprocess video but using 2D
        keypoints to find the edges in X and Y axis to crop the initial
        image.
        Parameters:
            img: (3D np.array) the input image
            keypoints: (2D np.array) for the given frame keypoints is a 2D
                np.array like [[x1, y1, _], [x2, y2, _], ...]
            keypoint expansion_factor: (float in 0. - 1.) to expand the cropped
                image with max/min of the keypoints.
        """
        try:
            # the maximum and minumum (x, y, _) values for all keypoints
            # e.g. kp_max[0] is the maximum in x direction
            kp_max = keypoints.max(axis=0)
            kp_min = keypoints.min(axis=0)
            width, height = kp_max[0] - kp_min[0], kp_max[1] - kp_min[1]
            x_start = max(0, int(kp_min[0] - width * expansion_factor))
            x_end = int(kp_max[0] + width * expansion_factor)
            y_start = max(0, int(kp_min[1] - height * expansion_factor))
            y_end = int(kp_max[1] + height * expansion_factor)
            # now get the frame and crop it
            # somehow x and y are swapped in the images...
            img_crop = img[y_start:y_end+1, x_start:x_end+1]

            # return the cropped image with shoes up to knees.
            logger.info("Cropped image shape: %s" % str(img_crop.shape))
            return img_crop
        except Exception:
            logger.error("Image keypoint crop not found! Return None!")
            return None

    def _read_frame(self, keypoints=None, expansion_factor=0.2):
        """
        Read the input frame image from preprocess video.
        """
        try:
            frame_names = load_h5_group_names(self.preprocess_file,
                                              tag="data")
            images = load_h5_group_vals(self.preprocess_file,
                                        tag="data", names=frame_names)
            assert len(images) > 0, "Must have at leaset 1 image."
            logger.info("Amount of the images found: %d" % len(images))
            # take the first image out of the image list
            img = images[0]
            logger.info("Reading the 1st image in %s." % self.preprocess_file)
            if keypoints is not None:
                img_kp = self._keypointcropped_image(img, keypoints,
                                                     expansion_factor)
                if img_kp is None:
                    logger.error("Keypoints not found, use raw image.")
                else:
                    img = img_kp
            return img
        except Exception:
            logger.error("No image is found in %s! Return None!" %
                         self.preprocess_file)
            return None

    def inference(self, img_raw, labels):
        """
        Test the XGBoost model and calculate the relevant performance metrics.
        Parameters:
            img_raw: (3D list of int) initial frame before resized to the
                target size.
            labels: (list of str) indicating the labels of classes. For
                example, if the model is a classification of two classes
                "shoes" (index=0) vs "feet" (index=1), then labels=["shoes",
                "feet"]
        """
        if self.model is None:
            logger.error("Model not found!")
            return None
        assert isinstance(labels, list), "labels must be a list"
        assert len(labels) == 2, "labels must have size of 2!"
        logger.info("The labels for classification: %s." % str(labels))

        # convert image to corresponding size
        img = cv2.resize(img_raw.copy(), (self.imgSize, self.imgSize))
        prediction = self.model.predict(np.asarray([img]))
        logger.info("Prediction: %s (0:%s; 1:%s)" %
                    (str(prediction), labels[0], labels[1]))
        logger.info("Shoes vs feet class definition is in model_train.toml")
        labels_ascii = [lb.encode('ascii') for lb in labels]
        labels_comb = "%s_vs_%s" % (labels[0], labels[1])
        with h5py.File(self.output_filename, 'w') as hf:
            dt = hf.create_group('data')
            # save the frame used for classification for debugging if needed.
            dt.create_dataset('frame_original', data=img_raw)
            dt.create_dataset('frame_converted', data=img)
            # there is only one image in the list, so use [0] for that image.
            dt.create_dataset('prediction', data=prediction[0])
            # "some-string".encode("ascii") is equvalent to b"some-string"
            dt.create_dataset("labels", data=labels_ascii)
        logger.info("Output file saved to: %s" % self.output_filename)

        # also update to preprocessing file
        with h5py.File(self.preprocess_file, 'r+') as hf:
            # there is only one image in the list, so use [0] for that image.
            if "classification" in hf:
                dt = hf["classification"]
                if "%s_labels" % labels_comb in dt:
                    del hf["classification/%s_labels" % labels_comb]
                    del hf["classification/%s_prediction" % labels_comb]
            else:
                dt = hf.create_group("classification")

            dt.create_dataset("%s_labels" % labels_comb, data=labels_ascii)
            dt.create_dataset("%s_prediction" % labels_comb,
                              data=prediction[0])
        logger.info("Prediction also updated to: %s" % self.preprocess_file)

        return prediction
