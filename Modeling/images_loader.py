import os
from sklearn.model_selection import train_test_split
import logging
from random import random, randint
import time
import cv2
import numpy as np

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s",
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class ImagesLoader:
    """
    Class to prepare training/testing/validation data sets.
    """

    def __init__(self, config):
        """
        Initialize the data preparation parameters. The following
        """

        self.folders = config["folders"]
        self.labels = config["labels"]
        self.nlabels = len(config["labels"])
        self.intlabels = {i: self.labels[i] for i in range(len(self.labels))}
        self.fTrain = config.get("fraction_train", 0.8)
        if self.fTrain < 1.e-5 or self.fTrain > 0.95:
            logger.warning("fraction_train=%f need to be in [0., 0.95]" %
                           self.fTrain)
        # fraction of validation out of test
        self.fValid = config.get("fraction_valid", -1.)
        logger.info("Data label and categorical number:")
        for ilabel, label in zip(self.intlabels, self.labels):
            logger.info("    %s : %d" % (label, ilabel))

        # split random state, use time in seconds unless it is fixed in config
        self.rndstate = randint(1, int(time.time())) + int(random()*1e4)
        if "random_state" in config:
            self.rndstate = config["random_state"]
        self.imgSize = config.get("image_size", 100)
        logger.info("Images are converted to %d x %d pixels." %
                    (self.imgSize, self.imgSize))
        # if image_padding is true, the use image padding or cropping.
        # otherwise, just perform a resizing
        self.imgPadding = config.get("image_padding", False)

        self.imgNorm = config.get("image_normalize", False)
        if self.imgNorm:
            logger.info("Image values are normalized to [0, 1.]")
        # decide how many images in the folder to read, if it is <=0, then
        # read all images
        self.n_images = config.get("n_images", -1)
        logger.info("Reading maximum of %d images from each input folder." %
                    self.n_images)

    def _convert_to_square(self, img):
        """
        If input image is not square, then convert it to square by padding to
        the lower dimension side.
        """
        height, width, color = img.shape
        if height == width:
            return img

        # 0 padding
        img_pad = np.zeros((max(width, height), max(width, height), color))
        offset = abs(width - height) // 2
        if height < width:
            img_pad[offset:height+offset, :] = img
        else:
            img_pad[:, offset:width+offset] = img
        return img_pad

    def _image_crop_or_pad(self, img):
        """
        Crop or pad the image to expected size.
        """
        # convert to square image if it is not already
        img = self._convert_to_square(img)
        height, width, color = img.shape
        assert height == width, "assuming same height and width n pixels"

        if height == self.imgSize:
            return img
        if height < self.imgSize:
            # 0 padding
            img_pad = np.zeros((self.imgSize, self.imgSize, color))
            offset = (self.imgSize - height) // 2
            img_pad[offset:height+offset, offset:height+offset] = img
            return img_pad

        # for height > self.imgSize, do center crop
        offset = (height - self.imgSize) // 2
        img_crop = img[offset:self.imgSize+offset, offset:self.imgSize+offset]
        return img_crop

    def _load(self, folder, label, ilabel):
        """
        Initialize
        """
        img_list = []
        # looping over the images from the giving folder.
        logger.info("Start to read max of %d images from folder: %s." %
                    (self.n_images, folder))
        for idx, img_name in enumerate(os.listdir(folder)):
            if self.n_images > 0 and idx >= self.n_images:
                logger.info("Reached the amount of images to read: %d." % idx)
                break

            imgfile = os.path.join(folder, img_name)
            # using img[...,::-1] to convert BGR to RGB format
            img = cv2.imread(imgfile)
            if img is None:
                logger.warning("%s has error! Skip!" % imgfile)
                continue

            logger.debug("reading inputs: %s with shape %s." %
                         (os.path.join(folder, img_name), str(img.shape)))

            if self.imgNorm:
                img = cv2.normalize(img, None, alpha=0, beta=1,
                                    norm_type=cv2.NORM_MINMAX,
                                    dtype=cv2.CV_32F)
            if self.imgPadding:
                img = self._image_crop_or_pad(img)
            try:
                resized = cv2.resize(img, (self.imgSize, self.imgSize))
            except Exception:
                logger.error("Image %s resizing failed!" % imgfile)
                continue

            img_list.append([resized, img_name, label, ilabel])
        logger.info("Read %d images from: %s." % (len(img_list), folder))

        return img_list

    def load_images(self):
        """
        Random splitting data based on the values set in the splits.
        """

        # split based on each label, say for each label 80% is training, 20%
        # testing, etc.
        logger.info("Data splitting random state: %d" % self.rndstate)
        data_train, data_valid, data_test = [], [], []
        _fvalid = True if self.fValid > 0 and self.fValid < 0.99 else False
        for fold, lb, ilb in zip(self.folders, self.labels, self.intlabels):
            logger.info("Loading images in folder %s with label %s (%d)" %
                        (fold, lb, ilb))
            data = self._load(fold, lb, ilb)
            logger.info("Data loaded...")

            _dt_train, _dt_valid_test = train_test_split(
                data, train_size=self.fTrain,
                shuffle=True,
                random_state=self.rndstate,
            )
            # train data
            data_train.extend(_dt_train)
            logger.info("Amount of training data: %d" % len(data_train))

            if _fvalid:
                # half-half for validation and testing dataset
                _dt_valid, _dt_test = train_test_split(
                    _dt_valid_test, train_size=self.fValid,
                    shuffle=True,
                    random_state=self.rndstate,
                )
                # validation data
                data_valid.extend(_dt_valid)
            else:
                data_valid = None
                _dt_test = _dt_valid_test
            # test data
            data_test.extend(_dt_test)
            logger.info("Amount of testing data: %d" % len(data_test))

        N = self.nlabels
        data_dict = {
            # image is the first element of the array
            # Y integer label is the last element
            "X_train": np.asarray([dt[0] for dt in data_train]),
            "y_train": np.asarray([dt[-1] for dt in data_train]),
            "X_test": np.asarray([dt[0] for dt in data_test]),
            "y_test": np.asarray([dt[-1] for dt in data_test]),
            # for test data, record also the image name and label
            "imgname_train": np.asarray([dt[1] for dt in data_train]),
            "imglabel_train": np.asarray([dt[2] for dt in data_train]),
            "imgname_test": np.asarray([dt[1] for dt in data_test]),
            "imglabel_test": np.asarray([dt[2] for dt in data_test]),
        }
        if _fvalid:
            data_dict["X_valid"] = np.asarray([dt[0] for dt in data_valid])
            data_dict["y_valid"] = np.asarray([dt[-1] for dt in data_valid])
            data_dict["imgname_valid"] = \
                np.asarray([dt[1] for dt in data_valid])
            data_dict["imglabel_valid"] = \
                np.asarray([dt[2] for dt in data_valid])

        if self.imgNorm:
            logger.info("Normalizing image values into [0., 1.]")
            for key in [k for k in data_dict if "X_" in k]:
                data_dict[key] = data_dict[key] / 255.

        for v in [k for k in data_dict if "y_" in k]:
            data_dict[v] = np.squeeze(np.eye(N)[data_dict[v].reshape(-1)])

        logger.info("Data images for modeling are ready!")

        return data_dict
