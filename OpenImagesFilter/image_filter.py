import cv2
import argparse
import numpy as np
import pandas as pd
import logging
import os
from time import time
import toml
import glob
from datetime import timedelta
logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s",
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenImageFilter:
    """
    Class to filtering images from openimage database.
    """
    def __init__(self, config):
        """
        Initialization of input files.
        remove_duplicate: bool, if True images with more than one object
            selected on the image are removed.
        """
        label_object_filename = config.get("label_object_filename", None)
        assert label_object_filename, "Must setup label_object_filename \
            (e.g. class-descriptions-boxable) in config!"

        object_annotation_filename = config.get("object_annotation_filename", None)
        assert object_annotation_filename, "Must setup object_annotation_filename \
            (e.g. oidv6-train-annotations-vrd) in config!"
        self.remove_duplicate = config.get("remove_duplicate", False)
        self.image_input_dir = config.get("image_input_dir", None)
        assert self.image_input_dir, "Must setup image_input_dir for source images in config."
        self.image_output_dir = config.get("image_output_dir", None)
        assert self.image_output_dir, "Must setup image_output_dir for source images in config."
        self.output_image_filtered = config.get("output_image_filtered",
                                                "output/image_filtered.csv")
        # save the amount of images per object filtering.
        self.n_images = config.get("n_images", -1)
        self.n_min_pixels = config.get("n_min_pixels", 128)
        logger.info("Filtering images or cropped images with at least %d pixels in X and Y" %
                    self.n_min_pixels)

        self.df = pd.read_csv(object_annotation_filename)
        self.object_label_dict = self._get_object_label_mapping(label_object_filename)

    def _get_object_label_mapping(self, label_object_filename):
        """
        # label to object mapping df, like:
        # LabelName ObjectName
        # /m/03bt1vf      Woman
        # /m/04yx4        Man
        """
        df_lb = pd.read_csv(label_object_filename,
                        names = ["LabelName", "ObjectName"])
        # convert the df into dict
        object_label_dict = pd.Series(
            df_lb["LabelName"].values,
            index=df_lb["ObjectName"].values).to_dict()
        return object_label_dict

    def _get_filters(self, object_keep, object_avoid):
        """
        Get the filters given the name of the variables to keep and to avoid
        on the image.
        """

        if object_avoid:
            # find man on the image but no woman in the image
            Q1 = "(LabelName1==\"%s\" & LabelName2!=\"%s\")" % \
                (self.object_label_dict[object_keep],
                self.object_label_dict[object_avoid])
            Q2 = "(LabelName2==\"%s\" & LabelName1!=\"%s\")" % \
                (self.object_label_dict[object_keep],
                self.object_label_dict[object_avoid])
            logger.info("Object to avoid: %s with label: %s" % 
                (object_avoid, self.object_label_dict[object_avoid]))
        else:
            Q1 = "LabelName1==\"%s\"" % self.object_label_dict[object_keep]
            Q2 = "LabelName2==\"%s\"" % self.object_label_dict[object_keep]

        logger.info("Object to keep: %s with label: %s" % 
            (object_keep, self.object_label_dict[object_keep]))
        return Q1, Q2

    def _crop_images(self, df_img, img_idx, output_dir):
        """
        Crop the images with given information and save to the given output
        folder.
        df_img: DataFrame having image id, coordicates to crop for certain object.
            ImageID,LabelName1,LabelName2,XMin1,XMax1,YMin1,YMax1,XMin2,XMax2,YMin2,YMax2,RelationshipLabel
            000ac95750ac7399,/m/04yx4,/m/019nj4,0.049375,0.560625,0.08427,0.999064,0.049375,0.560625,0.08427,0.999064,is
            000ac95750ac7399,/m/04yx4,/m/019nj4,0.485625,0.65875,0.496255,0.999064,0.485625,0.65875,0.496255,0.999064,is
        img_idx: either 1 or 2, for 1 the corresponding object is LabelName1 and XMin1, etc.
        self.image_output_dir: output image directory.
        n_images: the number of images to be operated. If it is negative, then do all.
        """
        assert img_idx in [1, 2], "Image index has to be 1 or 2 for object 1 or 2."
        # assert isinstance(self.image_input_dir, list), "Need to setup image dir as list for image cropping."
        os.makedirs(output_dir, exist_ok=True)
        img_counts = 0
        for idx, row in df_img.iterrows():
            # assuming a jpg file, which is the case for OpenImages dataset
            filename = "%s/%s.jpg" % (self.image_input_dir, row["ImageID"])
            img = cv2.imread(filename)
            if img is None:
                continue
            height, width, _ = img.shape
            w_start = int(row["XMin%d"%img_idx] * width)
            w_end = int(row["XMax%d"%img_idx] * width)
            h_start = int(row["YMin%d"%img_idx] * height)
            h_end = int(row["YMax%d"%img_idx] * height)
            nw = w_end - w_start + 1
            nh = h_end - h_start + 1
            # apply filter on the number of pixels to the images.
            if nw < self.n_min_pixels or nh < self.n_min_pixels:
                continue
            img_crop = img[h_start:h_end+1, w_start:w_end+1]
            imgname = "%s_X-%04d-%04d_Y-%04d-%04d.jpg" % (row["ImageID"], w_start, w_end, h_start, h_end)
            imgoutname = output_dir + "/" + imgname
            cv2.imwrite(imgoutname, img_crop)
            img_counts += 1
            if self.n_images > 0 and img_counts >= self.n_images:
                break
            if img_counts % 100 == 0:
                logger.info("N images cropped %d for object %d" % (img_counts, img_idx))
        logger.info("Number of images %d cropped saved to %s" % (img_counts, output_dir))

    def select_images(self, object_keep=None, object_avoid=None):
        """
        For example: object_keep="Man" and object_avoid="Woman", then only
        select the images with Man, but no woman was on the image
        self.image_output_dir: str, images output directory
        Obj_dict: Object to label mapping,
            e.g. {"/m/04yx4": "Man", "/m/03bt1vf", "Woman"}
        """
        assert "ImageID" in self.df.columns, "Must have image ID column"
        assert object_keep and object_keep in self.object_label_dict, \
            "Must have column to keep (%s) in the dict" % object_keep
        if object_avoid: 
            assert object_avoid in self.object_label_dict, \
                "Must have column to avoid (%s) in the dict" % object_avoid
    
        Q1, Q2 = self._get_filters(object_keep, object_avoid)
        if self.image_output_dir:
            output_dir = self.image_output_dir + "/Cropped_" + object_keep
            os.makedirs(output_dir, exist_ok=True)
            logger.info("Output directory: %s" % output_dir)
            logger.info("Filtering with object 1: %s" % Q1)
            df1 = self.df.query(Q1)
            logger.info("Filtering with object 2: %s" % Q2)
            self._crop_images(df1, 1, output_dir)
            df2 = self.df.query(Q2)
            self._crop_images(df2, 2, output_dir)

        df_out = pd.concat([df1, df2], axis=1)
        if self.remove_duplicate:
            # And the image should have only just the object to keep
            df_out = df_out.drop_duplicates(subset="ImageID", keep=False)
        logger.info("Selected %d images of %s from %d objects." % 
              (len(df_out), object_keep, len(self.df)))
        
        # output txt file containing the filtered images.
        if self.output_image_filtered:
            if os.path.dirname(self.output_image_filtered):
                os.makedirs(os.path.dirname(self.output_image_filtered), exist_ok=True)
            df_out.to_csv(self.output_image_filtered, index=False)
            assert "ImageID" in df_out.columns, "Must have ImageID in output file columns."
            df_out[["ImageID"]].to_csv(self.output_image_filtered.replace(".csv", "_imageid.csv"),
                index=False)
        return df_out

        
def main():
    """
    Exampe:
    python3 src/image_filter.py \
        -l ../datasets/openimages/meta-info/class-descriptions-boxable.csv \
        -o ../datasets/openimages/meta-info/oidv6-train-annotations-vrd_partial.csv

    Outputs are under: output/.
    """
    t_start = time()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        default="config.toml",
        type=str,
        required=True,
        help="Your config toml file.",
    )
    parser.add_argument(
        "--object-keep",
        "-k",
        default=None,
        type=str,
        required=True,
        help="Your object to keep.",
    )
    parser.add_argument(
        "--object-avoid",
        "-a",
        default=None,
        type=str,
        required=False,
        help="Your object to avoid.",
    )
    args = parser.parse_args()
    logger.info("Reading config file: %s" % args.config)
    config = toml.load(args.config)
    logger.info("Configurations: %s" % str(config))
    logger.info("Object to keep: %s" % args.object_keep)
    logger.info("Object to avoid: %s" % args.object_avoid)
    imgflt = OpenImageFilter(config)
    imgflt.select_images(object_keep=args.object_keep, object_avoid=args.object_avoid)
    tdif = time() - t_start
    logger.info("Time used: %s" % str(timedelta(seconds=tdif)))


if __name__ == "__main__":
    main()