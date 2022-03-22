import argparse
import toml
from sklearn import metrics
from time import time
from datetime import timedelta
from images_loader import ImagesLoader
from loader import Loader
import pandas as pd
import logging
import os
import numpy as np

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s",
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class Tester:
    """
    Class to test XGBoost modeling.
    """

    def __init__(self, config):
        """
        Different ways to load the model and data to test
        """
        self.model = None
        if "model" in config:
            self.model = config["model"]
        elif "model_file" in config:
            self.model = Loader(config["model_file"]).load_model()
        else:
            logger.error("Model is not loaded!")

        self.nlabels = config.get("nlabels", 2)
        self.X_test, self.y_test = None, None
        self.dp = None
        if "X_test" in config and "y_test" in config:
            self.X_test = config["X_test"]
            self.y_test = config["y_test"]
            assert all(
                [x in config for x in ["imgname_test", "imglabel_test"]]
                ), "Must include image names and labels."
            self.imgname_test = config["imgname_test"]
            self.imglabel_test = config["imglabel_test"]
        elif "images" in config:
            self.dp = ImagesLoader(config["images"])
            dt_dict = self.dp.load_images()
            # all images are used for testing, so combine train and test
            # prepared by data dictionary
            self.X_test = np.concatenate(
                [dt_dict["X_train"], dt_dict["X_test"]], axis=0
                )
            self.y_test = np.concatenate(
                [dt_dict["y_train"], dt_dict["y_test"]], axis=0
                )
            self.nlabels = self.dp.nlabels
            self.imgname_test = np.concatenate(
                [dt_dict["imgname_train"], dt_dict["imgname_test"]]
                )
            self.imglabel_test = np.concatenate(
                [dt_dict["imglabel_train"], dt_dict["imglabel_test"]]
                )

        self.out_metric = config.get("out_metric", None)
        if self.out_metric:
            os.makedirs(os.path.dirname(self.out_metric), exist_ok=True)

        self.out_res = config.get("out_res", None)
        if self.out_res:
            os.makedirs(os.path.dirname(self.out_res), exist_ok=True)

    def test(self):
        """
        Test the XGBoost model and calculate the relevant performance metrics.
        """
        if self.model is None:
            logger.error("Model not found!")
            return None

        if self.X_test is None or self.y_test is None:
            logger.error("Test data is not found!")
            return None

        # y_prob = self.model.predict_proba(self.X_test)
        y_pred = self.model.predict(self.X_test)
        predictions = [np.argmax(value) for value in y_pred]
        y_test_idx = [np.argmax(y) for y in self.y_test]

        accuracy = metrics.accuracy_score(y_test_idx, predictions)
        recall = metrics.recall_score(y_test_idx, predictions,
                                      average="weighted")
        precision = metrics.precision_score(y_test_idx, predictions,
                                            average="weighted")
        logger.info("Accuracy: %.2f%%" % (accuracy * 100.0))
        logger.info("Recall: %.2f%%" % (recall * 100.0))
        logger.info("Precision: %.2f%%" % (precision * 100.0))
        # f1score = metrics.f1_score(self.y_test, predictions,
        f1score = metrics.f1_score(y_test_idx, predictions,
                                   average="weighted")
        logger.info("F1 score: %.2f%%" % (f1score * 100.0))
        # cm = metrics.confusion_matrix(self.y_test, predictions)
        cm = metrics.confusion_matrix(y_test_idx, predictions)
        logger.info("Confusion Matrix: ")
        for truth_row in cm:
            logger.info("[ %s ]" % " ".join(["%d" % j for j in truth_row]))

        logger.info("End of testing model!")

        metric_dict = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "accuracy": round(accuracy, 3),
            "f1score": round(f1score, 3),
        }

        for i in range(self.nlabels):
            for j in range(self.nlabels):
                metric_dict["confusion_matrix_%d_%d" % (i, j)] = cm[i][j]

        y_dict = {
            "y_true": [v for v in y_test_idx],
            "y_pred": [v for v in predictions],
            "imgname": [name for name in self.imgname_test],
            "imglabel": [label for label in self.imglabel_test],
        }
        for ilb in range(self.nlabels):
            y_dict["y_prob_c%d" % ilb] = [p[ilb] for p in y_pred]
        if self.out_res is not None:
            df = pd.DataFrame(y_dict)
            df.to_csv(self.out_res, index=False)
            logger.info("Test sample prediction output: %s" % self.out_res)
        if self.out_metric is not None:
            df = pd.DataFrame(metric_dict, index=[0])
            df.to_csv(self.out_metric, index=False)
            logger.info("Test sample performance output: %s" % self.out_metric)

        return metric_dict


def main():
    t_start = time()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        default=None,
        type=str,
        required=True,
        help="Your model tester config file, e.g. config.toml.",
    )
    args = parser.parse_args()
    logger.info("Reading config file: %s" % args.config)
    config = toml.load(args.config)
    t = Tester(config)
    metric_dict = t.test()
    logger.info("Performance metrics for testing: %s" % str(metric_dict))

    tdif = time() - t_start
    logger.info("Testing model done!")
    logger.info("Time used: %s" % str(timedelta(seconds=tdif)))


if __name__ == "__main__":
    main()
