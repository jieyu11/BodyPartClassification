import argparse
import toml
from images_loader import ImagesLoader
from pretrained_models import load_pretrained_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from time import time
from datetime import timedelta
import logging
import os
import pandas as pd
from tester import Tester

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s",
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """
    Class to build EfficientNetB? modeling.
    """

    def __init__(self, config):
        """
        Initialize the run config and the hyper-parameters for tuning.
        """

        # model after training
        self.model = None

        # splitted training/validation/testing data
        # training data is retrieved: self.data_dict["X_train"], ["y_train"]
        # similarly for X_valid, X_test
        self.data_dict = None

        # instance of ImagesLoader class, which is used for loading the input
        # image files
        self.imgld = None

        # initialize the model parameters
        self._init_parameters(config)

    def _init_parameters(self, config):
        """
        Initialization of the parameters used in the model building. If any
        of the parameters needs to be updated, one can use a different config
        to do so.
        """
        assert "images" in config, "need to have image section in config."
        self.config_images = config["images"]
        self.image_size = self.config_images["image_size"]
        self.nlabels = len(self.config_images["labels"])
        assert self.nlabels >= 2, "At least 2 groups for classification!"
        # default output name: "model.h5"
        self.out_model = config.get("out_model", "model.h5")
        logger.info("Training model output name: %s" % self.out_model)
        self.retrain = config.get("retrain", True)
        self.nepoch = config.get("nepoch", 10)
        logger.info("Training with #epoch=%d" % self.nepoch)
        self.batch_size = config.get("batch_size", 32)
        logger.info("Training batch size: %d (reduce it if OOM)" %
                    self.batch_size)

        self.lr = config.get("learning_rate", 0.0001)
        logger.info("Optimizer learning rate: %f" % self.lr)

        self.out_metric_test = config.get("out_metric_test", None)
        if self.out_metric_test:
            os.makedirs(os.path.dirname(self.out_metric_test), exist_ok=True)

        self.out_res_test = config.get("out_res_test", None)
        if self.out_res_test:
            os.makedirs(os.path.dirname(self.out_res_test), exist_ok=True)

        self.out_history = config.get("out_history", None)
        if self.out_history:
            os.makedirs(os.path.dirname(self.out_history), exist_ok=True)

        # get the name of the pre-trained model
        self.pre_model_name = config.get("pre_model_name", "EfficientNetB0")
        logger.info("Setting the pre-trained model: %s" % self.pre_model_name)

    def _load_model(self):
        self.model = load_model(self.out_model)
        logger.info("Load existing model: %s" % self.out_model)

    def _get_model(self):
        logger.info("Init the pre-trained model: %s" % self.pre_model_name)
        model = load_pretrained_model(self.pre_model_name, self.image_size)
        if not model:
            logger.error("Pre-trained model: %s not found!" %
                         self.pre_model_name)
        return model

    def _train_model(self):
        logger.info("Start training model.")

        # loading data
        X_train = self.data_dict["X_train"]
        y_train = self.data_dict["y_train"]
        if "X_valid" in self.data_dict and "y_valid" in self.data_dict:
            X_valid = self.data_dict["X_valid"]
            y_valid = self.data_dict["y_valid"]
        else:
            X_valid = self.data_dict["X_test"]
            y_valid = self.data_dict["y_test"]

        # build model
        pretrained_net = self._get_model()
        assert pretrained_net, "Pretrained model is not found!"

        self.model = Sequential()
        self.model.add(pretrained_net)
        self.model.add(Dense(units=120, activation='relu'))
        self.model.add(Dense(units=120, activation='relu'))
        # output layer, 2 classes
        # Can also use Dense(1, activation='sigmoid'), but the y value
        #     format has to be integers instead of list of [0, 1]
        self.model.add(Dense(units=self.nlabels, activation='softmax'))

        optimizer = Adam(learning_rate=self.lr)
        loss = 'categorical_crossentropy',
        if self.nlabels == 2:
            loss = 'binary_crossentropy'
        # model compiling, can also use: optimizer='adam'
        self.model.compile(loss=loss, optimizer=optimizer,
                           metrics=['accuracy'])

        logger.info("Start fitting model!")
        history = self.model.fit(X_train, y_train,
                                 batch_size=self.batch_size,
                                 validation_data=(X_valid, y_valid),
                                 epochs=self.nepoch)
        if self.out_history:
            histdict = {k: history.history[k] for k in ["loss", "val_loss",
                                                        "accuracy",
                                                        "val_accuracy"]}
            histdict["epoch"] = [i for i in range(1, self.nepoch+1)]
            pd.DataFrame(histdict).to_csv(self.out_history, index=False)
            logger.info("History saved to %s." % self.out_history)

        logger.info("Save model as json file: %s" %
                    self.out_model.replace(".h5", ".json"))
        model_json = self.model.to_json()
        with open(self.out_model.replace(".h5", ".json"), "w") as json_file:
            json_file.write(model_json)

        # alternatively try to save only the weights:
        # self.model.save_weights(self.out_model)
        # which needs model.load_weights(/path/to/weight/file.h5) to load it
        self.model.save(self.out_model)
        logger.info("Model saved at: %s" % self.out_model)

    def train(self):
        """
        Train the XGBoost model with the given simulation data.
        """
        if self.config_images is None:
            logger.error("images for data split is not found in config. ")
            return None
        if self.imgld is None:
            self.imgld = ImagesLoader(self.config_images)
        if self.data_dict is None:
            self.data_dict = self.imgld.load_images()
        if not self.retrain:
            self._load_model()
        else:
            self._train_model()

    def test(self):
        """
        The input data is splitted into train/test(/valid), do a test on the
        testing dataset.
        """
        if self.model is None or self.data_dict is None:
            logger.error("No model or test data!")
            return None

        # do a test on the test data
        y_test = self.data_dict["y_test"]
        # setup the testing config
        tconfig = {
            "model": self.model, "X_test": self.data_dict["X_test"],
            "y_test": y_test, "nlabels": self.nlabels,
            "imgname_test": self.data_dict["imgname_test"],
            "imglabel_test": self.data_dict["imglabel_test"],
            "out_res": self.out_res_test
        }
        t = Tester(tconfig)

        # test part of data predicted result is saved in self.out_res_test
        # test result dict
        metric_dict = t.test()
        for key in metric_dict:
            logger.info("Metric %s: %g " % (key, metric_dict[key]))

        if self.out_metric_test is not None:
            df = pd.DataFrame(metric_dict, index=[0])
            df.to_csv(self.out_metric_test, index=False)
            logger.info("Saving performance metric: %s" %
                        self.out_metric_test)
        return None


def main():
    t_start = time()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        default=None,
        type=str,
        required=True,
        help="Your config toml file, e.g. config.toml.",
    )
    parser.add_argument(
        "--debug", dest='debug', action='store_true',
        required=False, help="Set debug=True for debug info")

    args = parser.parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    logger.info("Reading config file: %s" % args.config)
    config = toml.load(args.config)
    t = Trainer(config)
    t.train()
    t.test()
    tdif = time() - t_start
    logger.info("Time used: %s" % str(timedelta(seconds=tdif)))


if __name__ == "__main__":
    main()
