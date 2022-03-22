import argparse
import toml
from trainer import Trainer
from time import time
from datetime import timedelta
import logging
import os

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s",
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class HyperTuner:
    """
    Class to tune hyper parameters of EfficientNetB? modeling.
    """

    # supported parameters for parameter tuning
    tunables = {
        "image_size": int, "nepoch": int,
        "pre_model_name": str, "learning_rate": float
        }

    def __init__(self, config):
        """
        Initialize the run config and the hyper-parameters for tuning.
        """

        # model after training
        self.config_initial = config
        logger.info("Loading trainer for initilization.")
        self.trainer = Trainer(config)

    def tune(self, parameter_name, parameter_values):
        """
        Tune the giving parameter and values.
        """

        assert parameter_name in self.tunables.keys(), \
            "Parameter must be in [%s]" % ",".join(self.tunables.keys())

        vtype = self.tunables[parameter_name]
        # update the config file
        for val in parameter_values:
            if vtype is not str:
                val = vtype(val)
                logger.info("Param: %s -> %s." % (parameter_name, str(vtype)))

            # copy the original configuration
            config = self.config_initial.copy()
            # update the corresponding parameter
            config[parameter_name] = val
            logger.info("Tuning parameter: %s with %s" %
                        (parameter_name, str(val)))
            # also update the output names:
            for outkey in ["out_model", "out_metric_test",
                           "out_res_test", "out_history"]:
                if outkey not in config:
                    continue
                # split the name, e.g. my.model.h5 into [my, model, h5]
                keyslist = config[outkey].split(".")
                # put the parameter to second to last item:
                # model -> model_image_size_256
                assert len(keyslist) >= 2, "The '.' not in %s!" % outkey
                keyslist[-2] += "_%s-%s" % (parameter_name, str(val))
                # plug them together
                config[outkey] = ".".join(keyslist)
                logger.info("Output %s renamed: %s" % (outkey, config[outkey]))

            # update the config file for the trainer.
            self.trainer._init_parameters(config)

            # do the fitting testing
            self.trainer.train()
            self.trainer.test()


def main():
    t_start = time()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", default=None, type=str, required=True,
        help="Your trainer config toml file, e.g. config.toml.",
    )
    parser.add_argument(
        "--parameter", "-p", default=None, type=str, required=True,
        help="Your parameter to tune. ",
    )
    parser.add_argument(
        "--values", "-v", nargs="+",
        required=True, help="A list of parameters to be tuned.")

    args = parser.parse_args()

    logger.info("Reading trainer config file: %s" % args.config)
    config = toml.load(args.config)
    t = HyperTuner(config)
    t.tune(args.parameter, args.values)
    tdif = time() - t_start
    logger.info("Time used: %s" % str(timedelta(seconds=tdif)))


if __name__ == "__main__":
    main()
